#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SFT Training Script for Megatron Bridge

This script performs Supervised Fine-Tuning (SFT) using Megatron Bridge.
It loads a HuggingFace checkpoint, trains on JSONL data, and saves checkpoints.

Usage:
    Single node (8 GPUs):
        torchrun --nproc_per_node=8 scripts/sft_train.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --data-path /path/to/data_dir \
            --output-dir ./checkpoints

    Multi-node:
        torchrun --nnodes=2 --nproc_per_node=8 \
            --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            scripts/sft_train.py \
            --model Qwen/Qwen2.5-7B-Instruct \
            --data-path /path/to/data_dir \
            --tensor-parallel-size 4 \
            --pipeline-parallel-size 2

Data Format:
    The data file should be a JSONL file with chat-format messages:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Or standard SFT format:
    {"input": "...", "output": "..."}
"""

import argparse
import logging
import math
import os
from dataclasses import fields
from pathlib import Path

import torch
from tqdm import tqdm

# Configure logging with timestamp
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("megatron.core.distributed.param_and_grad_buffer").setLevel(
    logging.WARNING
)

# Enable debug logging for checkpointing
logging.getLogger("megatron.bridge.training.checkpointing").setLevel(logging.DEBUG)
logging.getLogger("megatron.core.dist_checkpointing").setLevel(logging.DEBUG)

from megatron.bridge import AutoBridge
from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
)
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.gpt_step import forward_step as gpt_forward_step
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory


class TqdmProgressCallback(Callback):
    """Callback to display tqdm progress bar during training."""

    def __init__(self, total_iters: int, log_interval: int = 1):
        """Initialize tqdm progress callback.

        Args:
            total_iters: Total number of training iterations
            log_interval: How often to update the progress bar
        """
        self.total_iters = total_iters
        self.log_interval = log_interval
        self.pbar = None
        self.current_loss = None

    def on_train_start(self, context: CallbackContext) -> None:
        """Initialize progress bar at training start."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0:
            start_iter = context.state.train_state.step
            self.pbar = tqdm(
                total=self.total_iters,
                initial=start_iter,
                desc="Training",
                unit="iter",
                dynamic_ncols=True,
            )

    def on_train_step_end(self, context: CallbackContext) -> None:
        """Update progress bar after each training step."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.pbar is not None:
            # Update progress bar
            self.pbar.update(1)

            # Update loss display if available
            if context.loss_dict:
                loss_val = context.loss_dict.get("lm loss", None)
                if loss_val is not None:
                    if hasattr(loss_val, "item"):
                        loss_val = loss_val.item()
                    self.current_loss = loss_val
                    self.pbar.set_postfix({"loss": f"{loss_val:.4f}"})

    def on_train_end(self, context: CallbackContext) -> None:
        """Close progress bar at training end."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and self.pbar is not None:
            self.pbar.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SFT Training with Megatron Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading HuggingFace model",
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to directory containing training.jsonl",
    )
    data_group.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: use model's max_position_embeddings)",
    )
    data_group.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to a Jinja2 chat template file to override the tokenizer's default chat template",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (mutually exclusive with --train-iters)",
    )
    train_group.add_argument(
        "--train-iters",
        type=int,
        default=1000,
        help="Number of training iterations (default: 1000)",
    )
    train_group.add_argument(
        "--global-batch-size",
        type=int,
        default=128,
        help="Global batch size (default: 128)",
    )
    train_group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size per GPU (default: 1)",
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    train_group.add_argument(
        "--min-lr",
        type=float,
        default=0.0,
        help="Minimum learning rate (default: 0.0)",
    )
    train_group.add_argument(
        "--lr-warmup-iters",
        type=int,
        default=50,
        help="Learning rate warmup iterations (default: 50)",
    )

    # Parallelism and model arguments - auto-generated from TransformerConfig
    # TransformerConfig inherits from ModelParallelConfig, so includes all parallelism settings
    # Uses ArgumentGroupFactory to automatically create args from the dataclass
    # Exclude list copied from _add_network_size_args in megatron/training/arguments.py
    transformer_exclude = [
        # cannot provide callables over CLI
        "timers",
        "finalize_model_grads_func",
        "grad_scale_func",
        "mtp_grad_scale_func",
        "no_sync_func",
        "grad_sync_func",
        "param_sync_func",
        "_cpu_offloading_context",
        "init_method",
        "output_layer_init_method",
        "embedding_init_method",
        "activation_func",
        # types affect docstring
        "pipeline_model_parallel_layout",
        "window_size",
        "window_attn_skip_freq",
        "no_rope_freq",
        "moe_layer_freq",
        "linear_attention_freq",
        "moe_router_load_balancing_type",
        "moe_aux_loss_coeff",
        "cp_comm_type",
        "cuda_graph_scope",
        "cuda_graph_modules",
        # no CLI argument exists for these
        "virtual_pipeline_model_parallel_size",
        "params_dtype",
        "enable_autocast",
        "autocast_dtype",
        "num_microbatches_with_partial_activation_checkpoints",
        "tp_comm_overlap_disable_qkv",
        "tp_comm_overlap_disable_fc1",
        "pipeline_dtype",
        "variable_seq_lengths",
        "batch_p2p_comm",
        "batch_p2p_sync",
        "deallocate_pipeline_outputs",
        "cpu_offloading",
        "cpu_offloading_activations",
        "cpu_offloading_weights",
        "cpu_offloading_double_buffering",
        "num_layers_in_first_pipeline_stage",
        "num_layers_in_last_pipeline_stage",
        "softmax_scale",
        "gated_linear_unit",
        "bias_activation_fusion",
        "activation_func_fp8_input_store",
        "test_mode",
        "memory_efficient_layer_norm",
        "fused_single_qkv_rope",
        "fp8_dot_product_attention",
        "fp8_multi_head_attention",
        "tp_only_amax_red",
        "use_kitchen",
        "moe_token_dropping",
        "cuda_graph_use_single_mempool",
        "cuda_graph_retain_backward_graph",
        "disable_parameter_transpose_cache",
        "inference_sampling_seed",
        "use_inference_optimized_layers",
        "heterogeneous_block_specs",
        "hetereogenous_dist_checkpoint",
        "quant_recipe",
        # deprecated and no CLI arg exists
        "tp_comm_atomic_ag",
        "tp_comm_atomic_rs",
        "moe_router_topk_limited_devices",
        # already generated by another config
        "inference_rng_tracker",
        "use_te_rng_tracker",
        "log_max_attention_logit",
        "barrier_with_L1_time",
        # args uses same var with a different name
        "num_moe_experts",
        "fp8_param",
        # incompatible defaults in dataclass
        "gradient_accumulation_fusion",
        "overlap_p2p_comm",
        "attention_softmax_in_fp32",
        "masked_softmax_fusion",
        "persist_layer_norm",
        "bias_dropout_fusion",
        "apply_rope_fusion",
    ]
    transformer_factory = ArgumentGroupFactory(
        TransformerConfig, exclude=transformer_exclude
    )
    transformer_factory.build_group(parser, "Model & Parallelism")

    # Checkpoint arguments
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--output-dir",
        type=str,
        default="./nemo_experiments/sft",
        help="Output directory for checkpoints and logs",
    )
    ckpt_group.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save checkpoint every N iterations, or every N epochs when --epochs is used (default: 100)",
    )
    ckpt_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory",
    )

    # Logging arguments
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (enables W&B logging)",
    )
    log_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    log_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name",
    )

    # Validation arguments
    val_group = parser.add_argument_group("Validation")
    val_group.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Evaluate every N iterations (default: 100)",
    )
    val_group.add_argument(
        "--eval-iters",
        type=int,
        default=10,
        help="Number of evaluation iterations (default: 10)",
    )

    # Advanced arguments
    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Random seed (default: 5678)",
    )
    advanced_group.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar",
    )

    # Fine-grained activation offloading (rednote contribution to mcore).
    # Module-level activation offload to CPU; the only activation-side offload
    # mechanism that works with PP>1 (cpu_offloading=True is PP=1 only).
    # Choose modules from {attn_norm, qkv_linear, core_attn, attn_proj,
    # mlp_norm, expert_fc1, moe_act}. Mutually exclusive with cpu_offloading.
    advanced_group.add_argument(
        "--fine-grained-offload-modules",
        type=str,
        default=None,
        help=(
            "Comma-separated list of submodules to offload to CPU during forward. "
            "Choices: attn_norm, qkv_linear, core_attn, attn_proj, mlp_norm, "
            "expert_fc1, moe_act. Enables fine_grained_activation_offloading."
        ),
    )

    return parser.parse_args()


def create_sft_config(args: argparse.Namespace) -> ConfigContainer:
    """Create SFT configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ConfigContainer with SFT configuration
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tb_logs"

    # Create model provider from HuggingFace model
    bridge = AutoBridge.from_hf_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    model_provider = bridge.to_megatron_provider(load_weights=True)

    # Copy all ModelParallelConfig fields from args to model_provider
    # This includes parallelism settings, precision, communication overlaps, etc.
    for field in fields(ModelParallelConfig):
        field_name = field.name
        if hasattr(args, field_name):
            arg_value = getattr(args, field_name)
            if arg_value is not None and hasattr(model_provider, field_name):
                setattr(model_provider, field_name, arg_value)

    # Set pipeline dtype if PP > 1
    if args.pipeline_model_parallel_size > 1:
        model_provider.pipeline_dtype = torch.bfloat16

    # calculate_per_token_loss=True multiplies the aux/router loss gradient attached
    # to router activations by (num_local_tokens * tp_cp_size) (router.py:506-511).
    # At long context + in-batch packing num_local_tokens is huge, scaling the router
    # gradient enormously. Disable for MoE to remove that amplifier. (Mcore default
    # is False; the dense path set it True but dense has no router/aux path.)
    model_provider.calculate_per_token_loss = (
        False if getattr(model_provider, "num_moe_experts", None) else True
    )

    # Disable Multi-Token Prediction (MTP) for SFT. Qwen3.5/3.6 HF configs carry
    # mtp_num_layers=1, which AutoBridge propagates. MTP is a pretrain-throughput
    # feature; for SFT it adds an extra prediction head whose gradients were
    # pathological on the 397B run (grad norm ~7.8e6 at iter 1 while lm loss was a
    # healthy 0.44 -> exploding grads came from the MTP/aux path, not the main LM),
    # causing inf-grad aborts / OOM. Turning it off removes that instability and is
    # standard for SFT. Set to None to fully disable.
    if getattr(model_provider, "mtp_num_layers", None):
        model_provider.mtp_num_layers = None
        print("Disabled MTP (mtp_num_layers=None) for SFT stability.")

    # MoE router stability for Qwen3.5-MoE SFT. Root cause of the grad explosion
    # (grad norm 1e6-1e13 + NaN by iter ~6-22, while the dense 27B with the SAME GDN
    # trains fine -> it's the MoE router path, not GDN/bf16):
    #   - No z-loss anywhere (moe_z_loss_coeff defaulted None) -> router logits grow
    #     unbounded -> topk-softmax grads blow up. z-loss is the standard ST-MoE fix
    #     that penalizes logsumexp(logits)^2 and keeps logits bounded.
    #   - global_aux_loss uses a running mean that is maximally imbalanced at SFT
    #     cold-start (256/512 experts) -> huge self-reinforcing aux gradient. The
    #     per-batch 'aux_loss' has no such pathological running statistic.
    # (Setting aux_loss_coeff=0 earlier only converted exponential blowup to a 1e6-1e8
    #  oscillation — it removed balancing pressure but left logits unbounded.)
    if getattr(model_provider, "num_moe_experts", None):
        model_provider.moe_z_loss_coeff = 1e-3
        model_provider.moe_aux_loss_coeff = 1e-3
        model_provider.moe_router_load_balancing_type = "aux_loss"
        print(
            "MoE router stability: z_loss=1e-3, aux_loss=1e-3, "
            "load_balancing_type=aux_loss (per-batch)."
        )

    # Keep the first/last transformer layers out of pure bf16. The Qwen3.5-MoE
    # provider sets num_layers_at_start/end_in_bf16=0 (all layers bf16); the Mcore
    # default is 1. The first (post-embedding) and last (pre-logit) layers are the
    # most prone to bf16 overflow -> NaN grads, which we saw on the MoE runs (grad
    # norm spiking to 1e8 then NaN). Restoring 1 keeps those layers higher-precision.
    if hasattr(model_provider, "num_layers_at_start_in_bf16"):
        model_provider.num_layers_at_start_in_bf16 = 1
    if hasattr(model_provider, "num_layers_at_end_in_bf16"):
        model_provider.num_layers_at_end_in_bf16 = 1

    # Configure recomputation if specified (from TransformerConfig)
    if args.recompute_granularity:
        model_provider.recompute_granularity = args.recompute_granularity
    if args.recompute_method:
        model_provider.recompute_method = args.recompute_method
    if args.recompute_num_layers is not None:
        model_provider.recompute_num_layers = args.recompute_num_layers

    # Fine-grained activation offloading (rednote contribution). Only path for
    # activation-to-CPU when PP>1 (cpu_offloading is PP=1 only). See
    # docs/megatron_parallel_constraints.md §13 (added) and the validator skill.
    if getattr(args, "fine_grained_offload_modules", None):
        modules = [
            m.strip() for m in args.fine_grained_offload_modules.split(",") if m.strip()
        ]
        model_provider.fine_grained_activation_offloading = True
        model_provider.offload_modules = modules
        print(
            f"Enabled fine_grained_activation_offloading for: {modules}"
        )

    # Determine data path
    data_path = Path(args.data_path)
    if not data_path.is_dir():
        raise ValueError(f"--data-path must be a directory, got: {data_path}")
    dataset_root = data_path

    dataset_kwargs = {}
    # Always use chat format with HF tokenizer chat template
    dataset_kwargs["chat"] = True
    dataset_kwargs["use_hf_tokenizer_chat_template"] = True

    # Load custom chat template if provided
    chat_template_content = None
    if args.chat_template:
        chat_template_path = Path(args.chat_template)
        if not chat_template_path.exists():
            raise FileNotFoundError(
                f"Chat template file not found: {args.chat_template}"
            )
        with open(chat_template_path, "r") as f:
            chat_template_content = f.read()

    # Create optimizer and scheduler config
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=args.lr_warmup_iters,
        lr_decay_iters=args.train_iters,
        max_lr=args.lr,
        min_lr=args.min_lr,
        adam_beta2=0.95,
    )

    # Configure wandb in logger if enabled.
    # Auto-generate a structured run name when --wandb-run-name is not given so wandb
    # runs are identifiable at a glance: <output-basename>-<tp>t<pp>p<ep>e-gbs<G>-seq<S>k.
    # E.g. myrun-pp6-2t6p8e-gbs36-seq16k
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    if args.wandb_run_name:
        wandb_exp_name = args.wandb_run_name
    else:
        seq_k = args.seq_length // 1024 if args.seq_length and args.seq_length >= 1024 else args.seq_length
        wandb_exp_name = (
            f"{Path(args.output_dir).name}"
            f"-{args.tensor_model_parallel_size}t"
            f"{args.pipeline_model_parallel_size}p"
            f"{args.expert_model_parallel_size}e"
            f"-gbs{args.global_batch_size}"
            f"-seq{seq_k}{'k' if args.seq_length and args.seq_length >= 1024 else ''}"
        )

    # Determine sequence length: use provided value or model's max_position_embeddings
    seq_length = (
        args.seq_length if args.seq_length is not None else model_provider.seq_length
    )

    # Update model provider's seq_length to match dataset (required for config validation)
    model_provider.seq_length = seq_length

    # Create the config container
    cfg = ConfigContainer(
        model=model_provider,
        train=TrainingConfig(
            train_iters=args.train_iters,
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        dataset=FinetuningDatasetConfig(
            dataset_root=str(dataset_root),
            seq_length=seq_length,
            seed=args.seed,
            memmap_workers=20,
            dataset_kwargs=dataset_kwargs,
            do_validation=False,
            do_test=False,
            dataloader_type="batch",
        ),
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=str(tensorboard_dir),
            log_timers_to_tensorboard=True,
            wandb_project=wandb_project,
            wandb_exp_name=wandb_exp_name,
            wandb_entity=wandb_entity,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=args.model,
            chat_template=chat_template_content,
        ),
        checkpoint=CheckpointConfig(
            save_interval=args.save_interval,
            save=str(checkpoint_dir),
            load=args.resume_from if args.resume_from else None,
            pretrained_checkpoint=None,  # Weights loaded in-memory via AutoBridge
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=True,
            use_persistent_ckpt_worker=True,
            ckpt_assume_constant_structure=True,  # Cache checkpoint structure for faster saves
            save_optim=False,  # Don't save optimizer state (much smaller checkpoints)
            save_rng=False,  # Don't save RNG state
        ),
        mixed_precision=bf16_mixed() if args.bf16 else None,
        rng=RNGConfig(seed=args.seed),
    )

    # Configure DDP settings
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    # Reduce/accumulate gradients in fp32. The official Qwen3.5-VL recipe sets this
    # (recipes/qwen_vl/qwen35_vl.py: cfg.ddp.grad_reduce_in_fp32=True). Without it,
    # the 397B bf16 run produced NaN grad-norms by iter ~6 (bf16 grad reduction
    # overflow). fp32 grad reduction is the standard fix for large bf16 MoE training.
    cfg.ddp.grad_reduce_in_fp32 = True

    # LOCAL PATCH for Qwen3-VL / Qwen3.5-VL models with pp>1 (or ep>1):
    # qwen3_vl_step.forward_step pads tokens to seq_length (force_to_pad_to_seq_len)
    # when pp>1, but only restores `packed_seq_params` into the model forward when
    # `state.cfg.dataset.pack_sequences_in_batch` is truthy (qwen3_vl_step.py:282-294).
    # Without it the model takes the BSHD path: combined_embeddings are converted to
    # the un-padded packed length while RoPE freqs are built for the full padded
    # length, crashing in attention with
    #   "size of tensor a (<packed_len>) must match tensor b (<seq_length>)"
    # The fork never exposes this flag as a FinetuningDatasetConfig field, so we set
    # it directly on the (non-frozen) dataset config object. Gated on VL models with
    # pp>1 since the THD packed path is what they require; the dense path (e.g. 27B
    # with pp1) trains fine without it.
    # NOTE: THD in-batch packing (pack_sequences_in_batch) is numerically BROKEN for
    # Qwen3.5-MoE in this fork (grad norm 1e7 -> NaN; the official team reverted
    # "enable THD packing by default for Qwen3.5-VL finetune", commit f8c61132). We
    # only set it when explicitly requested (SFT_ENABLE_PACK=1). The RoPE-mismatch
    # bug it was meant to work around is actually triggered by cp>1 (split_data_cp_rank
    # at model.py:504-505), NOT by pp>1 — so pp>1 with cp=1 should NOT need packing.
    provider_is_vl = "qwen_vl" in type(cfg.model).__module__
    enable_pack = os.environ.get("SFT_ENABLE_PACK", "0") == "1"
    if provider_is_vl and args.pipeline_model_parallel_size > 1 and enable_pack:
        cfg.dataset.pack_sequences_in_batch = True
        print(
            "Set dataset.pack_sequences_in_batch=True (SFT_ENABLE_PACK=1, VL + pp>1)."
        )

    # Raise the NCCL watchdog timeout. The default (10 min) can fire during a slow
    # first step (long-context compile + fp32 grad-reduce all-reduce on 128 GPU);
    # the 397B grad_reduce_in_fp32 run died on a Watchdog timeout before iter 1.
    cfg.dist.distributed_timeout_minutes = 30

    return cfg


def select_forward_step(model_provider):
    """Select the forward_step function appropriate for the model architecture.

    Qwen3-VL / Qwen3.5 / Qwen3.6 models (model_type ``qwen3_5``,
    ``Qwen3_5ForConditionalGeneration``) are mapped by AutoBridge to the
    ``Qwen3VLModel`` class. That model does its own context-/sequence-parallel
    split *inside* ``model.forward`` and expects the FULL (un-sliced) sequence as
    input. The generic ``gpt_step.forward_step`` already CP-slices the batch in
    ``get_batch``, so using it with a VL model double-slices the sequence (each
    rank sees ~1/cp**2 of the tokens, with labels misaligned) and trips the
    ``seq % (2*cp) == 0`` assert in ``split_data_cp_rank`` at cp>1.

    These models ship their own ``forward_step`` (``qwen_vl/qwen3_vl_step.py``)
    that keeps the input intact, pads to a CP-compatible length, and performs the
    single correct split. This mirrors upstream ``run_recipe.py --step_func
    qwen3_vl_step``. The VL step's ``get_batch`` is text-safe: with no images it
    sets ``multi_modal_inputs={}`` and reads tokens/labels/loss_mask exactly like
    our chat SFT dataloader produces.

    Falls back to the generic GPT step for text Qwen3/Llama models.
    """
    provider_module = type(model_provider).__module__
    if "qwen_vl" in provider_module:
        from megatron.bridge.models.qwen_vl.qwen3_vl_step import (
            forward_step as qwen3_vl_forward_step,
        )

        print(
            f"Detected VL model provider ({type(model_provider).__name__}); "
            "using qwen3_vl_step.forward_step (required for correct CP/SP handling)."
        )
        return qwen3_vl_forward_step

    return gpt_forward_step


def count_jsonl_samples(data_path: Path) -> int:
    """Count the number of samples in a JSONL file.

    Args:
        data_path: Path to directory containing training.jsonl

    Returns:
        Number of samples (lines) in the file
    """
    jsonl_path = data_path / "training.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"training.jsonl not found in {data_path}")

    count = 0
    with open(jsonl_path, "r") as f:
        for _ in f:
            count += 1
    return count


def calculate_train_iters(num_samples: int, global_batch_size: int, epochs: int) -> int:
    """Calculate the number of training iterations for a given number of epochs.

    Args:
        num_samples: Total number of samples in the dataset
        global_batch_size: Global batch size
        epochs: Number of epochs to train

    Returns:
        Number of training iterations
    """
    steps_per_epoch = num_samples // global_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    return steps_per_epoch * epochs


def main() -> None:
    """Main entry point for SFT training."""
    args = parse_args()

    # Handle epochs vs train_iters
    if args.epochs is not None:
        data_path = Path(args.data_path)
        num_samples = count_jsonl_samples(data_path)
        steps_per_epoch = num_samples // args.global_batch_size
        if steps_per_epoch == 0:
            steps_per_epoch = 1
        args.train_iters = calculate_train_iters(
            num_samples, args.global_batch_size, args.epochs
        )
        # When training by epochs, interpret --save-interval as every N epochs
        args.save_interval = args.save_interval * steps_per_epoch
        print(f"Dataset size: {num_samples} samples")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Training for {args.epochs} epochs = {args.train_iters} iterations")
        print(
            f"Save interval: every {args.save_interval // steps_per_epoch} epoch(s) = {args.save_interval} iterations"
        )

    # Validate parallelism settings
    if args.sequence_parallel and args.tensor_model_parallel_size <= 1:
        print(
            "Warning: --sequence-parallel requires --tensor-model-parallel-size > 1. Disabling sequence parallelism."
        )
        args.sequence_parallel = False

    # Set CUDA_DEVICE_MAX_CONNECTIONS for sequence parallelism speedup
    if args.sequence_parallel:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create configuration
    config = create_sft_config(args)

    # Create callbacks list
    callbacks = []

    # Add tqdm progress callback (rank 0 only will actually display)
    if not args.no_tqdm:
        callbacks.append(
            TqdmProgressCallback(
                total_iters=args.train_iters,
                log_interval=1,
            )
        )

    # Print configuration summary
    print("=" * 60)
    print("SFT Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {config.dataset.seq_length}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Micro batch size: {args.micro_batch_size}")
    if args.epochs is not None:
        print(f"Epochs: {args.epochs}")
    print(f"Training iterations: {args.train_iters}")
    print(f"Learning rate: {args.lr}")
    print(f"Tensor model parallel size: {args.tensor_model_parallel_size}")
    print(f"Pipeline model parallel size: {args.pipeline_model_parallel_size}")
    print(f"Context parallel size: {args.context_parallel_size}")
    print(f"Expert model parallel size: {args.expert_model_parallel_size}")
    print(
        f"Expert tensor parallel size: {args.expert_tensor_parallel_size or args.tensor_model_parallel_size}"
    )
    print(f"Sequence parallel: {args.sequence_parallel}")
    print(f"Output directory: {args.output_dir}")
    if args.wandb_project:
        print(f"W&B Project: {args.wandb_project}")
        print(f"W&B Run Name: {args.wandb_run_name or 'auto'}")
    print("=" * 60)

    # Select the forward_step appropriate for the model architecture (VL models
    # need their own step; see select_forward_step).
    forward_step_func = select_forward_step(config.model)

    # Start training
    # Use pretrain() instead of finetune() because weights are already loaded
    # in-memory via AutoBridge.from_hf_pretrained(load_weights=True).
    # finetune() is just pretrain() + an assertion that a checkpoint path is set,
    # which doesn't apply when weights are injected directly.
    pretrain(config=config, forward_step_func=forward_step_func, callbacks=callbacks)


if __name__ == "__main__":
    main()
