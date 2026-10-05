#!/usr/bin/env python3
"""Validate a Megatron-core parallel + recompute + offload config WITHOUT a GPU.

Builds a minimal :class:`TransformerConfig` with the requested fields and runs
its ``__post_init__`` validators. Either prints a green ✅ summary (the config
would survive init), or surfaces the exact ``ValueError`` / ``AssertionError``
that would crash a real launch.

This is the ~2-second feedback loop for "will my next 64-GPU job crash at
megatron init?". It does NOT detect runtime issues like OOM at the first
forward step or numerical bugs. It DOES detect every divisibility / mutual-
exclusion error that the post_init validators check (which is most of what
crashes in init).

The script also computes a back-of-envelope per-GPU memory footprint based on
HF config dims so you can compare recompute and offload settings before
launching. The activation estimate is a coarse linear model (proportional to
seq_length * num_layers_per_stage) — calibrated to a few real runs but
expected to be off by 30%+ on outliers.

Usage:
    python validate.py [model.json | hf_path] \\
        --tp 2 --pp 4 --cp 1 --ep 16 --etp 1 \\
        --gbs 16 --mbs 1 --seq 16384 --world 64 \\
        --recompute-granularity full --recompute-method uniform \\
        --recompute-num-layers 15

    Examples (built-in shorthand for the models we care about):
        python validate.py qwen35-122b --tp 2 --pp 6 --ep 8 --gbs 36 --seq 16384
        python validate.py qwen35-397b --tp 2 --pp 4 --ep 16 --gbs 16 --seq 16384 \\
            --recompute-granularity full --recompute-method uniform \\
            --recompute-num-layers 15
        python validate.py qwen36-35b  --tp 2 --pp 1 --ep 32 --gbs 32 --seq 16384

Exit code 0 = config valid. Non-zero = init would fail (with the error).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

# Built-in model shortcuts (HF config dims I've already verified in this repo).
# Add new ones by reading config.json from the HF dir; these are convenience.
KNOWN_MODELS = {
    "qwen35-9b":   {"num_hidden_layers": 36, "num_key_value_heads": 4,  "num_experts": 0,   "hidden_size": 4096, "moe_intermediate_size": 0,    "intermediate_size": 12288, "linear_num_key_heads": 0,  "linear_num_value_heads": 0,  "tie_word_embeddings": False, "vocab_size": 248320},
    "qwen35-122b": {"num_hidden_layers": 48, "num_key_value_heads": 2,  "num_experts": 256, "hidden_size": 3072, "moe_intermediate_size": 1024, "intermediate_size": 12288, "linear_num_key_heads": 16, "linear_num_value_heads": 64, "tie_word_embeddings": False, "vocab_size": 248320, "moe_topk": 8},
    "qwen35-397b": {"num_hidden_layers": 60, "num_key_value_heads": 2,  "num_experts": 512, "hidden_size": 4096, "moe_intermediate_size": 1024, "intermediate_size": 12288, "linear_num_key_heads": 16, "linear_num_value_heads": 64, "tie_word_embeddings": False, "vocab_size": 248320, "moe_topk": 10},
    "qwen36-27b":  {"num_hidden_layers": 64, "num_key_value_heads": 8,  "num_experts": 0,   "hidden_size": 5120, "moe_intermediate_size": 0,    "intermediate_size": 25600, "linear_num_key_heads": 0,  "linear_num_value_heads": 0,  "tie_word_embeddings": False, "vocab_size": 248320},
    "qwen36-35b":  {"num_hidden_layers": 40, "num_key_value_heads": 2,  "num_experts": 256, "hidden_size": 3072, "moe_intermediate_size": 1024, "intermediate_size": 12288, "linear_num_key_heads": 16, "linear_num_value_heads": 64, "tie_word_embeddings": False, "vocab_size": 248320, "moe_topk": 8},
}


def load_model_config(spec: str) -> dict:
    """Resolve --model spec to a dim dict. Accepts:
       - a built-in shorthand (qwen35-122b, ...)
       - a path to an HF config.json or a HF model directory
       - a path to a JSON file with the same keys as KNOWN_MODELS
    """
    if spec in KNOWN_MODELS:
        return KNOWN_MODELS[spec]
    p = Path(spec)
    if p.is_dir():
        p = p / "config.json"
    if p.is_file():
        cfg = json.loads(p.read_text())
        cfg = cfg.get("text_config", cfg)
        return {
            "num_hidden_layers": cfg.get("num_hidden_layers", 0),
            "num_key_value_heads": cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 1)),
            "num_experts": cfg.get("num_experts", 0),
            "hidden_size": cfg.get("hidden_size", 0),
            "moe_intermediate_size": cfg.get("moe_intermediate_size", 0),
            "intermediate_size": cfg.get("intermediate_size", 0),
            "linear_num_key_heads": cfg.get("linear_num_key_heads", 0),
            "linear_num_value_heads": cfg.get("linear_num_value_heads", 0),
            "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
            "vocab_size": cfg.get("vocab_size", 0),
            "moe_topk": cfg.get("num_experts_per_tok", 0),
        }
    raise SystemExit(f"Unknown model spec: {spec!r}. Known shortcuts: {sorted(KNOWN_MODELS)}")


def check_divisibility(world: int, tp: int, pp: int, cp: int, ep: int, etp: int,
                        gbs: int, mbs: int, model: dict) -> list[str]:
    """Run all paper checks from docs/megatron_parallel_constraints.md §11.

    Returns a list of (FAIL, msg) tuples; empty == valid.
    """
    fails = []
    NL = model["num_hidden_layers"]
    NE = model["num_experts"]
    KV = model["num_key_value_heads"]
    LKH = model["linear_num_key_heads"] or 0

    # 1. world_size % (tp*pp*cp) == 0   → derives dense dp
    if world % (tp * pp * cp) != 0:
        fails.append(f"world {world} % (tp*pp*cp = {tp*pp*cp}) != 0")
    dp_dense = world // (tp * pp * cp)

    # 2. world_size % (etp*ep*pp) == 0  → derives edp
    if NE > 0 and world % (etp * ep * pp) != 0:
        fails.append(f"world {world} % (etp*ep*pp = {etp*ep*pp}) != 0")
    edp = (world // (etp * ep * pp)) if NE > 0 else None

    # 3. GBS % (dp_dense * mbs) == 0
    if gbs % (dp_dense * mbs) != 0:
        fails.append(f"GBS {gbs} % (dp_dense*mbs = {dp_dense*mbs}) != 0")

    # 4. num_experts % ep == 0
    if NE > 0 and NE % ep != 0:
        fails.append(f"num_experts {NE} % ep {ep} != 0")

    # 5. num_kv_heads % tp == 0
    if KV % tp != 0:
        fails.append(f"num_key_value_heads {KV} % tp {tp} != 0")

    # 6. GDN: linear_num_key_heads % (tp*cp) == 0  (only for hybrid GDN models)
    if LKH > 0 and LKH % (tp * cp) != 0:
        fails.append(f"linear_num_key_heads {LKH} % (tp*cp = {tp*cp}) != 0  (GDN constraint)")

    # 7. num_layers % pp == 0
    if NL % pp != 0:
        fails.append(f"num_hidden_layers {NL} % pp {pp} != 0")

    # 8. with pp>1 + default order, etp*ep must == tp*cp so edp == dp_dense
    #    (this is waived by pp-last ordering, which Megatron uses for MoE+pp by default
    #    in 0.13+; real recipes routinely violate this and run fine. We surface as a
    #    WARNING only, not a hard fail, because the actual exception only fires if the
    #    user explicitly sets --no-pp-last-ordering.)
    warnings = []
    if pp > 1 and NE > 0 and (etp * ep) != (tp * cp) and edp != dp_dense:
        warnings.append(
            f"with pp>1 + default ordering, etp*ep == tp*cp would be required "
            f"(here {etp*ep} != {tp*cp}, edp={edp} vs dp_dense={dp_dense}). "
            f"Real Megatron recipes use pp-last ordering which waives this — only fails "
            f"if you explicitly disable pp-last."
        )
    return fails, warnings


def estimate_memory_gb(model: dict, tp: int, pp: int, ep: int, etp: int,
                       seq: int, mbs: int, recompute_granularity, recompute_num_layers,
                       recompute_method) -> dict:
    """Coarse per-GPU memory estimate, in GB. NOT precise. Use to compare configs."""
    NL = model["num_hidden_layers"]
    H = model["hidden_size"]
    INT = model["intermediate_size"]
    NE = model["num_experts"]
    MOE_INT = model["moe_intermediate_size"]
    TOPK = model.get("moe_topk", 0) or 0
    layers_per_stage = NL // pp

    # --- Weights ---
    # Dense per-layer params: ~12 H^2 (qkv, proj, gate/up/down, etc), simplified.
    # MoE per-layer: experts * (3 * H * MOE_INT) + non-expert (~6 H^2 attention/router)
    if NE > 0:
        per_layer_dense = 6 * H * H  # attention + norm + router roughly
        per_layer_expert = NE * 3 * H * MOE_INT
        # Sharding: dense by tp; expert by etp*ep
        dense_param_bytes = per_layer_dense * layers_per_stage * 2 / tp
        expert_param_bytes = per_layer_expert * layers_per_stage * 2 / (etp * ep)
        weights_gb = (dense_param_bytes + expert_param_bytes) / 1024 ** 3
    else:
        per_layer = 12 * H * H + 3 * H * INT
        weights_gb = per_layer * layers_per_stage * 2 / tp / 1024 ** 3

    # --- Optimizer state (Adam: 2 state + master fp32 weight + master fp32 grad ≈ 16 B/param fp16-mixed) ---
    # We approximate as 8x the bf16 weight bytes (Adam m + v fp32 = 8 B, grad+master = 8 B).
    # Distributed optimizer typically shards by dp_dense for dense weights, by edp for expert,
    # but we conservatively assume 1x sharding here (i.e. unsharded — upper bound).
    optim_gb = weights_gb * 8

    # --- Activations (very rough) ---
    # Activation memory for one micro-batch, no recompute = ~ seq * H * layers_per_stage * 12 B
    # (12 ≈ #activations stashed per layer, in bytes-equivalent units)
    act_full = seq * H * layers_per_stage * 12 * mbs / 1024 ** 3

    if recompute_granularity == "full" and recompute_method == "uniform":
        # uniform R: each chunk recomputed; saved per-chunk-boundary activation ≈ act_full / R
        R = max(recompute_num_layers, 1)
        # Bigger R → fewer save points → less memory. Boundaries ≈ ceil(layers_per_stage / R)
        boundaries = max(1, math.ceil(layers_per_stage / R))
        act_gb = act_full * (boundaries / layers_per_stage) * 0.4  # 0.4 = handwave constant
    elif recompute_granularity == "full" and recompute_method == "block":
        # block R: only first R layers recompute; remaining N-R store full activation
        R = max(recompute_num_layers, 0)
        non_recompute_layers = max(0, layers_per_stage - R)
        # recomputed layers: small; non-recomputed: full
        act_gb = act_full * (non_recompute_layers / layers_per_stage)
    elif recompute_granularity == "selective":
        # depends on which modules; rough -50% savings
        act_gb = act_full * 0.5
    else:
        act_gb = act_full

    return {
        "weights_gb": round(weights_gb, 2),
        "optim_gb": round(optim_gb, 2),
        "activations_gb_estimate": round(act_gb, 2),
        "static_total_gb": round(weights_gb + optim_gb, 2),
        "approx_peak_gb": round(weights_gb + optim_gb + act_gb, 2),
        "layers_per_stage": layers_per_stage,
    }


def try_build_transformer_config(args, model):
    """Actually instantiate megatron-core's TransformerConfig and run its
    post_init validators. This catches mutual-exclusion errors (cpu_offloading
    + recompute, fine_grained + cpu_offloading, etc.) that the paper checks
    don't catch.
    """
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError:
        return None, "megatron-core not installed (skip live validator)"

    import torch
    kwargs = dict(
        num_layers=model["num_hidden_layers"],
        hidden_size=model["hidden_size"],
        num_attention_heads=model["num_key_value_heads"],  # any divisor of tp works
        num_query_groups=model["num_key_value_heads"],
        ffn_hidden_size=model["intermediate_size"] or model["hidden_size"] * 4,
        kv_channels=128,
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=args.cp,
        sequence_parallel=args.sequence_parallel,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,  # required when pp > 1
        # NB: seq_length is NOT a TransformerConfig field — it lives on the dataset
        # config in mbridge. We don't pass it here.
    )
    if model["num_experts"] > 0:
        kwargs.update(
            num_moe_experts=model["num_experts"],
            moe_router_topk=model.get("moe_topk", 8) or 8,
            moe_ffn_hidden_size=model["moe_intermediate_size"],
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=args.ep,
            expert_tensor_parallel_size=args.etp,
        )
    if args.recompute_granularity:
        kwargs["recompute_granularity"] = args.recompute_granularity
        if args.recompute_granularity == "full":
            kwargs["recompute_method"] = args.recompute_method
            kwargs["recompute_num_layers"] = args.recompute_num_layers
        elif args.recompute_granularity == "selective" and args.recompute_modules:
            kwargs["recompute_modules"] = args.recompute_modules
    if args.cpu_offloading:
        kwargs["cpu_offloading"] = True
        kwargs["cpu_offloading_num_layers"] = args.cpu_offloading_num_layers
    if args.fine_grained:
        kwargs["fine_grained_activation_offloading"] = True
        kwargs["offload_modules"] = args.offload_modules or []
    try:
        cfg = TransformerConfig(**kwargs)
        return cfg, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument("model", help=f"Built-in: {sorted(KNOWN_MODELS)} or path to HF config.json/dir")
    p.add_argument("--tp", type=int, required=True, help="tensor_model_parallel_size")
    p.add_argument("--pp", type=int, required=True, help="pipeline_model_parallel_size")
    p.add_argument("--cp", type=int, default=1, help="context_parallel_size")
    p.add_argument("--ep", type=int, default=1, help="expert_model_parallel_size (MoE only)")
    p.add_argument("--etp", type=int, default=1, help="expert_tensor_parallel_size (MoE only)")
    p.add_argument("--world", type=int, default=None, help="world_size (default = tp*pp*cp*1)")
    p.add_argument("--gbs", type=int, default=8, help="global_batch_size")
    p.add_argument("--mbs", type=int, default=1, help="micro_batch_size")
    p.add_argument("--seq", type=int, default=4096, help="sequence length")
    p.add_argument("--sequence-parallel", action="store_true", default=True)
    p.add_argument("--no-sequence-parallel", dest="sequence_parallel", action="store_false")
    p.add_argument("--recompute-granularity", choices=["full", "selective", None], default=None)
    p.add_argument("--recompute-method", choices=["uniform", "block"], default="uniform")
    p.add_argument("--recompute-num-layers", type=int, default=1,
                   help="See docs/megatron_parallel_constraints.md §12 for semantics flip!")
    p.add_argument("--recompute-modules", nargs="*", default=None,
                   help="for selective; subset of {core_attn,mlp,moe,moe_act,layernorm,mla_up_proj}")
    p.add_argument("--cpu-offloading", action="store_true",
                   help="layer-level activation offload; PP=1 only, mutually exclusive with recompute")
    p.add_argument("--cpu-offloading-num-layers", type=int, default=0)
    p.add_argument("--fine-grained", action="store_true",
                   help="module-level activation offload (rednote); compatible with PP>1")
    p.add_argument("--offload-modules", nargs="*", default=None,
                   help="subset of {attn_norm,qkv_linear,core_attn,attn_proj,mlp_norm,expert_fc1,moe_act}")
    return p.parse_args()


def main():
    args = parse_args()
    model = load_model_config(args.model)
    world = args.world or (args.tp * args.pp * args.cp)

    print(f"=== Model: {args.model} ===")
    print(f"  layers={model['num_hidden_layers']} kv_heads={model['num_key_value_heads']} "
          f"experts={model['num_experts']} hidden={model['hidden_size']}")
    if model["linear_num_key_heads"]:
        print(f"  GDN: linear_key_heads={model['linear_num_key_heads']} (constrains tp·cp)")
    print(f"=== Parallel: tp{args.tp}·pp{args.pp}·cp{args.cp}·ep{args.ep}·etp{args.etp} on {world} GPU ===")
    print(f"=== Train: GBS={args.gbs} mbs={args.mbs} seq={args.seq} sp={args.sequence_parallel} ===")
    if args.recompute_granularity:
        print(f"=== Recompute: {args.recompute_granularity}/{args.recompute_method}/R={args.recompute_num_layers} ===")
    if args.cpu_offloading:
        print(f"=== CPU offload (layer): num_layers={args.cpu_offloading_num_layers} ===")
    if args.fine_grained:
        print(f"=== Fine-grained offload: modules={args.offload_modules} ===")
    print()

    # --- 1. Paper checks (divisibility, ordering) ---
    fails, warnings = check_divisibility(world, args.tp, args.pp, args.cp, args.ep, args.etp,
                                          args.gbs, args.mbs, model)
    if fails:
        print("❌ DIVISIBILITY / ORDERING:")
        for f in fails:
            print(f"   - {f}")
    else:
        print("✅ All divisibility/ordering rules pass.")
    if warnings:
        for w in warnings:
            print(f"⚠  WARNING: {w}")

    # --- 2. Live mcore validator ---
    cfg, err = try_build_transformer_config(args, model)
    print()
    if cfg is not None:
        print("✅ TransformerConfig.__post_init__ accepted.")
    elif err and "skip live validator" in err:
        print(f"⚠  {err}")
    else:
        print("❌ TransformerConfig validator REJECTED:")
        for line in (err or "").splitlines():
            print(f"   {line}")

    # --- 3. Memory estimate ---
    mem = estimate_memory_gb(
        model, args.tp, args.pp, args.ep, args.etp, args.seq, args.mbs,
        args.recompute_granularity, args.recompute_num_layers, args.recompute_method,
    )
    print()
    print("=== Memory estimate (per GPU, ROUGH; for relative comparison only) ===")
    for k, v in mem.items():
        print(f"  {k}: {v}")
    print()
    print("(H100 80 GB, H200 143 GB. If approx_peak_gb is within ~20% of capacity → expect OOM at runtime.)")

    if fails or (cfg is None and err and "skip" not in err):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
