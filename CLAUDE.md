# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project performs **Supervised Fine-Tuning (SFT)** of large language models (Qwen3, Llama 3.1, including MoE variants) using [Megatron Bridge](https://github.com/NVIDIA/Megatron-Bridge). It loads a HuggingFace checkpoint, trains on chat-format JSONL data with answer-only loss masking, and exports the result back to HuggingFace format. It targets long-context (up to 131072 tokens) multi-GPU / multi-node training.

The codebase is small — three scripts plus launch shims. The heavy lifting lives in pinned forks of `megatron-core`, `megatron-bridge`, and `megatron-energon` (see Dependencies).

## Environment & Dependencies

- Python is managed by **`uv`** with a locked `uv.lock`. Run everything via `uv run ...` or after `source .venv/bin/activate`.
- Core deps are **git forks pinned to the `sft-patch` branch** (`pyproject.toml`):
  - `megatron-core` → `github.com/leoleoasd/Megatron-LM.git@sft-patch-v2`
  - `megatron-bridge` → `github.com/leoleoasd/Megatron-Bridge.git@sft-patch-v2`
  - These forks contain the SFT-specific behavior (chat templating, loss masking, in-memory weight loading) the scripts rely on. Behavior differs from upstream Megatron — when a config field or function behaves unexpectedly, check the fork, not upstream docs.
- Requires a CUDA environment with `apex`, `transformer-engine`, and NVIDIA GPUs. Native extensions are built no-build-isolation (see Dockerfile); don't expect a clean `pip install` to work — use `uv sync`.
- The `Dockerfile` builds on the public `nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04` image (override via the `BASE_IMAGE` build arg); `docker buildx bake` (config in `docker-bake.hcl`) builds and pushes the training image — set the `REPO` variable (e.g. `REPO=ghcr.io/you/mbsft`) to your registry. The image carries no credentials; pass `HF_TOKEN`/`WANDB_API_KEY` at `docker run` time.

## Common Commands

There is no test suite, no linter config beyond a `.ruff_cache`, and no build step for the Python code itself.

**Single-node training (8 GPUs)** — via torchrun:
```bash
uv run torchrun --nproc_per_node=8 scripts/sft_train.py \
    --model /path/to/hf_model \
    --data-path /path/to/data_dir \
    --output-dir /path/to/output \
    --chat-template chat_templates/qwen3_2507_thinking.jinja \
    --tensor-model-parallel-size 4 --context-parallel-size 2 --sequence-parallel \
    --seq-length 131072 --global-batch-size 32 --micro-batch-size 1 \
    --lr 5e-5 --lr-warmup-iters 0 --epochs 3 --save-interval 1 \
    --wandb-project web-agent-sft
```
See `scripts.sh` for working multi-node (K8s) examples including a 30B MoE config with expert parallelism.

**Multi-node via Ray** (one actor per GPU, no torchrun):
```bash
ray job submit --address=auto --working-dir /path/to/mbsft -- \
    python scripts/sft_ray.py --model ... --data-path ... --output-dir ... [sft_train.py args]
```
`launch_ray.sh` bootstraps the Ray cluster (auto-detects AWS Batch vs Kubernetes from env vars, starts head/worker nodes). `sft_ray.py` discovers cluster topology, then forwards all unrecognized args straight to `sft_train.py`.

**Kubernetes via KubeRay**: `k8s/rayjob.yaml` is a ready-to-adapt RayJob manifest (head with `num-gpus: 0`, N×8-GPU workers, repo used from `/workdir` via `runtimeEnvYAML`). It doesn't use `launch_ray.sh` — the operator manages the cluster and submits `sft_ray.py`.

**Convert Megatron checkpoint → HuggingFace** (auto-run at end of `sft_ray.py` unless `--no-auto-convert-hf`):
```bash
uv run python scripts/convert_to_hf.py \
    --hf-model /path/to/base_hf_model \
    --input /path/to/output/checkpoints \
    --output-dir /path/to/hf_exports
```

## Architecture & Key Behaviors

### `scripts/sft_train.py` — the training entrypoint
- **Parallelism args are auto-generated** from Megatron's `TransformerConfig` dataclass via `ArgumentGroupFactory` (group "Model & Parallelism"). The explicit `transformer_exclude` list filters out fields that can't be CLI args (callables, deprecated, or set elsewhere). To expose a new transformer knob, ensure it's *not* in that exclude list rather than adding a new `add_argument`.
- After parsing, all `ModelParallelConfig` fields present on `args` are copied onto the model provider (`create_sft_config`). This is how CLI flags reach the model config generically.
- **Weights are loaded in-memory** via `AutoBridge.from_hf_pretrained(load_weights=True)`. Because of this, the script deliberately calls `pretrain()` (not `finetune()`) — `finetune()` only adds an assertion requiring a checkpoint path, which doesn't apply here. `pretrained_checkpoint=None` in `CheckpointConfig` for the same reason.
- **`--epochs` vs `--train-iters`**: when `--epochs` is given, the script counts lines in `training.jsonl` to derive `train_iters`, and reinterprets `--save-interval` as "every N epochs" (converted to iterations).
- Checkpoints are saved with `save_optim=False` and `save_rng=False` (smaller, inference-oriented checkpoints), `ckpt_format="torch_dist"`, async + fully-parallel save.
- `--chat-template` reads a Jinja file and overrides the tokenizer's template via `TokenizerConfig.chat_template`. The chat templates in `chat_templates/` add generation tags used for SFT loss masking (`answer_only_loss=True`).

### `scripts/sft_ray.py` — Ray multi-node launcher
- Creates a single `PACK` placement group, probes bundles with temporary `InfoActor`s to learn node/GPU topology, then **reorders bundle indices so consecutive global ranks land on the same node** (mimicking torchrun's `LOCAL_RANK` layout). Sets `RANK`/`WORLD_SIZE`/`LOCAL_WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` per actor and calls `sft_train.main()` directly.
- After training succeeds, kills actors, frees the placement group, then runs `convert_to_hf.py` in a fresh GPU actor. Output goes to `<output-dir>-hf`.

### `scripts/convert_to_hf.py` — checkpoint export
- **Monkey-patches** `megatron.bridge.training.model_load_save.load_model_config` so the Bridge provider (which has full model info) overrides the bare `TransformerConfig` saved in the checkpoint. Necessary because checkpoints don't store complete provider info.
- Accepts an `iter_*` dir, a `checkpoints/` dir (converts all iterations), or an experiment dir. Re-exports the tokenizer from the original HF model; **clears the chat template by default** (`--use-original-chat-template` to keep it; falls back to a `chat_template.jinja` saved in the checkpoint if present).

## Training Data Format

Detailed spec in `docs/training_data_format.md`. Key points:
- `--data-path` must be a **directory** containing `training.jsonl` (one JSON object per line). No validation/test files needed (`do_validation`/`do_test` are `False`).
- Always HuggingFace messages format: `{"messages": [{"role": ..., "content": ...}], "tools": [...]}`. Optional per-message `reasoning_content` (thinking models) and `tool_calls`; optional top-level `tools` (overrides global `--tool-schemas`).
- Loss mask = 1 on assistant tokens only. Sequences over `--seq-length` are right-truncated; EOS added, BOS not. Data shuffles within an epoch but not across epochs.

## Gotchas

- `.jsonl` files are **gitignored** — datasets live outside the repo (typically under `/data/...`).
- `--sequence-parallel` is silently disabled if `--tensor-model-parallel-size <= 1`. When enabled, `CUDA_DEVICE_MAX_CONNECTIONS=1` is set automatically.
- For pipeline parallel > 1, `pipeline_dtype` is forced to bfloat16.
