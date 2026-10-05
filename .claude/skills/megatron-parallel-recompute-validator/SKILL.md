---
name: megatron-parallel-recompute-validator
description: Validate a Megatron-core parallel + recompute + activation-offload config for Qwen3.5/3.6 SFT in this repo before launching it on real GPUs. Catches divisibility errors, mutual-exclusion violations (cpu_offloading + PP>1, fine_grained + cpu_offloading, recompute + cpu_offloading, kv_heads % tp), and gives a per-GPU memory back-of-envelope. Use when planning tp/pp/cp/ep/etp, choosing recompute_num_layers, deciding between cpu_offloading vs fine_grained_activation_offloading (rednote), or debugging an init-time crash.
license: Apache-2.0
when_to_use: planning a multi-GPU SFT/pretrain config; "tp pp ep recompute_num_layers", "uniform vs block recompute", "cpu_offloading", "fine_grained_activation_offloading", "offload_modules", "OOM at first step", "init-time ValueError"; before submitting a 64+ GPU job to verify it won't crash at megatron init.
---

# Megatron parallel-state + recompute + offload validator

Paths are relative to this repository's root.

## What this is

A 2-second offline (no GPU needed) validator that takes a parallel layout +
recompute/offload settings and either prints `✅ TransformerConfig accepted`
or surfaces the exact `ValueError`/`AssertionError` your real launch would
hit at megatron init. It also gives a coarse per-GPU memory estimate so you
can compare configs without burning a 20-minute model load.

The driver is `.claude/skills/megatron-parallel-recompute-validator/validate.py`.
It (1) runs the paper checks from `docs/megatron_parallel_constraints.md`
§11 (divisibility, kv-head limits, GDN limits) and (2) actually instantiates
`megatron.core.transformer.transformer_config.TransformerConfig` so the
mutual-exclusion validators in `__post_init__` fire — the same ones that
would fire on rank 0 of a real launch.

## Background you should know cold

These three docs in this repo are the source of truth — read them before
designing a new config. The validator script encodes their rules.

- `docs/megatron_parallel_constraints.md` — the dimensions (tp/pp/cp/ep/etp),
  every divisibility rule, the worked checklist, and **§12 the
  recompute-`num_layers`-semantics-flip**:
  - `recompute_method=uniform`: R = layers per recompute chunk (ALL layers
    recompute). **Larger R ⇒ MORE memory savings** (fewer save points).
    `R=1` is the *least* memory-efficient uniform setting.
  - `recompute_method=block`: R = number of layers (from front of stage)
    that recompute. **Larger R ⇒ MORE memory savings**.
  - Rule of thumb under uniform: set `R = num_layers / pp` (whole stage
    one chunk) for max savings.
- `docs/training/cpu-offloading.md` in the pinned Megatron-Bridge fork
  (see `pyproject.toml` for the fork/branch) — covers the
  **layer-level** `model.cpu_offloading=True` (PP=1 only) and the
  **fractional optimizer** offload `optimizer.optimizer_cpu_offload`.
- `docs/training/activation-recomputation.md` in the same fork — measured
  per-module recompute cost on Llama3 70B. The big finding: try
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first, then selective
  recompute (`core_attn`), then full only as last resort.

## The three offload mechanisms — pick the right one

| Mechanism | Config field | Granularity | PP > 1? | Notes |
|---|---|---|---|---|
| Layer-level activation offload | `cpu_offloading=True` + `cpu_offloading_num_layers=N` | per-layer | ❌ blocked | Mutually exclusive with recompute and CUDA graphs. |
| Fine-grained activation offload (**rednote**) | `fine_grained_activation_offloading=True` + `offload_modules=[...]` | per-submodule | ✅ works | Choose modules from `{attn_norm, qkv_linear, core_attn, attn_proj, mlp_norm, expert_fc1, moe_act}`. Mutually exclusive with `cpu_offloading`. |
| Optimizer state offload | `optimizer_cpu_offload=True` + `optimizer_offload_fraction=0.5` | optimizer state, fractional | ✅ works | Throughput penalty 1.9–4.2x scaling with fraction (Qwen3-30B-A3B). |

For a large MoE (122B/397B) needing PP > 1, **fine-grained activation
offloading is the only activation-side option**. Rednote contributed it
specifically to unblock that scenario. Validator usage:
`--fine-grained --offload-modules core_attn moe_act expert_fc1`.

Important rule from the validator (`transformer_config.py:1679-1685`): if
you put `attn_proj` in offload_modules, you must also put `core_attn`,
because `attn_proj`'s input is `core_attn`'s output and is consumed by
`core_attn.backward()`.

## Run the validator

```bash
# Built-in shortcut for known models. Returns 0 on valid, non-zero on fail.
.venv/bin/python .claude/skills/megatron-parallel-recompute-validator/validate.py \
    qwen35-397b --tp 2 --pp 4 --ep 16 --etp 1 \
    --gbs 16 --mbs 1 --seq 16384 --world 64 \
    --recompute-granularity full --recompute-method uniform \
    --recompute-num-layers 15
```

Built-in model shortcuts: `qwen35-9b qwen35-122b qwen35-397b qwen36-27b qwen36-35b`.
Or pass a path to an HF model directory / `config.json`.

The flag set mirrors `scripts/sft_train.py`:
```
--tp / --pp / --cp / --ep / --etp                       parallel dims
--world                                                  world_size (default tp*pp*cp)
--gbs / --mbs / --seq                                    batch + sequence
--sequence-parallel / --no-sequence-parallel
--recompute-granularity {full,selective}
--recompute-method {uniform,block}
--recompute-num-layers N            # uniform: layers/chunk; block: layers from front
--recompute-modules core_attn ...   # for selective
--cpu-offloading                    # layer-level; PP=1 only
--cpu-offloading-num-layers N
--fine-grained                      # module-level (rednote); compatible with PP>1
--offload-modules core_attn moe_act ...
```

### Interpreting output

```
=== Model: qwen35-397b ===
=== Parallel: tp2·pp4·cp1·ep16·etp1 on 64 GPU ===
=== Recompute: full/uniform/R=15 ===

✅ All divisibility/ordering rules pass.
⚠  WARNING: with pp>1 + default ordering, etp*ep == tp*cp would be required ...
              (Real recipes use pp-last ordering which waives this. Ignore the warning
              if your launch script doesn't override `order=`. mbridge does pp-last
              by default.)

✅ TransformerConfig.__post_init__ accepted.

=== Memory estimate (per GPU, ROUGH) ===
  weights_gb: 12.66
  optim_gb: 101.25
  activations_gb_estimate: 0.30
  static_total_gb: 113.91
  approx_peak_gb: 114.21
  layers_per_stage: 15

(H100 80 GB, H200 143 GB. If approx_peak_gb is within ~20% of capacity → expect OOM.)
```

The memory numbers are a coarse linear model; treat them as **relative**
indicators between configs you're comparing. Calibration anchors:
- 122B real run, pp6 ep8, seq16k, R=1 → predicted 46 GB, actual ran fine on H200.
- 397B real run, pp4 ep16, seq4k, R=1 → predicted 115 GB, actually ran on H200 at runtime.
- 397B real OOM, pp4 ep8, seq4k, R=1 → predicted ~135 GB, actually OOM'd at first step.

## Recompute decision tree (for OOM)

1. **Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` first.** Many
   borderline OOMs are fragmentation, not capacity (mbridge skill,
   measured on Llama3 70B).
2. **Selective recompute** with `recompute_modules=[core_attn]` —
   compatible with TE-scoped CUDA graphs. Add `layernorm` only if borderline.
   Avoid `mlp` on wide-FFN models (Llama3 70B: ~16% util drop).
3. **Full + uniform with R = layers_per_stage** is the most memory-frugal
   "everything recomputes" setting under uniform. Don't leave R=1 — that's
   the *least* memory-efficient uniform choice (see §12).
4. **`fine_grained_activation_offloading`** (rednote) — the next step
   when recompute alone isn't enough and you have PP > 1. Start with
   `core_attn` then add `moe_act` and `expert_fc1` for MoE models. **Do
   not** combine with `cpu_offloading=True` (mutually exclusive).
5. **Optimizer state offload** — orthogonal; reduces optimizer state
   memory at cost of step time. Use fractional (0.25–0.5) before going
   to 1.0.
6. **`cpu_offloading=True`** (layer-level) — last resort for PP=1
   models. Cannot combine with recompute or CUDA graphs.

## What the validator does NOT detect

- **Runtime OOM** — the memory estimate is rough; runtime activation
  patterns can differ by 2x for variable-length finetune data.
- **Numerical bugs in the fork** — e.g. our pp>1 RoPE crash at
  `qwen3_vl_step.py:261` (already fixed in this repo's installed wheel)
  or the THD-pack MoE grad explosion. These fire mid-step, not at init.
- **NCCL hangs / timeout issues** — these are launch / network problems,
  not config problems.
- **Ordering details** — the warning about `etp*ep != tp*cp` for pp>1
  default order is just a warning. Real recipes use pp-last ordering
  which silently waives the constraint, and we've validated this on real
  64-GPU runs.

## Quick sanity checks (run these)

```bash
# Known-good real configs (must all return 0):
.venv/bin/python .claude/skills/megatron-parallel-recompute-validator/validate.py \
    qwen35-122b --tp 2 --pp 6 --ep 8 --gbs 36 --seq 16384 --world 48 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1

# Known-bad mutual-exclusion (must return non-zero):
.venv/bin/python .claude/skills/megatron-parallel-recompute-validator/validate.py \
    qwen35-122b --tp 2 --pp 4 --ep 8 --gbs 16 --seq 4096 --world 64 \
    --cpu-offloading --cpu-offloading-num-layers 4
# Expected: ValueError: Currently there is no support for Pipeline parallelism with CPU offloading

# rednote fine-grained (PP > 1, MoE) — should pass:
.venv/bin/python .claude/skills/megatron-parallel-recompute-validator/validate.py \
    qwen35-397b --tp 2 --pp 4 --ep 16 --gbs 16 --seq 16384 --world 64 \
    --fine-grained --offload-modules core_attn moe_act
```

## Gotchas

- **The validator only runs `TransformerConfig.__post_init__`**, not the
  whole training pipeline. Some validators (e.g. dataset/PP-comm-shape
  checks) fire later. We catch the dataclass-level ones, which are most
  init-time crashes.
- **`pipeline_dtype` is mandatory for pp>1.** The driver always sets it
  (bf16) so config builds don't trip on this — same default our
  `sft_train.py` uses for pp>1.
- **Built-in models hardcode dims** verified against the models' published
  HF `config.json` files on 2026-06-10. If you add a new model, pass its
  config directly: `validate.py /path/to/hf/config.json ...`.
- **GDN models (Qwen3.5/3.6 hybrid):** the validator catches
  `linear_num_key_heads % (tp·cp) != 0` automatically when the dim is
  populated. For Qwen3.5-122B/397B that means `tp·cp ≤ 16` and must
  divide 16.
- **`recompute_num_layers` semantics flip between `uniform` and `block`** —
  see `docs/megatron_parallel_constraints.md` §12. The validator does
  not warn on suboptimal R, but will reject R out of `[1, layers/pp]`.
- **`cpu_offloading_num_layers` must be in `[0, num_layers-1)`** — i.e.
  strictly less than the model's total layer count. Setting it to
  `num_layers` triggers a ValueError.

## Troubleshooting

| Symptom from validator | Real meaning |
|---|---|
| `num_key_value_heads N % tp M != 0` | Reduce tp; for Qwen3.5/3.6 (kv=2) that means tp ≤ 2. |
| `num_hidden_layers N % pp M != 0` | Either pick pp that divides num_layers or set `num_layers_in_first/last_pipeline_stage` (validator doesn't model that). |
| `linear_num_key_heads N % (tp·cp) != 0` | GDN hybrid model. Reduce tp·cp; common values are 1, 2, 4, 8, 16. |
| `Currently there is no support for Pipeline parallelism with CPU offloading` | Drop `--cpu-offloading` or set pp=1. For pp>1 with offload needs use `--fine-grained` instead. |
| `fine_grained_activation_offloading cannot be enabled with cpu_offloading` | Pick one — the two mechanisms are mutually exclusive. |
| `attn_proj cannot be set to offload_modules alone without core_attn` | Add `core_attn` to `--offload-modules`. The input of `attn_proj` is the output of `core_attn`, which `core_attn.backward()` needs. |
| `world_size N % (tp*pp*cp) != 0` | Pick world that divides cleanly; often you want `world = tp·pp·cp·dp_dense`. |
| `WARNING: ... etp*ep != tp*cp` | Almost always safe to ignore — Megatron uses pp-last ordering by default which waives this. Only matters if you explicitly set `--no-pp-last-ordering` or your recipe overrides ordering. |
