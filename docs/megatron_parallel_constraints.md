# Megatron-core Parallelism Constraints

Source-derived reference for the parallel dimensions and the hard constraints
between them, as enforced by the pinned `megatron-core` fork in this repo. All
line numbers refer to
`.venv/lib/python3.12/site-packages/megatron/core/parallel_state.py`
(`initialize_model_parallel`) unless noted otherwise. Verified against the
installed fork on 2026-06-07.

> **Why this matters:** picking a parallel config that violates any of these
> makes the job crash *during init* (before training) — or worse, silently mis-shard.
> Use this to validate a config on paper before launching a multi-node job.

## 1. The parallel dimensions

| Symbol | CLI / config field | Meaning |
|--------|--------------------|---------|
| `tp`  | `--tensor-model-parallel-size` | Tensor (intra-layer) parallel for **dense/attention** weights |
| `pp`  | `--pipeline-model-parallel-size` | Pipeline (inter-layer) parallel |
| `cp`  | `--context-parallel-size` | Context (sequence) parallel |
| `dp`  | *(derived, not set directly)* | **Dense** data parallel |
| `ep`  | `--expert-model-parallel-size` (`expert_model_parallel_size`) | **Expert MODEL** parallel — shards the set of experts across ranks (a.k.a. EMP) |
| `etp` | `--expert-tensor-parallel-size` (`expert_tensor_parallel_size`) | **Expert TENSOR** parallel — TP *within* each expert's FFN |
| `edp` | *(derived)* | **Expert** data parallel |
| `vpp` | `virtual_pipeline_model_parallel_size` | Interleaved virtual pipeline |
| `ndoi`| `num_distributed_optimizer_instances` | Partial-shard distributed-optimizer (ZeRO) instances |

**`etp` ≠ `ep` — this is the most common confusion.** `ep` shards *which experts*
live on which rank (each rank holds `num_experts/ep` experts). `etp` shards *each
expert's weight matrices* the way `tp` shards dense layers. They multiply
together on the expert side. `etp` defaults to `tp` when left unset (line 781-782).

## 2. The two independent "worlds": dense vs expert

Megatron builds **two** rank generators (line 770-801): a dense/decoder one
(`ep=1`) and an expert one (`cp=1`). Both must tile the **same** `world_size`.

### Dense side (attention / non-expert layers)
```
model_size = tp * pp * cp                       # line 733
assert world_size % model_size == 0             # line 735  -> RuntimeError
dp = world_size // model_size                    # line 738  (dense data parallel)
```
So: **`world_size = tp · pp · cp · dp`**. EP does **not** enter `world_size`.

### Expert side (MoE FFN layers)
```
etp = expert_tensor_parallel_size or tp          # line 781-782 (defaults to tp)
expert_tmp = etp * ep * pp                        # line 783-784
assert world_size % expert_tmp == 0              # line 787  -> RuntimeError
edp = world_size // (etp * ep * pp)               # line 786  (expert data parallel)
```
So the experts are tiled as `etp · ep · pp · edp = world_size` (with `cp=1` on the
expert generator, line 798).

### Consequence (the trap)
Because both must tile the same `world_size`, **fixing `ep` forces a minimum
world size and therefore forces `dp`**. You cannot choose `dp` and `ep`
independently. Example (Qwen3.5-397B attempt): `ep=32, tp=2, pp=1, cp=1`, `etp=tp=2`:
- expert side requires `world % (etp·ep·pp) = world % 64 == 0` → smallest world = **64**
- dense side then gives `dp = world/(tp·pp·cp) = 64/2 = `**`32`** (NOT 2)
- the "dp=2" one might imagine is actually `edp = 64/64 = 1`, which is irrelevant to batching.

## 3. RankGenerator internal constraints (line 446-463)

```
assert ep == 1 or cp == 1                         # line 452-455
self.world_size = tp * dp * pp * cp * ep          # line 463 (per-generator)
```
- **EP and CP cannot both be > 1 in the same generator.** This is fine in practice
  because CP only lives in the dense generator and EP only in the expert generator —
  but it means there is no single group that is both context- and expert-parallel.

## 4. Pipeline-parallel ↔ expert-data-parallel coupling (line 803-812)

```
assert order.endswith("pp")
    or pp == 1
    or edp == dp                                  # line 803-807
assert decoder.get_ranks("pp") == expert.get_ranks("pp")   # line 809-812
```
- Unless you use pp-last rank ordering (default `order` is `"tp-cp-ep-dp-pp"`,
  which does **not** end with `pp`), then with `pp > 1` you must have
  **`edp == dp`** (expert and dense data-parallel sizes equal).
- `edp == dp` expands to `world/(etp·ep·pp) == world/(tp·pp·cp)`
  → **`etp · ep == tp · cp`** when `pp>1` and default ordering.
- Pipeline groups must be identical between dense and expert partitions.

## 5. Distributed-optimizer (ZeRO) sharding (line 827-834)

```
assert (dp * cp) % num_distributed_optimizer_instances == 0   # line 827-829
intra_partial_dp = (dp * cp) // ndoi                          # line 832-834
```
- The product `dp · cp` must be divisible by `num_distributed_optimizer_instances`.

## 6. Context-parallel hierarchical sizes (line ~968)

```
assert np.prod(hierarchical_context_parallel_sizes) == cp     # ~line 968
```
- If `hierarchical_context_parallel_sizes` is set, its product must equal `cp`.

## 7. Virtual pipeline (line 740-744)

```
if vpp is not None: assert pp > 1                 # line 740-744 -> RuntimeError
```
- Interleaved/virtual pipeline requires `pp > 1`.

## 8. Global-batch-size divisibility (NOT in parallel_state)

From `megatron/core/num_microbatches_calculator.py:367-388`:
```
micro_batch_times_data_parallel_size = micro_batch_size * data_parallel_size
assert global_batch_size % micro_batch_times_data_parallel_size == 0   # line 381
num_micro_batches = global_batch_size // (mbs * dp)
```
- **`data_parallel_size` here is the DENSE `dp`** (what the dataloader shards by).
- Therefore: **`GBS % (dp · mbs) == 0`**, i.e. dense `dp` must divide `GBS / mbs`.
- This is the constraint that most often clashes with a desired `ep`: a large `ep`
  forces a large `dp`, but `dp` must still divide `GBS`. E.g. `GBS=12` ⇒ `dp ∈ {1,2,3,4,6,12}`.

## 9. MoE expert-count divisibility (NOT in parallel_state)

From `megatron/core/transformer/moe/moe_layer.py:185-186`:
```
assert num_moe_experts % ep_size == 0
num_local_experts = num_moe_experts // ep_size
```
- **`num_experts % ep == 0`** (each EP rank holds `num_experts/ep` experts).
- In practice also keep `num_experts % (ep · etp) == 0` sensible so per-rank expert
  FFNs shard cleanly.

## 10. Model-architecture divisibility (per-model, from HF config + layer asserts)

These come from the model dimensions, not `parallel_state`, but gate `tp/cp` choices:
- **`num_attention_heads % tp == 0`** and **`num_key_value_heads % tp == 0`**
  (GQA — the KV-head count is usually the binding limit, e.g. `kv_heads=2 ⇒ tp ≤ 2`).
- **GDN / gated-delta-net hybrid models** (Qwen3.5/3.6): `linear_num_key_heads % (tp · cp) == 0`
  (`gated_delta_net.py`; e.g. `linear_num_key_heads=16 ⇒ tp·cp ≤ 16` and must divide 16).
- **`num_hidden_layers % pp == 0`** (layers split evenly across pipeline stages;
  uneven splits need `num_layers_in_first/last_pipeline_stage`, which this repo's
  `sft_train.py` excludes from the CLI).
- CP load-balancing splits each sequence into `2·cp` shards, so the per-microbatch
  sequence length must be divisible by `2·cp` (see `get_batch_on_this_cp_rank`,
  `core/utils.py`). For VLM-class models the model's own `forward_step` handles this
  padding — see [the parallel-constraints memory] and `qwen3_vl_step.py`.

## 11. Checklist to validate a config on paper

Given `(tp, pp, cp, ep, etp=tp by default, GBS, mbs, world_size, num_experts)`:

1. `world_size % (tp·pp·cp) == 0`            → derives dense `dp = world/(tp·pp·cp)`
2. `world_size % (etp·ep·pp) == 0`           → derives `edp = world/(etp·ep·pp)`
3. `GBS % (dp · mbs) == 0`                    (dense dp; §8)
4. `num_experts % ep == 0`                   (§9)
5. `num_attention_heads % tp == 0` and `num_key_value_heads % tp == 0`  (§10)
6. for GDN models: `linear_num_key_heads % (tp·cp) == 0`               (§10)
7. `num_hidden_layers % pp == 0`             (§10)
8. if `pp > 1` and default order: `etp·ep == tp·cp` (so `edp == dp`)    (§4)
9. `(dp·cp) % num_distributed_optimizer_instances == 0`                (§5)
10. if `vpp` set: `pp > 1`                    (§7)
11. EP and CP not both >1 in one group — automatically satisfied by Megatron's split (§3)

### Worked example — Qwen3.5-397B-A17B (512 experts, kv_heads=2, 60 layers, GDN)
- `tp ≤ 2` (kv_heads=2); `512 % ep == 0`; official SFT recipe = `tp2·pp4·ep32`.
- On 256 GPUs: `tp2 pp4 cp1 ep32 etp2 dp4` → dense dp=4, `12%4==0` ✓; `etp·ep=64`, `tp·cp=2` … but `pp>1` requires `etp·ep==tp·cp` → 64≠2, so **default ordering fails** unless `order` ends with `pp` (pp-last). With pp-last ordering the `edp==dp` assert is waived. Official recipe uses that ordering.
- Memory: ~807 GB bf16 weights, ~16 B/param training state ⇒ ~6.5 TB; needs `ep·etp·pp ≳ 128–256` to fit ~143 GB H200. **64 GPUs cannot hold it** (≈97 GB/GPU expert state floor + activations).

## 12. Activation recompute — the `--recompute-num-layers` semantics flip

The flag's meaning **inverts** between `uniform` and `block` recompute methods,
and the docstring is easy to misread. Source: `megatron/core/recompute.py:139-163`.

Let `N = num_layers_per_pipeline_rank` and `R = --recompute-num-layers`.

### `--recompute-method uniform` (most common; what `sft_train.py` uses)
```python
layer_idx = 0
while layer_idx < N:
    chunk_end = min(layer_idx + R, N)
    chunk_runner(layer_idx, chunk_end, checkpoint=True)   # whole chunk recomputed
    layer_idx += R
```
- **R = number of layers per recompute chunk.** ALL layers recompute; only the
  chunking changes.
- **Larger R ⇒ fewer chunk boundaries ⇒ fewer saved activations ⇒ MORE memory savings**
  (slower per chunk because each chunk re-runs more layers in a single backward).
- **Smaller R ⇒ more boundaries ⇒ more saved activations ⇒ LESS memory savings.**
- `R=1` (the value `sft_train.py` was using by default) is **the least memory-saving
  setting under uniform** — every layer boundary is a save point.
- `R=N` (one chunk per stage) is the **most memory-saving** uniform setting.

### `--recompute-method block`
```python
for layer_idx in range(N):
    use_checkpoint = (layer_idx < R)        # only the FIRST R layers checkpoint
    chunk_runner(layer_idx, layer_idx+1, use_checkpoint)
```
- **R = number of layers (counted from the front of each PP stage) that get
  recomputed at all.** The remaining `N − R` layers run normally (no recompute).
- **Larger R ⇒ more layers recompute ⇒ MORE memory savings.**
- `R=N` ⇒ every layer recomputes (equivalent in coverage to uniform R=1, but
  organised as N independent layer-checkpoints rather than N small chunks).
- `R=0` ⇒ recompute disabled (don't use this — set `--recompute-granularity` to
  `selective` or unset instead).

### Joint table

| Method | What R counts | Direction (more memory savings) | `R=1` |
|--------|---------------|---------------------------------|-------|
| `uniform` | layers per chunk; ALL layers recompute | **larger R = more savings** | least savings under uniform |
| `block`   | layers (from front) that recompute | **larger R = more savings** | only 1 layer recomputes |

So both directions are "larger R is more memory-saving", but the *meaning* of R
is different. Don't compare R values across methods.

### `--recompute-granularity selective` (the third option)
- Set `--recompute-modules` instead — a subset of `{core_attn, mlp, moe, moe_act,
  layernorm, mla_up_proj}`. **More modules listed ⇒ more recompute ⇒ more
  savings.** `recompute_num_layers` must be `None` here
  (`transformer_config.py:1570-1574`).

### Validation rules (`transformer_config.py:1562-1574`)
- If `recompute_granularity in {full, selective}`, `recompute_method` must be
  set when `granularity=full`.
- Under `granularity=full`: `recompute_num_layers` must be in
  `[1, num_layers / pipeline_model_parallel_size]`.
- Under `granularity=selective`: `recompute_num_layers` MUST be `None`.

### Practical takeaway
For an OOM-pressed full-recompute run: `--recompute-granularity full
--recompute-method uniform --recompute-num-layers $((num_layers / pp))` (i.e.
`R = N = layers_per_stage`) is the most memory-frugal "everything recomputes"
configuration. Going lower wastes memory; going higher is rejected by the
validator.

See also: the project memory notes on Qwen3.6/3.5 SFT parallel constraints and the
VLM `forward_step` requirement.
