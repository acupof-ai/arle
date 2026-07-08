# DP-attn: Shared KV Pool + Cross-Group Prefix Reuse

> Status: Draft
> Created: 2026-07-08
> Owner: ckl
> Tracking: #89

## Goal

Enable `attn_dp_size > 1` (e.g. 8卡 tp=8, attn_dp=2 → 2 groups of attn_tp=4)
so c>1 decode latency improves via **smaller attention collectives** (4-rank
allreduce/all-gather instead of 8-rank), with **shared KV pool and cross-group
prefix reuse** — a request admitted to DP group B can read prefix KV computed
by group A.

**Not data-parallelism**: all ranks compute ALL rows (unified submit). There is
2× redundant attention compute. Net win = faster collectives, only measurable
at c>1 where allreduce dominates step time.

## Non-goals

- PP (pipeline parallel) — separate axis.
- MoE-DP — EP stays global; `validate()` enforces `moe_dp=1` when `ep==tp`.
- Independent engine instances per DP group — one engine drives all groups.
- True DP batch splitting (pre-attn all-gather / post-attn reduce-scatter) — Phase 4.

## Architecture — Unified Submit

```
                    ┌─────────────────────────────────────────┐
                    │         Engine<E, K> (one instance)      │
                    │                                         │
                    │  waiting: VecDeque<RequestState>        │
                    │  active:  BTreeMap<slot, RequestState>  │
                    │  radix:   RadixCache (GLOBAL, shared)    │
                    │  kv:      KvPool (GLOBAL slot space)     │
                    │                                         │
                    │  per-request: req.dp_group: u32         │
                    │    (scheduler fairness tag, NOT          │
                    │     execution partition)                │
                    └────────────┬────────────────────────────┘
                                 │ ForwardPlan (ALL rows, all groups)
                                 ▼
                    ┌─────────────────────────────────────────┐
                    │         Dsv4Executor                    │
                    │                                         │
                    │  ALL ranks compute ALL rows             │
                    │  (weight replication → each DP group    │
                    │   independently produces correct output)│
                    │                                         │
                    │  Per layer:                             │
                    │  1. Attention allreduce:                │
                    │     tp.attn_tp sub-comm (4 ranks/group) │
                    │  2. MoE allreduce:                      │
                    │     tp.comm global (8 ranks, unchanged) │
                    └─────────────────────────────────────────┘
```

**Key insight**: DP group is a *tag on the request for scheduler fairness*,
not a partition of execution. All ranks compute all rows through both attention
and MoE. Weight replication across DP groups (P2 fix) ensures each group's
sub-comm allreduce independently produces the correct full output.

> **`dp_group` in Phase 1-5 is decorative**: it does NOT affect execution, KV
> allocation, or collective participation. All ranks enter all collectives, all
> ranks compute all rows regardless of `dp_group`. The tag exists for Phase 4
> (true DP batch splitting) readiness. Don't waste time looking for where it
> changes behavior — it doesn't, yet.

### Why unified submit (not per-group submit)

Per-group submit where only ranks in group `g` execute group `g`'s rows
**deadlocks** with MoE on global comm:

```
Per-group submit attempt:
  submit(group 0): ranks 0-3 enter attention allreduce ✓
                   ranks 0-3 enter MoE allreduce (global) ← DEADLOCK
                   ranks 4-7: empty plan, returned early, not in MoE allreduce
```

With `ep=tp=8`, `validate()` enforces `moe_dp=1` → MoE allreduce is on the
global comm requiring ALL 8 ranks. If only 4 ranks enter, the other 4 hang.

This is the same class of bug as vLLM issue #43547 (Ray/mp DP backends skip
per-step collectives on zero-token ranks → NCCL misalignment → ~1hr deadlock).
**Unified submit eliminates the entire class**: all ranks are always present
in all collectives.

---

## Known Pitfalls (Architecture-Level Coverage)

### P1 — MoE Global Allreduce Deadlock ★ CRITICAL

**Industry reference**: vLLM #43547 — DP ranks with zero scheduled tokens
skip `coordinate_batch_across_dp` collective → NCCL operation sequence
misaligns across ranks → MoE `reduce_scatter` deadlocks after ~1hr.
Signature: "100% SM util, 0% BW util" (NCCL spin-wait polling for peer).

**Our scenario**: `topology.validate()` enforces
`tp_size % (ep_size * moe_dp_size) == 0`. With `tp=8, ep=8`: `moe_dp` must be
1. `build_moe_ep_groups()` returns a single global group → `moe_ep` sub-comm
aliases the global comm → MoE allreduce requires all 8 ranks.

**Coverage**: Unified submit model. All ranks compute ALL rows, all ranks
enter both attention and MoE allreduces in the same order. No rank can be
absent from any collective. `dp_group` is a scheduler tag, not an execution
partition.

**Verification**: `INFER_TP_SIZE=8 INFER_ATTN_DP_SIZE=2 INFER_EP_SIZE=8` →
`validate()` passes, `moe_ep` groups == global group, all ranks enter MoE
allreduce. `needle_gate` PASS.

### P2 — Weight Sharding by tp_size Not attn_tp_size ★ CRITICAL

**Phenomenon**: Attention QKV/O projections sharded by `tp.world_size` (8)
instead of `attn_tp_size()` (4) → each rank has 16 heads instead of 32 →
rank 0 (heads 0-15) and rank 4 (heads 64-79) hold DIFFERENT heads, not the
same replicated set. Cross-group KV reuse would be invalid (different weights
→ different KV values).

**Evidence**: `load_dsv4_attention()` (loader.rs:3698) passes `tp.world_size`
to `shard_for()` and `column_shard()`. `head_shard()` (sharding.rs:201)
divides `num_q_heads` by `tp.world_size`. `attn_tp_size()` is used ONLY for
NCCL group construction, never for weight loading.

**Required fix**: Pass `attn_tp_cfg` (derived from `TpRuntime.attn_tp_config`,
not re-read from env) to the attention weight loading path.

At the call site (where `load_dsv4_attention` is invoked), the caller holds
`&TpRuntime` with its pre-computed `attn_tp_config: TpConfig`
(`{world_size: attn_tp_size, rank: attn_tp_rank}`). Thread this through to
the loader instead of re-deriving from env vars (avoids env-var drift, single
source of truth):

```rust
// In dsv4.rs model-load (caller holds &TpRuntime):
let attn_tp_cfg = tp.attn_tp_config();

// load_dsv4_attention() takes `attn_tp_cfg: &TpConfig`:
// - head_shard(num_q_heads, num_kv_heads, attn_tp_cfg) → local_q = 128/4 = 32
// - shard_for(config, &names.wq_b, attn_tp_cfg.world_size)
// - column_shard(wq_b_rows, attn_tp_cfg) → shard offset = attn_tp_rank * (rows/4)
// - kv_load_block_index(num_kv_heads, attn_tp_cfg) → KV head block index
// Same for wo_a, wo_b (wq_a is Replicated, unchanged)
```

**Indexer weights**: `load_dsv4_indexer()` uses `load_dsv4_global_matrix()`
(full replicated, not sharded). No change needed for indexer — see P15.

**Expert weights UNCHANGED**: `ExpertSplit::new(config.n_routed_experts,
tp_cfg.world_size, tp_cfg.rank)` stays on global `tp_size`. EP is global,
experts are unique per rank (256/8 = 32), NOT replicated across DP groups.

**attn_dp=1 baseline**: When `attn_dp_size == 1`, `attn_tp_cfg == tp_cfg`
(byte-identical). Zero risk to existing path.

### P3 — ncclCommSplit Ordering

**Industry reference**: NCCL docs — `ncclCommSplit` is a collective that must
be called by ALL ranks in the parent communicator in the SAME order relative
to other NCCL operations. Even ranks opting out (`NCCL_SPLIT_NOCOLOR`) must
participate.

**Already handled**: `from_env_with_nccl()` (tp.rs:167-204) does:
1. `attn_tp = split_sub_comm(&backend, &build_attn_tp_groups(cfg), rank)`
2. `moe_ep = split_sub_comm(&backend, &build_moe_ep_groups(cfg), rank)`

Same order on all ranks. `split_sub_comm()` skips the NCCL call only when
`groups.len() == 1` (single global group) — this decision is identical across
all ranks because `cfg` is resolved identically from env.

**No change needed**, but documented so future modifications don't reorder.

### P4 — CUDA Graph Capture with Sub-Communicators

**Industry reference**: NCCL CUDA graph docs — (1) capture status must be
collective (all ranks capturing or none), (2) graph launch ranks must match
capture ranks, (3) graphs from different communicators on the same thread
can deadlock (mitigated by `NCCL_GRAPH_MIXING_SUPPORT`).

**Our scenario**: Graph capture at dsv4.rs:6379 uses `tp.all_reduce_sum()`.
After switching to `attn_all_reduce_sum()`, it captures `self.attn_tp`
sub-comm.

**Coverage**:
- Capture happens AFTER `from_env_with_nccl()` — all sub-comms are initialized.
- All ranks capture the same graph with the same sub-comm (collective capture).
- `attn_tp` handle is stable for process lifetime — no re-capture needed.
- Only one graph at a time is replayed — no multi-comm mixing on same thread.

**Risk**: Low. If `attn_dp_size` changes between runs, process restarts and
re-captures.

### P5 — Numerical Equivalence of Cross-Group Reused KV

**Industry reference**: NVIDIA Megatron-LM + PyTorch DTP docs — bitwise
equivalence across different TP group sizes is NOT guaranteed (fp addition
non-associativity + different allreduce algorithms). bf16 discrepancy:
1e-3 to 1e-5. Does not meaningfully impact inference quality.

**Our analysis**: KV values are `W_kv * x` where `x` is the layer input
(after attention allreduce). `W_kv` is replicated across DP groups (P2 fix).
So `W_kv * x` is bit-identical regardless of which group computed it.

The attention allreduce sums partial outputs across `attn_tp` ranks. With
`attn_tp=4`: ranks 0-3 sum → full output. With `attn_tp=8`: ranks 0-7 sum
→ same full output (same values, different associative order for bf16).

**Conclusion**: Cross-group KV reuse is valid. bf16 rounding may differ by
~1 ULP — within inference noise floor (MoE non-determinism is much larger).
`needle_gate` is the correctness gate, not byte-identity.

### P6 — Engine State Consistency Across Ranks

**Phenomenon**: If different ranks see different tokens for the same request,
`RequestState` diverges and subsequent steps produce garbage.

**Coverage by unified submit**: All ranks compute ALL rows → all ranks see the
same `StepOutput` tokens for all requests. `apply_output()` (infer-core:894)
runs on all ranks with identical data → `RequestState` in `self.active` stays
consistent.

With per-group submit, this would be a risk: ranks in group 0 process group
0's tokens but not group 1's → `active` map diverges. Unified submit
eliminates this entire failure mode.

### P7 — tp_sync_min Must Stay on Global Comm

**Phenomenon**: `tp_sync_min()` (tp.rs:504) computes the minimum KV budget
across all ranks for scheduler admission. If only a subset of ranks
participate, the budget is wrong.

**Coverage**: `tp_sync_min()` uses `self.comm` (global), not `self.attn_tp`.
Called in `admit_waiting()` BEFORE execution — all ranks participate
regardless of `attn_dp_size`. No change needed.

### P8 — Oneshot Allreduce Limitation

**Phenomenon**: The oneshot small-message allreduce path (tp.rs:362-366) is
only available on the global comm. Sub-communicators fall back to NCCL.

**Impact**: With `attn_dp > 1`, attention allreduce (on sub-comm) loses the
oneshot optimization. For decode-sized messages, oneshot is measurably faster
than NCCL.

**Mitigation**: Acceptable for initial implementation. Attention allreduce is
on `attn_tp_size=4` ranks instead of 8 — the NCCL allreduce is already ~2×
fewer ranks, same message size per rank. Net effect is likely still a win.

**Future**: Port oneshot to sub-comms if profiling shows it matters.

### P9 — DeepEP-LL Token-Owned Dispatch

**Phenomenon**: `shard_rows_and_allreduce()` (tp.rs:392) splits `[seq_len]`
rows by `world_size` for DeepEP-LL token-owned dispatch. All ranks must
participate (it's a collective protocol over NVSHMEM).

**Coverage by unified submit**: All ranks are present in the executor → all
enter the DeepEP-LL dispatch/combine collectives. Correct.

Per-group submit would break this: only 4 ranks enter token-owned dispatch →
other 4 ranks' DeepEP combine hangs. Unified submit avoids it.

### P10 — DeepEP Normal Mode

**Phenomenon**: DeepEP normal mode handles EP combine internally via NVSHMEM
(dsv4.rs:5438: "DeepEP combine already reduces the EP-sharded routed output").
Non-DeepEP path needs explicit MoE allreduce.

**Coverage**: DeepEP normal mode is per-rank (each rank routes tokens to its
own 32 experts). Combine over NVSHMEM is rank-agnostic. With unified submit,
all ranks have all rows' tokens → correct routing to all 256 experts. No
change needed.

### P11 — Load Imbalance Across DP Groups

**Phenomenon**: Round-robin assignment may concentrate long-running requests
in one DP group (e.g. all 32K-token prefill requests land in group 0 by
chance).

**Mitigation**: Round-robin is fair in expectation. With unified submit,
"imbalance" doesn't affect execution (all ranks compute all rows regardless).
It only affects which group's sub-comm handles the allreduce — and all
sub-comms are symmetric.

If Phase 4 (true DP batch splitting) is implemented, add a per-group slot
cap: `max_slots_per_dp_group = num_slots / dp_size + 1`. Deferred.

### P12 — Empty-Plan Submit Overhead

**Phenomenon**: With per-group submit, if one group has no rows, its submit
call wastes kernel launch + setup overhead.

**Eliminated by unified submit**: One submit per step. No empty plans.

### P13 — ROCm / Non-H100 Compatibility

**Industry reference**: vLLM PR #47276 — memory access fault on ROCm when
using DPA with FP8 KV cache and AITER MLA backend.

**Our scope**: CUDA only (H20 / sm_90). Not applicable. Flagged for future
ROCm port.

### P14 — `attn_tp == TpComm::Single` Makes Allreduce a No-Op ★ CRITICAL

**Phenomenon**: When `attn_dp=1`, `build_attn_tp_groups()` returns a single
global group → `split_sub_comm()` skips NCCL and returns `TpComm::Single`
(tp.rs:224). `TpComm::Single` in `all_reduce_sum_over()` is a no-op
(`Ok(())`, tp.rs:469) — the buffer is NOT reduced.

If `attn_all_reduce_sum()` blindly calls `all_reduce_sum_over(&self.attn_tp,
...)`, then `attn_dp=1` (baseline) produces **partial attention output**
(never allreduced) → silent correctness bug.

**Fix in `attn_all_reduce_sum()`**:
```rust
pub fn attn_all_reduce_sum(&self, ctx: &CudaContext, buf: &mut CudaSlice<f32>) -> Result<()> {
    if matches!(self.attn_tp, TpComm::Single) {
        return self.all_reduce_sum(ctx, buf);  // attn_dp=1: global comm + oneshot
    }
    self.all_reduce_sum_over(&self.attn_tp, ctx, buf)
}
```

This also preserves the **oneshot optimization** for `attn_dp=1` (P8 said it
was lost — not so: delegate to `all_reduce_sum()` when sub-comm is Single).

**Verification**: `attn_dp=1` baseline `needle_gate` PASS (byte-identical
tok/s to pre-change).

### P15 — Indexer Weights Already Replicated (No Change Needed)

**Phenomenon**: DSv4 MLA indexer has its own `wq_b` (`index_n_heads=64` heads)
and `weights_proj`. If indexer weights were sharded by `tp.world_size`,
rank 0 and rank 4 would hold different head shards → cross-group KV reuse
invalid (different compressed KV → different MLA decompression).

**Already handled**: `load_dsv4_indexer()` (loader.rs:4310) loads both
`wq_b` and `weights_proj` via `load_dsv4_global_matrix()` — **FULL
replicated matrices, not sharded**. All ranks hold all 64 indexer heads.

No allreduce is needed for the indexer path because the output is already
complete on each rank. Verified: `load_dsv4_indexer` does not call
`load_dsv4_block_scaled_sharded` (the sharded loader).

**Conclusion**: No change needed. Flagged so future refactors don't
accidentally shard the indexer.

### P16 — `wo_a` Group-Per-Rank Behavior Change

**Phenomenon**: `wo_a` sharding (v4.rs:917):
`if config.o_groups.is_multiple_of(tensor_parallel_size)` → Column,
else → Replicated.

Assume `o_groups=8` (DSv4-Flash):
- `tp=8`: `8 % 8 == 0` → Column, 1 group/rank → `wo_a_deepgemm` (single cache)
- `attn_tp=4`: `8 % 4 == 0` → still Column, 2 groups/rank → `wo_a_group_deepgemm` (Vec of caches)

The multi-group DeepGEMM path already exists in the codebase
(`wo_a_group_deepgemm: Vec<...>`), so this is a **behavior change, not a
missing feature**. But the DeepGEMM cache structure changes from one entry
to `o_groups / attn_tp` entries — must verify:
1. Cache pre-allocation loop uses `o_groups / attn_tp_size` (derived from
   weight shape, so automatic if weight is loaded correctly).
2. The per-group dispatch in the MoE forward uses the right index.

**Risk**: Low — the code already handles multi-group. Flag for testing.

### P17 — `local_heads` Doubling → FlashMLA Kernel Compatibility

**Phenomenon**: `local_heads = wq_b.rows / head_dim`:
- tp=8: 128/8 = **16 heads/rank**
- attn_tp=4: 128/4 = **32 heads/rank**

FlashMLA kernel takes `num_q_heads` as a runtime parameter — no hardcoded
limit assumed. But verify:
1. `mla_head_dim` buffer pre-allocation: if derived from `wq_b` weight shape
   after loading, it auto-adapts to 32 heads.
2. Shared memory / tile layout in FlashMLA: tile size is a function of
   `head_dim` not `num_q_heads`, so safe.
3. KV cache layout: `kv_layout.rs` uses `num_kv_heads` (global, 8 for
   DSv4-Flash GQA), not `num_q_heads` — unchanged.

**Quick validation**: `INFER_TP_SIZE=4 INFER_ATTN_DP_SIZE=1` runs
`needle_gate` — this exercises attn_tp=4 with 32 heads/rank.

### P18 — HBM Memory Budget

**Phenomenon**: Weight replication across DP groups increases per-rank HBM
footprint. With `attn_dp=2`, each rank holds **2× the attention weight
shards** (32 heads instead of 16).

**Estimate** (DSv4-Flash, FP8 weights, per rank):
| Weight | tp=8 (16 heads) | attn_tp=4 (32 heads) | Delta |
|--------|-----------------|---------------------|-------|
| `wq_b` [heads×dim×hidden] | ~59 MB | ~118 MB | +59 MB |
| `wo_b` [hidden×dim×heads] | ~59 MB | ~118 MB | +59 MB |
| `wo_a` (o_groups scaled) | ~15 MB | ~30 MB | +15 MB |
| Indexer `wq_b` (64 heads) | ~30 MB | ~60 MB | +30 MB |
| **Total delta** | | | **~+160 MB/rank** |

H20: 96 GB HBM. **+160 MB is negligible** (~0.17%). Acceptable.

**When it matters**: `attn_dp=4` on 8 GPUs (attn_tp=2) → 64 heads/rank →
~+480 MB. Still fine. `attn_dp=8` (attn_tp=1, full replication) → ~+1120 MB.
Flag for large-model scenarios (DSv4-Full, not Flash).

### P19 — FlashMLA Q All-Gather + Output Slice Use Global TP ★ CRITICAL

**Phenomenon**: The FlashMLA attention path has TWO collectives, not one:
1. **Q all-gather** (before attention): each rank's `local_heads` Q rows are
   all-gathered so all ranks have all 128 Q heads.
2. **Output slice** (after attention): each rank extracts its
   `[tp_rank * local_width, +local_width]` portion from the full output.

Both use **global TP rank/size/comm**:
- `tp_world = tp.config().world_size` (8) for buffer sizing
- `tp_rank = tp.config().rank` (0-7) for output slice offset
- `tp.all_gather_bf16_raw()` on global comm

With `attn_tp=4, local_heads=32`:
- Q all-gather on global 8-rank comm: gathers `8 × 32 = 256` heads into
  `tp_gathered_q` sized for `h_q = 32 × 8 = 256` → FAILS `h_q ∈ {64,128}` check
- Even if it passed: output slice offset for rank 4 = `4 × 32 × 512 = 65536`
  = past end of `h_q × head_dim = 128 × 512 = 65536` → out-of-bounds read

**Affected sites** (all use global tp):
| Location | What uses global tp |
|----------|---------------------|
| `try_flashmla_prefill_attention` (attention.rs:2230) | `tp_world`, `tp_rank`, `all_gather_bf16_raw` |
| `try_flashmla_decode_attention` (attention.rs:2727) | `tp_world`, `tp_rank`, `all_gather_bf16_raw` |
| `gather_q_row` (flashmla.rs:1181) | `tp_world`, `all_gather_bf16_raw` |
| `slice_out_row` (flashmla.rs:1263) | `tp_rank` for slice offset |
| `slice_out_batched` (flashmla.rs:1326) | `tp_rank` for slice offset |
| `Dsv4FlashMlaDecodeShape::new()` callers | `tp_world` param |
| `Dsv4KvAdapter::new()` (dsv4.rs:1640) | `tp_world` param |
| `Dsv4LayerAttentionState::new()` (dsv4.rs:929) | `tp_world` param |
| `attn_sink` slicing (attention.rs:2537) | `tp_rank * local_heads` |

**Required fixes**:

1. **`TpRuntime`** — add `attn_tp_config: TpConfig` field:
   ```rust
   pub struct TpRuntime {
       config: TpConfig,        // global {world_size, rank}
       attn_tp_config: TpConfig, // {world_size: attn_tp_size, rank: attn_tp_rank}
       ...
   }
   pub fn attn_tp_size(&self) -> usize { self.attn_tp_config.world_size }
   pub fn attn_tp_rank(&self) -> usize { self.attn_tp_config.rank }
   ```

2. **`all_gather_bf16_raw_over(comm, ...)`** — sub-comm all-gather variant
   (mirrors `all_reduce_sum_over` pattern). `gather_q_row` and the FlashMLA
   functions use this with `&self.attn_tp`.

3. **FlashMLA functions** — replace `tp.config().world_size` →
   `tp.attn_tp_size()`, `tp.config().rank` → `tp.attn_tp_rank()`, and
   `tp.all_gather_bf16_raw(...)` → `tp.all_gather_bf16_raw_over(&tp.attn_tp, ...)`.

4. **`Dsv4KvAdapter::new()` / `Dsv4LayerAttentionState::new()`** — pass
   `attn_tp_size` instead of `tp.config().world_size` as `tp_world`.

5. **`attn_sink` slicing** — use `attn_tp_rank` not global `tp_rank`.

**Gate**: `if attn_dp_size > 1` — when 1, `attn_tp_config == config`
(baseline byte-identical). All FlashMLA code paths unchanged for attn_dp=1.

### P20 — KV Cache Per-Rank Memory ★ RESOLVED: No Change

**Initial concern**: If KV heads are sharded by `attn_tp` (changing from tp=8 →
attn_tp=4), each rank holds 2 KV heads instead of 1 → KV cache could double.

**Resolution**: DSv4 FlashMLA stores **full compressed latent KV** on every rank
(replicated, not head-sharded). `Dsv4FlashMlaDecodeShape::total_blocks =
sw_blocks + comp_blocks` depends solely on `max_seq_len` / `compress_ratio` /
`sliding_window`, **not** on `tp_world` or `num_kv_heads`.

Evidence: `kv_layout.rs:1655-1672` — `flashmla_slot_pages` derived from
`Dsv4FlashMlaDecodeShape::total_blocks`; `tp_world` only validates `h_q =
local_heads * tp_world ∈ {64,128}`.

**Conclusion**: `attn_dp=2` does NOT change KV cache per-rank memory. P18's
+160 MB weight replication is the only memory delta.

### P21 — `Dsv4LayerAttentionState(tp_world)` + DSA Audit ★ RESOLVED: Covered by P19

**Initial concern**: `Dsv4DsaSharedScratch` and `Dsv4LayerAttentionState` both
take `tp_world`. If they use it for internal buffer sizing or TP-relative
indexing, they need `attn_tp_size` not global `tp_size`.

**Resolution** (code audit `dsa.rs`):
- `Dsv4DsaSharedScratch::new()` (dsa.rs:149) takes **5 params, no `tp_world`**.
  DSA indexer scratch is sized by `max_seq_len` / `compress_ratio` /
  `num_slots` / `index_n_heads` / `index_head_dim` — all config values, no TP.
- `Dsv4LayerAttentionState::new()` (dsa.rs:1089) takes `tp_world` and passes
  it **only** to `Dsv4FlashMlaDecodeState::new()` (dsa.rs:1157) →
  `Dsv4FlashMlaDecodeShape::new()` for `h_q` validation. Already covered by
  P19 fix chain (caller passes `attn_tp_size` at dsv4.rs:929).

**Conclusion**: No additional changes needed beyond P19.

---

## Changes per File

### 1. `crates/infer-cuda/src/tp.rs` — attn sub-comm infrastructure

**New fields on `TpRuntime`**:
```rust
pub struct TpRuntime {
    config: TpConfig,         // global {world_size, rank}
    attn_tp_config: TpConfig, // {world_size: attn_tp_size, rank: attn_tp_rank}
    comm: TpComm,
    attn_tp: TpComm,
    moe_ep: TpComm,
    ...
}
```

Populate `attn_tp_config` in `from_env_with_nccl()`:
```rust
let attn_tp_config = if multi_axis.attn_dp_size > 1 {
    TpConfig { world_size: multi_axis.attn_tp_size(), rank: coord.attn_tp_rank }
} else {
    cfg  // byte-identical to global when attn_dp==1
};
```

**New accessors**:
```rust
pub fn attn_tp_size(&self) -> usize { self.attn_tp_config.world_size }
pub fn attn_tp_rank(&self) -> usize { self.attn_tp_config.rank }
```

**New method `attn_all_reduce_sum`** (after `all_reduce_sum`, line ~368):
```rust
/// Attention all-reduce over the attn_tp sub-communicator.
///
/// When attn_dp_size == 1 the sub-comm IS `TpComm::Single` (no NCCL split
/// happened) — delegate to `all_reduce_sum` for the global comm + oneshot
/// optimization. When attn_dp_size > 1 each DP group reduces only within
/// its own attn_tp=tp_size/attn_dp_size ranks.
///
/// MoE allreduce keeps using the global `all_reduce_sum` (EP stays global,
/// moe_dp=1 enforced by validate()).
pub fn attn_all_reduce_sum(
    &self,
    ctx: &CudaContext,
    buf: &mut CudaSlice<f32>,
) -> Result<()> {
    if matches!(self.attn_tp, TpComm::Single) {
        return self.all_reduce_sum(ctx, buf);  // baseline: global comm + oneshot
    }
    self.all_reduce_sum_over(&self.attn_tp, ctx, buf)
}
```

**New method `all_gather_bf16_raw_over`** (sub-comm all-gather):
```rust
/// Like `all_gather_bf16_raw` but on an explicit communicator.
/// Used by FlashMLA Q all-gather over the attn_tp sub-comm (P19).
pub unsafe fn all_gather_bf16_raw_over(
    &self,
    comm: &TpComm,
    ctx: &DeviceContext,
    sendbuf: *const c_void,
    sendcount: usize,
    recvbuf: *mut c_void,
) -> Result<()> {
    // Same body as all_gather_bf16_raw but match on `comm` not `self.comm`.
    // Oneshot: skip for sub-comms (only available on global comm).
    match comm {
        TpComm::Single => bail!("single-rank all_gather not needed"),
        TpComm::Nccl(backend) => {
            backend.all_gather(sendbuf, recvbuf, sendcount, DType::BF16,
                ctx.stream.cu_stream().cast())?;
            Ok(())
        }
    }
}
```

**Convenience for FlashMLA**:
```rust
/// Q all-gather over the attn_tp sub-comm.
/// When attn_dp==1 delegates to `all_gather_bf16_raw` (global + oneshot).
pub unsafe fn attn_all_gather_bf16_raw(
    &self,
    ctx: &DeviceContext,
    sendbuf: *const c_void,
    sendcount: usize,
    recvbuf: *mut c_void,
) -> Result<()> {
    if matches!(self.attn_tp, TpComm::Single) {
        return self.all_gather_bf16_raw(ctx, sendbuf, sendcount, recvbuf);
    }
    self.all_gather_bf16_raw_over(&self.attn_tp, ctx, sendbuf, sendcount, recvbuf)
}
```

Remove `#[allow(dead_code)]` from `attn_tp()` (line 255).

### 2. `crates/infer-cuda/src/dsv4.rs` — 6 attention allreduce sites

Replace `self.tp.all_reduce_sum(&self.ctx, &mut attn_out)?`
with     `self.tp.attn_all_reduce_sum(&self.ctx, &mut attn_out)?`

at lines: **3969, 4500, 4930, 5348, 5873, 6379**.

MoE allreduce sites (4125, 4588, 5008, 5557, 5912, 6446): **UNCHANGED**.

Also:
- Line 929 (`Dsv4LayerAttentionState::new()`): pass `self.tp.attn_tp_size()`
  instead of `model.tp.config().world_size` as `tp_world`.
- Line 1640 (`Dsv4KvAdapter::new()`): pass `self.tp.attn_tp_size()` instead
  of `self.tp.config().world_size` as `tp_world`.

### 2b. `crates/infer-cuda/src/attention.rs` — FlashMLA tp_world/tp_rank

In `try_flashmla_prefill_attention` (line ~2230) and
`try_flashmla_decode_attention` (line ~2727):
- `tp_world = tp.config().world_size` → `tp.attn_tp_size()`
- `tp_rank = tp.config().rank` → `tp.attn_tp_rank()`
- `tp.all_gather_bf16_raw(ctx, send, count, recv)` → `tp.attn_all_gather_bf16_raw(ctx, send, count, recv)`
- `attn_sink` slicing: `(sink_base as *const f32).add(tp_rank * local_heads)` → use `attn_tp_rank`

Gate: `if tp.attn_tp_size() == tp.config().world_size` → old code path
(byte-identical).

### 2c. `crates/infer-cuda/src/attention/flashmla.rs` — batched FlashMLA

In `gather_q_row` (line ~1181):
- `tp_world = tp.config().world_size` → `tp.attn_tp_size()`
- `tp.all_gather_bf16_raw(...)` → `tp.attn_all_gather_bf16_raw(...)`

In `slice_out_row` (line ~1263) and `slice_out_batched` (line ~1326):
- `tp_rank = tp.config().rank` → `tp.attn_tp_rank()`

In `Dsv4FlashMlaDecodeShape::new()` callers (kv_layout.rs:1176, 1656):
- `tp_world` param: pass `tp.attn_tp_size()` (need to thread `tp` through
  or pass the size explicitly).

Gate: same `attn_tp_size == world_size` check — baseline unchanged.

### 3. `crates/infer-cuda/src/loader.rs` — weight sharding by attn_tp_size

In `load_dsv4_attention()` (line ~3698): accept `attn_tp_cfg: &TpConfig` as an
additional parameter (caller in dsv4.rs model-load derives it from
`tp.attn_tp_config()`). Use `attn_tp_cfg` for attention weight sharding
(`wq_b`, `wo_a`, `wo_b`) instead of `tp_cfg`. See P2 for the pattern.

Single source of truth: `TpRuntime.attn_tp_config` is computed once in
`from_env_with_nccl()`, never re-read from env vars.

Indexer weights (`load_dsv4_indexer()`) are already loaded as full
replicated matrices via `load_dsv4_global_matrix()` — **no change needed**
(P15).

Gate: `if attn_dp_size > 1` — when 1, `attn_tp_cfg == tp_cfg` (baseline
byte-identical).

### 4. `crates/infer-core/src/lib.rs` — dp_group tag (Phase 4 readiness)

**`RequestState`** (line ~290): add `pub dp_group: u32`.

**`Engine<E, K>`** (line ~402): add `dp_size: u32`, `dp_rr: u32`.

**`submit_request_with_options()`**: round-robin `dp_group` assignment.

**`build_forward_plan()`**: No change — all rows in one plan.

> **Note**: `dp_group` does NOT affect execution in Phase 1-5. It exists
> solely for Phase 4 (true DP batch splitting) where each DP group processes
> only its own rows. Don't look for where it changes scheduler/executor
> behavior — it doesn't, yet.

### 5. `crates/infer-plan/src/lib.rs` — OPTIONAL dp_group tag

Not needed for execution. Add if useful for KV bookkeeping:
```rust
/// DP-attn group that "owns" this request (for KV accounting).
pub dp_group: u32,
```

### 6. `crates/infer-topo/src/topology.rs` — no changes

All group builders + `RankCoord` already implemented and tested.

---

## Execution Flow (8卡, attn_dp=2, attn_tp=4)

```
Engine::step() on ALL ranks (0-7):
  1. admit_waiting()
     - tp_sync_min() on global comm (all 8 ranks) ✓ [P7]
     - round-robin dp_group assignment to new requests
  2. build_forward_plan()
     - ALL active requests → one unified plan
  3. allocate_for_plan(plan)
     - KV alloc from global pool (all ranks, same slots) ✓
  4. executor.submit(plan)  ← ONE submit, all ranks
     Per layer:
     a. Q projection: all 8 ranks compute local Q from wq_b shard
        - Rank 0 (group 0, heads 0-31): 32 Q rows
        - Rank 4 (group 1, heads 0-31): same 32 Q rows (replicated wq_b) [P2]
     b. Q all-gather on attn_tp sub-comm:
        - Group 0: ranks 0-3 allgather 32×4=128 Q heads (full) [P19]
        - Group 1: ranks 4-7 allgather same 128 Q heads (numerically equiv)
     c. Attention compute (FlashMLA): all 8 ranks compute ALL 128 heads
     d. Output slice: extract [attn_tp_rank * local_width, +local_width]
        - Rank 0 (attn_tp_rank=0): offset=0 → heads 0-31
        - Rank 4 (attn_tp_rank=0): offset=0 → heads 0-31 (same as rank 0)
     e. attn_all_reduce_sum on attn_tp sub-comm:
        - Group 0: ranks 0-3 allreduce (4×32=128 heads summed) → full output
        - Group 1: ranks 4-7 allreduce → same full output (numerically equiv) [P5]
     f. MoE: all 8 ranks process ALL rows
        - Each rank routes to its 32 unique experts (EP=8, global)
        - all_reduce_sum on global comm (all 8 ranks) ✓ [P1, no deadlock]
  5. apply_output()
     - All ranks see same tokens for all requests ✓ [P6]
     - RequestState stays consistent
```

---

## Configuration

```bash
# 8卡, TP=8, attn_DP=2 (→ attn_TP=4 per group), EP=8 (global)
INFER_TP_SIZE=8 \
INFER_ATTN_DP_SIZE=2 \
INFER_EP_SIZE=8 \
arle serve --backend cuda --model deepseek-ai/DeepSeek-V4-Flash
```

Validation:
1. `MultiAxisConfig::validate()` — `8 % (2*1) == 0` ✓, `8 % (8*1) == 0` ✓
2. `build_attn_tp_groups()` → `[[0,1,2,3], [4,5,6,7]]`
3. `build_moe_ep_groups()` → `[[0,1,2,3,4,5,6,7]]` (global)
4. Weight sharding: rank 0 and 4 both have heads 0-31 (verify via log)

---

## Risks

| Risk | Mitigation |
|------|------------|
| Weight sharding change breaks attn_dp=1 | `attn_dp > 1` gate: when 1, `attn_tp_cfg == tp_cfg` (byte-identical) |
| `attn_tp==Single` → allreduce no-op (P14) | `attn_all_reduce_sum()` delegates to `all_reduce_sum()` when `TpComm::Single` |
| Indexer weights not replicated (P15) | Already replicated via `load_dsv4_global_matrix()` — no change needed |
| FlashMLA Q all-gather + output slice use global TP (P19) | Use `attn_tp_size`/`attn_tp_rank`/`attn_all_gather_bf16_raw` in all FlashMLA paths |
| `attn_sink` slice OOB on rank 4+ (P19) | `sink_offset = attn_tp_rank * local_heads`, not global `tp_rank` |
| KV cache memory doubles (P20) | **RESOLVED**: FlashMLA latent KV replicated, not head-sharded. No change. |
| DSA / `Dsv4LayerAttentionState` tp_world (P21) | **RESOLVED**: DSA no `tp_world`; `Dsv4LayerAttentionState` covered by P19 chain. |
| bf16 drift between 4-rank and 8-rank allreduce | Within noise floor. MoE non-det dominates. `needle_gate` is the gate. |
| DeepEP-LL with attn_dp > 1 | Unified submit → all ranks present, collective protocol satisfied |
| Graph capture stale comm | Capture after init. Sub-comm stable for lifetime. |
| `wo_a` group-per-rank change (P16) | Multi-group path already exists; verify via `needle_gate` at attn_dp=2 |
| `local_heads` 16→32 kernel compat (P17) | FlashMLA `num_q_heads` is runtime param. Quick test: `TP=4, DP=1` |
| HBM budget (P18) | +160 MB/rank on H20 96GB → negligible |
| Redundant attention compute (2×) offsets allreduce gain | Measure with guidellm. If net loss → Phase 4 (true DP split) |

---

## Phase Rollout

**Phase 1 — TpRuntime attn_tp_config + sub-comm all-gather (0.5 day)**:
- `tp.rs`: add `attn_tp_config: TpConfig` field, populate in `from_env_with_nccl()`
- `tp.rs`: `attn_tp_size()` / `attn_tp_rank()` accessors
- `tp.rs`: `all_gather_bf16_raw_over()` + `attn_all_gather_bf16_raw()`
- `tp.rs`: `attn_all_reduce_sum()` method
- Test: `INFER_TP_SIZE=8 INFER_ATTN_DP_SIZE=1` → attn_tp_config == global config

**Phase 2 — FlashMLA attn_tp plumbing (1 day)**:
- `attention.rs`: `try_flashmla_prefill/decode_attention` use attn_tp_size/rank/allgather
- `flashmla.rs`: `gather_q_row` / `slice_out_row` / `slice_out_batched` use attn_tp
- `kv_layout.rs`: `Dsv4FlashMlaDecodeShape::new()` callers pass attn_tp_size
- `dsv4.rs`: `Dsv4LayerAttentionState::new()` + `Dsv4KvAdapter::new()` pass attn_tp_size

**Phase 3 — Attention allreduce + weight sharding (0.5 day)**:
- `dsv4.rs`: replace 6 attention allreduce sites with `attn_all_reduce_sum()`
- `loader.rs`: attention weight sharding by `attn_tp_size` (wq_b, wo_a, wo_b)
- Gate: `attn_dp > 1` — baseline byte-identical when attn_dp==1

**Phase 4 — Engine DP tagging (0.5 days)**:
- `RequestState.dp_group`, `Engine.dp_size/dp_rr`
- Round-robin assignment in `submit_request_with_options()`

**Phase 5 — Validation + perf (1 day)**:
- `attn_dp=1` baseline: `needle_gate` PASS (byte-identical tok/s to pre-change)
- `attn_dp=2`: `needle_gate.py` PASS
- `bench_guidellm.sh` c-sweep: ITL should improve (4 vs 8 rank allreduce)
- Measure throughput: redundant compute vs allreduce speedup

**Phase 6 (FUTURE) — True DP batch splitting**:
- Pre-attn all-gather across DP groups (collect tokens from all groups)
- Each DP group processes only its own rows (1/dp_size compute)
- Post-attn reduce-scatter or all-gather to share results
- Pattern: SGLang `dp_gather()` / `dp_scatter()` (dp_attention.py)
- Requires `all_gather` + `reduce_scatter` primitives on `attn_dp` sub-comm
- Only if Phase 5 shows redundant compute is the bottleneck

---

## Open Questions

1. **Throughput vs latency**: Unified submit does 2× attention compute but
   gets 2× faster allreduce. Net win? Measure with guidellm.

2. **MoE-DP**: Blocked by `validate()` when `ep == tp`. If `ep < tp`,
   `moe_dp > 1` becomes possible. Defer — attention DP is the primary lever.

3. **Cross-node DP-attn**: Works but allreduce bandwidth worse. Start
   intra-node. Configurable via `INFER_ATTN_DP_SIZE`.
