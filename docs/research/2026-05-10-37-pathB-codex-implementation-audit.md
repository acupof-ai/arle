---
title: #37 Path B codex implementation in-progress audit
date: 2026-05-10
type: research
status: in-progress-tracking
audience: Claude (manager) + reviewers
---

# #37 Path B codex implementation in-progress audit

> Track codex's Path B device-memory `start_pos` implementation against
> my brief `docs/plans/M_37-pathB-device-mem-startpos.md` (2c43bc7)。
> Tick: 7m+ into implementation, working tree dirty on 5 files,running
> CUDA release check。

## Codex implementation choices(audit vs my brief)

### ✅ FFI change matches Path B exactly

`crates/cuda-kernels/src/ffi/attention.rs` diff:
```rust
- start_pos: i32,
+ start_pos_ptr: *const i32,
```

This is **textbook Path B device-pointer pattern** per my brief §2.2:
"prep kernel reads from device pointer instead of launch scalar"。

### ✅ Per-row offset computation

Earlier diff snippet showed:
```rust
let sp_ptr_offset = (sp_ptr as usize + seq_idx * i32_size) as *const i32;
```

→ Caller computes per-row pointer offset。Pattern:base array of
`start_pos[seq_idx]` values pre-uploaded to device,kernel takes single
`*const i32` per call(this row's start_pos pointer)。

This is **conservative correct pattern**:per-row dispatch loop with per-row
device pointer。No risk of multi-row batch race in kernel。Slightly less
efficient than fully-batched single launch with full array,but functionally
sound。

### ✅ Scope expansion to batch_decode.rs

Codex WIP includes `infer/src/model/qwen3/batch_decode.rs` — meaning
device-pointer pattern propagates to **decode path** too。

This is **correct architectural decision**:if prefill uses device-pointer
start_pos,decode must also use device-pointer for consistency(otherwise
graph capture across decode+prefill mixed batches would have inconsistent
metadata layout)。

Per my brief §2.4 "Tests + greedy_consistency 30-50 LOC" — codex likely
needs both prefill + decode passes for greedy_consistency to PASS。

### ⚠ Single pointer vs array — verify safety

`*const i32` 是 single pointer。Two interpretations:
A) Pointer to array of N values(safe,if caller passes array base + kernel
   reads `start_pos_ptr[seq_idx]`内 kernel)
B) Pointer to single value per call(safe,if caller per-row dispatches
   with offset:`sp_ptr_offset` per row)

Codex chose **B**(per-row dispatch with offset)— safer + simpler than A,
but more launches。

**Predicted slight perf cost** vs A:per-row launch overhead(~1 us each,N
seqs = N us extra)。Negligible compared to kernel compute time per row。

### Pending verification(post codex commit)

1. Capture key tuple narrow per my brief §2.1:does codex remove
   `start_positions` + `seq_lens` from key?(必 verify in `prefill.rs` diff)
2. Replay refresh hook:does device tensor get refreshed before each launch?
3. greedy_consistency 数值等价:per row dispatch device-mem read vs original
   scalar read

## Predicted bench outcome(per Path B design)

If codex's implementation follows my brief faithfully(seems to):
- Capture key reuse should approach 100%(no per-request varying fields in key)
- Per skill anti-pattern check:`cudaGraphLaunch` count ≫ `cudaGraphInstantiate` count
- TTFT 4k/c=4 close 30-50% of +76.6% SGLang gap → 1639 ms → 1100-1300 ms range

## Cross-references

- Path B brief:`docs/plans/M_37-pathB-device-mem-startpos.md`(2c43bc7)
- Path A KILL:`docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md`(e462c53)
- Original design:`docs/research/2026-05-09-37-multikey-vs-device-startpos-design.md`(9a477c7)
- Codex's #24 base:`docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`(35fc3cf)
- Pre-built bench template:`docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`(1168381)

## 状态

Codex Path B impl 7m+ in,FFI device-pointer change matches my brief
verbatim,scope expanded to decode path(correct architectural decision)。
Awaits CUDA release check completion + commit + Phase 0v3 5-gate
validation(`scripts/validate_p24_phase0v3.sh`)+ throughput bench A vs B
(`scripts/post_p24_commit_pipeline.sh`)。
