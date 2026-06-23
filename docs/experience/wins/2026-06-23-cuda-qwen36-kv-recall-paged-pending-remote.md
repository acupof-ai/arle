# Qwen3.6 hybrid CUDA — opt-in paged full-attn KV-recall path (`--kv-recall`) — pending-remote

`pending-remote`: implemented + Mac-cross-compiled (no nvcc/GPU locally). The
H20 correct-inference (needle) + decode-cost gate below is the bench/correctness
gate; the human runs it via `bin/pod`. This stub lands per the §Benchmarks rule
(every runtime change → a wins/ entry or pending-remote stub). The path is
**opt-in, default-off** — the contiguous full-attn decode is byte-identical when
`--kv-recall` is absent, so this is not a default flip.

## Context

Extends session KV-recall (the dense-Qwen3 arm shipped `898bb979` +
`docs/experience/wins/2026-06-23-cuda-kv-recall-pending-remote.md`) to the
**Qwen3.6 hybrid CUDA executor** (`Qwen35CudaExecutor`). The dense arm already
runs paged attention over a page table; the Qwen3.6 hybrid arm did **not** — its
full-attn layers used per-slot **contiguous** K/V caches
(`Qwen35SlotState::k_caches`/`v_caches`), so recall (restricted page table +
page-granular evict) had no page-addressable storage to act on. `set_kv_recall`
on the Qwen35 arm previously warned and ignored.

This wires an opt-in **device-only** `PagedKVPool` for the full-attn layers
(HD256, `num_full` layers, BF16) and routes the full-attn forward through the
registered HD256 paged kernels when `--kv-recall` is on. Linear-attn
(GatedDeltaNet) layers are untouched — they hold recurrent/conv state in the
slot, not KV. Cross-links the
[session-infinite-kv plan](../../plans/2026-06-23-session-infinite-kv-memory.md).

## What's real vs stubbed

| Piece | Status |
|---|---|
| Lazy opt-in BF16 `recall_kv: PagedKVPool` (HD256, `num_full` layers, device-only) on first `set_kv_recall(true)` | **real** (`Qwen35CudaExecutor::set_kv_recall`) |
| Paged full-attn forward: `decode_prep_paged_hd256` → `resolve_paged_attn_v1(256,q,kv,Decode/Prefill)` → `attention_gate_paged_hd256` | **real** (`Qwen35Model::full_attention_paged`) |
| Per-head sigmoid gate preserved (read from `q_full`'s gate half) | **real** (step 3 of `full_attention_paged`) |
| Prefill writes KV into the recall pool (`alloc_tokens` + `PageMeta::for_slot` + paged-prefill kernels) | **real** (`prefill_row_recall`) |
| Decode reads recall-restricted page table (`PageMeta::for_recall_decode`) + stale-Q rescore + evict-drop | **real** (`decode_row_recall`, mirrors dense `try_recall_decode`) |
| Reps + `q·rep` scoring + `plan_recall` reused verbatim (`CudaRecallState`, head-dim/kv-agnostic) | **real** (`crate::recall`, no changes) |
| Mid-decode device-page **free** out of HBM (`PagedKVPool::evict_slot_page`) | **real** — and this arm has **no host tier**, so the evict is an outright device-page free (the flat-VRAM lever the dense arm documented as blocked) |
| Default-off byte-identity (contiguous path is the `else` of a clean `if recall_active`) | **real** (every non-recall caller passes `None`) |
| GPU correctness (needle) + flat-VRAM / decode-cost numbers | **pending-remote** (this doc) |

### Flat-VRAM note (differs from the dense arm)

The dense-Qwen3 recall landing documented the device-page free as **blocked** by
the host single-allocator (`mirror_slot` re-publishes the contiguous page table
each step). This Qwen3.6 arm has **no host `KvPool` for full-attn KV** — the
recall pool is device-only and self-allocating — so `evict_slot_page` frees the
HBM page immediately with no allocator contention. There is no host tier here, so
an evicted block is **dropped** (not mirrored): the recall plan masks evicted
blocks to `-inf` in `recompute_recall_plan`, so a dropped middle block is never
re-attended (decode is tier-free; re-recall would need a prefill prefetch — the
accepted write-through boundary). The pod test must therefore confirm VRAM
**flattens** for recall ON past the working-set budget (this arm, unlike dense,
should bend the curve now).

### Concurrency / batched-decode routing

The recall decode path is **inherently serial per slot** (host stale-Q readback
+ `q·rep` rescore + evict-drop between steps, and the full-attn KV lives in the
paged pool, not the contiguous caches the batched HD256 decode kernel reads). So
when `--kv-recall` is on, `submit_decode_batch` (rows>1) routes **per-row**
through `submit_decode_row` → `decode_row_recall` instead of the batched
`fused_gqa_attention_decode_batched` kernel (which has no page table). With
recall off, batched decode is unchanged. The whole-step decode graph lane is also
bypassed under recall (it bakes contiguous-cache addresses). Net: recall trades
the batched-decode / graph throughput levers for the flat-VRAM + recall win —
the pod decode-cost gate should measure recall-ON c=1 ITL vs recall-OFF c=1 (the
single-user shape recall targets), not a batched throughput sweep.

## Design (load-bearing, one sentence)

When `self.kv_recall && self.recall_kv.is_some()`, the full-attn KV path is
**always** the recall pool (prefill writes pages, decode reads pages, no
mid-session switch); when off, **always** contiguous — a clean `if/else` in
`forward_hidden_staged` so the default baseline is byte-identical. The captured
decode graph (which bakes contiguous-cache addresses) is bypassed when recall is
active (recall needs the host query read-back + restricted table between steps —
eager-only, same as the dense arm).

Verbatim default-off gating branch (`Qwen35Model::forward_hidden_staged`):

```rust
if let Some(rc) = recall.as_deref_mut() {
    self.full_attention_paged(full_attn, normed, full_idx, full, attn_out, rc, seq_len == 1)
} else {
    self.full_attention(full_attn, normed, slot, full_idx, start_pos, start_pos_dev, full, attn_out)
}
```

`recall` is `None` on every non-recall path (`forward_hidden`, captured-decode,
spec-verify), so the contiguous branch is byte-for-byte unchanged with
`--kv-recall` off.

## Files changed

- `crates/infer-cuda/src/qwen35.rs` — `Qwen35RecallForward<'a>` (pool + per-step
  `PageMeta` + `layer0_query`); `full_attention_paged` (the HD256 paged
  prep→TileLang→gate sequence into the recall pool, layer-0 post-RoPE Q
  read-back); `local_kv_heads()`/`local_q_heads()` accessors; threaded
  `Option<&mut Qwen35RecallForward>` through
  `forward_hidden`/`forward_hidden_capture`/`forward_hidden_staged`; new
  `forward_tokens_recall`.
- `crates/infer-cuda/src/executor.rs` — 5 recall fields on `Qwen35CudaExecutor` +
  constructor init; `set_kv_recall` (lazy BF16 recall-pool alloc, page_size
  guard); `recall_active`/`prefill_row_recall`/`decode_row_recall`; prefill/decode
  dispatch routed to the recall path when active + slot-reset frees the pool;
  `RealCudaExecutor::set_kv_recall` now returns `Result<()>` and routes Qwen35.
- `crates/infer-cuda/src/lib.rs` — `CudaExecutor::set_kv_recall` returns
  `anyhow::Result<()>` (propagates the lazy pool-alloc error).
- `crates/infer-api/src/loaded.rs` — CUDA build path propagates the `Result`
  (`set_kv_recall(config.kv_recall)?`); Metal path untouched.

## Kernel triplet (decode, `--kv-recall` on)

1. `ffi::decode_prep_paged_hd256_cuda` — RMSNorm + partial-RoPE on Q/K; writes the
   new token's RoPE'd K + raw V into the last page of the recall-restricted page
   table (HND pool layout). `stride_page = pool.kv_dim * pool.page_size`.
2. `ffi::resolve_paged_attn_v1(256, local_q_heads, local_kv_heads, AttnPhase::Decode)`
   → the registered HD256 decode kernel (q16_kv4 / q16_kv2 / q8_kv2 per shard).
   18-arg base BF16 ABI (`[abi.paged_attn_v1]`, `kernels.toml`). Q = the prep's
   RoPE'd `q_prepped`.
3. `ffi::attention_gate_paged_hd256_cuda` — applies `sigmoid(gate)·attn_out` in
   place, reading the gate half of `q_full`.

Prefill uses `prefill_attention_paged_prep_hd256_cuda` + the `Prefill`-phase
resolve + the same gate.

## Pod test plan (H20, run via `bin/pod`)

Serve Qwen3.6 (the canonical MoE hybrid) on a single H20 (recall is eager-only;
disable the decode graph explicitly so the path is unambiguous):

```bash
INFER_CUDA_DEVICES=<gpu> ARLE_QWEN35_DECODE_GRAPH=0 \
arle serve --backend cuda --kv-recall --kv-cache-dtype bf16 \
  --model-path <Qwen3.6 checkpoint> --port 8000
```

Baseline control (recall OFF, same binary/shell/model/device): drop `--kv-recall`.

### 1. Correctness — long-context needle (correct-inference gate, NOT byte-identity)

Recall + restricted attention deliberately deviates from a single full-KV run →
gate is **needle retrieval = the full-attention answer**, not token-exact vs
baseline (`feedback_correct_inference_not_baseline_identity`). Plant a passkey at
mid-depth in a prompt LONGER than the working-set budget (sink 32 + local 256 +
8×32 = **544 tokens** with `default_recall_config`); e.g. ~6K-token context,
passkey at depth 0.5. Pass = recalled answer contains the planted passkey. Run x3
same-config repeats vs the baseline envelope (absorbs MoE non-determinism). A
streaming control (sink+local, no recall) should MISS at the same budget — that's
what makes recall load-bearing.

### 2. Flat-VRAM / decode-cost vs history (the decisive evidence)

Drive ONE session far past the working-set budget (4K → 8K → 16K → 32K via the
same `session_id`), recording per-step decode ITL and `nvidia-smi` VRAM for both
arms. **Assertions:** (a) recall ON VRAM holds ~flat past 544 tokens (device-page
free, no host tier) while OFF grows with `cache_len` — this arm should bend the
curve NOW (unlike dense); (b) recall ON decode ITL holds ~flat (bounded working
set) while OFF rises; (c) the early needle is still retrieved at 32K under recall
ON.

### 3. Default-off regression (mandatory, byte-identical)

Recall OFF arm with the decode graph ON (default) — tok/s matches the latest
Qwen3.6 CUDA baseline within noise (the recall code is behind
`if self.recall_active()`; the graph path and contiguous full-attn are untouched).

## Gates (local, Mac — no nvcc)

- Mac CUDA typecheck: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release
  --no-default-features --features cuda,no-cuda --lib` → **clean** (0 errors, 0
  warnings beyond the expected no-cuda skip notice).
- `cargo test -p infer-core -p infer-seam` → **78 + 28 passed**, 0 failed.
- cpu/no-cuda builds: `cargo check -p infer-cuda --no-default-features
  --features no-cuda` and `cargo check -p infer-api --no-default-features
  --features cpu,no-cuda --lib` → **clean** (the `Placeholder` arm returns
  `Ok(())`).

## Rule

The Qwen3.6 hybrid full-attn KV can be paged with the **existing** registered
HD256 paged kernels + the **existing** device-neutral recall core
(`PageMeta::for_recall_decode`, `CudaRecallState`) — no new kernel, no new policy.
Because this arm has no host KV pool for full-attn KV, recall's device-page
evict is an outright HBM free (the flat-VRAM lever the dense arm documented as
blocked). Opt-in only; the contiguous decode is the `else` of a clean
`if recall_active`, byte-identical with `--kv-recall` off. GPU needle +
decode-cost + VRAM-flatten verification is the human's pod gate before any default
flip (no flip proposed here).
