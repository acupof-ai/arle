# DSv4 TRUE batched decode — FFN/all-reduce amortized over N rows (pending-remote)

## SLO-shape probed?  N — pending-remote (CUDA bench runs on the 8×H20 pod, not this Mac worktree)

## Status

`pending-remote` — Rust+typecheck landed in this worktree (Mac, no nvcc).
Pod build + correctness-parity + c-sweep bench is the next step (see
**Correctness gate** + **Bench to run on pod** below). No silent skip: this
entry is the stub the verify-phase requires; it upgrades to a real bench entry
after the pod run.

## Context

DSv4 `forward_decode_batch` (`infer/src/model/deepseek/forward.rs`) LOOPED
`forward_decode` per request → N concurrent decode requests ran N separate full
43-layer forward passes. Measured (campaign I1,
`docs/projects/2026-05-29-dsv4-beat-sglang-30pct-campaign.md`): decode step
88 ms at c=1, c≥8 timeouts (>300 s, step×512 ≈ 264 s, **zero batching
speedup**). The I1 decode profile attributes the per-step cost to **attn_core**
(per-sequence CSA-select + compressor + indexer + FP8 pool) **plus a serial
~21 ms NCCL all-reduce** (ffn_all_reduce + attn_all_reduce × 43 layers, NOT
overlapped). The all-reduce + routed-MoE expert GEMMs are the **batchable**
lever; the attention core is irreducibly per-sequence (each sequence's KV
caches live in `DeepseekState.incremental.layers[l].attention` — the FP8 pool,
SW ring, FlashMLA decode arena are all per-state, and the FlashMLA decode FFI
is hard-coded `b=1, s_q=1`).

## What changed (file:line)

1. **`infer/src/model/deepseek/weights.rs`**
   - `forward_transformer_layer_stream_incremental_into` — split into an
     **attention half** (`forward_attention_half_incremental_into`, new) +
     the existing FFN half. The single-row path is unchanged behaviorally:
     attention-half writes the post-attention residual stream into the
     `attn_post` scratch, FFN-half consumes it. Pure refactor (extract method).
   - `compute_top_level_logits_incremental_batch` (new) — TRUE batched decode
     for N single-token sequences: batched token embeddings + initial HC
     stream; per layer { **attention half per-row** (each into its row of a
     batched `attn_stream` via `write_hidden_row`, using that row's own state's
     per-layer KV cache at its own `start_pos`) + **FFN half ONCE over the
     N-row stream** (MHC/hc_pre/RMSNorm/expert route+GEMMs/**NCCL all-reduce
     over `[N, hidden]`**/shared expert/hc_post) }; then per-row head HC +
     lm_head logits. Returns `Vec<DeviceVec>` (one `[1, vocab]` per row).
   - `write_hidden_row` (new helper) — d→d scatter of a `[1, width]` row into
     row i of a batched `[N, width]` `HiddenStates`.
2. **weights.rs `try_decode_batch`** (new, `pub(super)`) — eligibility gate +
   logits scatter (`state.decode_logits = logit; prefill_logits = None;
   kv_cache.advance_seq_len(1)` per row, mirroring the per-row path exactly).
3. **`infer/src/model/deepseek/forward.rs`** `forward_decode_batch` — calls
   `try_decode_batch`; on `Ok(false)` falls through to the **retained per-row
   loop** (correctness reference + fallback, never deleted).

## What batches vs. falls back

**Batches over the N rows (one launch per layer instead of N):**
token embeddings + initial HC expand; FFN half = MHC(ffn), hc_pre, RMSNorm,
routed-MoE expert route + GEMMs, **NCCL all-reduce over `[N, hidden]`**, shared
expert, hc_post. (Head HC + lm_head are issued per-row because the head HC
kernel is last-token / single-row only — cheap, exact parity.)

**Stays per-row (looped inside the batched layer, byte-identical to the per-row
decode path):** the **attention core** — MHC(attn), hc_pre, RMSNorm, sparse /
sliding-window attention, hc_post — because each sequence's KV caches (SW ring,
compressed, FP8 pool, FlashMLA decode arena) are per-`DeepseekState` and the
FlashMLA decode FFI (`arle_flashmla_sm90_sparse_decode_sched_meta(b=1, s_q=1)`,
`dsv4_flashmla_decode_build_indices_raw` single-row, `dsv4_flashmla_pack_one_sw_token`)
is hard-`b=1`. A single batched `b=N` FlashMLA decode would need NEW CUDA
kernels (batched indices builder with per-row `start_pos[]`, batched pack,
batched sched_meta, a shared cross-sequence KV pool) — deferred to a CUDA
tranche; not attempted here (cannot build/verify CUDA on this Mac).

**Falls back to the per-row loop (returns `Ok(false)`):** N == 1; CPU reference
model active (`self.reference.is_some()`); `ARLE_DSV4_INCREMENTAL_KV` off; any
of embed/head/norm/lm_head/layers not loaded.

## Correctness gate (why greedy output is byte-identical to per-row)

Every batched op is either row-independent (embed / MHC / RMSNorm / GEMM —
identical math whether issued per-row or stacked) or a sum-reduce (the NCCL
all-reduce result is identical per-row vs. over the stacked batch). The
attention core is the **unchanged** per-row path. Hash-routed MoE layers route
on the per-row **token id** (`gate_tid2eid`); the batch passes `tokens` in slot
order so row i gets `tokens[i]` — same id the per-row path uses. ⇒ batched
greedy output == per-row greedy output, byte-for-byte.

## Bench to run on pod (8×H20 TP=8, DSv4-Flash)

Working serving config per
`wins/2026-05-29-dsv4-gpu-native-coherent-output-pd-handoff.md` but with
`--num-slots ≥ 8`:
```
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7
ARLE_DSV4_LOAD_LAYER_WEIGHTS=1 ARLE_DSV4_GPU_FULL_LAYERS=43
ARLE_DSV4_INCREMENTAL_KV=1 ARLE_DSV4_FLASHMLA_PREFILL=1 ARLE_DSV4_FLASHMLA_DECODE=1
ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_EXPERT_BACKEND=native
--num-slots 8 --max-seq-len 4096 --mem-fraction-static 0.10 --kv-cache-dtype fp8
--deepseek-distributed-layers 43
```

**Correctness-parity test (mandatory before any default claim):** with the same
seed + greedy sampling, decode the same prompts at c={2,4,8} two ways —
(a) batched path (this change, the default once eligible) and
(b) force per-row by disabling the batch entry (temporarily make
`try_decode_batch` return `Ok(false)`, OR set N=1 by serializing requests).
**Assert the decoded token streams are byte-identical per request.** Expected:
identical (the math is row-independent / sum-reduce + unchanged attention).

**Perf c-sweep:** `scripts/bench_guidellm.sh dsv4-batched-decode` vs. the I1
baseline (c=1 = 5.65 tok/s; c≥8 timeout). Expected: c≥8 no longer times out;
per-step time flattens vs. batch size to the degree the FFN + all-reduce
(amortized) dominated. Attention core still scales ~linearly with N, so the win
is bounded by the FFN/all-reduce fraction of the step — report the measured
flattening, don't assume.

## Typecheck (this worktree, Mac)

```
CUDARC_CUDA_VERSION=12090 cargo check -p infer --lib \
  --no-default-features --features cuda,no-cuda,nccl   # clean (0 errors)
CUDARC_CUDA_VERSION=12090 cargo check -p infer --lib \
  --no-default-features --features cuda,no-cuda        # clean (0 errors)
cargo test --release -p infer --lib                    # 594 passed
```
clippy on the deepseek module: my new code (try_decode_batch /
compute_top_level_logits_incremental_batch / write_hidden_row /
forward_attention_half_incremental_into) is warning-clean; remaining warnings
are pre-existing.

## Rule

For a model whose KV caches are **per-state** and whose attention FFI is
**hard-`b=1`**, "TRUE batched decode" splits each layer into the per-row
attention core (unchanged, correctness reference) and a **batched FFN +
single all-reduce over `[N, hidden]`** — attack the serial collective, not the
irreducible per-sequence attention. A single batched FlashMLA decode is a
separate CUDA-kernel tranche, not a Rust-only change.
