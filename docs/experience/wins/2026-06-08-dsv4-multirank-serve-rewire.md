# DSv4 multi-rank serve re-wired into the rewrite stack (cutover-debt repair)

**Date:** 2026-06-08. **Backend:** CUDA, DSv4-Flash FP8 TP=8/EP=8, 8×H20.
**Status:** foundation landed + verified for the single-short-request case. Three known
follow-ups (one pre-existing, two new) documented below; the headline 900K needle is
blocked by the pre-existing one, NOT by the serve infra.

## Context

The cutover commit `e81b98fb` ("delete legacy infer/ crate — rewrite is the sole serving
stack") **deleted the DSv4 multi-rank HTTP serve** (the legacy `infer` crate's
`MULTIPROC_SERVE` / `deepseek-distributed` path) and the rewrite did not re-implement it:
current `arle serve` single-process loader BAILED on DSv4 (`infer-api/src/loaded.rs:436`,
"multi-GPU only; launch via the parity script"). So DSv4 could not be served at all on the
rewrite — only the prefill-parity example (one-shot, gated decode) existed.

## What worked (verified on 8×H20)

Re-implemented the multi-process serve, grafted onto the rewrite Engine/executor:
- **Coordinator/worker** (`crates/cli/src/serve_multiproc.rs`): `arle serve` detects DSv4+CUDA →
  becomes rank 0, mints the NCCL unique id (`infer-cuda` `mint_nccl_unique_id_hex`, published
  via `INFER_NCCL_UNIQUE_ID` so workers inherit it), forks N-1 workers via `current_exe()` +
  `ARLE_WORKER_RANK`/`WORLD_SIZE`/`INFER_TP_RANK`/`INFER_CUDA_DEVICE` + a parent-fd death pipe.
- **Request relay** (`crates/infer-server/src/multiproc_relay.rs`, ported from
  `e81b98fb^:infer/src/multiproc_relay.rs`): rank 0 broadcasts each admitted request to workers
  at the single FIFO admission point (`execution.rs` `admit_submission`); workers submit to their
  rank-R Engine and run the same step loop → the per-step NCCL executor forward is the lockstep
  barrier (deterministic planner + identical request order ⇒ identical batches).
- **Un-bail DSv4** (`loaded.rs`): build the rank-0 executor+Engine instead of bailing.
- **HTTP body limit** (`infer-server/src/http.rs`): `DefaultBodyLimit::max(256 MiB)` — a 900K-token
  prompt is several MB, over axum's 2 MiB default.
- **DSv4 prefill scratch memory fixes** (so a 920K-max-seq-len slot loads): the DSA-indexer
  per-layer scratch is sized by a 1024-query tile (was 4096 → ~74 GB OOM at 900K) and
  `raw_indices` by the prefill chunk (was `max_seq_len × topk` ≈ 1.9 GB/layer); the
  `Dsv4PrefillDeepGemmLinearScratch` M dimension is chunk-bounded. Net: DSA prefill scratch
  ≈ 19 GB at 900K, fits the ~31 GB free after weights (~40 GB/rank, replicated dense) + KV (~22 GB).

**Verified:** `arle serve --backend cuda --model-path DeepSeek-V4-Flash` on 8×H20 launches all
8 ranks (relay connect + NCCL form), HTTP up; smoke `"The capital of France is"` →
**" Paris. The capital of France is Paris…"** (correct, coherent, continuous-batched decode —
the production decode path, not the gated example). Serve loads at `INFER_DSV4_MAX_SEQ_LEN=920000`
(66 GB/rank, no OOM).

## Known follow-ups (NOT fixed here)

1. **(pre-existing, the real 900K blocker) DSA-active long-sequence decode = non-deterministic
   garbage.** At ≥~1.5K context (once the compressed/sparse DSA path activates), generated tokens
   are garbage and non-deterministic. This is NOT the serve: the **parity example** (single
   process group, one-shot, the validated DSA path) shows the SAME garbage + `ref_self_parity=false`
   at 1500, and its launcher NOTE states incremental decode (`start_pos>0`) is unverified.
   Matches `reference_dsv4_longctx_decode_broken_and_deepgemm_skew` (deterministic+retrieves at
   ~37 tok, garbage beyond). Buffer allocation (shared vs per-layer) was ruled OUT — both garbage.
   → separate investigation: DSv4 decode long-sequence stability (CSA/HCA/SWA KV logic).
2. **(serve) extend prefill (≥2 chunks) crashes** the engine thread: `DSv4 slot seq_len N !=
   start_pos M; decode requires contiguous appends` — a chunked-prefill start_pos/slot-state
   tracking bug in the multi-rank driving. Single-chunk (≤ chunked_prefill_size) is fine.
3. **(serve) concurrency c>1 crashes** the engine thread — the lockstep continuous-batching across
   ranks desyncs when multiple requests batch together. Single request is fine.

## Rule

- DSv4 serving works on the rewrite again (single short request); the multi-rank coordinator/relay/
  NCCL-lockstep foundation is real. The 900K needle is gated by the **pre-existing DSv4 decode**
  correctness (1), verified by the parity example exhibiting it independently of the serve.
- pending-remote: full sweep / c-sweep deferred until (1)-(3) land. cargo check (cuda,nccl,no-cuda)
  PASS; on-pod smoke PASS.
