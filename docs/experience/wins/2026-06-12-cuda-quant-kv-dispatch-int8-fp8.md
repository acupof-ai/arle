# INT8/FP8 paged quant-KV wired into the dense-Qwen3 CUDA path (#68 T3)

**Date:** 2026-06-12. **Backend:** CUDA, dense Qwen3 (gate model Qwen3-0.6B).
**Scope:** `cuda-kernels` (2 new per-channel-K dequant kernels + wrappers),
`infer-cuda` (executor dtype admission, loader PageMeta quant fields,
attention refill/calibrate/quantize/fused-decode dispatch), `infer-api`/`cli`
threading. **Commits:** `d9f930b6` (port), `b2415c01` (codex-review P2:
workspace gate on the 1-split floor).
**Status: VERIFIED on H20** (same day) — **correctness LICENSED for INT8 and
FP8.** The initially reported decode −77% at B=1 was root-caused same-day to
an uncached `cudaGetDeviceProperties` in the quant decode shim (see
§Perf row); post-fix the lanes sit at **−27% vs the bf16+graph default,
−7% vs eager bf16 apples-to-apples**. Opt-in only; no default flip.

## Context

#68 left Qwen3.5's BF16/INT8/FP8/TQ4 matrix BLOCKED until the monolith's
quant-KV dispatch was re-ported to the rewrite's paged-KV path. The pool
plumbing (T2) and `--kv-cache-dtype` threading (T4) were already in; T3 is
the missing op layer: prefix refill, KIVI calibration, row quantize, and
fused-dequant decode behind the seam dispatch.

## What Worked

Not a 1:1 monolith port — four design corrections over the template:

1. **NEW per-channel-K dequant kernels**
   (`dequantize_paged_kv_{fp8,int8}_per_channel_k_to_hnd`): the monolith
   refilled K through the per-token dequant path, but under per-channel K
   quantize the per-token K scales are never written → prefix refill read
   zeros. K refill now consumes `k_static_scales[kv_head*head_dim+d]`.
2. **Latch-once KIVI calibration** (`k_kivi_calibrated` AtomicBool per
   layer) for both INT8 and FP8 — recalibrating on a later chunk would
   silently corrupt earlier chunks' already-quantized K under chunked
   prefill.
3. **Decode CUDA-graph hard-disabled for quant pools** (warmup early-return
   before `decode_graph_enabled()`); decode runs eager through
   `decode_attention_{fp8,int8}_per_channel_k`. V1 tradeoff, revisit only
   with a wall-clock license.
4. **1-token-prompt edge:** decode path calibrates-from-single-row when the
   latch is unset, so a prompt that never hits the prefill finalize still
   gets valid scales.

Explicitly deferred: **TQ4** (TurboQuant pools are page_size=1; the TileLang
paged prefill kernels are compiled for PAGE_SIZE=16 — no paged-prefill path
ever existed; `resolve()` bails with a pointer to #68). Accepted dead
weight: per-token `k_scales` plane still allocated under per-channel K
(unused; removal is a pool-layout change, not V1).

Pool transports needed **zero changes**: tier store
(`copy_pages_to_host/from_host`) and radix COW
(`copy_pages_device_to_device`) already carry data plane + scales + norms
per layer.

## Verification

Mac-side: `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api` (cuda,no-cuda)
PASS; `cargo test -p agent-infer` (cpu,no-cuda,cli) PASS; `cargo test -p cli`
147/147; clippy `-D warnings` clean; fmt clean.

H20 pod (sm_90a, CUDA 12.9, `scripts/dsv4_toolchain.sh build`, 2026-06-12):

- **nvcc first compile of both new kernels: PASS** (full `cuda,nccl` link).
- **GPU kernel tests 3/3**: `hnd_refill_{fp8,int8}_per_channel_k_matches_scale_table`
  (exact bf16-bit match vs the scale table; head_dim 8 = vectorized path,
  head_dim 7 = scalar fallback) + the pre-existing reference test.
- **Needle gates** (`GATE_PROFILE=generic MODEL=/data01/models/Qwen3-0.6B
  SERVE_FLAGS="--kv-cache-dtype <dt>" scripts/lever_gate.sh qwen06b_<dt>_raw
  RAW=1`; RAW=1 mandatory — chat template burns the budget in `<think>`):

  | dtype | len 115/300/446/2000/8000 | verdict |
  |---|---|---|
  | bf16 (envelope, hd128 entry) | exact=3 partial=0 miss=0 DET ×5 | baseline |
  | int8 | exact=3 partial=0 miss=0 DET ×5 | **LICENSED (correctness)** |
  | fp8 | exact=3 partial=0 miss=0 DET ×5 | **LICENSED (correctness)** |
  | tq4 | — | DEFERRED (`resolve()` bails; page_size=1 vs PAGE_SIZE=16) |

- **Path probe** (RUST_LOG=info, not inferred): pool boots `format=INT8`
  (data 268.4 MB/layer vs 536.9 MB bf16 — the 2× KV-memory win) and warmup
  logs "decode graph disabled for quantized KV … fused-dequant kernels".

## Perf row (same-binary, same-shell, same-prompt, side-by-side ×3)

**Correction (same day).** The first version of this entry reported int8/fp8
at ~100 tok/s = −77% vs bf16 439, with the parenthetical "no decode graph in
any lane (INFER_CUDA_DECODE_GRAPH unset)". That parenthetical was **wrong**:
unset falls back to the default, and the decode graph default is **ON**
(`DECODE_GRAPH_DEFAULT_ENABLED`, landed `6a01e8cc`, predates the pod build) —
so the bf16 439 was graph-ON while quant lanes hard-disable the graph
(correction #3 above). The −77% conflated three separate effects:

1. **The actual bug (−61% of the quant step):** `choose_decode_num_splits`
   in `decode_attention_quantized.cu` called the full ~100-attribute
   `cudaGetDeviceProperties` **per layer per decode step**. nsys
   `cuda_api_sum` evidence: 112 calls @ avg 218 µs in the int8 warmup trace
   (112 = 28 layers × 4 warmup decode steps — exact per-layer-per-step
   signature); the bf16 lane had 0 calls. 28 × 218 µs ≈ **6.1 ms/token** of
   the ~10 ms quant step. Fixed with a static SM-count cache
   (`cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` once, mirrors
   `device_num_sm()` in `arle_fa3_shim.cu`); identical split choices, pure
   overhead removal. Shared by the int8/fp8/int4 decode variants.
2. **Graph-vs-eager (−21%):** quant pools run eager (V1 tradeoff above);
   the bf16 default replays the whole-step graph.
3. **Quant kernel cost (−7%):** the genuinely extra work — calibrate-check +
   row-quantize + two-phase split-KV vs TileLang's single tuned bf16 kernel.

Post-fix 4-lane A/B (B=1 decode, 256 tokens ×3, Qwen3-0.6B, H20 GPU 0,
same binary/shell/prompt, path markers verified in serve logs):

| lane | tok/s (steady) | ms/token | Δ vs bf16+graph |
|---|---|---|---|
| bf16 + graph (default) | 439–440 | 2.27 | — |
| bf16 eager (`INFER_CUDA_DECODE_GRAPH=0`) | ~346 | 2.89 | −21% |
| int8 (eager, forced) | ~322 | 3.11 | −27% (−7% vs eager bf16) |
| fp8 (eager, forced) | ~321 | 3.11 | −27% (−7% vs eager bf16) |

Post-fix needle gate re-run (int8, RAW=1, len 446/2000/8000): exact=3
partial=0 miss=0 DET — the cache changes no kernel math, correctness
licensing stands.

**Verdict:** quant recovers 3.2× (≈100 → ≈322 tok/s). The remaining −27% is
fully attributed: −21% graph (lever P1 — re-enable the whole-step decode
graph for quant pools; post-KIVI-latch the step sequence is shape-static;
belongs with the batched-lane work per the 2026-06-11 graph entry) and −7%
quant kernels (lever P2 — fuse quantize into the decode-prep kernel). The
lane's value proposition is **2× KV capacity** (INT8 pool 8.3 GB vs BF16
15 GB at boot); B=1 tok/s on a 0.6B model is launch-bound everywhere
(roofline ~3300 tok/s; even bf16+graph sits at 13%) and is a regression
signal, not the lane's SLO. guidellm is absent on the pod; this
same-session side-by-side A/B is the Δ% row.

## Rule

- A quant-KV port is four dataflows (refill, calibrate, quantize, decode),
  not one kernel swap — enumerate every buffer each dataflow writes and
  prove the scale plane it READS is the one the quantize path WRITES.
  The monolith's refill-reads-unwritten-per-token-K-scales bug survived
  there because the monolith never refilled a per-channel pool.
- **A −2× to −5× lane gap is a bug until traced, not a cost model.** The
  "~3 extra launches/layer" story was source-survey hypothesis and absorbed
  a 6.1 ms/token host-API stall without question. One nsys `cuda_api_sum`
  found it in minutes: per-lane API-call *counts* (112 vs 0) localize a
  per-layer-per-step culprit faster than kernel timings do.
- **"env unset" ≠ "feature off."** Before framing an A/B around a flag,
  read the default-resolution path (`DECODE_GRAPH_DEFAULT_ENABLED` is ON);
  the original perf row's "no decode graph in any lane" parenthetical was
  false and mis-attributed the graph delta to the quant kernels.
