# DSv4 decode batched compressor pre-pass (bf16 cublasLt) — perrow −39%, default-off

## Context

DSv4-Flash decode throughput is kernel-bound. Live measurement on the .62 8×H20
TP=8 serve (own build, `ARLE_DSV4_DECODE_PHASE_TIME=1`) decomposed the decode step
and pinned the bottleneck to **`perrow`** — the per-row m=1 compressor + indexer
projection GEMVs run inside `mla_attention_prepare_compressed_only` (one
`compressor_forward` per decode row, each doing `dsv4_linear(wkv)` +
`dsv4_linear(wgate)` at m=1). Measured **dead-linear in batch size n**:

| n | perrow (ms) | compidx | step (sw_attn+moe) |
|---|-------------|---------|--------------------|
| 4 | 32.9 | 33.9 | ~86 |
| 8 | 61.5 | 62.6 | ~134 |
| 16 | 121.2 | 122.4 | ~227 |
| 22 | 161.9 | 163.1 | ~302 |

`perrow ≈ 7.4 × n ms` ⇒ the step grows ∝ n ⇒ aggregate throughput **saturates**
(n=4 ≈ 47 tok/s → n=22 ≈ 73 tok/s, sub-linear). The per-row GEMVs are
bandwidth-bound (each re-reads the full weight per token, N reads for N rows, zero
amortization). The batched DSA *read* (`csa_select_official_batched`, `2aec76e7`)
was already landed and is NOT the cost (`read=1.2ms`); the per-row projection GEMVs
are.

## What worked

Batch the per-row m=1 compressor/indexer GEMVs into **one m=N GEMM each**, in a
pre-pass mirroring `mla_attention_prepare_proj_batch`. Opt-in
`ARLE_DSV4_DECODE_COMPRESSOR_BATCH`, **default OFF (byte-identical baseline)**.

Key realisation (corrected from an initial wrong FP8 spec): the DSv4 compressor
weights are **all bf16** (verified in the checkpoint: 41 main-compressor + 21
indexer-compressor layers, every `wkv`/`wgate` is BF16, no FP8). And `dsv4_linear`
already dispatches per `weight_format`: `DenseBf16 → ops::gemm_batch → gemm_cuda →
gemm_cublaslt_impl → cublasLtMatmul`, which **amortizes the weight read at m=N**
(read once for all N rows). So the lever is simply "call `dsv4_linear` once at m=N
instead of N times at m=1" — **no FP8 quant** (zero selection-shift correctness
risk), **no DeepGEMM weight cache**, **no loader change**. It is
**mixed-precision-safe by construction**: `dsv4_linear` picks bf16-cublasLt vs
fp8-deepgemm per the weight's own format, per layer.

The "dsv4_linear m=N does not amortize" caveat in the proj-batch fallback is
**FP8-scalar-gemv-specific** (`dsv4_fp8_gemv_batch`, per-(out,token) grid); the
bf16 path goes through cublasLt and DOES amortize.

Implementation (2 files, opt-in, default OFF):
- `attention.rs`: `dsv4_decode_compressor_batch_enabled()` (env gate),
  `compressor_batch_prepass()` (two m=N `dsv4_linear` calls → `kv_raw_batch[width,N]`
  + `score_raw_batch[width,N]`), `compressor_forward(precomputed: Option<…>)` (Some
  → skip the two m=1 GEMVs and use the per-row slice; None → byte-identical legacy),
  `mla_attention_prepare_compressed_only(compressor_precomputed)`.
- `dsv4.rs`: decode loop runs the pre-pass (gated) after the proj batch, slices
  column r per row, threads it through. Gate OFF → None throughout (byte-identical).

## Measured A/B (same bf16 binary, .62 8×H20 TP=8, c=48)

End-to-end decode step at the saturated batch n=22 (gate OFF vs ON, same binary,
same session — both ran c=48 cleanly, no crash):

| metric @ n=22 | gate OFF | gate ON | Δ |
|---------------|----------|---------|---|
| perrow (lever target) | 162.1 | 92.0 | **−43%** |
| sw_attn | 257.2 | 191.4 | −26% |
| **decode step** (sw_attn+moe) | **302.6 ms** | **237.2 ms** | **−22%** |
| **decode throughput** (n/step) | **72.7 tok/s** | **92.7 tok/s** | **+28%** |

perrow Δ across n (gate OFF→ON): n=4 31.1→19.0 (−39%), n=8 61.5→34.5 (−44%),
n=22 162.1→92.0 (−43%); slope `7.4 → ~4.4 ms/row`.

Roofline note: this entry licenses the component A/B and end-to-end direction,
not a roofline-efficiency verdict. Achieved-vs-peak is deferred to a follow-up
nsys/ncu pass per `docs/bench-and-trace-spec.md` §7.7.

**Both gate states serve c=48 (n→22) cleanly — no crash.** Correctness (gate ON):
"capital of France" → "Paris" (deterministic across two greedy runs), needle
"GREEN-5521-CAT" retrieved exact, "17+25" → "42". bf16 ⇒ no numerical shift vs the
per-row path (same cublas math, batched M), so needle is exact, not just within the
non-determinism floor.

## Full-flatten follow-on (landed, same gate)

Batched the remaining per-row decode kernels too — per-slot `compressor_update`
(new stateful batched CUDA kernel, array-of-N state pointers), inverse-RoPE, and
sw-window write — under the same `ARLE_DSV4_DECODE_COMPRESSOR_BATCH` gate. Same
n=22 c=64 A/B: step **237.2 → 218.8 ms**, decode **92.7 → 100.5 tok/s** — i.e.
**+8% over the GEMV lever, +38% vs baseline (72.7→100.5)**. Correct (needle
"AMBER-7788-LION" exact, deterministic ×2). Component deltas vs baseline @ n=22:
perrow 162→76 (update-batching shaved the lever's 92→76), finish 86→81, moe
unchanged.

**The flatten thesis (flat step → linear scaling) did NOT hold.** The step is
still ∝ n (sw_attn 111→173 from n=13→22). The residual `perrow`+`finish` are
**irreducible per-row compute** (compressor compressed-cache writes + indexer
top-k select + sw-window writes) — no shared weight to amortize like the GEMV's
weight-read, so batching only removed launch overhead (small). True linear
scaling needs a different axis (DP-attention, or a compressor-cache/indexer-select
redesign), not more per-slot batching. Kept anyway — every measured gain counts —
but do not expect the next per-slot batching to flatten further.
- **An earlier silent c=48 crash was the FP8-version binary — RESOLVED.** An
  initial (wrong-spec) FP8-DeepGEMM-cache build of this lever hard-crashed gate-OFF
  at c=48 (n→~22). The bf16 rewrite does not: **both gate OFF and gate ON serve
  c=48 (n=22) cleanly** (measured this session). So local-main's high-batch path is
  fine; the crash was specific to the discarded FP8 path.

## Rule

bf16 `dsv4_linear` at m=N amortizes the weight read via cublasLt (`gemm_cuda`); the
"not amortized" caveat is FP8-scalar-gemv-specific. **Check `weight_format` before
assuming a weight needs FP8/DeepGEMM** — DSv4 compressor weights are bf16, so the
batched pre-pass needs neither a quantize nor a deepgemm cache, and carries no FP8
correctness risk. `dsv4_linear`'s per-format dispatch makes the pre-pass
mixed-precision-safe for free.
