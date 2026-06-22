# Qwen3.6-27B-FP8 decode 5.5× — the MMA GEMV was occupancy-starved at B=1

**Commit:** 7669ff69 (`perf(qwen35): default FP8 decode GEMV to coalesced scalar kernel`).
**Measured:** Qwen3.6-27B-FP8, 1×H20, `arle serve` (no-think prompt, max_tokens=200, ignore_eos).

## Context

ckl flagged 27B-FP8 decode as "比 Mac 还差". Measured single-stream **7.5 tok/s**
— ~19× off the H20 memory roofline (~140 tok/s for a 27B-FP8 weight read at
~4 TB/s). §0 profile (`ARLE_QWEN35_QUANT_PROFILE`) attributed the entire step to
the FP8 block-scaled GEMVs.

## Root cause (measured, not inferred)

The decode GEMV ran the **`mma.m16n8k16` tensor-core path**
(`gemv_fp8_block_scaled_batch_mma`), which was built for **DSv4 batched decode
(B≤16)** and is the wrong tool at **B=1**:

- **Occupancy-starved:** `grid = (N/32, B/16)` → at B=1 only `N/32` blocks
  (160–544) on the H20's 132 SMs (~25% occupancy). Bandwidth scaled *linearly*
  with block count — the smoking gun:

  | proj | N | blocks | GB/s | % of 4 TB/s |
  |------|---|--------|------|-------------|
  | gate/up | 17408 | 544 | 341 | 9% |
  | down | 5120 | 160 | 126 | 3% |

  gate and down read the **same 89 MB** but down (160 blocks) is 2.7× slower —
  fully explained by block count, not bytes.
- **Uncoalesced:** the MMA col-major B-operand makes each warp read 8 different
  N-rows (stride K), scattering HBM transactions.

Both come from forcing a batched-GEMM MMA onto a B=1 GEMV. Graph (`ARLE_QWEN35_DECODE_GRAPH=1`)
and launch overhead were **confirmed washes** (identical 7.5 tok/s) — it is
GPU-bound on the kernel, not host.

## Fix

The repo already had a **coalesced scalar warp-per-row** FP8 GEMV
(`dsv4_fp8_gemv_kernel` / `dsv4_fp8_gemv_batch_kernel`, `quantized_gemv.cu`):
`grid = N/GEMV_ROWS` (~16× more blocks) and threads stride K reading consecutive
bytes (coalesced) + warp-reduce. Flipped the Qwen FP8 decode dispatch
(`quant_linear.rs`) to default to it; MMA is now opt-in (`ARLE_QWEN35_FP8_GEMV_MMA=1`).

## Measured

| | MMA (old default) | scalar (new default) | speedup |
|---|---|---|---|
| single-stream | 7.5 tok/s | **41.3 tok/s** | **5.5×** |
| conc=8 | 17.6 tok/s | **50.4 tok/s** | **2.9×** |

Scalar GEMV bandwidth jumped from 3–9% to **32–42%** (down-proj 0.705→0.053ms =
13×; gate 0.262→0.054ms = 4.9×). (The A/B's 27 tok/s was 3-serve contention;
isolated default is 41.3.)

Correctness: the scalar kernel **is the reference** the MMA was built to match
(cosine 1.0 / max_abs 0); decoded output coherent (17×23 = 391). Parity-safe flip.

## Rule

A "fast" tensor-core kernel for batched decode (B≤16) can be a **5× regression**
at B=1 — the MMA's batch tile + col-major weight layout starve occupancy and
coalescing. For a memory-bound B=1 GEMV the right kernel is coalesced
warp-per-row with `N/rows` blocks. Always profile the **actual decode shape**;
the block-count↔bandwidth correlation is the occupancy fingerprint.

## Second-front results (parallel-analysis Workflow + measured A/B)

A 5-agent Workflow analyzed the remaining fronts. **Key §0 correction it caught:**
the live Qwen FP8 decode kernel is `fp8_f32_block_gemv_batch_kernel`, which is
**already 16-wide vectorized** (`fp8_f32_dot16`, committed `26014f4e`) — NOT the
1-byte `dsv4_fp8_gemv_kernel` I'd been reading. So "vectorize the byte loop" is
void for Qwen. The plateau at ~41% HBM is latency/occupancy, not load width.

Measured A/B on H20 (single-stream, vs the 41.3 baseline):

| lever | change | result | verdict |
|-------|--------|--------|---------|
| one-warp-per-row | `FP8_GEMV_ROWS=8`, drop cross-warp `__syncthreads` | 39.9 tok/s | **WASH/slightly worse → KILLED** (halved blocks offset the barrier removal) |
| ILP-unroll K-loop | 4-way unroll of the `dot16` walk | **unmeasured** (pod reclaimed mid-A/B) | patch saved `/tmp/lever2-ilp-unroll.patch`; re-test next GPU |

The one-warp-per-row lever (the synthesis' top pick) measured a wash — lowering
confidence that the ~41% plateau has an easy GEMV-micro-opt win. **The 5.5× scalar
default is the structural win; further GEMV tuning is low-yield.**

## DEFINITIVE: the scalar GEMV is near-optimal (ncu, Colab Pro 6000 proxy)

After the pod was reclaimed, re-ran the kernel A/B on a **Colab RTX PRO 6000
Blackwell** (sm_120, native FP8) via the standalone `tools/gemv_bench.cu`
micro-bench (no 27B needed). The Pro 6000 with an L2-flush reproduces the H20
**occupancy-bound regime** (avg ~48% of an optimistic 1790 GB/s roofline, vs L4's
86% bandwidth-bound — L4 is the wrong regime). On that proxy:

- `GEMV_ROWS` sweep {1,2,4,8}: **all wash** (~48-49%).
- 4-way **ILP unroll**: **wash** (~48%).
- **`ncu` verdict** (the definitive one): `gpu__dram_throughput = 80.8%`,
  `sm__throughput = 22.6%` (compute idle), `warps_active = 84%`. The kernel is
  **memory-bound at 80% of DRAM peak with 84% occupancy** → near-optimal. The "55%
  of 1790" was an optimistic marketing roofline; 80% of achievable DRAM is the real
  ceiling for a read-once streaming GEMV. **No kernel-level headroom remains.**

So the 5.5× scalar default IS the structural win; **further GEMV micro-opt is
futile** (ncu-confirmed, not just A/B-washed). All 4 shapes cosine=1.0 — correctness
of the kernel (and the no-headroom verdict) verified on representative HW cheaply,
without burning H20 time. The reusable harness + proxy method is the lasting infra.

**One residual (untestable here):** the H20 (78 cut-down SMs vs Pro 6000's ~188)
may be slightly *under*-saturated (~50% of *achievable* DRAM vs the proxy's 80%),
so ILP/more-loads-per-SM *might* help on H20 specifically even though it washed on
the already-saturated proxy. Worth one quick H20 A/B when the pod returns — but
low-confidence; the GEMV is essentially done.

## Follow-ups (next GPU — pod reclaimed 2026-06-22)

- **ILP-unroll** re-test (patch saved); if wash, the GEMV is at its practical floor.
- **lm_head** 1.37ms/step (bf16, vocab 248320) → FP8 reload halves the read (~+1 tok/s).
- **norm+residual fusion** (`fused_add_rms_norm_offset_cuda` exists, unwired) — ~1%.
- **linear-attn `in_proj`** 0.176ms — re-profile whether it's off the FP8 fast path.
- **Deletion-refactor** (ckl): collapse the dead MMA Qwen path to one scalar path
  (zero perf; `quant_linear.rs` mma branch + `quantized_gemv_mma.cu` Qwen clone +
  FFI + parity test) — needs a full `--features cuda` build to verify, so deferred.
- **Colab continuation caveat**: the 27B-FP8 (~27 GB) needs a big GPU; the Colab
  `<8s-exec` verification lane can't serve it (model load ≫8s). Colab is for kernel
  *correctness* unit-tests (sm_80+), not 27B serving-perf — that needs an H20/A100-80.
