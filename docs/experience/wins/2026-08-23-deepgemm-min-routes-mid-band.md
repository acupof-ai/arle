# DeepGEMM min-routes mid-band — characterized, no flip

2026-08-23 · CUDA · bench

## Context

`--qwen35-deepgemm-min-routes` (default 1024) gates the DeepGEMM grouped expert
path on routed-row count `R = top_k * B`. FP8 experts are forced to the hand
CUDA-core kernels at `R ≤ 128` (`DEEPGEMM_MASKED_BAND`), so the uncharacterized
regime was `R = 129..1023` (decode `c = 17..127` with top_k=8). The flag's own
doc said "lower it to reach the uncharacterized mid-band." An industry-path
sweep (`2026-08-22-tp-decode-graph-industry-path.md`) found SGLang/vLLM run
masked grouped GEMM at all decode shapes — no floor — motivating a measurement.

## Setup

- Model: Qwen3.6-35B-A3B-FP8 (256 experts, top_k=8, 40 layers), 1×H20, fp8 KV.
- Same binary (`47e80f06f`), same GPU, sequential A/B, only the flag differs.
- Prompts: 128 agent prompts, ~33k tokens each. `--spec-type none`, decode graph on.
- Cells: c=16 (boundary control, R=128=MASKED_BAND, both arms forced to hand
  kernels), c=32 (R=256, treatment arm switches to DeepGEMM), c=64 (R=512).
- 120 s per concurrency, `bench_throughput.py`, ITL p50 as the decode metric.

## Result

| c | base (1024) itl_p50 ms | treat (129) itl_p50 ms | verdict |
|---|---|---|---|
| 16 | 36.54 | 36.36 | identical (control ✓) |
| 32 | 64.77 | 66.56 | wash (mean 148.4 vs 147.6; p90 406 vs 370) |
| 64 | 2993.9 | 142.4 | confounded — see below |

c=16 control confirms the gate: both arms run the hand kernels at R=128 and the
ITL matches. c=32 is the clean mid-band cell: DeepGEMM contiguous at R=256 does
not beat the hand kernels (treat 2.8 % slower on p50, 9 % faster on p90, mean
identical — noise). The JIT/TMA overhead on small bands that lost the R=8 BF16
measurement (37.5 vs 40.8 tok/s) persists into the FP8 contiguous mid-band.

c=64 is not a viable config: 33k-token prompts × 64 concurrent ≈ 84 GB KV +
35 GB weights > 96 GB. The baseline arm also ran during a concurrent pod build
and collapsed to 0.4 tok/s; the treatment arm ran later and did not. The
comparison is confounded and the cell is over capacity regardless.

## Verdict

**No flip.** The 1024 floor stays. The mid-band (R=129..1023) is hand-kernel
territory on H20; DeepGEMM only pays off at prefill-scale R (the existing
needle measurement: R=16384, 9.07 → 6.10 s). The flag stays as the documented
escape hatch — the regime it reaches is now characterized, not unknown.

## Rule

A "no floor in industry" finding is a hypothesis to test, not a license to
flip. ARLE's vendored DeepGEMM JIT/TMA cost on small bands is a local property
that industry defaults don't capture; measure on the target hardware before
lowering a floor.
