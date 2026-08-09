# FlashQLA `block_DV=32` failed numerical parity

## Context

On Qwen3.6-27B at `Q=2048`, `H=48`, the shipped FlashQLA `fq_fwd` launches 96
CTAs across 78 H20 SMs. Nsight Compute measured 794.4 us, 24.73% achieved
occupancy, 2.80% DRAM throughput, and 12.87% SM throughput. No eligible warp
was available in 92.67% of cycles; long-scoreboard stalls accounted for 87.2%
of the issued-instruction interval. The measured stall split licensed one
candidate: halve `block_DV` from 64 to 32, increasing the grid to 192 CTAs.

## Root Cause

The first candidate build exposed a generator bug. The device kernel read
`FQ_BLOCK_DV`, while `gen_tilelang_aot.py` hardcoded the public wrapper grid to
`2 * H`. A 32-wide kernel therefore launched 96 CTAs and left half of the value
tiles uncomputed. Those numerical results are invalid for judging the complete
candidate.

`e2a837ff6` derives the wrapper grid from the loaded kernel module's `FQ_DV`
and `FQ_BLOCK_DV`. Nsight Compute then confirmed the complete candidate at
grid `(192, 1, 1)`, block `(512, 1, 1)`.

The complete 192-CTA candidate still failed numerical parity against the
recurrent reference. The specific numerical mechanism is unknown. The same
fixed `Q=2048` request produced 144 in-forward comparisons across 48 linear
layers and segments 2048, 272, and 16. Relative to the shipped 64-wide arm,
790 metrics exceeded the 5% regression budget. At layer 0, segment 2048:

| metric | `block_DV=64` | `block_DV=32` |
|---|---:|---:|
| state max absolute error | 6.676 | 63.542 |
| state RMSE | 0.02546 | 0.13805 |
| output max absolute error | 0.21875 | 2.75 |
| output RMSE | 0.001334 | 0.012994 |

No comparison contained a non-finite mismatch.

## Fix

Keep `FQ_BLOCK_DV=64`. The 32-wide candidate is rejected before timing or
model-level acceptance. Keep the wrapper-grid derivation fix so future tile
changes cannot silently launch an incomplete output domain.

`3582c881a` also makes the two numerical paths explicit: the ignored CPU
reference test is named and pinned to the recurrent kernel, while
`ARLE_FQ_PARITY=1` reports `max_abs`, `p99_abs`, RMSE, floored max-relative
error, relative L2, cosine, and non-finite mismatches from FlashQLA and the
recurrent reference inside the same forward.

Artifacts on the validation host:

- baseline NCU:
  `/host/fq-fwd-9a6ca91ac9-g0/artifacts/fq64-q2048-h48-full.ncu-rep`;
- corrected candidate grid:
  `/host/fq-fwd-9a6ca91ac9-g0/artifacts/fq32-grid192-smoke.ncu-rep`;
- corrected candidate parity:
  `/host/fq-fwd-9a6ca91ac9-g0/artifacts/fq32-grid192-parity-1-serve.log`;
- comparison report:
  `/host/fq-fwd-9a6ca91ac9-g0/artifacts/fq32-grid192-vs-fq64-parity-run1.txt`.

## Rule

Every generated kernel gate proves both domains: tensor values and launch
geometry. A kernel constant and its wrapper launch grid share one source of
truth. Numerical A/B uses an in-forward reference so MoE run-to-run
non-determinism cannot become an operator verdict.
