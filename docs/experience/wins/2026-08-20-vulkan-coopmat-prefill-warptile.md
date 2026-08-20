# Vulkan prefill 3.45 → 37.0 tok/s: llama.cpp's coopmat warptile is the *worst* of eleven on RDNA 3.5

## Context / Goal
Qwen3.8-27B-Q4_K_M on Vulkan / Strix Halo prefilled token-serially at
**3.45 tok/s** — one GEMV per token (`executor.rs:203`), so TTFT scaled at the
decode rate and a 2000-token turn cost minutes. The goal was to put prefill on
a real batched GEMM and then on the device's KHR cooperative-matrix units.

## Hypothesis
Going in: (1) batching the GEMM is most of the win, and (2) for the coopmat
kernel we can lift llama.cpp's `{s,m,l}_warptile_mmq` tiles verbatim, since
`mul_mm.comp` is the same shader. **(2) was exactly backwards** — the copied
tile made coopmat *slower than the scalar kernel it replaced*.

## Params
- Backend: Vulkan, `arle serve --backend vulkan`
- Model: `Qwen3.8-27B-Q4_K_M.gguf` (15.93 GiB)
- Routes: `ARLE_VULKAN_BATCHED_PREFILL=0` (GEMV loop) / `=1` (batched), and
  within batched, `mul_mmq` (scalar integer dot) vs `mul_mm` COOPMAT
- Profiler: `ARLE_GPU_TIMESTAMPS=1` for the per-op split, off for throughput

## Env
- Host: Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, RDNA 3.5, 40 CU), 128 GB
  LPDDR5X @ 256 GB/s, Windows 11, Vulkan 1.4.349
- Device coopmat: **f16×f16→f32, scope Subgroup, 16×16×16**;
  `maxComputeSharedMemorySize = 32768`; subgroup size 64
- Power: Armoury Crate **Silent** mode — read every absolute below as a ratio
- Date: 2026-08-20

## Results

### End to end
| Prefill route | tok/s | vs previous |
| --- | ---: | ---: |
| per-token GEMV loop (`executor.rs:203`) | 3.45 | — |
| `mul_mmq` batched chunk | 23.7 | 6.9× |
| `mul_mm` COOPMAT | **37.0** | 1.57× |

**10.7× overall.** A/B at chunk width 256, same GGUF, same prompt: routes agree
token-for-token (`'We need to respond to user: "'`).

### TTFT parity (`scripts/vulkan_prefill_parity.py`, 48 tokens greedy)
| Prompt | serial TTFT | batched TTFT | speedup | text identical |
| ---: | ---: | ---: | ---: | --- |
| 1 rep (~55 tok) | 25.12 s | 3.06 s | 8.22× | **yes** |
| 8 reps (~250 tok) | 72.89 s | 7.39 s | 9.86× | **yes** |

### The warptile sweep (`tests/device_mm_coopmat_bench.rs`), geomean vs `mul_mmq`
```text
           tile     n=32     n=64    n=128    n=192    n=256    all n
 l 128x128 w128x64  0.32x    0.25x    0.89x    0.97x    0.89x    0.57x   <- llama.cpp
    128x32  w32x32  1.80x    0.85x    2.14x    2.42x    1.90x    1.72x   <- narrow
     64x64  w32x32  1.31x    1.17x    2.93x    2.82x    2.40x    1.98x   <- medium
    128x64  w32x32  1.53x    1.16x    2.55x    3.12x    2.60x    2.06x   <- wide
     32x32  w32x32  1.34x    0.84x    1.71x    1.78x    1.47x    1.38x   <- tiny
```
`l_warptile_mmq` is the worst of eleven candidates, and the old `choose` picked
it for every `n > 64` — i.e. every prefill chunk. That is the whole 0.75×
regression.

### Numerics
12/12 device cases (Q4_K / Q6_K / Q8_0 × 4 shapes exercising narrow/medium/wide),
worst error **9.31e-5** against a 1e-2 tolerance.

## Verification
| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy -- -D warnings` | PASS |
| `cargo clippy --workspace --exclude infer-cuda --no-deps -- -D warnings` | PASS |
| `cargo test -p vulkan-kernels -p infer-vulkan --features vulkan --lib` | PASS, 18 passed / 2 ignored |
| `cargo test -p vulkan-kernels --features vulkan --tests` | PASS, all device suites |
| coopmat vs host f32 oracle | PASS, 12/12, max err 9.31e-5 |
| `vulkan_prefill_parity.py --serial` | PASS, text identical both lengths |
| `needle_gate.py "115,446,2000" 2 0.0` | PASS, exact=2/2 every length, DET |
| `eval_harness resident_reuse` | PASS, on_hit 536/536 at ceiling, 5.68× |
| `eval_harness prefix_reuse token_reuse` | FAIL — pre-existing, see below |

`infer-cuda` is excluded because it fails to build at HEAD (E0432/E0433/E0599 in
`tp.rs`) — reproduced in a clean worktree, unrelated to this change.

`prefix_reuse` / `token_reuse` gate the *page-attached* radix cache, which this
lane does not implement and never has — recorded as FAIL 0/2 at the roofline
commit, before any of this work. The Vulkan lane's flat absolute-position KV
can't be page-attached, so its reuse is resident-sequence keyed; `resident_reuse`
is the gate that actually covers it, and it passes.

**`NEEDLE_MAX_TOKENS` defaults to 16, which auto-fails any thinking model.**
The first needle run reported miss=2 at every length with outputs cut off
mid-reasoning (`"...access code is 7"` — the needle is `738291`). The budget is
tuned for non-thinking checkpoints; at 200 tokens the same build scores exact
2/2 everywhere. A gate default can manufacture a failure.

## Problems

**The bottleneck moved off the GEMM, so the projected ~87 tok/s did not land.**
Per-op GPU profile, 256-token chunk (total 6430.77 ms):

| Op | ms | share | dispatches |
| --- | ---: | ---: | ---: |
| mmq | 3665.81 | 57% | 400 |
| **lin_gdr** | **2442.79** | **38%** | 48 |
| swiglu | 69.72 | 1.1% | 112 |
| gemv | 65.89 | 1.0% | 24576 |
| everything else | 186.6 | 2.9% | — |

`lin_gdr` is the gated-delta recurrence: **token-serial by construction** — one
workgroup per value head, stepping tokens in order. Corroborating evidence that
it now sets the ceiling: prefill tok/s is **flat at 36–37 across chunk widths
128 / 192 / 256 / 384**. Widening the batch no longer buys anything. Filed as
the chunkwise-matrix-form rewrite (intra-chunk parallel, inter-chunk serial).

It also required splitting the profiler label: the conv1d and the recurrence
share a layer but not a scaling law, and the combined `linear` category hid
which half to attack. `lin_conv` turns out to be 42.52 ms — negligible.

## Learnings

**Vendor tile constants do not port across vendors, and the failure is silent —
it looks like "coopmat is just slow here."** The mechanism is per-subgroup
accumulator count, not warp width. `mul_mm.comp:179` declares
`sums[(WM/TM) * (WN/TN)]` live across the whole K loop, each `coopmat` costing
`TM*TN/WARP` VGPRs per lane. `l_warptile_mmq` at warp 64 gives `8 * 4 = 32`
accumulators ≈ **128 VGPRs/lane** before operands — occupancy collapse on
RDNA 3.5. Cap `WM = WN = 32` (4 accumulators, 16 VGPRs) and buy the tile area
back with *more subgroups per workgroup* instead: 2–3×.

**Two earlier diagnoses were confidently wrong, and each cost a build+measure
cycle.** Recorded so they are not re-walked:
- *"the f16 B-operand pack kernel dominates"* — killed by the per-op profile:
  19.63 ms of 12448 ms.
- *"`WARP` is hardcoded 32 on a wave64 device"* — plausible (llama.cpp does
  derive every tile from `max(subgroup_size, 8)`), fully implemented, and a
  **measured non-effect**: 0.75× before, 0.75× after.

**The "72× vs llama.cpp" number that started this was against the wrong model.**
It compared llama.cpp on a dense 27B to ARLE on a 3B-active MoE. Re-run
dense-vs-dense, the real gap was 2.3×.

**Also: `BLOCK_SIZE` should be derived, not passed.** The shader hands warp tile
`gl_SubgroupID` to each subgroup and never loops, so the workgroup must hold
exactly `(BM/WM) * (BN/WN)` subgroups. A separate parameter is only a way to get
it wrong; `MmSpec::new` now computes it.

## Rule
Before porting a tuned kernel's tile/tiling constants from another vendor's
backend, sweep them on the target part and compare against the kernel you are
replacing. A constant tuned for one register file is not a constant — it is a
measurement someone else took on different hardware, and it can land *below*
the naive path it was supposed to beat.

And when a GEMM optimization stops paying, check whether throughput has gone
flat in batch width before tuning further: flatness means the ceiling has moved
to a serial op, and more GEMM work is wasted effort.
