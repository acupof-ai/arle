# Vulkan prefill: llama.cpp's coopmat warptile is the *worst* of eleven on RDNA 3.5 (1.60x once fixed)

## Context / Goal
Qwen3.8-27B-Q4_K_M on Vulkan / Strix Halo prefilled token-serially — one GEMV
per token (`executor.rs:203`) — so TTFT scaled at the decode rate and a
2000-token turn cost minutes. (3.45 tok/s as first measured, but see the power
mode caveat under Results: that figure is throttled and only the ratios below
are portable.) The goal was to put prefill on
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
- Power: measured in **both** Armoury Crate modes. Silent (throttled) for the
  original investigation; Performance / on AC (`powercfg` scheme
  `27fa6203-…`, "Performance") for the re-measure. Every absolute is annotated
  with its mode; ratios are quoted from a single sitting.
- Date: 2026-08-20

## Results

### End to end — read the ratios, not the absolutes

Every absolute below is a function of the box's Armoury Crate power mode, and
this entry was measured across two of them. **The ratios reproduce; the
absolutes do not.** Same code, same GGUF, same prompt, chunk width 256:

| Prefill route | Silent tok/s | Performance tok/s | ratio |
| --- | ---: | ---: | ---: |
| `mul_mmq` batched chunk | 23.7 | 81.2 | 1.00× |
| `mul_mm` COOPMAT | 37.0 | **129.9** | **1.57× / 1.60×** |

The coopmat win measured **1.57×** under Silent and **1.60×** under Performance —
the same number twice, three months of throttle apart. Routes agree
token-for-token in both (`'We need to respond to user: "'`).

Against the per-token GEMV loop it replaced (`executor.rs:203`), batched prefill
cuts TTFT **10.5×–11.6×** (Performance; 8.2×–9.9× under Silent), text identical.

### TTFT parity (`scripts/vulkan_prefill_parity.py`, 48 tokens greedy)
| Prompt | serial TTFT | batched TTFT | speedup | text identical |
| ---: | ---: | ---: | ---: | --- |
| 1 rep (~55 tok), Silent | 25.12 s | 3.06 s | 8.22× | **yes** |
| 8 reps (~250 tok), Silent | 72.89 s | 7.39 s | 9.86× | **yes** |
| 1 rep (~55 tok), Performance | 9.75 s | 0.93 s | 10.52× | **yes** |
| 8 reps (~250 tok), Performance | 28.44 s | 2.44 s | 11.64× | **yes** |

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
All re-run on the tree rebased onto `main` (493 commits of drift).

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo clippy -- -D warnings` | PASS |
| `cargo clippy --workspace --no-deps -- -D warnings` | PASS |
| `cargo test -p vulkan-kernels -p infer-vulkan --features vulkan --lib` | PASS, 17 passed / 2 ignored |
| `cargo test -p vulkan-kernels --features vulkan --tests` | PASS, all 13 device suites |
| coopmat vs host f32 oracle | PASS, 12/12, max err 9.31e-5 |
| `vulkan_coopmat_ab.py --width 256` | PASS, 1.60×, routes agree token-for-token |
| `vulkan_prefill_parity.py --serial` | PASS, text identical, TTFT 10.5×/11.6× |
| `needle_gate.py "115,446,2000" 2 0.0` | PASS, exact=2/2 every length, DET |
| `eval_harness resident_reuse` | PASS, on_hit 536/536 at ceiling, 5.68× |
| `eval_harness prefix_reuse token_reuse` | FAIL — pre-existing, see below |

`prefix_reuse` / `token_reuse` gate the *page-attached* radix cache, which this
lane does not implement and never has — recorded as FAIL 0/2 at the roofline
commit, before any of this work. The Vulkan lane's flat absolute-position KV
can't be page-attached, so its reuse is resident-sequence keyed; `resident_reuse`
is the gate that actually covers it, and it passes.

Two things that were true of the pre-rebase tree and are no longer: `infer-cuda`
failed to build (E0432/E0433/E0599 in `tp.rs`) and had to be excluded from the
workspace sweep — `main` has since fixed it, so the sweep now runs whole. And
the lib-test count went 18 → 17 because `main`'s dead-code sweep `622bae739`
deleted `moe_launcher_sequence_marks_host_topk_and_device_experts`.

**`NEEDLE_MAX_TOKENS` defaults to 16, which auto-fails any thinking model.**
The first needle run reported miss=2 at every length with outputs cut off
mid-reasoning (`"...access code is 7"` — the needle is `738291`). The budget is
tuned for non-thinking checkpoints; at 200 tokens the same build scores exact
2/2 everywhere. A gate default can manufacture a failure.

## Problems

**The bottleneck moved off the GEMM.** Per-op GPU profile, Performance mode,
64-token chunk (total 524.74 ms):

| Op | ms | share | dispatches |
| --- | ---: | ---: | ---: |
| mmq | 324.11 | 62% | 400 |
| **lin_gdr** | **177.65** | **34%** | 48 |
| gemv | 5.33 | 1.0% | 6144 |
| swiglu | 4.87 | 0.9% | 112 |
| everything else | 12.8 | 2.4% | — |

`lin_gdr` is the gated-delta recurrence: **token-serial by construction** — one
workgroup per value head, stepping tokens in order. Its 34% share is stable
across power modes (it read 38% under Silent), so the share, unlike the
milliseconds, is a property of the workload.

Splitting the profiler label was a precondition for seeing this: the conv1d and
the recurrence share a layer but not a scaling law, and the merged `linear`
category hid which half to attack. `lin_conv` is 3.42 ms — negligible.

**The kernel is already at its roofline, so the fix is algorithmic, not a
tuning pass.** `crates/vulkan-kernels/tests/device_gated_delta_bench.rs` settles
it at the production shape (48 heads, key_dim = val_dim = 128):

- Addressed traffic runs at ~610 GB/s — **238% of the 256 GB/s LPDDR5X peak** —
  so the 56% of bytes that are pure per-thread redundancy (all 128 threads
  re-reading the same `q[j]`/`k[j]`, and each re-running the `q_sumsq`/`k_sumsq`
  reduction) are cache-resident and free. The obvious micro-optimization is
  worth zero.
- What remains is irreducible **in the recurrent formulation**: 2 reads + 2
  writes of the `[key_dim, val_dim]` state per token per head — 3.22 GB per
  dispatch at T=256, sustained at ~270–290 GB/s.
- Production agrees with the bench once the clock does: 3.70 ms/dispatch at
  T=64 measured in-engine vs ~3.0 ms in the bench.

So the only lever is the **chunkwise matrix form** (intra-chunk parallel,
inter-chunk serial), which touches the state once per chunk rather than once per
token — a ~C× cut in the one term that is actually being paid. The bench is the
license-or-kill gate: it must show state GB dropping by ~C, not just ms falling.

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

**Every absolute number in the first version of this entry was throttled, and
the conclusion drawn from them was wrong.** The whole coopmat investigation ran
with the box in Armoury Crate **Silent**; re-measured under **Performance**,
identical code went 37.0 → 129.9 tok/s (3.5×). Two things follow:

- The *ratio* survived exactly — 1.57× vs 1.60× for coopmat-over-mmq. Ratios
  measured in one sitting are portable across clock states; absolutes are not.
- The *reasoning* did not. Under Silent, prefill looked **flat at 36–37 tok/s
  across chunk widths 128/192/256/384**, which reads as "the batch axis is
  exhausted, the ceiling is elsewhere." Under Performance it rises monotonically
  (86.9 / 113.4 / 113.9 / 124.3 / 125.9 at 32/64/96/128/256). The flatness was
  the power limit, not the algorithm. A throttled box does not just make numbers
  smaller — it makes curves flat, and flat curves invite exactly the wrong
  structural conclusion.

This was a *documented* trap in this project's own notes ("read the Armoury
Crate mode before trusting any benchmark — the '3× regression' was Silent") and
it still cost a full investigation, because nothing in the measurement announces
itself as throttled.

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

**Record the machine's power/thermal mode next to every absolute number, and
prefer ratios measured in a single sitting.** If a throughput curve comes out
flat in the parameter you are sweeping, suspect a power limit before you
conclude the axis is exhausted — flatness is what throttling looks like from the
inside.

And when a kernel looks like the next bottleneck, measure it against its own
roofline before rewriting it. "38% of the profile" licenses investigation;
only "N% of peak bandwidth, with the redundant traffic proven free" licenses a
rewrite — and it also tells you *which* rewrite, since the term that is actually
being paid is the one to attack.
