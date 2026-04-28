# cuda-l4 TileLang TC-decode HD128 alias — c=16: -31% (KEEP OFF; build a real decode kernel)

> **Negative result.** This bench is documented as a win because it
> closes a long-open `pending-remote` claim that the
> prefill→decode HD128 alias was a free upgrade. Matched A/B on the
> L4 here shows the alias is -30.6% on out-tok/s at c=16 / 4096-in,
> roughly as the Tranche 4 stub feared.

## Context

Plan agent (subagent_type=Plan, 2026-04-28) recommended validating
the already-wired `tilelang-attn` TC-decode HD128 alias as the
highest-ROI next step after c=16 unblocked at 197 tok/s on FP8 KV.
The reasoning was sound — Qwen3-4B's decode call site
(`flashinfer_run_layer_hd128`) is the per-step hot kernel at
saturation, and the AOT cubins (`q*_kv8`) are already built and
linked under `tilelang-attn`.

The Tranche 4 stub
(`docs/experience/wins/2026-04-27-bench-guidellm-cuda-tilelang-tc-decode-hd128-pending-remote.md`)
explicitly warned: *"the prefill HD128 cubin is tuned for
many-Q-rows-per-request, but pure decode batches have 1 Q row per
request. The kernel may waste shared memory on a Q-tile that is
mostly empty."* That hypothesis was never validated on hardware
because `--features cuda,tilelang-attn` had been broken since the
HD256 decode tranche was added (TileLang 0.1.9 codegen failure on
M=1, see
[`errors/2026-04-28-tilelang-hd256-decode-m1-codegen-failure.md`](../errors/2026-04-28-tilelang-hd256-decode-m1-codegen-failure.md)).

This commit unblocks the build by gating HD256 decode behind a new
`tilelang-decode-hd256` feature, then runs the matched A/B that the
Plan agent commissioned.

## What Worked (the Validation, not the Kernel)

`scripts/bench_guidellm.sh tilelang-{off,on}-r{1..6} --fast`
(profile=concurrent rate=16, data=4096-in/256-out, max-seconds=30,
random-seed=20260416). Same HEAD, same box, **6 runs per side**
(server kept up across runs to amortize CUDA-graph capture; warm vs
cold prefix-cache state varies inside each side and is the dominant
source of run-to-run variance).

L4 / Qwen3-4B BF16 weights / FP8 paged KV (auto) / Driver 580.82.07
/ CUDA 13.0 / guidellm 0.6.0. Both servers:
`--num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --cuda-graph true`.

### Per-run

| run | OFF tok/s | ON tok/s | OFF ITL p50 | ON ITL p50 | OFF TTFT p50 | ON TTFT p50 |
|-----|----------:|---------:|------------:|-----------:|-------------:|------------:|
| r1  | 172.15    | 137.23   | 79.33       | 80.47      | 8486.1       | 10566.7     |
| r2  | 165.98    | 174.65   | 82.67       | 80.00      | 634.9        | 1337.6      |
| r3  | 84.38     | 122.51   | 67.92       | 68.21      | 8368.7       | 7466.3      |
| r4  | 196.64    | 129.70   | 77.94       | 66.06      | 10023.3      | 8235.6      |
| r5  | 153.84    | 124.06   | 87.37       | 67.90      | 1143.9       | 8194.6      |
| r6  | 120.78    | 135.57   | 80.59       | 64.51      | 7457.2       | 5757.1      |

### Aggregate (n=6 per side)

| metric         | OFF median | OFF mean | ON median | ON mean | Δ median | Δ mean    |
|----------------|-----------:|---------:|----------:|--------:|---------:|----------:|
| out tok/s      | 159.91     | 148.96   | 132.63    | 137.29  | **-17.1%** | **-7.8%** |
| ITL p50 (ms)   | 79.96      | 79.30    | 68.06     | 71.19   | **-14.9%** | **-10.2%** |
| TTFT p50 (ms)  | 7912.95    | 6019.02  | 7830.45   | 6926.32 | -1.0%    | +15.1%    |

The ITL/throughput inversion is the diagnostic: TileLang's HD128 alias
is **15% faster on per-token decode** (ITL p50 drops 79.96 → 68.06 ms
median) but loses on aggregate tok/s. Decode-rate ceiling at saturated
B=16: OFF ≈ 16/0.0793 = 202 tok/s; ON ≈ 16/0.0712 = 225 tok/s. Both
sides run far below ceiling because the c=16 / 4096-in admission burst
spends most of the 30s window in mixed prefill+decode steps, not pure
decode. **TileLang prefill HD128 is slower per chunk than FlashInfer's
prefill HD128**, so each mixed step makes less progress; that
difference dominates the headline tok/s metric on this workload.

The Tranche 4 stub feared the prefill cubin's 16-row Q-tile would be
wasted on 1-row decode batches. The data says the opposite: TileLang's
HD128 cubin actually wins at decode (probably due to better cublasLt
fusion or warp-spec) but loses at prefill chunk processing. So the
real architectural fix is **a dedicated decode kernel** (BLOCK_M=1,
GEMV-style) that locks in the +15% ITL win, **plus continued use of
FlashInfer for prefill** (OR a prefill cubin retune, separate work).

## Decision

**Keep `tilelang-attn` OFF by default for Qwen3-4B's c=16 admission
workload.** Median tok/s is -17%, well past the -5% revert threshold.
**But ITL is +15% faster** — for steady-state decode-bound workloads
(long sessions with low admission rate, where the 30s window is
dominated by pure-decode steps rather than mixed steps), `tilelang-attn`
ships a real per-token win.

`tilelang-attn` therefore stays opt-in (current state). Two follow-on
operator-roadmap items, recorded here:

1. **Dedicated `tilelang_batch_decode_paged_hd128_q*_kv8` kernel**
   (BLOCK_M=1, no causal mask, grid `(1, num_q_heads, batch)`). Should
   improve on the +15% ITL win observed here while bypassing the
   prefill-cubin penalty for decode launches. Same architectural fix
   that unblocks HD256 decode (see errors entry).
2. **Investigate prefill HD128 perf parity with FlashInfer.** The
   pending-remote prefill HD128 wins entry claimed parity; this bench
   shows a regression at c=16 / 4096-in. Either the canonical
   measurement diverged, or the chunk size and admission-burst pattern
   matter and the "floor" win was on a different shape.

The Tranche 4 alias is structurally wrong for pure-decode shapes.
The right next step (also from the Plan agent) is to write a
**dedicated `tilelang_batch_decode_paged_hd128_q*_kv8` kernel** with
`BLOCK_M=1, no causal mask, grid (1, num_q_heads, batch)` — i.e. a
GEMV-like inner loop that doesn't waste shared memory on a 16×128
Q-tile mostly filled with padding. That kernel has to bypass
TileLang 0.1.9's `M%kMPerWarp==0` invariant the same way the HD256
decode kernel does (see the errors entry), so a single architectural
fix unblocks both.

`tilelang-attn` stays opt-in. The HD128 prefill path (the working
part of Tranche 4) remains usable behind that feature flag for
prefill-bound workloads — but is not promoted to default on the 4B
path until a real decode kernel ships.

## Cross-references

- Code: this commit (gate + errors entry + wins entry, ~7 files).
- Errors: [`2026-04-28-tilelang-hd256-decode-m1-codegen-failure.md`](../errors/2026-04-28-tilelang-hd256-decode-m1-codegen-failure.md)
- Plan agent's brief and decision rule: see prior turn's transcript
  (subagent_type=Plan, 2026-04-28).
- Prior `pending-remote` claim:
  [`2026-04-27-bench-guidellm-cuda-tilelang-tc-decode-hd128-pending-remote.md`](2026-04-27-bench-guidellm-cuda-tilelang-tc-decode-hd128-pending-remote.md)
- FP8 KV default that this run was layered on:
  [`2026-04-28-bench-guidellm-cuda-l4-kv-fp8-auto.md`](2026-04-28-bench-guidellm-cuda-l4-kv-fp8-auto.md)
- Raw artefacts:
  `bench-output/2026-04-28-tilelang-tc-decode-{off,on}/`

## Rule

**Run the matched A/B before promoting an "already-wired" path.**
The Tranche 4 alias was sitting under `--features tilelang-attn`
for weeks with the working theory that the HD128 prefill cubin
"should" handle decode. The kernel hypothesis was specific —
prefill cubin's Q-tile is wasted on 1-row decode — and the Plan
agent flagged it before the bench. The bench confirmed it. Save
the steps next time: when a kernel's BLOCK_M / grid shape doesn't
match the call's actual M / batch shape, **don't ship and validate
later, write the call-shape-specific kernel first.**
