# DSv4 FP8 decode GEMV: uint4 vectorization + TILE-templated batch accumulator (#141, #142)

> Status: LICENSED 2026-07-04 — measured on 8×H20 (see Results; shared campaign
> with the mhc levers, so deltas below are the three-lever aggregate).

## Goal
Cut the FP8 GEMV stack's share of DSv4 B=1 decode GPU-busy (nsys TP=4/EP=4:
batch_tiled 27.7% + batch 9.8% = the scalar per-byte inner loops running at
~4% HBM bandwidth vs a ~0.66ms roofline floor). 6ms/token plan lever G1.

## Hypothesis
uint4 loads (16 FP8 bytes + paired bf16 vectors, the needle-validated
`fp8d_dot16` / `fp8_f32_dot16` idiom already in-tree) lift the inner loop from
latency-bound scalar to bandwidth-bound vector; the `template<int TILE>`
accumulator (Qwen sibling measured fixed-32 at 2.15x/3.59x/6.50x vs templated
1.04x/1.07x/1.14x for B=2/4/8) removes the 32-register fixed cost at the
B=2..8 shapes decode actually uses.

## Params
- Kernels: `dsv4_fp8_gemv_kernel`, `dsv4_fp8_gemv_batch_kernel`,
  `dsv4_fp8_gemv_batch_tiled_kernel` (now `template<int TILE>`,
  TILE ∈ {2,4,8,16,32} smallest ≥ min(B,32)).
- Fast-path guard: `K%16==0 && block_w%16==0` (production block_w=128), else
  the original scalar loop verbatim. FP4/MMA/Qwen kernels untouched.
- extern "C" dispatch signatures byte-identical.

## Env
8×H20, TP=4/EP=4 GPUs 0-3, DSv4-Flash-FP8 (274G FP8-native dir), greedy T=0,
HEAD 6d238b7e (`/host/arle-kern141-bin`) vs base 5bbeeaac (`/host/arle-base-bin`),
same quad/env/scripts, back-to-back serve cycles, stats-trace method.

## Results (aggregate with the #143 mhc levers — same campaign, same binaries)
- First nvcc compile of both .cu diffs: zero errors/warnings.
- Correctness: spec-none count gates verbatim-clean ×2, needle 3/3 first-token
  exact (ZEPHYR-74915 / MULBERRY-3382 / GRANITE-5519); MTP-on needle 2/2. MTP-on
  count drift observed on BOTH binaries at same rate (HEAD 2/4 vs base 1/4
  clean) — pre-existing verify-lane behavior, not a regression.
- **MTP-off c=1 (467-in/256-out, ×3): TPOT 39.57 → 24.90 ms (−37.1%), decode
  25.27 → 40.16 tok/s (+58.9%).**
- **MTP-on c=1 (2015-in/256-out, ×3): 31.27 → 20.94 ms/committed-tok (−33.0%),
  31.98 → 47.75 tok/s (+49.3%);** implied ms/step 59.1 → 41.4.
- nsys (HEAD, MTP-off 30s decode window): dsv4_fp8_gemv_batch now 9.6% share
  (8.2 µs avg); top shares now gemv_handwritten bf16 16.6%, NCCL AllReduce
  8.6%, mhc_params 7.8%. `dsv4_fp8_gemv_kernel` (non-batch) 0 instances —
  B=1 decode routes entirely through the batch variant. tiled<32> appears only
  in chunked prefill in this lane (7392 inst). Audit-window shares (27.7/9.8)
  are a different lane mix — not directly comparable.
  Report: `/host/kern141_decode2.nsys-rep`.

## Problems
- Raw `__nv_fp8x4_e4m3→float4` skips the scalar decoder's NaN→±448 remap —
  same licensed behavior as `fp8d_dot16` on the same weight format; noted
  in-kernel.
- nsys system-wide attach on a running process captured no kernels (empty
  report) — use `nsys launch` + `start/stop` instead.

## Learnings
- Blind-written CUDA is verifiable by READING when every construct has an
  in-file compiling precedent — the deterministic pre-read predicted the
  zero-error first compile; the compiler is just a faster reader.
- Next decode levers by the fresh share table: gemv_handwritten (bf16 dense,
  16.6%), NCCL allreduce (8.6%), mhc_params eager tail (7.8%), MoE
  swiglu/down (13%).
