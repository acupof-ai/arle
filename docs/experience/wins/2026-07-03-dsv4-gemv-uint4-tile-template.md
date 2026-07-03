# DSv4 FP8 decode GEMV: uint4 vectorization + TILE-templated batch accumulator (#141, #142)

> Status: pending-remote — H20 build + needle gate + matched A/B ride the next pod run.

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
pending-remote (8×H20, TP=4/EP=4, DSv4-Flash-FP8).

## Results
pending-remote: needle gate ×3 + same-shell binary-pair A/B (decode tok/s,
ITL, B=1 and B=5..8) + nsys share re-measure for the three kernels.

## Problems
- Raw `__nv_fp8x4_e4m3→float4` skips the scalar decoder's NaN→±448 remap —
  same licensed behavior as `fp8d_dot16` on the same weight format; noted
  in-kernel.

## Learnings
pending-remote.
