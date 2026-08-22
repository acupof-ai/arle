# Reading a shared NVFP4 base in its Marlin layout is bit-exact

## Context

`--share-frozen-base` lets an OPD student borrow the serving engine's frozen
weights instead of holding a second copy. It worked for FP8 (DeepGEMM eats the
block-scaled bytes as-is) but not for NVFP4: the loader repacks NVFP4 into the
Marlin tile layout and frees the group-layout source, so the student had
nothing to borrow. Sharing an NVFP4 base saves 9 GB per rank versus FP8
(21 GB vs 30 GB resident).

## What Worked

Teach autograd to read the Marlin layout directly rather than keeping a second
group-layout copy alive. `marlin_fp4_to_bf16`
(`crates/autograd/src/backend_cuda/kernels/fp8_block_scaled.cu`) inverts the
repack in one pass: the `{0,2,4,6,1,3,5,7}` nibble map, the per-64-run 8x8
scale transpose followed by the quad swap of lanes 1 and 2, and the
`2^-119` global-scale compensation.

Two bugs were caught before any GPU time, by replaying the repack's index map
in Python and diffing against the inverse:

- the de-permutation undid the quad swap first when the repack applies it last
  (384 of 512 scales wrong; the nibble map was already 8192/8192 exact);
- `new_borrowed_marlin` stored the global scale as a device allocation inside
  storage whose `Drop` deliberately forgets every buffer, leaking one
  allocation per weight per import.

`cuda_marlin_fp4_dequant_matches_group_layout` compares `I @ W^T` from both
layouts and requires exact equality. On sm_90 it passes: the Marlin read and
the group read produce identical bf16.

## Rule

An index-map inverse is cheap to verify offline and expensive to debug on a
GPU. Replay the forward permutation in a scratch script and diff against the
inverse before spending a build cycle -- both bugs here were visible with no
device at all.
