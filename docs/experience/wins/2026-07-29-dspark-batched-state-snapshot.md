# DSpark snapshot/restore in one launch — D2D is a host cost, not a bandwidth one

## Context

After the varlen replay landed, an nsys capture of the shipped binary put
`cuMemcpyDtoDAsync` at **336,315 calls / 3,681 ms = 12.3% of the wall**. The
on-GPU time for the same copies is **683 ms = 2.3%** — the cost is issuing them,
~11 µs of host driver time each, not moving the bytes.

Splitting them by transfer size names them exactly:

| bytes | calls | what |
|---:|---:|---|
| 3,145,728 | 89,995 | GDR recurrent state (48 v_heads × 128 × 128 f32) |
| 61,440 | 90,263 | conv ring |
| 576 | 102,144 | b/a capture |
| 122,880 | 51,072 | qkv capture |

Snapshot + restore is the first two = **54% of every D2D call**: 48 layers × B
slots × 2 buffers, twice per tick.

## What Worked

`batched_copy_uniform_cuda(dst_ptrs, src_ptrs, bytes, count)` — `count`
equal-sized copies in one launch, `blockIdx.y` picking the buffer, `uint4`
words. The snapshot and the restore each become two launches (gdr, conv) instead
of `2 * num_linear * B` memcpys.

## Measurement

Matched, one binary, GPU 0, ThinkingCap-Qwen3.6-27B-FP8 + 27B-DFlash, block 6,
48 req/point, max_tokens 214, seed 20260416. `varlen` is the previous arm.

| c | no-spec | varlen | **+batched copy** | TPOT no-spec | varlen | **+batched** |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 13.9 | 18.8 | **18.9** | 28.86 | 10.58 | **10.40** |
| 2 | 47.1 | 83.2 | **86.2** | 37.38 | 19.54 | **18.76** |
| 4 | 60.7 | 104.0 | **106.6** | 59.26 | 33.15 | **31.56** |
| 8 | 94.5 | 120.7 | **122.9** | 71.93 | 57.01 | **55.89** |
| 16 | 122.7 | 133.1 | **133.3** | 101.88 | 95.70 | **95.04** |

+3.6% / +2.5% / +1.8% at c=2/4/8, TPOT −4.0% / −4.8% / −2.0%. Each point is
small — c=2 and c=4 TPOT are the only ones outside the ±3% drift band — but all
**ten** measurements move the same way, which a drift cannot do. Gate exact=3 DET
at 512/4k/16k, 0 errors. The run carried no `--spec-max-batch`, so it also
verifies the new default of 16 end to end.

Against no-spec, shipped: **c=1 +36% / c=2 +83% / c=4 +76% / c=8 +30% /
c=16 +8.6%** tok/s; TPOT −64 / −50 / −47 / −22 / −6.7%.

## Still open

The other 46% of the D2D calls is the linear capture — 3 copies per slot per
layer out of the packed verify rows. One packed capture per layer would make it
3 per layer and would let the varlen replay take row offsets instead of pointer
tables: a deletion, not another table.

## Rule

**Separate a transfer's bandwidth from its issue cost before optimizing it.**
300 GB of D2D in 30 s reads like a bandwidth problem and is 2.3% of the GPU's
time; the same line is 12.3% of the wall on the CPU that issued it. The fix that
follows is "fewer calls", which is the opposite of what a byte-count would
suggest.
