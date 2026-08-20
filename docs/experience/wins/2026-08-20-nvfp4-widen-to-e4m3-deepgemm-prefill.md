# NVFP4 prefill on the FP8 tensor cores: widen the nibbles to E4M3 — CUDA, 2026-08-20

> Status: Shipped. `a5df06c7c` (prefill arms) + `30171f8be` (derive from Marlin,
> the VRAM fix) + `9f1987f25` (delete the arms that were not the serving path).

## Context

[`2026-08-20-marlin-source-freed-18gb.md`](2026-08-20-marlin-source-freed-18gb.md)
left Qwen3.8-27B-NVFP4 ahead of Qwen3.6-27B-FP8 at every decode point and
**33.9% behind at c=1 end-to-end** on the 32K long-agent workload, which is
154:1 prefill to decode. Decode was never the problem. Prefill was, and the
reason is structural rather than a tuning miss.

sm_90 has no FP4 tensor core. Any real GEMM has to widen the nibbles first, and
Marlin widens them to BF16 — so an NVFP4 prefill GEMM runs at the BF16 rate
while the FP8 checkpoint's own prefill runs at the FP8 rate. Measured on
`gate_up [34816, 5120]` at M=2048:

| path | ms | TFLOPS | share of that format's H20 peak |
|---|---:|---:|---:|
| Marlin, NVFP4 | 8.678 | 84 | 57% of BF16's 148 |
| Marlin, per-channel FP8 | 8.457 | 86 | 58% |
| DeepGEMM, FP8 | 2.664 | **274** | 93% of FP8's 296 |

Two gaps, not one: the format is worth 2x and Marlin's own prefill efficiency is
worth another 1.6x.

## What worked

Widen to **E4M3** instead of BF16 and hand the bytes to DeepGEMM. The widened
copy lives in scratch and is rebuilt per call, so the FP4 layout stays the
resident one and decode keeps reading half the bytes through Marlin.

The dequant is not free but it is small: 278 MB of traffic against a 2.664 ms
GEMM is ~3.4% at M=2048, 13% at M=512. Effective 265 TFLOPS against Marlin's 84.

**The group scale cannot ride inside the E4M3 value.** This checkpoint's
`weight_scale` uses E4M3's full range — a scan of all 168 NVFP4 scale tensors
puts the max at exactly 448 — and an E2M1 value reaches 6, so the product
reaches 2688 against a 448 ceiling. A per-128x128-block **power of two** is
divided out at load (`prepare_fp4_deepgemm_sfb`) and handed back as DeepGEMM's
`sfb`, which multiplies it into the fp32 accumulator. A power of two is exact
both ways.

The floor that makes this safe is measured, not assumed. Over the same 168
tensors the widest 128x128 block spans **6.81 binades**, which leaves the
smallest folded value at 0.332 against E4M3's 0.0156 normal minimum — 4.4
binades of headroom, so nothing lands in the subnormal range.

Same shape as the per-channel FP8 arm that shipped in the same tranche, and it
shares that arm's `dense_deepgemm_prefill_floor`: the floor sits above the
engine's declared decode row count, so a batched decode step can never reach it,
and the route disables itself — retention included — when the engine's prefill
chunk is shorter than the floor.

## What it costs, and why that turned out to be free

Both DeepGEMM arms pin their pre-repack source: **16,992 MB** measured
(NVFP4 8.4 GB + per-channel FP8 10.6 GB, less the shapes each arm declines).

That would have been a bad trade on its own — it is the same VRAM the previous
entry recovered. It is paid for by a flag the workload always justified:

```
                       256 slots (no flag)   16 slots (--max-running-requests 16)
recurrent reservation        37,584 MB              2,349 MB
post-weights free            75,515 MB             58,523 MB   (after retention)
max_total_tokens                790,603            1,302,407
```

`hot_workspace_slots()` is `max_running_requests.unwrap_or(num_slots)`, and the
executor pre-allocates one ~146 MB recurrent block per slot eagerly. The chain
bench never passed the flag, so it reserved 240 slots it could not admit —
35,235 MB. The archived FP8 anchor in `docs/baselines.md` already runs 16 slots;
the chain runs were the deviation.

Net: the KV pool is **1,302,407 tokens against the 790,603 this entry started
from and the FP8 checkpoint's 593,995**, with both sources retained.

## Engagement

2,608-token prompt, `--max-running-requests 16`, FP8 KV:

```
cuda.fp4.widen_fp8_deepgemm          224     <- new
cuda.qwen.fp8_per_channel_deepgemm   288
cuda.fp4.marlin_tensorcore           336     <- decode still Marlin
cuda.qwen.fp8_marlin_tensorcore      437
cuda.qwen.fp8_gemv                   ABSENT
```

224 is exactly 2 prefill chunks x 56 NVFP4 MLP layers x 2 GEMMs.

## Correctness

The arm is **not** bit-parity and cannot be argued from the scale algebra. Two
separate losses, both intrinsic to the entry point:

1. The E2M1 x E4M3 product needs 4 mantissa bits and E4M3 stores 3, so about a
   quarter of the nonzero weights round at half an ulp.
2. `dsv4_deepgemm_fp8_gemm_nt` has no BF16-activation entry, so the activation
   is quantized to E4M3 per 128-K block — the same W8A8 the FP8 checkpoint
   already runs at prefill.

Sized rather than argued, as RMS output error over 4,000 random activations at
K=5120, against the exact product Marlin forms:

| block spread | fold cost |
|---|---:|
| 2.47 binades (measured median) | 2.38% |
| 3.72 (measured p99) | 2.11% |
| 6.81 (measured worst) | 1.94% |
| *bf16 activation -> E4M3, same metric* | *2.65%* |

The fold costs less than the activation rounding the FP8 baseline already
carries, which is the honest reference — not bit-parity with Marlin.

Needle ladder, `RAW=1 TEMPLATE=qwen3_nonthink`, 3 runs per length:

```
len=512    exact=3 miss=0 DET
len=4096   exact=3 miss=0 DET
len=16384  exact=3 miss=0 DET
len=32768  exact=3 miss=0 DET
```

## Result

1xH20, TP=1, FP8 KV, `--max-running-requests 16`, no spec. Both arms on the same
binary, NVFP4 on GPU 0 and Qwen3.6-27B-FP8 on GPU 1, points taken back to back.
`bench-agent-32k-16x8.jsonl` (sha 8867f63e), 32 req/point, `--max-tokens 214`.
32/32 complete and `SERVER_ERRORS=0` at every cell.

| c | NVFP4 ITL ms | FP8 ITL ms | ITL | NVFP4 out tok/s | FP8 out tok/s | end-to-end |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **20.45** | 24.81 | **+21.3%** | 19.49 | 19.65 | −0.8% |
| 4 | **40.04** | 47.31 | **+18.2%** | **84.13** | 75.53 | **+11.4%** |
| 8 | **71.16** | 79.47 | **+11.7%** | **98.00** | 91.62 | **+7.0%** |
| 16 | **130.68** | 135.84 | **+3.9%** | 106.24 | 106.53 | −0.3% |

The c=1 end-to-end leg is what this change was for: it was **−33.9%** before the
prefill arms and −5.9% with them but with both sources retained; the VRAM fix
took it to parity. ITL leads at every concurrency and decays monotonically
21.3 → 18.2 → 11.7 → 3.9, which is the `dense_ffn` residue
[the occupancy entry](../errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md)
measured, unchanged by this work — the DeepGEMM arms sit above an M floor no
decode batch reaches.

**VRAM, the actual point:**

```
resident        39,348 MB -> 22,356 MB   (-16,992, exactly the retention removed)
vs FP8          +9,984 MB -> -7,008 MB   (a 4-bit model finally smaller than the 8-bit one)
KV pool         1,302,407 -> 1,779,114   (FP8: 1,582,506)
```

**Engagement, read from the same process** — resident falling to 22,356 MB is
also what a silent arm failure looks like, so the counters are not optional:

```
cuda.fp4.widen_fp8_deepgemm          224   = 2 prefill chunks x 56 NVFP4 MLP layers x 2 GEMMs
cuda.qwen.fp8_per_channel_deepgemm   288
cuda.fp4.marlin_tensorcore           336   decode still Marlin
cuda.qwen.fp8_marlin_tensorcore      437
cuda.qwen.fp8_gemv                   ABSENT
```

**Quality**, GSM8K-shaped, 200 items, three arms on one binary — `ev_off` is the
same NVFP4 checkpoint with `--chunked-prefill-size 128`, which puts every GEMM
below the floor and turns both DeepGEMM arms off, so it is a same-binary control:

| arm | exact | |
|---|---:|---|
| NVFP4, arms on | 188/200 (94.0%) | |
| NVFP4, arms off | 189/200 (94.5%) | control |
| Qwen3.6-27B-FP8 | 169/200 (84.5%) | |

On/off agree on 196/200 answers. The numerics do move — the fold and the E4M3
activations are real — but not enough to change an answer: one item in 200.
**The absolute scores are not a GSM8K result**: the pod has no network, so the
items come from the repo's own `examples/opd/gsm8k-train.jsonl`, the TRAIN split.
Only the arm-to-arm difference on identical inputs carries.

Two runs before this one produced no numbers and were discarded rather than
reported: one serve took an external SIGTERM (traced — no ERROR, no fatal path,
PID absent from the cgroup OOM record), and one lost `target/release/arle`
mid-flight to a concurrent `cargo build`. A third c=16 row is discarded because
the FP8 control arm moved 30.42 → 106.53 out tok/s with unchanged code and
identical resident bytes; a control that moves invalidates the run it is in.

## Rule

When the hardware has no tensor core for the stored format, the question is not
whether to widen but **what to widen to**. Marlin's choice of BF16 is what put
NVFP4 prefill at half the FP8 checkpoint's rate on the same card, and the fix
was one output type, not a new kernel — the fused W4A8 kernel that looked
necessary would have bought the last 3.4%.

A format whose scale granularity is finer than the GEMM's scale granularity
cannot fold that scale into the value without a range argument. Get the
distribution off the checkpoint before designing around it: `max = 448` said the
naive fold overflows 6x, and `worst block = 6.81 binades` said the power-of-two
fix has 4.4 binades to spare. Both were one pass over the safetensors headers.
