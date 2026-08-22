# NVFP4 prefill on the FP8 tensor cores: widen the nibbles to E4M3 — CUDA, 2026-08-20

> Status: Shipped. `a5df06c7c` prefill arms · `30171f8be` derive from Marlin
> (the VRAM fix) · `9f1987f25` delete the arms that were not the serving path ·
> `905fc4fc2` + `ec5edf987` the two kernels.

## Context

[`2026-08-20-marlin-source-freed-18gb.md`](2026-08-20-marlin-source-freed-18gb.md)
left Qwen3.8-27B-NVFP4 ahead of Qwen3.6-27B-FP8 at every decode point and
**behind at c=1 on prefill** on the 32K long-agent workload, which is
154:1 prefill to decode. The cause is structural.

sm_90 has no FP4 tensor core, so any real GEMM widens the nibbles first. Marlin
widens to BF16, so an NVFP4 prefill GEMM runs at the BF16 rate while the FP8
checkpoint's prefill runs at the FP8 rate. `gate_up [34816, 5120]`, M=2048:

| path | ms | TFLOPS | share of that format's H20 peak |
|---|---:|---:|---:|
| Marlin, NVFP4 | 8.678 | 84 | 57% of BF16's 148 |
| Marlin, per-channel FP8 | 8.457 | 86 | 58% |
| DeepGEMM, FP8 | 2.664 | **274** | 93% of FP8's 296 |

Two gaps: the format is worth 2x and Marlin's own prefill efficiency another 1.6x.

## What worked

Widen to **E4M3** and hand the bytes to DeepGEMM. The widened copy lives in
scratch and is rebuilt per call from Marlin's resident layout, so no weight is
resident twice and decode keeps reading half the bytes through Marlin.

**The group scale cannot ride inside the E4M3 value.** This checkpoint's
`weight_scale` uses E4M3's full range — a scan of all 168 NVFP4 scale tensors
puts the max at exactly 448 — and an E2M1 value reaches 6, so the product
reaches 2688 against a 448 ceiling. `prepare_fp4_deepgemm_sfb` divides out a
per-128x128-block **power of two** at load and hands it back as DeepGEMM's
`sfb`, which multiplies it into the fp32 accumulator. A power of two is exact
both ways.

The range that makes this safe is measured. Over the same 168 tensors the widest
128x128 block spans **6.81 binades**, leaving the smallest folded value at 0.332
against E4M3's 0.0156 normal minimum — 4.4 binades of headroom, nothing
subnormal.

Both DeepGEMM arms share `dense_deepgemm_prefill_floor`: it sits above the
engine's declared decode row count, so a batched decode step can never reach it,
and the route disables itself when the engine's prefill chunk is shorter than
the floor.

## Result

1xH20, TP=1, FP8 KV, `--max-running-requests 16`, no spec, runtime `ec5edf987`.
Both arms on the same binary, NVFP4 on GPU 0 and FP8 on GPU 1, points back to
back. `bench-agent-32k-16x8.jsonl` (sha 8867f63e), 32 req/point, `--max-tokens
214`. 32/32 complete, `SERVER_ERRORS=0` everywhere.

| c | NVFP4 ITL ms | FP8 ITL ms | ITL |
| ---: | ---: | ---: | ---: |
| 1 | **20.46** | 24.81 | **+21.3%** |
| 4 | **39.40** | 47.57 | **+20.7%** |
| 8 | **69.81** | 79.00 | **+13.2%** |
| 16 | **130.11** | 137.23 | **+5.5%** |

The ITL lead decays 21.3 → 20.7 → 13.2 → 5.5, which
is the `dense_ffn` decode residue
[the occupancy entry](../errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md)
measured; this work does not touch it, since the arms sit above an M floor no
decode batch reaches.

VRAM:

```
resident   39,348 MB -> 22,356 MB   (-16,992, the retention the derive removed)
vs FP8     +9,984 MB -> -7,008 MB   (a 4-bit model finally smaller than the 8-bit one)
KV pool    1,302,407 -> 1,779,114   (FP8: 1,582,506)
```

Engagement, read from the same process — resident falling to 22,356 MB is also
what a silent arm failure looks like, so the counters are not optional:

```
cuda.fp4.widen_fp8_deepgemm          224   = 2 prefill chunks x 56 NVFP4 MLP layers x 2 GEMMs
cuda.qwen.fp8_per_channel_deepgemm   288
cuda.fp4.marlin_tensorcore           336   decode still Marlin
cuda.qwen.fp8_marlin_tensorcore      437
cuda.qwen.fp8_gemv                   ABSENT
```

Needle ladder, `RAW=1 TEMPLATE=qwen3_nonthink`, 3 runs each: 512 / 4096 / 16384 /
32768 all `exact=3 miss=0 DET`.

## Correctness

The arm is not bit-parity, and the scale algebra does not make it so. Two losses:
the E2M1 x E4M3 product needs 4 mantissa bits where E4M3 stores 3, and
`dsv4_deepgemm_fp8_gemm_nt` has no BF16-activation entry, so activations are
quantised to E4M3 per 128-K block — the same W8A8 the FP8 checkpoint runs.

RMS output error over 4,000 random activations at K=5120, against the exact
product Marlin forms:

| block spread | fold cost |
|---|---:|
| 2.47 binades (measured median) | 2.38% |
| 3.72 (measured p99) | 2.11% |
| 6.81 (measured worst) | 1.94% |
| *bf16 activation -> E4M3, same metric* | *2.65%* |

The fold costs less than the activation rounding the FP8 baseline already
carries, which is the reference that matters.

GSM8K-shaped, 200 items. `arms off` is the same binary with
`--chunked-prefill-size 128`, which puts every GEMM below the floor and turns
both DeepGEMM arms off — a same-binary control:

| arm | exact |
|---|---:|
| NVFP4, arms on | 188/200 (94.0%) |
| NVFP4, arms off (control) | 189/200 (94.5%) |
| Qwen3.6-27B-FP8 | 169/200 (84.5%) |

On/off agree on 196/200 answers. **The absolute scores are not a GSM8K result**:
the pod has no network, so the items come from the repo's own
`examples/opd/gsm8k-train.jsonl`, the TRAIN split. Only the arm-to-arm
difference on identical inputs carries.

## The per-call cost was 8x the estimate

The design was justified with a roofline: 278 MB against a 2.664 ms GEMM,
"~3.4%". `ARLE_QWEN35_QUANT_PROFILE`, three 14K-token prefills, ms/call:

| op | as shipped | tables out | vectorised | |
|---|---:|---:|---:|---|
| `qwen/deepgemm/dense_gemm` | 1.4465 | 1.4426 | 1.4427 | control |
| `qwen/fp4/dense_widen_fp8` | **0.7559** | **0.1871** | 0.1871 | −75.3% |
| `qwen/fp8/dense_channel_scale` | 0.1169 | 0.1153 | **0.0313** | −73.2% |
| `qwen/deepgemm/dense_pack_quantize` | 0.0424 | 0.0409 | 0.0410 | control |
| `qwen/fp8/dense_materialize` | 0.0508 | 0.0505 | 0.0505 | control |

The widen was **52% of the GEMM it feeds**, not 3.4% — 0.756 ms against a 0.093
ms bound, an effective 370 GB/s. Non-GEMM overhead 838 ms → 303 ms, 24.5% →
10.5% of the profiled total.

The tell was not the roofline but the FP8 materialiser beside it at 0.0505
ms/call for comparable traffic: a 15x gap between two kernels doing the same
kind of work. Two divergently-indexed tables explained it — the `__constant__`
E2M1 LUT (16 reads per work item, replayed per distinct value) and a
`const int R[4]` with a dynamic index, which nvcc puts in local memory. Both
became arithmetic.

**The first ALU form of the decoder was wrong**, and an exhaustive 16-input
comparison caught it before it compiled: E2M1's `exp == 0` is the subnormal step
(0 and 0.5), and treating it as normal returns 0.75 where the encoding means
0.5. That is slightly-wrong weights rather than a crash — the needle ladder
would not reliably catch it and the eval would read it as noise.

## Rejected on measurement

`ncu`, 12 launches each:

| | ms/launch | bank conflicts | SM throughput |
|---|---:|---:|---:|
| `dequantize_fp4_marlin_to_fp8` | 0.199 | 13,236,164 | **87.2%** of peak |
| `marlin_fp8_to_e4m3` | 0.041 | 16,571,970 | 45.8% |

**Bank-conflict padding.** The conflicts a review predicted would dominate are
real and not the limiter: the FP4 kernel is issue-bound at 87% of SM peak, and
the FP8 kernel — 2.6x more conflicts per sector, and the headroom to use a fix —
is 1.8% of the profiled total. Skewing the banks takes FP4's shared memory from
4.1 KB to 10.3 KB and its occupancy limit from 19 blocks/SM to ~7.

**Folding the column scale into its consumer.** Counted from the checkpoint,
`linear_attn` produces 52.1% of the bytes that scale touches and its consumer is
the gated-delta recurrence. SwiGLU and the residual add cover 20.1%, which is
0.2% of the step.

**Stream overlap of the materialisation.** Worth 2.5% once the widen dropped to
146 ms, against a second stream, double buffering (+178 MB) and a dispatch API
change.

**Moving the 170 MiB scratch off `thread_local`.** The thread-locality is load
bearing: two threads sharing one materialisation buffer and one stream can have
the second widen overwrite the first before its GEMM runs.

The path is now GEMM-dominated at 89.5%, with `dense_gemm` at 93% of this card's
FP8 peak.

## Rule

When the hardware has no tensor core for the stored format, the question is
**what to widen to**. Marlin's choice of BF16 put NVFP4 prefill at half the FP8
checkpoint's rate on the same card, and the fix was one output type — the fused
W4A8 kernel that looked necessary would have bought the last 3.4%.

A format whose scale granularity is finer than the GEMM's cannot fold that scale
into the value without a range argument. Get the distribution off the checkpoint
first: `max = 448` said the naive fold overflows 6x, and `worst block = 6.81
binades` said the power-of-two fix has 4.4 to spare. Both were one pass over the
safetensors headers.

A roofline bounds a bandwidth-bound kernel and says nothing about an issue-bound
one. A freshly written kernel has no evidence either way, so profile it before
quoting its cost — one serve, no `ncu` needed.
