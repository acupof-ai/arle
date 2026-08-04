# Where an 80K training step goes: one kernel is 71%, and FA3 is worth 3.54× not 2.17×

**Date:** 2026-08-05 · **Pod:** 8×H20, commit `e675f031b`, ThinkingCap-Qwen3.6-27B-FP8, LoRA attention-qv

> Status: Characterization. No runtime change in this entry; it is the map the
> 80K campaign is ranked against, and it re-ranks the campaign.

## Context

Every prior ranking rested on one nsys capture at **seq=8192, cp=2, before FA3**.
The target is 80K and attention is O(s²), so the shares could not survive the
extrapolation. Three runs at seq=81920 plus one nsys profile.

## A1 — one card does not fit 80K

cp=1, seq=81920. Forward completes; backward does not.

| stage | |
|---|---|
| forward_hidden_states | 3972.216 s |
| fused_ce | 3.620 s |
| backward | `cuda alloc_zeros failed` |
| peak VRAM (sampled) | ~85 / 97.9 GB |
| peak host RSS | 104.5 GB |

The `merge_grad` in-place accumulate fix holds — the forward and fused-CE path
that used to be the wall now clears 80K. The remaining wall is in backward, and
the binary logs no op/shape/bytes beyond that string, so **which tensor is
unknown**.

Consequence: cp=1 × dp=8 is not available at 80K, CP stays, and the "bf16 tape
lets one card hold the sequence" argument now has a price it has to beat — one
we cannot state until the failing allocation is named.

## A2 vs A3 — FA3 at 80K

cp=2, seq=81920, GPUs 4/5, same binary, same devices, back to back.

| | FA3 ON | FA3 OFF | ratio |
|---|---|---|---|
| forward (r0/r1) | 80.671 / 81.009 s | 512.283 / 516.438 s | **6.35×** |
| backward | 670.275 / 670.319 s | 2151.332 / 2151.435 s | **3.21×** |
| step wall | 752.956 / 753.294 s | 2670.055 / 2669.886 s | **3.54×** |
| peak VRAM/rank | 91547 MiB | 91707 MiB | wash |
| host RSS peak | 55.5 GB | 55.4 GB | wash |
| loss | 4.536131 | 4.534415 | |
| grad_norm | 7.202155 | 8.037487 | 11.6% apart |

The same A/B gave 2.17× at seq=32768. The win grows with sequence length, as
O(s²) attention against O(s) everything else predicts.

The 11.6% grad-norm gap between two cp=2 arms reproduces the 14% seen at 32768
(#85). **That divergence is not sequence-specific**, and it is not caused by FA3:
at 32768 with a cp=1 anchor available, FA3 (2.265) sits closer to single-card
(3.745) than the scalar ring (1.984) does.

## The nsys table — one kernel is the step

One A2 step, both ranks combined, `cuda_gpu_kern_sum`:

| share | time | instances | kernel |
|---|---|---|---|
| **71.0%** | 707.345 s | 90 | `linear_attention_chunked_scan_backward_f32` |
| 6.7% | 66.316 s | 238,080 | `gated_delta_rule_prefill_recurrent_kernel` |
| 3.9% | 38.365 s | 7,436 | nvjet GEMM 128×256 |
| 3.2% | 32.096 s | 4,194 | nvjet GEMM 320×128 TNT |
| 1.9% | 19.134 s | 2,886 | nvjet GEMM 320×128 NNT |
| 1.5% | 15.271 s | 11,664 | `transpose_axes_swap_f32` |
| 1.5% | 14.635 s | 47 | `FlashAttnBwdSm90` |
| 1.4% | 13.553 s | 25,196 | `slice_f32` |

At seq=8192 this kernel family was 26% and attention was 31%. At 80K the GDN
backward is 71% and attention, once FA3 is on, is 1.5%. The earlier note that
"GDN is O(s) so its share drops at 80K" was wrong in direction.

Together the two GDN rows are 77.7%. Both are on the route the FlashQLA port
replaces, so the projection for that port at upstream's ~2× backward is a step
around 440–490 s — roughly 5.5× against the shipped default this session started
from (FA3 off + the chunked-scan backward = 2670 s).

## Build hygiene — the run that nearly published a wrong number

The first A2 launch failed with "CP parallelism requires the nccl feature". Root
cause: the shared `/host/arle-build/target` had been overwritten 24 min after our
build by another actor's `cuda`-only build, confirmed via `.fingerprint`
timestamps and `nm`/`ldd` showing zero nccl symbols. A2/A3/nsys were rebuilt in a
private tree and the binary frozen to `/host/opd80k-out/arle.a2a3.snap`.

## Rule

- Re-derive the share table at the target shape before ranking against it. An
  8192 capture ranked GDN backward third; at the real 80K it is the entire step.
  O(s) vs O(s²) reasoning got the direction wrong because the two GDN rows do
  not scale the way the label "linear attention" suggests.
- On a shared box, verify the binary you are about to measure — `nm -D` for the
  symbol the feature implies, `ldd` for the library. A silently-overwritten
  shared target produced a failure that reads as a code bug.
