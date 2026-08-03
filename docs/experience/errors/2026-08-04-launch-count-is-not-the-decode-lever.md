# Cutting 192 decode launches/step moved the GPU wall 0.00 ms — 2026-08-04

> Status: **Built, measured, reverted.** Refutes the "launch count is the
> lever" conclusion this same investigation reached hours earlier, and kills
> two more fusions that were queued on the same reasoning.

## The prediction and the result

The two-sided ledger put ARLE at 1059 launches/step against SGLang's 928, with
~4.3 ms of intra-GPU idle against 1.68 — **~4.0 µs of dead time per launch vs
~1.8**. From that I predicted a residual-add + RMSNorm fusion would return its
kernel time *plus* its dispatch gap, and ranked it the top lever.

Fused every residual-add + input-norm pair in all three layer loops
(single-row, contiguous batched, paged batched): **192 launches removed per
decode step**. Measured with the same in-process instrument, same protocol,
same 32K c=1 W8A16 champion:

| | `gpu=` (GPU wall, explicit sync) |
|---|---:|
| champion | 18.973 ms |
| fused, 500 steps | 18.978 |
| fused, 1000 steps | 19.006 |

**Zero.** Not the predicted ~0.7 ms, not even the 0.156 ms of kernel time the
ledger attributed to that path.

Treatment engagement verified before concluding (the T5b lesson): the build
log shows `cuda-kernels` and `infer-cuda` both recompiled, all three call
sites converted, and zero remaining `add_batch(hidden, attn_out, hidden_mid)`
in the pod tree.

## Two things were wrong

**The dispatch-gap model.** 4.0 µs/launch was a derived average
(idle ÷ launches), and I treated it as a marginal cost. It is not: removing
18% of the launches removed none of the idle. Whatever the GPU is doing
between kernels during a graph replay, it does not scale with node count.

**The kernel itself was a wash by construction.** The fused kernel's second
pass re-reads the sum from global memory (`uint2 xv = sum_ro[i]`), so its
traffic is `read a + read b + write sum + read sum + read w + write norm`
= 6×H — *identical* to the unfused pair's 3×H + 3×H. It saved exactly one
launch and nothing else. My "5×H → 4×H" estimate had miscounted both sides
(forgot the weight read on the norm side, forgot my own re-read).

A register-resident second pass (the [`T6 GDN`](2026-08-03-t6-gdn-decode-kernel.md)
pattern) would make it 5×H and is worth ~0.1 ms — but that is 0.5% of the
step, and it is not where the 4.3 ms lives.

## What this kills, and what is still open

Dead, on this evidence: the conv1d 2-kernel merge (~48 launches) and the
split2/split_qkv fusion (~64), both of which were ranked purely on the
4 µs/launch model.

**Still unexplained: ~4.3 ms/step of intra-GPU idle** (GPU wall 18.97 vs
Σ kernel ~14.7). It is not host time (0.061 ms, measured), and it is not
per-launch dispatch (this entry). The next probe should ask what the device
is waiting on — dependency serialization between adjacent kernels, tail
effects on small grids, or memory-system stalls the kernel-duration sum
already counts as "busy" — and it needs `ncu`, not another fusion.

## Rule

**A per-unit cost derived by division is a hypothesis, not a rate.** "4.3 ms
of idle ÷ 1059 launches = 4 µs per launch" only becomes a marginal cost if
removing launches removes idle. That is a one-experiment test, and it is
cheaper than the fusion it justifies — run it on the smallest change that
moves the divisor, before building the one that depends on the answer.

Related: [[feedback_no_ungrounded_estimates]],
[[feedback_measured_floor_is_not_physical_floor]],
[[feedback_prove_the_treatment_arm_engaged]].
