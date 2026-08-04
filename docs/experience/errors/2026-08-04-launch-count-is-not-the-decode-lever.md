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
18% of the launches removed none of the idle. Node timestamps later priced the
real gap at **0.084 µs per node** — 88.6 µs across a 1060-node step that is
99.7% busy ([budget](../wins/2026-08-04-w8a16-decode-step-kernel-budget.md)).
There was never 4 µs of anything to reclaim.

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

**The ~4.3 ms of "intra-GPU idle" did not exist.** It came from GPU wall 18.97
minus `Σ kernel ≈ 14.7`, and that 14.7 was the T4 trace's 15.886 minus the
*expected* gains of two later tranches, never re-measured. An nsys run on the
champion put Σ kernel at **19.184 ms** with occupancy 0.95 — the step is kernel
time end to end.

The launch cost itself was then measured directly: `--qwen35-decode-graph
false` runs the same work eagerly and costs **+1.55 ms** over ~1000 launches,
i.e. ~1.6 µs per eager launch, consistent with T4's 1.84 ms graph win. Inside a
captured graph node count is effectively free, which is why removing 192 nodes
bought nothing.

The real lever was in one kernel:
[FA3 decode splits](../wins/2026-08-04-fa3-decode-splits-fill-the-sms.md).

## Rule

**A per-unit cost derived by division is a hypothesis, not a rate.** "4.3 ms
of idle ÷ 1059 launches = 4 µs per launch" only becomes a marginal cost if
removing launches removes idle. That is a one-experiment test, and it is
cheaper than the fusion it justifies — run it on the smallest change that
moves the divisor, before building the one that depends on the answer.

Related: [[feedback_no_ungrounded_estimates]],
[[feedback_measured_floor_is_not_physical_floor]],
[[feedback_prove_the_treatment_arm_engaged]].
