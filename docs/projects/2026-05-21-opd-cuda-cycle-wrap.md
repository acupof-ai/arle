# ARLE OPD CUDA Cycle Wrap

## Headline

ARLE OPD CUDA today: **~30 s naive -> 0.164387 s/step** on the real
Qwen3-0.6B checkpoint path, a conservative **~170x** speedup headline, while
the real-checkpoint eval moved from a failed high-LR recipe to **CONVERGES**
at `lr=1e-7`.

Ground-truth wall-clock from this CUDA cycle:

- OPD naive reference from the CPU cycle: `~30 s/step`.
- First real-checkpoint CUDA profile in this cycle: `10.411479 s/step`.
- Current real-checkpoint CUDA profile after decode prepare/layout fusion:
  `0.164387 s/step`, `0.42%` sigma over n=3.
- Speedup vs `10.411479 s` measured real-checkpoint profile: **63.3x**.
- Speedup vs `~30 s` naive reference: **~170x conservative** (`30.0 /
  0.164387 = 182.5x` arithmetic, but the naive baseline is coarse).
- Moderate-shape CUDA now runs at `0.047103 s/step` mean, faster than the
  PyTorch CUDA reference `0.083179 s/step` by **1.77x**.
- Real Qwen3-0.6B OPD eval at `lr=1e-7`: step 200 train overlap `81.25%`,
  step 500 held-out overlap `64.06%`, and step 200 train KL down `41.27%`.

## Cycle Table

| Commit | Axis | Impact |
| --- | --- | --- |
| `6f9e9b9` | Trainable student checkpoint load | Added the real-checkpoint student path needed for later OPD eval. |
| `6ff5f3e` | Checkpoint reload OPD step test | Proved the trainable checkpoint path can run an OPD step after reload. |
| `cf8f03c` | CPU profile warmup | Made measured OPD CPU profile runs stable enough for matched controls. |
| `319aa5a` | KL grad correctness | Pinned wide-range `kl_distill_loss` finite-diff gradients before CUDA perf claims. |
| `b1c53cc` | PyTorch CUDA baseline | Established the moderate-shape perf target: `0.083179 s/step`. |
| `e7ca73d` | First ARLE CUDA OPD step | Ran moderate OPD on CUDA at `0.099181 s/step`, CPU/CUDA loss relerr `1.276e-6`. |
| `7e67b92` | CUDA convergence correctness | Verified numerical substrate and showed scratch eval signal was weak, not enough to claim usefulness. |
| `0589959` | Real-checkpoint profile | Attributed Qwen3-0.6B step `10.411479 s`; host-mirror retention was the binding constraint. |
| `4a631c0` | Host-mirror fix | Dropped real-checkpoint step to `1.304788 s`; bottleneck shifted to optimizer. |
| `7c669eb` | AdamW device/in-place update | Dropped real-checkpoint step to `0.294321 s`; optimizer `1.036148 s -> 0.025408 s`. |
| `cebd013` | Rollout KV cache | Dropped real-checkpoint step to `0.253540 s`; removed rollout prefix re-encoding. |
| `8fa3d8b` | Real-checkpoint eval, high LR | Found the substrate ran, but `lr=5e-5` destroyed the model. |
| `0a74221` | H1 vs H2 diagnosis | Distinguished recipe failure from algorithm bug; `lr=1e-7` stabilized and improved. |
| `5939cc7` | Real-checkpoint eval, `lr=1e-7` | Licensed CONVERGES: step 200 train overlap `81.25%`, held-out step 500 `64.06%`. |
| `68477fc` | SDPA mask-softmax KILL | Killed Option B: Qwen step stayed `0.253420 s`, moderate regressed to `65.936 ms`. |
| `92de30e` | Fused grad clip | Dropped real-checkpoint step to `0.231698 s`; grad-clip moved off the top line. |
| `36606d5` | Device RoPE | Dropped real-checkpoint step to `0.208804 s`; exposed launch-count limits in rollout. |
| `b0d0cc8` | CUDA Graph rollout KILL | Capture was possible, but replay produced wrong tokens because transient HtoD buffers were captured. |
| `938e19b` | Post-step cleanup KILL | Proved cleanup is a 21k device-free storm, not a host-mirror sweep. |
| `5fb212a` | Device rollout argmax | Dropped real-checkpoint step to `0.195945 s`; rollout readback path `-90.98%`. |
| `9852fa0` | Rollout inner attribution | Identified rollout attention as the next binding component. |
| `67607a0` | Fused decode SDPA | Dropped real-checkpoint step to `0.177385 s`; rollout attention `-27.89%`. |
| `9d3db99` | Rollout attribution v2 | Showed remaining attention time was prepare/layout, not fused SDPA math. |
| `f3af58c` | Decode attention prepare/layout fusion | Dropped real-checkpoint step to `0.164387 s`; targeted prep cluster `-55.80%`. |
| `df1e09f` | SwiGLU `silu_multiply` KILL | Reverted before code commit: step `0.164387 s -> 0.164397 s`, no measurable win. |

## Phase Attribution Arc

The bottleneck moved every time a real wall-clock constraint was removed.

| Stage | Step seconds | Dominant finding |
| --- | ---: | --- |
| Real-checkpoint profile | `10.411479` | `rollout_student_forward=6.521386 s`, with host mirrors causing massive allocator/page pressure. |
| Host-mirror fix | `1.304788` | `optimizer_step=1.036148 s`, `79.4%` of step. |
| AdamW in-place/device moments | `0.294321` | `rollout_student_forward=0.125247 s`, `43%` of step. |
| KV cache | `0.253540` | Rollout remained top; repeated prefix encoding was gone, launch count became visible. |
| Fused grad clip | `0.231698` | Rollout decode launches became the clear next phase. |
| Device RoPE | `0.208804` | Rollout still dominated; host/device position plumbing was removed. |
| Device argmax | `0.195945` | Per-token logits readback collapsed to one final tiny token readback. |
| Fused decode SDPA | `0.177385` | SDPA math moved; surrounding attention prepare/layout dominated. |
| Decode prepare/layout fusion | `0.164387` | Remaining rollout MLP and residual paths are small; micro-fusion is now diminishing return. |

The useful framing was step-level wall-clock first, then phase tables, then
sub-phase counters. Narrow-window wins were treated as hypotheses until the
whole OPD step moved under matched controls.

## Killed Axes

| Axis | Evidence | Root-cause one-liner |
| --- | --- | --- |
| `forward_last_logits` / last-logits-only local path | Not retained as a standalone commit; it did not move enough wall-clock versus the full rollout path. | Reducing a small logits slice path does not help if the rollout layer stack and sync points still dominate. |
| `merge_grad` shared-first | [`2026-05-20-opd-merge-grad-shared-first-revert.md`](../experience/errors/2026-05-20-opd-merge-grad-shared-first-revert.md) | A local `merge_grad` counter improved, but full-step wall-clock regressed `+2.6%`; wall-clock won. |
| SDPA Option B `{scale + mask + softmax}` | [`2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse-kill.md) | Middle-stack fusion was only about `0.08%` of real step and regressed moderate shape. |
| CUDA Graph rollout | [`2026-05-21-arle-cuda-opd-rollout-graph-capture-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-rollout-graph-capture-kill.md) | Capture succeeds is not enough; replay recorded transient host memcpy pointers and broke greedy tokens. |
| Post-step cleanup small fix | [`2026-05-21-arle-cuda-opd-post-step-cleanup-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-post-step-cleanup-kill.md) | Cleanup is a device allocator/free-count problem, not a host-resident sweep. |
| MLP `silu_multiply` fusion | [`2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md) | The local MLP timer improved, but the whole step changed `+0.006%`; sub-10ms fusion was below the wall-clock noise floor. |

## Open Axes

1. Allocator pooling / CUDA ephemeral tensor arena.
   The post-step cleanup probe found `21,716` device tensor frees in one
   Qwen3-0.6B OPD step. A real arena axis should be licensed only by a
   multi-step wall-clock run, not by moving time out of the cleanup label.

2. Decode prefill / residual decode-block specialization.
   After decode prepare/layout fusion, the next plausible cluster is
   `append_kv + merge + o_proj` and residual/norm plumbing. Pre-license from
   the latest wins entry: Qwen3-0.6B mean step `<= 0.150 s`, rollout
   `<= 0.042 s`, tokens unchanged, loss relerr `<= 1e-4`; kill if mean step
   remains `> 0.160 s`.

3. Longer-rollout and broader-prompt eval.
   `lr=1e-7` converged on the 8-train / 4-held-out token-id setup, but prompt
   4 remained fragile. The next eval axis should increase prompt diversity and
   track per-prompt regressions, not only mean overlap.

## Evidence Index

- Baselines:
  - [`2026-05-20-pytorch-cuda-opd-baseline.md`](../experience/wins/2026-05-20-pytorch-cuda-opd-baseline.md)
  - [`2026-05-20-arle-cuda-opd-moderate-first-run.md`](../experience/wins/2026-05-20-arle-cuda-opd-moderate-first-run.md)
- Correctness and eval:
  - [`2026-05-21-arle-cuda-opd-convergence-correctness.md`](../experience/wins/2026-05-21-arle-cuda-opd-convergence-correctness.md)
  - [`2026-05-21-arle-cuda-opd-realckpt-convergence.md`](../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence.md)
  - [`2026-05-21-arle-cuda-opd-convergence-h1-vs-h2.md`](../research/2026-05-21-arle-cuda-opd-convergence-h1-vs-h2.md)
  - [`2026-05-21-arle-cuda-opd-realckpt-convergence-lr1e-7.md`](../experience/wins/2026-05-21-arle-cuda-opd-realckpt-convergence-lr1e-7.md)
- Perf wins:
  - [`2026-05-21-arle-cuda-host-mirror-fix-qwen3-06b.md`](../experience/wins/2026-05-21-arle-cuda-host-mirror-fix-qwen3-06b.md)
  - [`2026-05-21-arle-cuda-fused-adamw-qwen3-06b.md`](../experience/wins/2026-05-21-arle-cuda-fused-adamw-qwen3-06b.md)
  - [`2026-05-21-arle-cuda-opd-rollout-kvcache.md`](../experience/wins/2026-05-21-arle-cuda-opd-rollout-kvcache.md)
  - [`2026-05-21-arle-cuda-opd-fused-grad-clip.md`](../experience/wins/2026-05-21-arle-cuda-opd-fused-grad-clip.md)
  - [`2026-05-21-arle-cuda-opd-device-rope.md`](../experience/wins/2026-05-21-arle-cuda-opd-device-rope.md)
  - [`2026-05-21-arle-cuda-opd-rollout-device-argmax.md`](../experience/wins/2026-05-21-arle-cuda-opd-rollout-device-argmax.md)
  - [`2026-05-21-arle-cuda-opd-rollout-fused-decode-sdpa.md`](../experience/wins/2026-05-21-arle-cuda-opd-rollout-fused-decode-sdpa.md)
  - [`2026-05-21-arle-cuda-opd-decode-prepare-fusion.md`](../experience/wins/2026-05-21-arle-cuda-opd-decode-prepare-fusion.md)
- Research / attribution:
  - [`2026-05-21-arle-cuda-opd-realckpt-profile-attribution.md`](../research/2026-05-21-arle-cuda-opd-realckpt-profile-attribution.md)
  - [`2026-05-21-arle-cuda-opd-rollout-inner-attribution.md`](../research/2026-05-21-arle-cuda-opd-rollout-inner-attribution.md)
  - [`2026-05-21-arle-cuda-opd-rollout-inner-attribution-v2.md`](../research/2026-05-21-arle-cuda-opd-rollout-inner-attribution-v2.md)
- Kills / lessons:
  - [`2026-05-20-opd-merge-grad-shared-first-revert.md`](../experience/errors/2026-05-20-opd-merge-grad-shared-first-revert.md)
  - [`2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-sdpa-mask-softmax-fuse-kill.md)
  - [`2026-05-21-arle-cuda-opd-rollout-graph-capture-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-rollout-graph-capture-kill.md)
  - [`2026-05-21-arle-cuda-opd-post-step-cleanup-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-post-step-cleanup-kill.md)
  - [`2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`](../experience/errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md)
  - [`2026-05-20-opd-cpu-perf-cycle-wrap.md`](2026-05-20-opd-cpu-perf-cycle-wrap.md)

## SOLID Check

This wrap uses wall-clock step measurements as ground truth. Phase and
sub-phase counters are used only to choose the next axis. The remaining open
items are explicitly deferred because the last local micro-fusion did not move
whole-step wall-clock under n=3 matched controls.
