# OPD corrected arm "crash" was SIGTERM plus host-side rollout cost, not a CUDA fault

## Context

The corrected OPD capability arm was configured to test the real transfer
recipe: Qwen3.5-4B teacher to Qwen3.5-0.8B student, GSM8K question-only
prompts, `--rollout-len 256`, temperature sampling, completion-only KL, and
all-linear LoRA rank 32. The first launch appeared to freeze after the model
load lines, with no step output, and an older dmesg sample showed an Xid 31 MMU
fault attributed to `arle`.

That was not enough evidence to call the corrected arm a CUDA crash. The
original launch only captured stdout, so stderr was missing, and the run was
sharing a pod where an unrelated `srv62` DeepSeek-V4 TP=8 serve repeatedly
occupied all eight GPUs.

## Root Cause

The reproducible failure observed during the RCA was process SIGTERM, not a
CUDA illegal-memory crash.

- The corrected `launch.sh` was fixed to route stderr into `driver.log`
  (`exec > .../driver.log 2>&1`).
- Clean detached one-step probes with merged stdout/stderr showed no fresh Xid,
  no CUDA stderr, and no compute-sanitizer-worthy faulting kernel.
- `--rollout-len 128` completed successfully: `rc=0`, `loss 0.000008`,
  reported total `rollout_len 200`, and saved a full materialized checkpoint.
- `--rollout-len 256` was repeatedly terminated with `rc=143` and launcher
  output `Terminated`. One run survived about 25 minutes before SIGTERM; a
  second was terminated after about 2.5 minutes. In both cases dmesg had no
  fresh Xid.
- The external `srv62` DSv4 TP=8 serve relaunched during the 256 probes and
  claimed all eight GPUs. That made the long probe environment non-exclusive
  and invalidated the "crash" read.

Candidate #1, a legacy cached-decode SDPA dynamic shared-memory overflow, was
falsified for this corrected run. Both teacher and student configs report
`head_dim=256`, and the probe environment did not set
`ARLE_AUTOGRAD_DECODE_ATTN_LEGACY`, so the Qwen rollout path selects the fixed
shared-memory `causal_sdpa_decode_gqa_cache_online_f32_hd256` kernel rather
than the legacy visible-length-scaled kernel.

The remaining implementation issue is performance: sampled 256-token rollout
is extremely slow on the current path. While the 256 probe was alive, the main
process was one CPU thread near 100%, CUDA helper threads were idle in `poll`,
and GPU memory stayed low until late in the run. Code inspection explains the
shape:

- `student_rollout_only_with_keep` uses the device-argmax fast path only when
  `sampling.is_none()`.
- Temperature sampling falls into the host-token loop and calls
  `greedy_next_token` for every generated token.
- `greedy_next_token` calls `store.to_host(logits_id)`, which performs CUDA
  readback and stream synchronize, then scans/samples the vocab row on the CPU.
- For Qwen3.5-0.8B that means up to 256 per-token sync/readback/sample loops
  over a 248,320-token logits row, plus host allocation/copy overhead.

This is a serious rollout-performance bug for long sampled OPD/SOPD, but it is
not evidence that OPD itself is broken.

## Fix

Immediate environment fixes:

- Keep stderr merged into OPD run logs.
- Do not judge the corrected arm from a shared pod where `srv62` can relaunch a
  TP=8 serve and preempt every GPU.
- Wait for an explicit exclusive GPU window before running the corrected
  `rollout_len=256` arm.

Code-side next step, when a GPU window is available:

- Profile the sampled rollout loop at `rollout_len=256` with per-token timing
  split into student forward, logits readback/sync, host sampling, KV-cache
  write/retain, teacher forward, loss, backward, and checkpoint save.
- Replace or bypass the sampled host readback loop with a device-side sampling
  path or batched/async sampling path before running long corrected-arm
  capability curves.

## Rule

Do not call a long GPU train run a model/kernel crash from a frozen coarse log.
First prove the process exit mode, stderr, fresh dmesg, GPU ownership, and
which kernel path is actually selected. `rc=143` plus an all-GPU external serve
is an environment/setup failure; it is not a verdict on OPD or on a candidate
CUDA root cause.
