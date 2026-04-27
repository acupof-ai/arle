# DFlash should be evaluated on ultra-long sequence workloads

## Context

After a short Qwen3.6 Metal DFlash quick check, I summarized the result as if
short prompt/decode TPOT was useful for deciding DFlash direction.

The user corrected the scope: DFlash work should only be judged as an
ultra-long-sequence optimization going forward.

## Root Cause

I treated DFlash like a generic decode-throughput knob. That is too broad for
this project. Short 32-token prompt checks are useful only as load/smoke
diagnostics; they do not represent the workload DFlash is supposed to improve.

## Fix

- Keep short DFlash checks labeled as diagnostics only.
- Do not use short prompt/decode runs to decide whether DFlash is a win.
- For DFlash optimization claims, require long-context or ultra-long-sequence
  workloads with explicit prompt length, generation length, TTFT, TPOT, and
  end-to-end numbers.

## Rule

For Metal DFlash, optimize and report performance against ultra-long sequence
workloads only. Short-sequence runs may prove that the path loads and executes,
but they are not acceptance evidence for DFlash performance.
