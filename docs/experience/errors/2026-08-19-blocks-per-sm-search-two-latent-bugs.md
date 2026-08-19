# The blocks-per-SM search woke two latent Marlin bugs — CUDA, 2026-08-19

> Status: Two real bugs fixed. Cause NOT established — the revert experiment was invalid.

## Context

`cfec5827e` let `determine_exec_config` search `blocks_per_sm` instead of
pinning it to 1. Measured +4.5% decode, needle 6/6, shipped. It also made
`blocks_per_sm > 1` reachable for the first time, and two pieces of surrounding
code had only ever been correct because it was always 1.

Neither showed up in the decode benches. Both need `prob_m` large enough that
`marlin_mm` splits it across `while (rest_m)` iterations — that is prefill, and
every gate run after the change was a decode measurement.

## Bug 1 — the shared-memory budget is lowered in place and never restored

`gptq_marlin.cuh:641` declares `max_shared_mem_new` OUTSIDE the chunk loop:

```c
int max_shared_mem_new = max_shared_mem;
int rest_m = prob_m;
while (rest_m) {
  ...
  if (exec_cfg.blocks_per_sm > 1) max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;
```

The assignment only ever lowers it. A chunk that picks `blocks_per_sm = 3`
leaves `max_shared_mem/3 - 1024` in place; the next chunk picks
`blocks_per_sm = 1`, the `if` does not fire, and the budget stays at the
previous chunk's value. `RuntimeCheck(is_valid_config(..., max_shared_mem_new))`
then rejects a configuration that is in fact valid, `marlin_mm` throws, and the
extern-C shim maps the C++ exception to `CUDA_ERROR_INVALID_VALUE`.

Observed as, on a 33K prefill:

```
infer-server engine step failed: NVFP4 Marlin GEMM failed:
  DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")
```

`cudaFuncSetAttribute` and the launch read the same stale value, so a run that
got past the check would have launched with the wrong dynamic shared memory.

Fix: recompute per chunk instead of lowering in place.

## Bug 2 — the lock buffer is sized for one block per SM

`marlin_workspace_ints` returned `sms * 4`, documented as "upstream sizes locks
at `sms * max_blocks_per_sm` (default 1); 4x is headroom". The search made the
grid `sms * blocks_per_sm` with `blocks_per_sm` up to 5, and the kernel indexes
`locks[blockIdx.x]` (`marlin_template.h:388`) and increments past it
(`:417`, `:420`). At `blocks_per_sm = 5` that is an out-of-bounds WRITE.

Fix: `sms * MARLIN_MAX_BLOCKS_PER_SM * 4` — scale with the search bound, keep the
same headroom. 78 SMs x 20 ints is 6 KB.

This is the likely source of the first 32K crash, which reported
`CUDA_ERROR_ILLEGAL_ADDRESS` and was attributed at the time to the FP8 dequant
arm. Two independent failures were stacked; fixing the dequant arm removed the
first and exposed the second.

## Why the gates missed it

The change was licensed on a decode measurement (`+4.5%`, needle 512/4096 x3).
Both bugs need `prob_m` past `max_thread_m_blocks * 16`, so decode can never
reach them, and the needle ladder at 512/4096 prefills below the chunk split
that triggers bug 1. The 32K long-agent run is the first workload that exercises
multi-chunk prefill, and it was not run against this change until three commits
later.

## RETRACTED: the revert experiment never engaged its treatment arm

Pinning `MARLIN_MAX_BLOCKS_PER_SM` to 1 completed 32/32 at c=1 and c=4, and I
read that as confirming the search was the cause. It confirms nothing. Counting
the condition that precedes both crashes — a PARTIAL prefix restore, where
`prefix-attach` reports `restored < matched` — across every run:

| run | blocks_per_sm | requests | partial restores | outcome |
|---|---:|---:|---:|---|
| v3 | 5 | 226 | **2** | crashed |
| v7/v8 | 1 (revert) | 207 | **0** | "passed" |
| v9 | 5 | 18+ | **0** | not yet crashed |

The revert arm never met the failing condition in 207 requests, so its clean run
is not evidence about `blocks_per_sm`. Re-running the search at 5 with the
diagnostics went 18 requests with no partial restore and no crash either.

The correlation the data actually supports is **crash iff partial prefix
restore**, not crash iff blocks_per_sm. Both crashes landed on a request whose
`prefix-attach` restored fewer tokens than it matched
(`matched=31856 restored=24576` of a 33059-token prompt), leaving ~8.5K to
prefill — four full 2048 chunks plus a tail that lands on Marlin. Requests with
a full restore prefill a few hundred tokens, never split across chunks, and
never fail.

Next step is to force the trigger rather than wait for it: find what makes a
restore partial and drive it directly. A bisect on `blocks_per_sm` is only
meaningful once each arm is shown to hit the condition.

## What the two bugs below still are

Both are real and both are fixed; neither is demonstrated to be the crash.

Both bugs above are real and both are fixed. Neither stopped the crash. The 33K
prefill still threw `CUDA_ERROR_INVALID_VALUE` out of `marlin_mm` with the
shared-memory recompute and the resized lock buffer in the binary (object
rebuilt 09:34, verified).

Pinning `MARLIN_MAX_BLOCKS_PER_SM` back to 1 — and restoring upstream's
first-valid-tile selection, since capping the bound alone still selects by waves
and is therefore not a revert — cleared it on the first try:

| | 32K, 32 req/point | c=1 | c=4 |
|---|---|---|---|
| search at 5 | completed | 25/32 then dead | 0/32 |
| pinned to 1 | completed | **32/32** | **32/32** |

Zero server errors, zero `[marlin]` diagnostics, server alive at the end. This
is the first complete 32K run this checkpoint has produced.

## The revert costs 6-9x what the change was licensed on

| c | pinned to 1 | search at 5 | delta |
|---:|---:|---:|---:|
| 1 | 47.3 | 66.4 | **-28.8%** |
| 4 | 96.4 | 163.5 | **-41.0%** |
| 16 | 138.1 | 236.1 | **-41.5%** |

`cfec5827e` was licensed at +4.5%. The true value is 29-42%, and the reason the
original number was so low is that it was measured when only `gate_up` was on
Marlin — `down_proj` and `o_proj` joined at `5499e20a7`, roughly doubling the
surface the search acts on.

**An optimisation's licensing number only holds for the surface it was measured
on.** When a later change widens that surface, the number understates both the
gain and the cost of reverting. Re-measure before trading it away; the 4.5% on
file nearly bought a permanent 40% loss.

So the revert is not even a bisect result — it is an unengaged arm. The 29-42%
is a real measurement of what pinning costs, and it stands; what does not stand
is any claim that pinning fixes anything. The `marlin_gemm.cu` exception message is now wired up
(`catch (const std::exception& e)` + `fprintf`) precisely to capture which
`RuntimeCheck` fires when it is re-enabled.

## First real 32K comparison (bps=1, no spec, both arms same box)

| c | NVFP4 decode | FP8 decode | delta |
|---:|---:|---:|---:|
| 1 | **43.7** | 37.5 | **+16.5%** |
| 4 | 20.2 | 21.1 | -4.3% |

The gap is much narrower than on an 8-token prompt because attention dominates
at 33K and is format-independent.

## Rule

A change that makes a previously-constant value variable must be gated on the
code paths that read it, not on the path it was written for. `blocks_per_sm` was
1 everywhere in the tree; two readers had silently been written against that
constant. Grep every reader of the value before licensing the change, and gate
on a workload that reaches each one — here, one 32K prefill.

Corollary: a decode-only gate cannot license a kernel-selection change. Prefill
splits M across chunks and decode does not, so the two exercise different
control flow in the same function.
