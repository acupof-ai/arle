# The blocks-per-SM search woke two latent Marlin bugs — CUDA, 2026-08-19

> Status: Root-caused, fixed, 32K gate re-running

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

## Rule

A change that makes a previously-constant value variable must be gated on the
code paths that read it, not on the path it was written for. `blocks_per_sm` was
1 everywhere in the tree; two readers had silently been written against that
constant. Grep every reader of the value before licensing the change, and gate
on a workload that reaches each one — here, one 32K prefill.

Corollary: a decode-only gate cannot license a kernel-selection change. Prefill
splits M across chunks and decode does not, so the two exercise different
control flow in the same function.
