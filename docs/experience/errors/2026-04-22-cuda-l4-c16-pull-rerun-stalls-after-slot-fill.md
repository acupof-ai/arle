# CUDA L4 c16 pull rerun stalls after slot fill

## Context

- Pulled `origin/main` to `f15f87c` (`feat(scheduler): overlap decode readback and fetch waits`).
- Rebuilt `infer` in release mode and reran the same serial `guidellm` concurrency sweep used for the earlier `infer` vs `sglang` comparison:

```bash
GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh \
  infer-qwen3-4b-l4-c1-c16-serial-f15f87c \
  --target http://127.0.0.1:8000 \
  --model Qwen/Qwen3-4B \
  --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --concurrencies 1,2,4,8,16 \
  --max-seconds 60 \
  --warmup 5
```

- The run advanced far enough to admit a full `c16` working set, then stopped making forward progress.
- Artefacts are kept under `bench-output/2026-04-22-infer-qwen3-4b-l4-c1-c16-serial-f15f87c/`; the useful evidence is `service_stats_trace.jsonl` plus the partial `guidellm.log` setup transcript.
- Last scheduler-side admission burst:
  - `Request 123 → slot 0`
  - `...`
  - `Request 135 → slot 15`
  - final scheduler line: `Scheduler step: assign=237964us step=13us cleanup=1us total=237979us active=16`
- After that point, `/v1/stats` stayed frozen for at least `30s+` with:
  - `active=16`
  - `waiting=3`
  - `running_batch=0`
  - `prefill_queue=0`
  - `kv_fetch_q=0/16`
  - `kv_fetch_waiters=0`
  - `tokens_out=13069` unchanged

## Root Cause

- **Suspected, not yet proven:** the overlap change leaves the CUDA scheduler with live requests but no runnable batch after slot assignment at high concurrency.
- Evidence points away from KV fetch backpressure:
  - `kv_fetch_q=0/16`
  - `kv_fetch_waiters=0`
  - `kv_bp=fetch:0,store:0`
- Evidence points toward a scheduler wakeup / handoff bug between:
  - pending decode readback carried across loop turns
  - `runtime.rs::run()` sleep path when work appears parked
  - the next `execution.rs::step()` batch planning / launch decision
- The critical symptom is not “slow”; it is **idle with active work present**: `active=16` and `waiting=3`, but `running_batch=0` and no completions or token growth.

## Fix

- Do not trust the pulled scheduler change as performance-positive until it survives a targeted `c16` progress smoke.
- Use the new end-to-end trace plan in `docs/plans/2026-04-22-cuda-end-to-end-trace.md` to instrument:
  - `scheduler_iter`
  - admission
  - launch/readback
  - `WaitingFetch` enter/exit
  - finish/incomplete reasons
- Add a live-progress gate for high-concurrency reruns:
  - if `active>0` and `running_batch=0` persists for `N` consecutive `/v1/stats` polls while `tokens_out` is flat, fail the run as a scheduler stall.
- Reproduce with a narrower diagnosis run first (`c16` only, same model/flags), then profile the exact iteration where progress stops.

## Rule

- Scheduler loop reorderings need a **progress** gate, not just compile/test green and a pending remote bench stub.
- For CUDA serving regressions, record the stuck `/v1/stats` window and the last scheduler log line immediately; that evidence is more valuable than letting `guidellm` hang indefinitely.
