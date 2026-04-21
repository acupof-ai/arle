# CUDA greedy waiting prefill batch bench backlog

## Context

Local regression-check run for the CUDA scheduler refactor that:

- deletes the `runtime.rs::assign_slots()` pre-step waiting admission path
- keeps one planned path per tick
- lets `PrefillOnly` plan and execute a greedy batch of waiting prefills under one `PrefillBudget`

Environment and command:

- Date: 2026-04-21
- Host: NVIDIA L4, driver 580.82.07
- Commit: `45dc8ad`
- Build: `CUDA_HOME=/usr/local/cuda ZIG=/tmp/zig-tool/zig-x86_64-linux-0.15.2/zig cargo build -p infer --release --bin infer`
- Server:
  `target/release/infer --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --port 8017 --num-slots 16 --max-seq-len 4608 --mem-fraction-static 0.94 --chunked-prefill-size 4096 --max-prefill-tokens 16384 --enable-mixed-chunk true`
- Bench:
  `PATH=/root/.local/bin:$PATH ./scripts/bench_guidellm.sh cuda-l4-c16-greedy-waiting-prefill-batch --target http://127.0.0.1:8017 --model Qwen/Qwen3-4B --processor /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`

The release build succeeded and the server came up cleanly, but the canonical
GuideLLM sweep never finished cleanly in-session. After more than 15 minutes,
`bench-output/2026-04-21-cuda-l4-c16-greedy-waiting-prefill-batch/` still only
contained `command.txt` and `guidellm.log`; no `benchmarks.json`, `.csv`, or
`.html` were emitted.

The scheduler trace did confirm the structural change:

- new requests stayed in `waiting` until `step()` admitted them
- a no-decode `PrefillOnly` tick admitted and advanced multiple waiting
  prefills in one planned path

Representative trace from the local server log:

- `Request 9 → slot 1 (prompt=4097 tokens, queue=59)`
- `Request 10 → slot 2 (prompt=4097 tokens, queue=58)`
- `Request 11 → slot 3 (prompt=4097 tokens, queue=57)`
- same tick summary:
  `step breakdown: plan=prefill admission=60829us decode=0us emit=0us prefill=2073152us total=2133981us batch=1`
- shortly after:
  `Mixed batch: fallback (Ok(false))`
- backlog signal later in the same run:
  `Request 15 done: 256 tokens (active=0, waiting=511)`

## Root Cause

The scheduler refactor did what it was supposed to do structurally, but the
canonical c16 long-context workload still did not converge to a usable steady
state:

1. The new `PrefillOnly` batching path can now spend >2s in one serial
   prefill-heavy tick.
2. The mixed path still falls back (`Ok(false)`), so the workload does not
   reliably convert those admissions into decode overlap.
3. Under the throughput leg, waiting backlog grows into the hundreds and the
   benchmark never reaches a clean point where GuideLLM flushes final result
   artefacts in a reasonable time window.

This is not evidence that the refactor is wrong. It is evidence that the
refactor moved the bottleneck into "batched waiting admission without enough
mixed/decode overlap" for the c16 canonical profile.

## Fix

- Keep the admission refactor: it removed the bypass and the trace shows the
  intended greedy waiting batch behavior is live.
- Treat the next iteration as a diagnosis task, not a regression-check-only
  task:
  - instrument the mixed fallback reason instead of only logging `Ok(false)`
  - measure how often no-decode batched prefills produce >1s / >2s ticks
  - rerun the c16 sweep only after that overlap story is clearer
- Do not claim a clean GuideLLM delta from this run; there are no final
  artefacts to compare.

## Rule

When a CUDA scheduler change intentionally increases per-tick prefill admission
width, a local `scripts/bench_guidellm.sh` c16 sweep is allowed to terminate as
an `errors/` diagnosis entry if both of these hold:

1. The server trace proves the new planner behavior is active.
2. The canonical run still has not emitted `benchmarks.json` after an extended
   backlog window, so there is no trustworthy headline table to publish.

In that case, record the attempted command, the trace evidence, and the queue
growth signal, then follow with a narrower diagnosis pass instead of waiting
indefinitely for GuideLLM to drain.
