# /v1/models connect-refused under sustained load — NOT REPRODUCIBLE at HEAD

## Context

Prior session observed /tmp/repro2.sh failing: 27 parallel fresh-process
curls to `/v1/models` after 15 s of 1000 concurrent streaming chat
workers saw connect-refused at TCP level. Queued as task #34 ("fix the
server, don't work around it").

Today (2026-04-20, commit 38c6c31) re-ran the same workload against
metal_serve on `mlx-community/Qwen3.5-4B-MLX-4bit`:

- Stage 1: 1000 chat completions streaming, spaced 0.03 s apart.
- Stage 2 (15 s into Stage 1): 27 parallel fresh-process curls to
  `/v1/models` with `--connect-timeout 5 --max-time 5`.

## What Worked

**All 27 probes succeeded.** Max wall-clock 213 ms, all CONNECT < 1 ms,
HTTPSTATUS 200 for every worker. Server log shows no ERROR beyond the
expected "stream consumer dropped" when I `pkill`-cleaned up the load
generator at the end.

`submit_request` path is lock-free (atomic cmpxchg + unbounded mpsc
send), the `/v1/models` handler is a trivial `Json<...>` return,
tokio worker pool (default `#[tokio::main]`) on M4 Max saturates
neither the accept loop nor the handler. `kern.ipc.somaxconn=128`
was sufficient because axum's accept loop drains the queue fast
enough under this workload.

## Hypothesis for prior failure

Either:
1. **Environment-dependent** — the prior box was short on memory/fd/etc.
   and the OS level was refusing SYNs. No regression at our layer.
2. **Fixed by intervening work** — candidates: e22eddc (batch sampling
   collapsed many per-request kernel launches into one), de7b687
   (de-duplicated eval in StepDriver). Both reduce wall time spent
   inside worker tasks, freeing the tokio runtime to run the accept
   loop.

Either way, task #34 closes as **not reproducible at HEAD**. Re-open
only if the bug resurfaces with a reproducer against the current
commit.

## Rule

**Re-run the reproducer against HEAD before committing to a fix.**
Bugs logged across sessions drift — a tight loop can close them before
you spend time patching.

Bench: exempt (investigation-only, no code change).
