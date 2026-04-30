# SGLang longctx smoke harness fixed

## Context

Phase 1 S4 needs a pinned SGLang baseline runner for the longctx-32k gate.
The first local smoke reached the cloned SGLang checkout but failed before
launching the server when the caller environment did not define `PYTHONPATH`.

## What Worked

- Hardened `scripts/bench_sglang_longctx.sh` so the SGLang checkout is prepended
  with `${PYTHONPATH:-}` instead of requiring the variable to exist under
  `set -u`.
- Installed the pinned SGLang Python package from
  `/tmp/sglang-arle-214c35b03184c354acf1f86f99746799e1c9b3a9/python` after the
  next server attempt exposed the missing `pybase64` dependency.
- Verified the server path with:

```bash
scripts/bench_sglang_longctx.sh longctx-s4-sglang-smoke60 --smoke-seconds 60
```

Smoke result on the local L4:

| shape | out tok/s | total tok/s | req/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512/128/c1/60s | 29.26 | 146.53 | 0.22 | 140.00 ms | 186.43 ms | 33.43 ms | 33.48 ms |

Artefacts:

- Raw: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-s4-sglang-smoke60`
- Primary: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-s4-sglang-smoke60/guidellm-primary/benchmarks.json`
- Server log: `/content/workspace/agent-infer/bench-output/2026-04-30-sglang-longctx-s4-sglang-smoke60/sglang_server.log`

## Rule

Baseline wrappers must be safe under `set -u` with an empty caller
environment. The 5-second SGLang smoke path is too short on a fresh process
because startup-time graph work can consume the window; use `--smoke-seconds 60`
for local SGLang validation before the canonical 32k run.
