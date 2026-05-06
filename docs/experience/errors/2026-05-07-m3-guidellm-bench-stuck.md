# M3 GuideLLM Bench Stuck

## Context

Track B M3 S1-S6 landed shared scheduler IR and default-on CUDA happy-path logical lowering. S7 verification attempted canonical `scripts/bench_guidellm.sh` on the local RTX 4070 Ti SUPER.

Server launch used for the meaningful runs:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo run --release --manifest-path infer/Cargo.toml \
  --no-default-features --features cuda,unified_scheduler \
  -- \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 4 \
  --max-seq-len 4096
```

## Symptoms

Canonical run:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
scripts/bench_guidellm.sh cuda-m3-unified \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B
```

Problems encountered:

- `guidellm` was not installed; fixed with `/home/ckl/projects/arle/.venv/bin/pip install -e '.[bench]'`.
- `httpx` failed under the current SOCKS proxy env; fixed with `/home/ckl/projects/arle/.venv/bin/pip install socksio`.
- Canonical 4096-token synthetic prompts were tokenized/admitted as 4097 tokens and rejected:
  `Rejecting prompt with 4097 tokens: scheduler max_input=4090 max_request=4095`.

Fallback quick run:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
GUIDELLM__MP_CONTEXT_TYPE=spawn \
scripts/bench_guidellm.sh cuda-m3-unified \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --quick
```

The quick run reached the server and generated tokens, but stalled at the c=8 stage. Final service snapshot before terminating the run:

```text
requests=198 active=4 waiting=8 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=4 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=24983 ... engine_queue_depth=8 engine_active_requests=4 engine_batch_occupancy=0.6183
```

`Ctrl-C` could not shut down the server cleanly; it blocked while waiting for the scheduler thread and required `kill -KILL`.

Artifacts:

- Canonical preflight/proxy failure: `bench-output/2026-05-07-cuda-m3-unified/`
- Canonical rejected-prompt/stall attempt: `bench-output/2026-05-07-cuda-m3-unified-run2/`
- Quick c=8 stuck-active attempt: `bench-output/2026-05-07-cuda-m3-unified-run3/`
- Quick service summary: `bench-output/2026-05-07-cuda-m3-unified-run3/service_stats_trace_summary.md`

## Current Read

This blocks M3 performance acceptance. It is not enough to publish tok/s or claim a no-regression result.

Two separable issues were exposed:

- Canonical 4096-token guidellm workload is one token over the server's effective request limit for `--max-seq-len 4096`.
- Under the fallback fixed-concurrency quick workload, the scheduler/runtime can stop making progress with full active slots and queued work.

The Qwen3.5 e2e baseline mismatch also reproduced with `--no-default-features --features cuda`, so that specific failure is not caused by the default-on `unified_scheduler` path.

## Red-Light Recheck

On 2026-05-07, the same quick GuideLLM command was rerun at baseline commit
`788b15a` (M3 design docs only; unified scheduler IR was not connected):

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=/home/ckl/projects/arle/.venv/bin/python \
cargo run --release --manifest-path infer/Cargo.toml \
  --no-default-features --features cuda \
  -- \
  --model-path infer/models/Qwen3-4B \
  --port 8000 \
  --num-slots 4 \
  --max-seq-len 4096

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
GUIDELLM__MP_CONTEXT_TYPE=spawn \
scripts/bench_guidellm.sh cuda-m3-baseline-788b15a \
  --target http://localhost:8000 \
  --model Qwen3-4B \
  --processor infer/models/Qwen3-4B \
  --quick
```

The run made normal progress at lower concurrency, then reproduced the same
c=8 stuck-active shape:

```text
requests=196 active=4 waiting=8 scheduled=0 decode_rows=0 prefill_rows=0 running_batch=4 prefill_queue=0 batch_width=0 decode_tokens=0 prefill_tokens=0 tokens_out=24967 ... engine_queue_depth=8 engine_active_requests=4 engine_batch_occupancy=0.6183
```

This excludes M3 logical IR lowering as the cause of the GuideLLM c=8 stuck
bench. The remaining scheduler/server issue is pre-existing: four active slots
stay resident while no runnable decode or prefill rows are produced, and queued
requests cannot be admitted.

Artifacts:

- Baseline quick reproducer: `bench-output/2026-05-07-cuda-m3-baseline-788b15a/`
- Baseline quick service summary: `bench-output/2026-05-07-cuda-m3-baseline-788b15a/service_stats_trace_summary.md`

The independent Codex review found one M3-specific recoverability regression:
CUDA logical lowering skipped a malformed decoding slot with no generated
tokens, where the legacy decode path logs and calls `finish_slot()`. That issue
is separate from the GuideLLM baseline reproducer and was fixed by falling back
to legacy launch whenever a decode-bearing `StepPlan` lowers to zero logical
decode rows.

## Next Checks

- Add a targeted scheduler trace around `running_batch`, `pending_decode`, closed `delta_tx`, and request phases when `active>0` but `slot_is_runnable_decode` produces no rows.
- Decide whether canonical guidellm should launch the server with additional headroom or lower the locked prompt token target to the server's true `max_input`.

## Rule

Do not count a benchmark run unless `benchmarks.json` is produced and `/v1/stats` drains to `active=0 waiting=0` after the client finishes. A stuck-active service trace is a regression artifact, not a perf number.
