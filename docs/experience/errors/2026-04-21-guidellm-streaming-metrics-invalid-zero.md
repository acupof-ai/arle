# GuideLLM Streaming Metrics Invalid Zeroes

## Context

`guidellm` produced a seemingly "successful" long-context `c16` run where
`output_tokens.mean = 256`, but every sampled response body was empty and the
headline table rendered `TTFT/ITL = 0` with impossible throughput.

The bad run was not a GuideLLM crash. It wrote `json/csv/html`, and the wrapper
accepted those artefacts as if the benchmark were valid.

## Root Cause

Two issues stacked:

1. The server can currently finish a request with non-zero usage while emitting
   no visible text chunks. Under GuideLLM's streaming accounting, that means the
   request still looks "successful" on usage, but `first_token_iteration` /
   `last_token_iteration` stay unset because no non-empty text delta was ever
   observed.
2. `scripts/bench_guidellm.sh` had no validity gate after GuideLLM finished.
   It also rendered missing timing fields as literal `0`, which turned a bad run
   into a fake headline table.

## Fix

- Added a pre-benchmark `/v1/completions` streaming probe in
  [scripts/bench_guidellm.sh](/content/workspace/agent-infer/scripts/bench_guidellm.sh:205)
  that requires both:
  - at least one non-empty streamed text chunk
  - a terminal usage chunk
- Added post-run JSON validation in the same script. The wrapper now hard-fails
  if a benchmark reports successful non-zero output tokens but sampled outputs
  are empty or `TTFT/ITL` stay zero.
- Added a scheduler warning in
  [infer/src/scheduler/cuda/request.rs](/content/workspace/agent-infer/infer/src/scheduler/cuda/request.rs:167)
  when a request finishes with generated tokens but no visible decoded text.

## Rule

Bench wrappers must validate semantic correctness, not just process exit codes.
If streaming text is missing, the run is invalid even when usage counters and
output files look complete.
