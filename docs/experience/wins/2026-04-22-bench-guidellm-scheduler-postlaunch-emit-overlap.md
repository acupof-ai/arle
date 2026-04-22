# Scheduler Post-Launch Emit Overlap

## Context

CUDA scheduler overlap cleanup continued on 2026-04-22. Decode readback was
already split from decode launch, but detokenize/streaming `emit_delta` still
ran entirely before the next decode launch.

## What Worked

- Scheduler-side decode emission is now split by control-flow impact.
- Requests with non-empty textual stop sequences still emit before launch so
  stop detection can gate the next decode row.
- Requests without textual stop sequences now emit after `step_decode_launch()`,
  letting their detokenize/streaming work overlap the current step's GPU decode.

## Rule

Status: `pending-remote`

- This is a scheduler-only runtime change. Any performance claim still needs a
  CUDA GuideLLM before/after snapshot.
