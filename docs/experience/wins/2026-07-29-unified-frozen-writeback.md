# Frozen writeback uses the shared training path

## Context

Frozen-prompt CE kept a second copy of validation, forward, loss, backward,
optimizer, cleanup, and telemetry after the shared path gained frozen-prefix
support.

## What Worked

Keep the public function as a compatibility wrapper and delete the duplicate
implementation. The CUDA boundary stays streaming; CPU parity reuses the host
reference.

## Rule

One training step owns validation, backward, optimizer, and cleanup.
