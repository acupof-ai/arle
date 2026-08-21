# `--spec-type auto` skipped the MTP head on the multiproc path — CUDA, 2026-08-21

> Status: Fixed in the same-day cleanup; shipped latent in 0.5.8, caught by a
> code-quality review before any multiproc DSv4 run.

## Context

0.5.8 flipped the serve default to `--spec-type auto`, which resolves to MTP
speculation when the checkpoint declares an MTP head. The measured win
(c=1, 20.50 → 11.94 ms per committed token, +21.6% end-to-end) was on the
single-process path.

## Root Cause

`auto` was resolved only inside `serve_http`. The multiproc coordinator
serializes `engine_config` into `ARLE_WORKER_ENGINE_CONFIG` and never runs
`serve_http`'s lowering, so worker ranks inherited `mtp_draft_tokens=None`
and loaded without the MTP head — plain decode, silently, on every multi-GPU
DSv4 serve, the family that ships the head and is multiproc by design.

## Fix

Resolve `auto` in `cli::resolve_config` before the engine-config lowering,
using the same `checkpoint_has_mtp_head` probe; the serialized config then
carries the draft depth. `serve_http`'s arm stays for direct API callers. A
sibling hole was closed in the same pass: explicit `--mtp-draft-tokens` /
`--mtp-draft-topk` is now a loud error when `auto` resolves to no head,
instead of being silently dropped.

## Rule

A default flip must be exercised on every serve path it travels, not just the
measured one — the multiproc coordinator is a second, lower-resolution path
that reuses only the engine config.
