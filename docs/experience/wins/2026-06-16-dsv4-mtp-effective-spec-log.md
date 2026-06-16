# DSv4 MTP head-load log uses effective spec decision

## Context

Serving with `--spec-type mtp` / `--mtp-draft-tokens` already sets the effective
DSv4 spec-decode decision, but the MTP head-load preflight log still checked only
`ARLE_DSV4_SPEC_DECODE`. That could print "MTP draft head deferred" while the
head actually loaded, making the env look required.

## What Worked

`ensure_loadable` now logs from the same `spec_decode_on` boolean that controls
the MTP head load. The env remains a backward-compatible fallback, but a CLI MTP
request is enough for the log to say the draft head will load.

## Verification

- `rg` confirms `ensure_loadable(&config, spec_decode_on)` passes the effective
  decision from `from_dsv4_fp8_safetensors_with_tp`.
- `rg` confirms the old env-only `ARLE_DSV4_SPEC_DECODE=1` log string is gone.
- Local no-nvcc typecheck: `CUDARC_CUDA_VERSION=12080 cargo check -p infer-api
  --release --no-default-features --features cuda,no-cuda --lib`.

## Benchmark

Pending remote. This is a log-only loader fix and should not change decode or
load wall-clock. If a CUDA serve bench is needed for release bookkeeping, run it
on the gcc>=10 MTP lane; do not use the clang-11 host for MTP.

## Rule

Logs for gated runtime paths must report the effective decision that the code
uses, not one legacy env fallback. Otherwise operator diagnosis chases a flag
that is no longer load-bearing.
