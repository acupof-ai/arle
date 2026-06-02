# DSv4 Internal MTP Draft Mode Contract

## Context

The DSv4-Flash target is the warm single-node TP8 + EAGLE shape:
256K/1500, TTFT about 0.44 s, TPOT about 4.85 ms, E2E about 7.7 s,
and output throughput about 196 tok/s. ARLE previously only exposed
`self-spec` and `external:<path>` draft modes. That mixed two different
semantics:

- `self-spec` is MagicDec-style sparse self speculation.
- DSv4 EAGLE is an internal checkpoint `mtp.N` draft head over frozen target KV.

Using `self-spec` for DSv4 EAGLE would hide the real missing path and could
silently run the wrong runtime structure.

## What Worked

- Added `DraftMode::InternalMtp` with CLI aliases `internal-mtp`, `mtp`,
  `eagle`, and `internal-eagle`.
- Added a model trait hook for batched internal MTP draft proposals:
  `forward_internal_mtp_draft_batch`.
- Added startup validation for `--spec-draft-model internal-mtp/eagle` so
  unsupported models fail before serving opens instead of falling back silently.
- Added a CUDA scheduler branch that routes internal MTP draft proposals through
  the existing target verifier and commit logic.
- Added DSv4-specific fail-closed messages that distinguish "MTP weights not
  loaded" from "frozen-KV MTP draft forward/graph capture not implemented yet".

This is not a performance win yet. It is the correct control-plane and
scheduler contract needed before implementing real DSv4 frozen-KV MTP draft
math.

## Verification

- `cargo test -p infer internal_mtp --no-default-features --features no-cuda`
  - passed, including `internal_mtp_allows_multi_token_without_sparse_kv`
    and `request_spec_internal_mtp_honors_aliases_and_opt_outs`.
- `cargo check -p infer --no-default-features --features no-cuda`
  - passed.
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda`
  - passed with pre-existing DSv4 warnings.

Remote DSv4 fast-build and startup contract verification are pending for the
next tranche.

## Rule

Do not overload `self-spec` for DSv4 EAGLE. Internal checkpoint MTP draft,
external draft models, and sparse self speculation are separate runtime
structures and need separate startup contracts.
