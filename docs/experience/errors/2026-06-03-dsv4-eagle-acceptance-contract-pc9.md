# DSv4 EAGLE Acceptance Contract PC9

## Context

Target workload remains DSv4-Flash TP8 + EAGLE, 256K/1500, hot GPU cache:
TTFT ~0.44 s, TPOT ~4.85 ms, E2E ~7.7 s, output throughput ~196 tok/s.

PC5-PC8 moved startup contract evidence from missing owner-group transport to
`owner-groups-collectives-ready`. The remaining target blockers include
full-decode CUDA graph replay, DeepEP/NCCL graph safety, EAGLE graph safety,
and attention metadata replay.

While inspecting PC9, the scheduler-side internal MTP/EAGLE path had another
target-workload confounder: even with `--spec-enabled --spec-draft-model
eagle`, accepted draft tokens were cleared by default unless
`ARLE_INTERNAL_MTP_ACCEPT_DRAFTS=1` was set.

## Root Cause

The default was conservative for parity bring-up: draft tokens could be
generated and verified, but the verifier committed only a target bonus token.
That is useful for debugging, but it is not the DSv4-Flash + EAGLE performance
workload. It reports target-only effective output even though an EAGLE draft
path ran before the verifier.

The startup contract did not carry enough speculative-decode state for the
model-specific best-practice check to catch this. It could print graph and
transport blockers while staying silent about whether EAGLE acceptance was
actually enabled.

## Fix

The scheduler startup contract now carries:

- `spec_enabled`
- `internal_mtp_draft_requested`
- `spec_draft_k`
- `internal_mtp_accepts_drafts`

The DSv4 SGLang best-practice contract now fails closed if the target route is
not EAGLE/internal-MTP, or if internal-MTP acceptance is disabled. The runtime
spec path uses the same env parsing helper as the startup contract so there is
one interpretation of `ARLE_INTERNAL_MTP_ACCEPT_DRAFTS`.

This does not make the path fast. It prevents a weaker EAGLE-debug run from
being counted against the 256K/1500 hot-cache target.

## Verification

Local:

- `cargo fmt --check`
- `git diff --check`
- `cargo test -p infer internal_mtp --no-default-features --features no-cuda`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check --tests -p infer --no-default-features --features cuda,no-cuda`

Remote DSv4 pod, `/data01/build/arle`, temp patch on `af59123e`:

- Build passed with the prebuilt CUDA fast path in 21.68 s.
  Log: `/tmp/dsv4_pc9_eagle_acceptance_build.log`.
- Probe without `ARLE_INTERNAL_MTP_ACCEPT_DRAFTS`:
  `/tmp/dsv4_pc9_eagle_acceptance_no_accept_1780455000`.
  - `STATUS=101`, expected fail-closed startup contract.
  - `contract_accept_false=8`
  - `contract_accept_true=0`
  - `acceptance_blocker=8`
  - `full_graph_blocker=8`
  - `deepep_graph_blocker=8`
  - `eagle_graph_blocker=8`
  - `metadata_blocker=8`
- Probe with `ARLE_INTERNAL_MTP_ACCEPT_DRAFTS=1`:
  `/tmp/dsv4_pc9_eagle_acceptance_accept_1780455030`.
  - `STATUS=101`, expected fail-closed startup contract.
  - `contract_accept_false=0`
  - `contract_accept_true=8`
  - `acceptance_blocker=0`
  - `full_graph_blocker=8`
  - `deepep_graph_blocker=8`
  - `eagle_graph_blocker=8`
  - `metadata_blocker=8`

The temp patch was reversed after build. Remote source remained clean at
`af59123e`, and post-probe process checks showed no lingering infer/timeout
compute process output.

## Rule

For DSv4-Flash TP8 + EAGLE, `--spec-draft-model eagle` is not enough. The
target comparison needs accepted draft tokens in the effective output stream,
and the report must include TTFT, TPOT, E2E, output throughput, and EAGLE
acceptance together.
