# P2.4 multi-token spec canary reported fake acceptance

## Context

P2.3/P2.4 exposed that `--spec-draft-k=5` did not fan out into a real
multi-token speculative verifier path. The scheduler routed decode through the
single-token verifier canary, then recorded one verified token and one accepted
token per row.

Observed symptom:

- `acceptance_rate = 100%`
- `draft_tokens_total == verified_tokens_total == accepted_tokens_total`
- effective throughput `23.89 tok/s`, below the Phase 1 c4 baseline
  `26.169 tok/s`

## Root Cause

`infer/src/scheduler/cuda/spec_path.rs::draft_then_verify` delegated to the
normal decode path. `infer/src/scheduler/cuda/decode.rs` checked only
`spec_draft_k > 0`, launched a one-token decode, and hard-coded one accepted
token per eligible row at readback.

This made `spec_draft_k=5` look enabled while no K-token draft proposal, target
K-position verifier, accepted-prefix commit, bonus token, or paged-KV rollback
existed.

## Fix

Restrict the P2.3 canary route to `spec_draft_k == 1` and reject
`spec_enabled + self-spec + spec_draft_k > 1` during scheduler configuration
validation. Request-level `speculative.draft_k > 1` and non-self draft-model
overrides are also excluded from the single-token canary route. Multi-token
configuration must fail explicitly or stay out of the canary path until a real
verifier API lands.

The next implementation unit must add:

- model-side verifier output for K target positions
- paged-KV provisional append and rollback/truncate
- scheduler pending verifier metadata
- commit of `0..K` accepted tokens plus the target fallback/bonus token

## Rule

Do not let a canary route count as a multi-token speculative decoder. Metrics
must report zero speculative tokens when K-token verifier machinery is absent.
