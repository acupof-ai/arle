# Report per-request prefix-cache tokens in OpenAI `cached_tokens` — 2026-09-12

> Status: Pending remote. Mac unit gates cover the JSON absence/presence and
> relay round trip. The value is captured in the CUDA worker path
> (`infer-api/src/loaded.rs`), which is `#[cfg(feature = "cuda")]`, so a pod
> real-CUDA build is required and recorded below. No GPU run needed (compile
> path only).

## Context

The OpenAI-compatible `usage.prompt_tokens_details.cached_tokens` was hard-coded
to 0 (schema.rs), even though the engine measures per-request prefix reuse. The
engine-wide `prefix_cache_stats.hit_tokens` is cumulative since engine start,
so wiring that into a per-response field would print a since-start total on
every request — a wrong nonzero, worse than an honest zero.

## What the number is

At prefix attach, `restored_len` is the cross-rank-min-aligned restore boundary
— exactly this request's prompt tokens served from prefix cache, per-request,
token units, and never larger than the prompt (`infer-core/src/prefix.rs`
attach). It was previously written only into `prefill_start_pos`, which chunked
prefill then overwrites per row, so it is a moving cursor, not a stable count.

## Change

- Capture it once at attach into a new `RequestState.cached_prompt_tokens:
  Option<usize>` (None = prefix cache disabled/unmeasured; Some(0) = measured,
  no reuse), carried to `CompletedRequest`.
- Thread it to the OpenAI response through both deployments: single-process via
  the local relay terminal `StreamItem::Done`, and multiproc via a new
  `#[serde(default)] cached_prompt_tokens` on `RelayCompletionDelta`.
- `Usage` takes `Option<usize>`; when `None` the whole
  `prompt_tokens_details` is omitted rather than emitted as zero. `Some(0)`
  emits 0.
- Programmatic `infer-api::TokenUsage` is unchanged (non-HTTP callers don't use
  the OpenAI breakdown).

## Why absent is not zero

A field a deployment cannot report (old worker in a version-skewed cluster, or
a response assembled without the terminal delta) must be absent; a `0` would be
manufactured by serde defaults and indistinguishable from a real no-reuse
request.

## Verification

Unit gates (`cargo test -p infer-server`):

- known reuse serializes the breakdown with `cached_tokens` equal to the
  restore count and `<= prompt_tokens` (assertion, no clamp);
- measured zero reuse emits `cached_tokens: 0` (present, a measurement);
- unreported (`None`) omits `prompt_tokens_details` entirely;
- a relay delta without the field deserializes to `None`, and an explicit value
  round-trips.

Pending remote: pod `cargo check -p infer-api --features cuda,nccl` and
`cargo clippy --workspace --all-targets --features cuda,nccl -- -D warnings`
for the `#[cfg(feature="cuda")]` worker sink.
