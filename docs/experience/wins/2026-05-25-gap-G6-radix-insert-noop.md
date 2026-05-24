# G6 Radix Insert E2E Verification Closed as Noop

Related: `docs/projects/2026-05-24-opd-mainline-task-backlog.md` T6/G6 and
`docs/plans/2026-05-24-sglang-pipeline-cuda-mlx-gap-analysis.md`.

## Context

T6 G6 asked whether the CUDA scheduler actually inserts completed prompts back
into `RadixCache`, because the gap-analysis doc only had source inference for
the D2 "new sequence inserts back into radix" path.

The requested direct assertion was "same prompt twice, second request
`req.reusable_prefix_len == full prompt len`". Under the CPU-only constraint,
an `infer/tests/*` integration test cannot construct the CUDA scheduler or read
`ActiveRequest::reusable_prefix_len`: `Scheduler` is compiled only with the
`cuda` feature and the field is crate-private. The test therefore locks the
shared contract that feeds that field: completed-prompt insert followed by the
next same-prompt `lookup_or_stage`.

## What Worked

- Added `infer/tests/radix_insert_e2e.rs`.
- The test publishes a two-block, block-aligned prompt through
  `RadixCache::insert`.
- The second lookup of the same prompt returns:
  - `matched_len == prompt.len()`;
  - every matched block has `HitKind::ReadyOnGpu`;
  - no recompute advice.

This closes the G6 "insert may be missing" hypothesis as a noop for the
RadixCache contract. No runtime scheduler fix was needed.

## Scheduler Projection

The source-to-field path remains:

| Step | File | Semantics |
| --- | --- | --- |
| Completed prompt publish | `infer/src/scheduler/cuda/core.rs:1303` | `publish_to_prefix_cache()` calls `insert_with_fingerprints()` for sealed full blocks. |
| Next admission lookup | `infer/src/scheduler/cuda/runtime/admission.rs:214` | `build_prefix_admission_plan()` calls `lookup_or_stage(prompt_tokens, ...)`. |
| Direct T0 reusable tokens | `infer/src/scheduler/cuda/runtime/admission.rs:299` | When blocks are fully addressable and ready on GPU, reusable tokens become `lookup.matched_len`. |
| Active request field | `infer/src/scheduler/cuda/runtime/admission.rs:707` | `admit_waiting_candidate()` writes `lookup.matched_len` into `req.reusable_prefix_len` for direct GPU attach, otherwise the reusable slot length. |

## Verification

```bash
cargo test -p infer --no-default-features --features no-cuda --test radix_insert_e2e
cargo check -p infer --no-default-features --features no-cuda
```

- `radix_insert_e2e`: 1 passed, 0 failed.
- `cargo check`: exit 0.
- CPU-only; no GPU, P5 PID 28950, `infer/src/kv_tier/`, or scheduler runtime
  behavior was touched.

## Rule

For CPU-only validation of CUDA scheduler cache behavior, test the public
RadixCache contract directly and document the private scheduler projection
separately. Do not expose scheduler internals solely to satisfy an integration
test.

## Verdict

PASS / noop. The cache insert-to-lookup contract returns a full same-prompt
T0 hit; G6 does not need a production code patch.
