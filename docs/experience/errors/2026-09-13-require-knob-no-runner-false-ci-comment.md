# A require-resource knob that no runner sets, and a comment claiming a CI lane that does not exist

Date: 2026-09-13. Found while closing the `skip-shaped-pass` agenda row and
running the same day's device-test census. The two knobs this entry
describes as unbacked were deleted the same day; the tests they guarded
stay in the tree as manual-only gates.

## Context

Two different audits, card-free, converged on one class: **tests that no
lane can force to run, by construction.**

1. The skip-shaped row enumerated every unmet-precondition verdict and,
   instead of counting the knobs that *read* an environment variable,
   enumerated the runners that *set* it.
2. The device-test census classified cuda-kernels tests by what the
   function body does rather than by `cfg`, finding 16
   device-requiring functions that had never executed.

## The knobs and who set them

There were three require-resource knobs; the Vulkan one was the original
template.

| knob | read at (at the time) | tests gated | a runner set it |
| --- | --- | --- | --- |
| the Vulkan device knob (deleted 2026-09-13) | `vulkan-sys` test helper, `vulkan-kernels/tests/common/mod.rs` | 8 vulkan-kernels device tests (`device_gemv`, `device_gemv_id`, `device_full_attention`, `device_q8_1`, `device_elementwise`, `device_router_topk`, `device_linear_attention`, `pipeline_cache`) plus the `vulkan-sys` device lib tests | **none** |
| the train fixture knob (deleted 2026-09-13) | `crates/train/tests/common/qwen35_test_support.rs` | 2 train fixture tests (`test_qwen35_loader::loader_smoke`, `test_infer_teacher`) | **none** |
| `ARLE_REQUIRE_CUDA_DEVICE` | `crates/infer-cuda/src/qwen35.rs` | `device_lora_merge_matches_host_reference` | `scripts/parity_gpu_batch.sh` on the claimed card |

Only the CUDA knob was backed by a runner. The other two were mechanisms
that could not fire anywhere.

## The false comment

`vulkan-kernels/tests/common/mod.rs` stated that CI set the Vulkan device
knob. No workflow did, and no lane provided the device the knob would
demand. `metal-ci.yml` is Metal-only: it never installs the Vulkan SDK or
MoltenVK, never sets `VK_ICD_FILENAMES`, and never builds a `vulkan-*`
crate; the only Vulkan reference in CI is a clippy typecheck
(`ci.yml`, no ICD). The comment described a lane that has never
existed in the current workflows. It was wrong rather than stale, and it
is worse than no comment: a reader trusts the wiring and does not check.

This is the second false-comment finding of the day. 7d independently
found `metrics.rs` asserting that five counters "remain in /v1/stats
and JSONL" while `from_counters` hardcodes them to `None`. Both are a
comment asserting a behavior the code does not have; neither is caught
by any existing check, and they were found by two different habits —
enumerating the setters rather than the readers here, reading the code
the comment describes there.

## Outcome: the two unbacked knobs were deleted

An earlier draft of this entry argued for keeping both knobs as the
mechanism a future resource-providing runner would use. The controller
decision the same day went the other way, on the same first-principles
ground the audit used: a mechanism no runner sets is dead surface, and
the re-addition cost (one env read in a test helper) is smaller than the
cost of carrying a safety control that cannot act and of comments that
claim it does. The deletion covered every read site, the helper code
that existed only for the knobs, and every doc or comment that named
them. The guarded tests remain: a missing device or fixture is a plain,
visible skip (`--nocapture`), and a box with the resource runs them
unchanged. The CUDA knob survives untouched because a runner sets it.

## The merged population

Across two backends, tests that cannot execute in any lane by
construction:

- 16 device-requiring functions in `cuda-kernels` (10 gemm FFI tests,
  2 `device_matrix`, 2 `device_context` fences,
  `fa2_sm70_matches_host_reference`, `moe::w4a16_grouped_gemv`) — see
  the `device-tests-never-executed` agenda row; execution additionally
  blocked on the all-8-GPU assignment to the 27B/30B serving job.
- 8 Vulkan device tests plus the `vulkan-sys` device lib tests, for want
  of an ICD lane; manual-only after the deletion.
- 2 train fixture tests, for want of a provisioned small dense model.
  The one small model a CI lane touches is
  `mlx-community/Qwen3.5-0.8B-MLX-4bit` in the Metal needle gate — MLX
  4-bit, not an HF safetensors directory the train loaders accept, so it
  is not a usable fixture; manual-only after the deletion.

## Secondary defect, accepted rather than open

The skip note is an `eprintln!`; libtest captures passing-test output, so
on a plain `cargo test` the skip is invisible, and no repo invocation
passes `--nocapture`. With the loud-failure half deleted there is no
runner-side remedy left either: these gates are manual-only, and the skip
message is written for the capable-box user who runs them with
`--nocapture`.

## Rule

A require-resource variable is only as real as the runner that sets it:
when auditing skips, enumerate setters, not readers. A knob with zero
setters is a candidate for deletion, not retention. And treat a comment
claiming a wiring ("CI sets X", "X remains in Y") as an assertion to
verify against the workflow and the code, not documentation to trust —
a false comment about a safety mechanism actively prevents the next
person from discovering the gap.
