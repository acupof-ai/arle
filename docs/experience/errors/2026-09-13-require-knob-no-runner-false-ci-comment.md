# A require-resource knob that no runner sets, and a comment claiming a CI lane that does not exist

Date: 2026-09-13. Found while closing the `skip-shaped-pass` agenda row and
running the same day's device-test census.

## Context

Two different audits, card-free, converged on one class: **tests that no
lane can force to run, by construction.**

1. The skip-shaped row enumerated every unmet-precondition verdict and,
   instead of counting the knobs that *read* an environment variable,
   enumerated the runners that *set* it.
2. The device-test census classified cuda-kernels tests by what the
   function body does rather than by `cfg`, finding 16
   device-requiring functions that had never executed.

## The knobs and who sets them

There are three require-resource knobs; the Vulkan one is the original
template.

| knob | read at | tests gated | a runner sets it |
| --- | --- | --- | --- |
| `ARLE_REQUIRE_VULKAN_DEVICE` | `vulkan-sys/src/lib.rs:2112`, `vulkan-kernels/tests/common/mod.rs:15` | 8 vulkan-kernels device tests (`device_gemv`, `device_gemv_id`, `device_full_attention`, `device_q8_1`, `device_elementwise`, `device_router_topk`, `device_linear_attention`, `pipeline_cache`) plus the `vulkan-sys` device lib tests | **none** |
| `ARLE_REQUIRE_TEST_FIXTURE` | `crates/train/tests/common/qwen35_test_support.rs:148` | 2 train fixture tests (`test_qwen35_loader::loader_smoke`, `test_infer_teacher`) | **none** |
| `ARLE_REQUIRE_CUDA_DEVICE` | `crates/infer-cuda/src/qwen35.rs:392` | `device_lora_merge_matches_host_reference` | `scripts/parity_gpu_batch.sh:215` on the claimed card |

Only the CUDA knob is backed by a runner. The other two are mechanisms
that cannot fire anywhere today.

## The false comment

`vulkan-kernels/tests/common/mod.rs:5` stated:

> CI sets `ARLE_REQUIRE_VULKAN_DEVICE=1` …

No workflow does, and no lane provides the device the knob would demand.
`metal-ci.yml` is Metal-only: it never installs the Vulkan SDK or
MoltenVK, never sets `VK_ICD_FILENAMES`, and never builds a `vulkan-*`
crate; the only Vulkan reference in CI is a clippy typecheck
(`ci.yml:113`, no ICD). The comment described a lane that has never
existed in the current workflows. It was wrong rather than stale, and it
is worse than no comment: a reader trusts the wiring and does not check.

This is the second false-comment finding of the day. 7d independently
found `metrics.rs:338` asserting that five counters "remain in /v1/stats
and JSONL" while `from_counters` hardcodes them to `None`. Both are a
comment asserting a behavior the code does not have; neither is caught
by any existing check, and they were found by two different habits —
enumerating the setters rather than the readers here, reading the code
the comment describes there.

The `common/mod.rs` comment now states the truth: no workflow sets the
knob, no CI lane has an ICD, the gates skip everywhere, and the knob is
retained so a future device-providing runner can force a loud failure.

## Why the knobs are kept

A knob nothing sets is superficially the same defect as a statistic
nothing gates, but the repair costs are not symmetric. A discarded
statistic is re-gated with a one-line predicate; deleting a require knob
removes the only mechanism by which these tests could ever fail loudly
and costs more to re-add later. Both knobs stay. What is removed is the
false claim that they are already wired.

## The merged population

Across two backends, tests that cannot execute in any lane by
construction:

- 16 device-requiring functions in `cuda-kernels` (10 gemm FFI tests,
  2 `device_matrix`, 2 `device_context` fences,
  `fa2_sm70_matches_host_reference`, `moe::w4a16_grouped_gemv`) — see
  the `device-tests-never-executed` agenda row; execution additionally
  blocked on the all-8-GPU assignment to the 27B/30B serving job.
- 8 Vulkan device tests plus the `vulkan-sys` device lib tests, for want
  of an ICD lane.
- 2 train fixture tests, for want of a provisioned small dense model and
  a runner that points the knob at it. The one small model a CI lane
  touches is `mlx-community/Qwen3.5-0.8B-MLX-4bit` in the Metal needle
  gate — MLX 4-bit, not an HF safetensors directory the train loaders
  accept, so it is not a usable fixture.

## Secondary defect, accepted rather than open

The skip note is an `eprintln!`
(`qwen35_test_support.rs:137`); libtest captures passing-test output, so
on a plain `cargo test` the skip is invisible, and no repo invocation
passes `--nocapture`. This is the Vulkan template's own accepted
design: quiet on a developer box that does not have the resource, loud
on a runner that expects it. The fix is a runner-wide `--nocapture` in a
lane that wants skips shown, not a louder print per test.

## Unblocking cost (decisions for the controller, not code)

- Fixture knob: provision a small **dense HF-format** model on the pod or
  a CI runner and set `ARLE_REQUIRE_TEST_FIXTURE=1` in a train-test
  runner. The pod currently holds only 27B/35B FP8/MoE checkpoints and
  2-layer nvfp4 slices.
- Vulkan knob: stand up a lane with a Vulkan ICD (MoltenVK on the
  Apple-Silicon runner, or a Linux SwiftShader/LAVA lane) and set
  `ARLE_REQUIRE_VULKAN_DEVICE=1`. New infrastructure; deliberately not
  started in this round.

## Rule

A require-resource variable is only as real as the runner that sets it:
when auditing skips, enumerate setters, not readers. And treat a comment
claiming a wiring ("CI sets X", "X remains in Y") as an assertion to
verify against the workflow and the code, not documentation to trust —
a false comment about a safety mechanism actively prevents the next
person from discovering the gap.
