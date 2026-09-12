# A test that skips by returning Ok is a green that means nothing

Date: 2026-09-13. Third instance in two days of the same shape, found
because #408 executed tests that had never run.

## Context

A test function that cannot build its required state and returns `Ok(())`
reports the same verdict as a test whose every assertion passed. The skip
note it prints goes to the test log, not the result. Three instances:

1. `crates/train/tests/test_qwen35_loader.rs::loader_smoke_qwen3_0_6b` —
   runs in the CPU CI lane; when neither `INFER_TEST_QWEN3_06B_DIR` nor the
   ModelScope cache exists it prints a skip note and returns early, so it is
   `ok` on every CI run with zero assertions. Its header says it avoids
   `#[ignore]` deliberately so CI can "see" the skip, but a skip note is not
   a verdict.
2. `crates/train/tests/test_infer_teacher.rs` — the same
   missing-fixture-to-`Ok` return, additionally gated behind the cuda
   feature no CI lane enables, so it neither compiles nor runs.
3. `crates/infer-cuda/src/qwen35.rs:501-504
   device_lora_merge_matches_host_reference` —
   `let Ok(ctx) = DeviceContext::new() else { eprintln!("no CUDA device;
   skipping"); return; }`. First-ever pod run with no visible device
   reported `1 passed`; the numeric comparison the test exists for never
   ran.

## Root Cause

Rust has a distinct, counted skip verdict (`#[ignore]`, or a skipped test
reported by the harness), and all three sites instead encode a skip as the
success value of the function. A `Result<(), E>` return type makes this
syntactically natural — `Ok(())` reads as "nothing went wrong" — but a
device test with no device did not verify anything, so "nothing went wrong"
is the wrong frame: the precondition was unmet, which is neither pass nor
fail. `ok` is the most misleading output available for it: a pass is
exactly the signal a regression must turn red, and a missing GPU/model/ICD
then makes the gate structurally incapable of going red on any runner that
lacks the resource.

The existing correct pattern is the Vulkan tests:
`crates/vulkan-sys/src/lib.rs::skip_or_panic` and
`crates/vulkan-kernels/tests/common/mod.rs::require_device`. A test
acquires its resource through one helper; the helper returns `None` and
prints a visible skip when the resource is absent, but if
`ARLE_REQUIRE_VULKAN_DEVICE` is set it panics. A runner that expected a
device (GPU CI, the pod) sets the variable and gets a hard failure when
none is present; an ordinary host that never set it still skips visibly.
The environment variable is the precondition made explicit at the call
site rather than guessed from the exit code.

## Fix

Entry only; the code change is a separate lane. Apply the Vulkan pattern
to CUDA device tests: one `require_cuda_device()` helper keyed on a
`ARLE_REQUIRE_CUDA_DEVICE` variable (the CUDA analogue; pod-build-env
already sets `CUDA_VISIBLE_DEVICES=""` for builds and the run scripts set
it to a claimed card), returning a context or panicking when the variable
says a device was expected. Convert the three sites to it, starting with
`device_lora_merge_matches_host_reference`. The loader smoke test is a
model fixture rather than a device: the same rule, keyed on its model-dir
variable.

## Rule

A test whose required resource is absent must not report the pass verdict.
Encode the precondition in the verdict: `#[ignore]` for a permanently
optional test, or a require-resource environment variable that turns
absence into a panic on the runner that set it and into a visible skip
elsewhere. Never return `Ok(())` from an unmet precondition — a gate that
cannot go red on the runner that executes it is not a gate.
