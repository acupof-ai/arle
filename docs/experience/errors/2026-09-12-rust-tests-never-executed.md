# Rust tests that compile (or not) and never run

Date: 2026-09-12. Found by auditing the Rust test surface the same way the
shell suite was audited: every `#[test]` compared against the actual `cargo
test` invocations in `.github/workflows/` and `scripts/pre_push_checks.sh`,
including their feature sets.

## Context

The pre-push/CI test commands were extracted and enumerated:

- `ci.yml` test-backend: `cargo test -p infer-api -p arle -p cli -p autograd
  -p train --features cpu,no-cuda,cli`.
- `ci.yml` test-neutral: a fixed `-p` list of 20 device-neutral crates.
- `metal-ci.yml`: `cargo test -p infer-api -p cli -p arle --lib
  --features metal,no-cuda`, `cargo test -p infer-metal --lib`, and
  `cargo test -p mlx-sys`; autograd/train Metal are `cargo check --lib`.
- Hook: the same crate lists plus clippy.

Comparing the test functions in the tree against those commands produced
five populations that never execute:

1. Crates absent from every `cargo test`. `cuda-kernels` and `infer-cuda`
   appear in no test invocation anywhere (clippy `--tests` compiles them
   only). Eight host-only `cuda-kernels` test functions need no GPU and have
   never run (ring host math in `tests/`, weight-format validation,
   device-ordinal parsing); eleven host-only `infer-cuda` functions likewise
   (quant routing, TP budget, slot-image byte inverse, prefix-pool meta).
   `vulkan-sys` / `vulkan-kernels` / `infer-vulkan` are clippy-only; the
   vulkan tests' own anti-skip (`ARLE_REQUIRE_VULKAN_DEVICE=1`) is armed by
   nobody. `hip-sys` tests are not even compiled (`--lib` clippy).
2. Feature-gated tests whose feature no lane enables: five functions under
   `autograd/src/safetensors_io.rs` gated `feature = "safetensors"`.
3. The Metal backend layer: ~52 `autograd` test functions and one `train`
   test function gated `feature = "metal"` are excluded everywhere. The
   `mlx-sys` kernel-parity tests do run; the backend tests against Metal do
   not.
4. A test that turns a missing fixture into a pass:
   `crates/train/tests/test_qwen35_loader.rs::loader_smoke_qwen3_0_6b` runs
   in the CPU backend lane and, when neither `INFER_TEST_QWEN3_06B_DIR` nor
   the ModelScope cache exists, prints a skip note and `return Ok(())` —
   reported ok with zero assertions executed on every CI run. Its header
   says it is deliberately not `#[ignore]` so CI "see[s]" the skip.
   `test_infer_teacher.rs` has the same shape and is additionally cuda-gated.
5. Explicit `#[ignore]`: three CLI live tests, two on-box GGUF/vulkan tests
   — labeled with the resource they need, the honest form.

## Root Cause

Three independent mechanisms produce the same outcome — a test that reports
no result while the file exists in the tree.

1. **Absence from the invocation list.** Nothing derives the tested crate
   set from the workspace; the lists are hand-maintained per lane. A new
   kernel crate with host-side math tests gets clippy coverage (it
   typechecks) that reads as test coverage (it does not run).
2. **`--lib` justified by a wrong comment.** `metal-ci.yml:78-80`:

   > one merged Metal test run for the binary crates (none have tests/ dirs,
   > so --lib is a no-op) + one Metal typecheck for the train/autograd libs
   > (their tests/ hold CUDA cases skipped on macOS).

   Both claims are false. The binary-crate statement was true when written
   but the crates' `tests/` later existed; and the autograd/train `tests/`
   hold the Mac cases themselves (`test_device_handle.rs`,
   `test_metal_device_matmul.rs`, the metal-gated functions in
   `test_linear_attention.rs`, `test_qwen35_hybrid_forward_metal.rs`), not
   only CUDA cases. `--lib` excludes the `tests/` target entirely, and the
   recorded reasoning made the exclusion look deliberate. A wrong comment
   that justifies a wrong flag keeps the flag alive.
3. **A missing fixture converted to `Ok`.** Returning `Ok(())` from a
   `#[test]` is indistinguishable from a pass; the skip message goes to the
   test log, not the verdict. A skip needs `#[ignore]` (counted and
   overridable) or a require-fixture env that makes absence panic, the
   pattern the vulkan tests already implement.

Separately: `AGENTS.md` documents `INFER_TEST_MODEL_PATH` as a unit-test
opt-out, and no Rust code reads it — grep across `crates/`, `src/`, `tests/`
is empty. Only `scripts/train_and_chat.sh` reads it; `.env.example` and
`docs/environment.md` document it. The documented test knob does not exist
in test code. `AGENTS.md` is ckl's; the discrepancy is raised with him, not
edited here.

## Fix

Not started; this entry is the record. The order is fixed: run before
wiring. A never-executed test is not known to pass, so execute each
population first (`cargo test -p cuda-kernels`, `cargo test -p infer-cuda`
host functions — no GPU; `cargo test -p autograd --features metal` on a
Mac lane), record pass/fail per function, wire only the green ones, and
file each red separately. Converting the fixture-missing test to a real
skip or require-fixture failure is independent of the wiring work.

## Rule

A test's existence is not evidence it runs. The check is against the actual
invocation list — command, `-p` set, target (`--lib` excludes `tests/`),
and feature flags — not against the presence of a test file, and not
against clippy `--tests`, which proves compilation only. Any skip-by-return
inside a test function must change the verdict (`#[ignore]` or panic),
never report ok.
