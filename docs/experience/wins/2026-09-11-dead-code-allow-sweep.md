# Dead-code allow sweep (infer-cuda + cuda-kernels)

Date: 2026-09-11. Lane `simplify-dead-code-allows`. PR #303.

Status: **Shipped.** This is a net deletion with no intended behaviour
change: the only deleted items were ones the compiler proved unreferenced
in the full cuda feature set, and the gate for that claim is
`clippy -D warnings`, not a GPU run. There is no runtime path to validate
remotely.

## Context

The two CUDA crates carried 65 plain `#[allow(dead_code)]` attributes (54 in
infer-cuda, 11 in cuda-kernels). Several had outlived the callers they once
named — a removed allow that compiles clean is a field/method the current
build actually reads. `cfg_attr(not(feature=…), allow(…))` guards were left
untouched (those are correctly feature-conditional).

## What changed

Deleted every plain allow in one pass, then let
`cargo clippy -p infer-cuda -p cuda-kernels --features cuda,no-cuda,nccl,deepep
--all-targets -D warnings` adjudicate. Of the 65, only 9 surfaced:

- **Deleted as genuinely unused (3):** `Dsv4DraftLatent::reset` (thin wrapper
  over the surviving `rebase(0)`),
  `load_dsv4_block_scaled_bf16_copy` (bf16 dequant copy never wired),
  `physical_token_rows` (its sibling `contiguous_page_table_byte_range` is
  used; this identity-flatten helper had no caller after its inline tests
  were removed).
- **Feature-conditional (0):** none — every flagged item compiled under the
  cuda feature set.
- **Kept with a one-line why (3 allow attrs covering 6 symbols):**
  - module-scoped `#[allow(dead_code)]` on the two `include!` wrappers of the
    generated `qwen_fp8_dense_projection.rs`. One generated file is included
    into both `quant_linear.rs` (reads `POLICY_ID`) and
    `quant_linear_fp8.rs` (reads `Route`/`select_exact`/`fallback`/
    `HAS_EXACT_CELLS`); the unused half in each module is dead-but-shared by
    construction. 5 generated symbols.
  - `Dsv4GemvTables::scale_storage` — owned `Vec<CudaSlice<u8>>` for the
    W4AFP8 decode lane; the GPU kernel reads it by device pointer, so Rust
    never reads the field after construction (held for ownership/Drop).

Also moved a pre-existing live `impl From<&DeviceMatrix> for WeightLayoutQuery`
from after `mod tests` to before it in `cuda-kernels/src/tensor/device_matrix.rs`
— four infer-cuda loader call sites use it, but `--tests` rejects non-test
items placed after the test module. No behavior change.

Net: 29 files, +36/−161 (−125).

## Rule

After deleting a blanket `#[allow(dead_code)]`, trust the compiler per feature
set; never restore an allow without naming why (wire/FFI layout, Drop-held
field, or a shared generated include). A field populated but never read in Rust
is often a GPU-pointer-owned buffer — keep it for Drop, don't delete it.

## Host checks (Mac, no GPU)

- `clippy -p infer-cuda -p cuda-kernels --features cuda,no-cuda,nccl,deepep
  --all-targets -D warnings`: clean.
- pre-push shape `clippy -p infer-api --features cuda,no-cuda,nccl --lib
  -D warnings`: clean.
- `cargo check -p arle --features cpu,no-cuda,cli`: clean.

## Verdict

Net deletion; the gate is `clippy -D warnings` under the CI Lint mirror
(clean in CI on #303) plus the three host commands above. A deleted
plain-allow is, by construction, an item no compiled feature set reads —
there is no serving path whose behaviour could change, so no GPU run is
owed. Shipped.
