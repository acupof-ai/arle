# The MarlinW4A8 weight format and its producer scripts are gone

Date: 2026-09-13 (deletion merged 2026-09-11 as `bacd5c5f1`; exit criterion
verified 2026-09-13).

## Context

`WeightFormat::MarlinW4A8` was a quantization format with no serving path. It
survived as an enum variant, a build-script branch, two backward-pass matches,
and five Python producer scripts that converted or diagnosed checkpoints
nothing could load. The agenda row `delete-w4a8-variant` asked for its removal
with a specific exit: zero hits for `MarlinW4A8`, `marlin_w4a8` or
`w4a8-marlin` across `src/`, `crates/`, `scripts/` and `tests/`.

## What Worked

A deletion refactor rather than a deprecation. The variant was removed from
`crates/cuda-kernels/src/tensor/weight_format.rs` and the compiler located
every consumer — `crates/autograd/src/backend.rs`, `crates/train/src/opd.rs`
and `opd/backward.rs` — instead of a grep being trusted to find them. The
producer scripts went with it: `convert_gptq_w4a16_to_w4a8_marlin.py`,
`diag_w4a8_pack_roundtrip.py`, `merge_w4_hybrid_checkpoint.py`,
`quantize_qwen3_w4a8.py` and `verify_gptq_w4a8_repack_quality.py`, with the
w4a8 branch cut out of `quantize.py`. Thirteen files, 1003 deletions against
11 insertions.

Verification used the exit criterion's own three spellings rather than a
broader pattern, and took a positive control first: the pattern was confirmed
to match the literal string `MarlinW4A8` before its zero was believed. Result:
zero files.

A first attempt at that check also included the bare token `w4a8` and returned
five files. All five are `crates/cuda-kernels/csrc/moe/w4a8/w4a8_grouped_gemm.cu`
and its build wiring, which is a live CUTLASS grouped GEMM kernel and not the
weight format. Reading the hits rather than the count is what kept a working
kernel from being swept up in a cleanup.

## Rule

Delete a dead format at its definition and let the type checker enumerate the
callers; a grep can only find the spellings you thought of. When the exit
criterion names its own patterns, verify with exactly those and not with a
broader one — a wider pattern turns unrelated live code into apparent debt, and
the cleanup that follows is the damage. Confirm any zero with a positive
control before recording it.
