# AGENTS.md named a unit-test opt-out that no unit test reads

Date: 2026-09-13.

## Context

`AGENTS.md` pinned the Metal canonical model and then offered an escape
hatch: "Unit-test opt-out: `INFER_TEST_MODEL_PATH=models/Qwen3.5-0.8B-MLX-4bit`
(document why)." A reader would conclude that Metal unit tests default to the
35B MoE model and that this variable switches them to a small dense one.

## Root Cause

Neither half is true. No Rust code anywhere in the tree reads
`INFER_TEST_MODEL_PATH`; its only reader is `scripts/train_and_chat.sh`, a demo
script, whose default is `models/Qwen3-0.6B` — a different model from the one
the doc prescribes, and a directory that does not exist locally.

And there is nothing for it to opt out of. The three test-bearing files under
`crates/infer-metal/` and `crates/mlx-sys/` cover KV page round-trip, KV seq-axis
growth, and kernel parity. None of them loads a model, so no Metal unit test has
a model default to override.

The tests that do take a model path use differently-named variables
(`INFER_TEST_QWEN3_06B_DIR` in `crates/train/tests/`, `INFER_GGUF_TEST_MODEL` in
`crates/infer-gguf/`), each scoped to its own crate. There was never a global
test-model knob.

## Fix

Sentence deleted; the canonical-model rule it hung off is unchanged. The
replacement states that no Metal unit test loads a model, which is the fact a
reader needs in order to stop looking for the knob.

## Rule

A documented escape hatch is a claim about wiring, and it decays the same way a
documented gate does. Two greps settle it: does anything read the variable, and
does the thing it claims to override exist. Both were negative here, and the
sentence had survived two doc restructures.
