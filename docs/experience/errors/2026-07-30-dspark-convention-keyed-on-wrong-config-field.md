# DSpark proposal convention re-keyed onto a constant, accept 0.54 -> 0.03

## Context

`DsparkConfig::from_dir` derives the proposal convention — `next_token_heads`,
i.e. block drafts `block_size` tokens (rows are next-token heads) vs
`block_size − 1` (same-position, row 0 carries the known anchor). It originally
read the `dflash_config` nesting. While aligning our implementation with the
DeepSpec reference I re-keyed it on `architectures[0].contains("DSpark")`,
because that is what DeepSpec's `eval.py` and sglang dispatch on
(`92afc4a17`).

## Root Cause

**Our checkpoints do not come from DeepSpec — they come from
`scripts/convert_dspark_speculators.py`, which hardcodes
`"architectures": ["DSparkDraftModel"]` for every conversion** (line 35) and
encodes the real convention in the `dflash_config` nesting (line 58, driven by
the upstream `speculative_tokens == block_size - 1`). So the new discriminator
was keyed on a constant: every checkpoint became next-token, and
`dspark-fr-native` — a same-position model — had its whole chain shifted one
position forward.

The reference's dispatch axis was right for the reference and meaningless for
our corpus. I checked the *reference*, not the *producer of the files being
read*.

Measured on the pod (H20, GPU 3, ThinkingCap-27B-FP8, block 6, 128 reqs):

| arm | c=1 TPOT | c=16 TPOT | accept c=1 | chains c=1 |
|---|---|---|---|---|
| no-spec | 78.25 ms | 262.49 ms | — | — |
| DFlash blk6 (control) | 21.77 ms | 162.50 ms | 0.504 | 4492 |
| fr-native blk6, flipped | 66.88 ms | 277.21 ms | **0.026** | 13634 |

Spec decode stays lossless under either reading — the needle ladder passed
`exact=3 DET` at 512/4k/16k/32k on the broken arm. Only accept moves, and no
assertion read it.

## Fix

Revert to the truthful signal and make it explicit: the converter now writes
`speculative_tokens` into config.json, and `from_dir` prefers that field,
falling back to the `dflash_config` nesting for checkpoints converted before it
existed. `architectures` is never consulted. Unit test
`convention_from_speculative_tokens_then_nesting` pins all four cases.

## Rule

**A discriminator must be validated against the thing that WRITES the file, not
the thing the file resembles.** When adopting a reference implementation's
dispatch, first check whether our inputs are produced by that reference. Ours
are produced by a converter in this repo — one `grep architectures
scripts/convert_dspark_speculators.py` would have shown it emits a constant.
And a convention that only survives as a structural side effect (a nesting) is
one refactor away from being lost: give it a field.
