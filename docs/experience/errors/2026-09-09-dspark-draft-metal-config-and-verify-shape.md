# DSpark draft on Metal: config rejected, then verify shape mismatch — 2026-09-09

> Status: Fixed in lane c. Validation: `scripts/spec_parity.py` negative
> control + green run on Qwen3.5-0.8B + r3lax/Qwen3.5-0.8B-DSpark.

## Context

`scripts/spec_parity.py` (design note 2's reproduction) pairs
`models/Qwen3.5-0.8B-MLX-4bit` with `r3lax/Qwen3.5-0.8B-DSpark`. The draft
arm failed twice before a token was verified.

## Bug 1 — config parse rejected the checkpoint

```
engine build failed: cannot parse .../config.json: missing field `rope_theta`
```

The checkpoint was exported with transformers 5.10.2: RoPE settings nest
under `rope_parameters` and the DFlash-specific fields (`mask_token_id`,
`target_layer_ids`) sit at the top level with no `dflash_config` object.
`RawDraftConfig` in `crates/infer-metal/src/dflash.rs` required the old
layout (top-level `rope_theta`, nested `dflash_config`). Two sibling parsers
already handled the new layout — `qwen3-spec`'s `Qwen3ConfigRaw` and the MTP
path's `RawMtpRopeParams` — but the DFlash path was never updated. The
earlier Metal DSpark work (wins 2026-08-26) used a z-lab-layout checkpoint,
so the gap was never exercised.

Fix: `rope_theta` optional with a `rope_parameters` fallback (default 1e6,
matching every shipped Qwen3 config); `dflash_config` optional with a
top-level fallback; a config with neither fails loudly.

## Bug 2 — verify kernel shape mismatch

With the config parsing, the draft arm built and loaded, then exited on the
first request:

```
DFlash target verify summary failed: MLX error:
qwen35_compiled_verify_block_summary requires token_ids shape [block_size]
```

The Markov-head (DSpark) path builds a verify block of `block_size + 1`
tokens — `[real, d1..d_bs]`, verifying `block_size` drafts, per the DeepSpec
layout the code comments cite. The compiled verify kernel
(`qwen35_compiled_verify_block_summary`) takes exactly `block_size` tokens
and verifies `block_size - 1` drafts: that is the non-Markov DFlash layout,
and the kernel's matched-prefix loop is hard-wired to it
(`drafted_len = block_size - 1`). The Markov path shipped (
2026-08-24) against a compiled forward baked for S=5; a later refactor
collapsed the verify path onto the fixed-shape summary kernel and the
Markov path was never re-validated on Metal — the 2026-08-26 prefix-reuse
work used a non-Markov draft.

Fix (minimal): the Markov and no-Markov paths now build a `block_size`-token
block — `[real, d1..d_{bs-1}]` — verifying `block_size - 1` drafts. A
confidence-truncated tail pads with the last draft token; padding cannot
extend the matched prefix past the real drafts. The full `block_size + 1`
verify needs a variable-length kernel — follow-up, not this lane.

## Rule

Two rules. First: when one parser path grows a compatibility fallback for a
new checkpoint layout, the sibling paths parsing the same family's configs
get audited in the same change — the MTP path was updated for nested RoPE,
the DFlash path was not. Second: a draft path that was validated against one
checkpoint family is not validated against another; the Markov path's last
Metal validation used a non-Markov draft, and nothing in the tree said so.
A draft pairing that has never run a request is an unvalidated pairing,
however much code surrounds it.
