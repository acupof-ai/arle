# Metal DSpark + KV prefix cache: usable together — metal, 2026-08-26

> Status: Confirmed (local M4 Pro 48 GB)
>
> Baseline: `4c1ff355d`, `arle serve --backend metal --model-path
> LiquidAI/LFM2.5-8B-A1B-MLX-4bit --draft-model LiquidAI/LFM2.5-8B-A1B-DSpark
> --kv-cache-dtype bf16 --kv-disk=<dir> --chunked-prefill-size 1024`,
> M4 Pro 48 GB. One 15 030-token prompt, three questions over a shared prefix,
> then the same run after a server restart.

## Context

Turning on the Metal draft model silently turned the whole prefix cache off.
`MetalExecutor::reusable_prefix_blocks` returned 0 whenever `dflash.is_some()`,
and engine-core uses that same predicate on **both** sides of the cache — the
attach path (`lookup_prefix_for_attach`) and the publish path
(`publish_prefix`, `local_sealed = reusable_prefix_blocks(...)`). So with
DSpark on, nothing was ever inserted into the radix, nothing was ever written
through to `--kv-disk` (the store stayed at **0 bytes**), and every request
re-prefilled the full 15 k prompt — including after a restart, which is the
one thing `--kv-disk` exists to prevent.

The reason given for the bail was that a DFlash restore boundary is
incomplete: the snapshot carries only target K/V + recurrent state, not the
target-hidden feature store or the draft KV. Both halves of that turn out to be
false in practice:

- **Draft KV is not prefix state.** DSpark calls `DFlashDraftState::reset()` at
  the end of *every* block — its target context changes each block, so stale KV
  is thrown away anyway. There is nothing across a prefix boundary to restore.
- **Target hidden is re-seeded by the tail.** A prefix match is capped at
  `attach_cap = prompt_len - 1` (`infer-core/src/prefix.rs`), so a matched
  prompt always re-prefills a non-empty tail, and that tail's capture is exactly
  what the first draft block needs.

A shorter first-block context can only cost acceptance, never correctness — the
target verify decides every token either way.

## What worked

Two edits, no new state and no new snapshot payload:

- `reusable_prefix_blocks`: drop the `dflash.is_some() -> 0` bail. DFlash rides
  the same boundary as everything else.
- `submit_prefill`: seed `dflash_draft_state` on first sight instead of only at
  `start_pos == 0`. A prefix-restored prompt never sees chunk 0, so the old
  placement left decode preflighting with no draft state (`DFlash decode for
  slot N has no draft cache state`). `start_pos == 0` now explicitly clears both
  dflash fields, keeping the old "a new prompt restarts the stream" semantics.

Not needed, though it looks like it should be: restating the draft's RoPE base
after a restore. DSpark runs exactly one `forward_draft` per block over an empty
draft KV, and the bridge derives both offsets from the same base
(`k_offset = rope_offset`, `q_offset = rope_offset + context_len`,
`mlx_dflash_draft_model.cpp`). Every row therefore shifts together and RoPE
scores depend only on the difference, so the absolute base cancels. The LFM2
lane's existing `set_rope_offset` is inert for the same reason. A base that did
not cancel would need a lane that keeps draft KV written under *different*
bases — which is the DFlashEagle trim/window lane, not DSpark.

Measured, LFM2.5-8B-A1B-MLX-4bit + DSpark, 15 030-token prompt:

| request | baseline TTFT | patched TTFT (2 runs) |
|---|---|---|
| cold (empty store) | 10.11 s | 13.66 s / 10.97 s |
| same prefix, new question | 10.77 s | **0.645 s / 0.584 s** |
| same prefix, new question | 11.17 s | **0.638 s / 0.721 s** |
| after restart | 10.60 s | **1.54 s / 0.69 s** |

Patched runs license 896 of 938 blocks and restore 14 336 of 15 030 tokens; the
restart adopts all 938 cold blocks by content key. Disk store: baseline
**0 B**, patched **194 MB**.

Decode is unchanged — 98.6 / 97.0 / 91.9 / 86.7 tok/s baseline (mean 93.6) vs
89.2 / 100.3 / 93.0 / 82.1 and 108.9 / 105.9 / 94.9 / 97.4 patched (means 91.2
and 101.8). The shorter first-block draft context does not show up.

Cold TTFT pays the write-through of 194 MB: +3.6 s and +0.9 s over the two
patched runs, the same cost in kind the content-keyed tier entry measured
(14.3 -> 16.0 s). The baseline was not faster, it was just not doing the work.

Correctness:

- Output identity, greedy, 160 tokens: cold vs a 14 336-token restored prefix
  over the same question — byte-identical.
- Needle ladder 512 / 2000 / 4000 / 8000, x3, `NEEDLE_MAX_TOKENS=400`:
  exact 3/3 DET at every length. Runs 1 and 2 of each length are identical
  prompts, so they attach the restored prefix — the gate is the restore path.
- `cargo test -p infer-metal --features metal`: 12/12.

## Rule

- A capability predicate that gates reuse usually gates *publication* too. Ours
  did (`publish_prefix` calls the same `reusable_prefix_blocks`), so "reads are
  disabled" quietly meant "writes never happen" — a `--kv-disk` that stayed at
  0 bytes with no error anywhere.
- Before declaring a restore boundary incomplete, check whether the missing
  state is *carried* or *rebuilt*. DFlash's two extra pieces are both rebuilt
  every block, so there was nothing to snapshot — the bail cost a whole feature
  for state that does not exist across the boundary.
- Speculative decode fails safe on context: the verify step owns correctness, so
  a degraded draft context is a throughput question, not a licensing one.
- "The positions are now wrong" is a claim to check, not assume. Moving the
  draft's context to an arbitrary start looked like it needed a RoPE fix; RoPE
  is relative and every row moved together, so it needed nothing. The measured
  decode parity said the same thing before the derivation did.

## Known limits (pre-existing, not DSpark-specific)

Prompts that fit in one prefill chunk get one boundary, and it is at a
non-page-aligned `cache_len`, so `publish_slot` writes no restore snapshot and
the second identical prompt licenses 0 blocks. Reproduced identically with
`--no-speculative`, so it is snapshot granularity, not the draft model. Same
root cause as the "without chunking there is one boundary" note in
`2026-08-26-metal-kv-disk-content-keyed-restart-cache.md`.
