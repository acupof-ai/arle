# A prefix cache for hybrid models

Design note 1 of 5 ([plan](../plans/2026-09-02-design-theses.md)). Metal
and CUDA, Qwen3.5 / Qwen3.6. Date: 2026-09-02.

## Problem

A coding agent sends the whole conversation on every turn: the system prompt,
every prior tool result, every prior reply. Turn 12 of a typical session is
8.6K tokens, of which about 350 are new. A server that re-prefills all of it
pays a full prefill per turn; on a 35B MoE on an M4 Pro that is over a second
before the first token, every turn, for the life of the session.

The standard fix is a prefix cache: keep the KV of earlier requests, match the
new prompt against it, prefill only the tail. Qwen3.5 and Qwen3.6 break the
standard fix. Three of every four layers are gated delta networks (linear
attention). A linear-attention layer has no per-token KV. Its state after `n`
tokens is one fixed-size matrix per head plus a short convolution window, and
the state after `n+1` tokens is computed from the state after `n`. It cannot
be sliced, trimmed, or addressed by page.

## Standard practice and where it fails

Paged KV caches (vLLM, SGLang, the radix cache here) key on blocks of
attention KV: a block of `page_size` tokens is one unit of storage and one
unit of matching. Reusing a prefix of `k` blocks means handing the new request
those `k` pages. Nothing in that model holds a recurrent state, so a hybrid
model gets one of two outcomes: no reuse for the recurrent layers, which
makes the attention reuse useless because the recurrent layers still need
the full prompt, or a separate mechanism.

mlx-lm 0.31.2 takes the other route: it caches whole conversation objects
(`LRUPromptCache`, one deep copy per hit) and trims a longer cached object
down to the common prefix. `ArraysCache`, the recurrent layers' cache class,
reports `is_trimmable() == False`, so a hit requires the cached prompt to be
a prefix of the new one. In the agent workload that holds within one
conversation and fails across conversations sharing a system prompt, and
every hit copies the full cache object.

The requirement, stated once: **reuse length is set by where a recurrent
state exists, and the state must be stored under the same identity the
attention pages are matched by.** Both halves failed here at least once.

## The design

**Snapshots at page boundaries, bound to the attention pages.** Each prefill
chunk that ends on a page boundary publishes a snapshot of the recurrent
state alongside its pages. On CUDA the snapshot is
`Qwen35RecurrentSnapshot { gdr, conv }`
([`qwen35_state.rs:54`](../../crates/infer-cuda/src/qwen35_state.rs)), copied
device to host and stored in the slot tier under its own namespace keyed by
the token-prefix hash ([`executor.rs:852`](../../crates/infer-cuda/src/executor.rs)).
On Metal it is `MetalPrefixSnapshot { cache_len, gdr_flat }`
([`kv_ssd.rs:78`](../../crates/infer-metal/src/kv_ssd.rs)), held resident
under the logical-id chain of the pages it ends on and written through to the
disk tier under the engine's content key.

**The radix stays device-neutral.** `infer-core` matches pages and asks the
backend how many of the matched blocks are complete restore boundaries
(`PrefixReuse::reusable_prefix_blocks`,
[`infer-seam/src/lib.rs:466`](../../crates/infer-seam/src/lib.rs)). The match
is then clamped to that answer. A 487-block page match licensed 0 blocks
until intermediate snapshots were persisted
([wins 2026-08-26](../experience/wins/2026-08-26-metal-kv-disk-content-keyed-restart-cache.md));
the clamp is what makes that a slow turn instead of a wrong one.

**The last token always prefills.** The match is capped at `prompt_len - 1`
([`prefix.rs:114`](../../crates/infer-core/src/prefix.rs)). A full-prompt
match that jumped straight to decode had no forward pass to sample the first
token from; the planner fell back to the prompt's last token as the decode
seed, duplicating its KV and shifting every later position by one
([wins 2026-07-08](../experience/wins/2026-07-08-prefix-cache-wrong-seed-token-fix.md)).
The same cap gives the block drafter its first-block context: a restored
prompt always has a non-empty tail, and the tail's prefill re-seeds the
target hidden state the draft needs
([wins 2026-08-26 DSpark](../experience/wins/2026-08-26-metal-dspark-prefix-reuse.md)).

**Content keys on disk, page identity in memory.** The durable tier addresses
pages and snapshots by a hash of the token prefix, so a restarted server
serves a prior prompt from disk with no page ids in common. The resident
Metal store keys snapshots by the chain of logical page ids, which is cheap to
compute at publish and invalidates correctly when a page is recycled
(`release_pages`, [`kv_ssd.rs:207`](../../crates/infer-metal/src/kv_ssd.rs)).
That split is where the failure below lived.

## The failure

Turn 2 of a multi-turn conversation restored its prefix. Turns 3 through 12
licensed 0 blocks and re-prefilled everything
([wins 2026-09-02](../experience/wins/2026-09-02-metal-prefix-restore-survives-turns.md)).
Two defects, both about identity:

1. **A restored page is the same page.** `publish_slot` minted a new logical
   id for any page whose owner changed. A restored slot republishes the
   radix's pages under a new slot epoch, so every one of them read as
   recycled, and the alias-hazard prune deleted every earlier snapshot
   (`prefixes 5->0` in the debug log). Fix: the slot records how many leading
   tokens it was materialized from (`restored_len`,
   [`slot.rs:17`](../../crates/infer-metal/src/slot.rs)); pages below that
   keep their id ([`kv_ssd.rs:433`](../../crates/infer-metal/src/kv_ssd.rs)).
2. **The radix keeps the original page.** When a slot recomputes a block the
   radix already holds, dedup keeps the radix's page and drops the slot's.
   The slot's snapshots were keyed to a chain the radix would never hand out.
   The seam contract already passed `slot_pages` for exactly this repair
   ([`infer-seam/src/lib.rs:553`](../../crates/infer-seam/src/lib.rs)); the
   Metal implementation had it as `_slot_pages`. Fix:
   `alias_snapshots_to_canonical_chain`
   ([`kv_ssd.rs:338`](../../crates/infer-metal/src/kv_ssd.rs)) re-keys each
   boundary snapshot onto the radix-canonical chain where the two diverge.

Neither defect shows on a two-turn test. Turn 2 exercises restore; only turn
3 exercises republish-after-restore.

## The number

Qwen3.5-0.8B-MLX-4bit, M4 Pro 48 GB, `arle serve --backend metal`, default
flags. 12 turns: a 4.8K-token system prompt, then one tool result of about
350 tokens per turn, 8.6K tokens at turn 12. Greedy, `max_tokens` 32, TTFT to
the first streamed delta, identical request bytes to both servers
(`scripts/bench_multiturn_ttft.py`).

| Arm | Turn 1 | Turns 2–12 median | Turn 12 |
|---|---:|---:|---:|
| Before the fix | 1.61 s | 2.01 s | 2.27 s (turn 8, run stopped) |
| After the fix | 1.95 s | **180 ms** | 202 ms |
| mlx-lm 0.31.2, `--prompt-cache-size 4`, same weights | 1.26 s | 249 ms | 248 ms |

Correctness: greedy output of turns 3 and 6 in the restored chain equals the
cold single-prompt output token for token. Needle ladder 115 to 8000 tokens,
three runs per length, 18/18 exact, every length deterministic, 17 restored
attaches. The 35B row is pending a machine without swap pressure
([roadmap Goal 0](../plans/2026-08-24-roadmap.md)).

## What would be done differently

Key the resident snapshots by content from the start. The logical-id chain
was chosen because it is free at publish time and invalidates by construction
when a page is recycled; both defects above are consequences of having two
identities for one block. Content keys already exist for the disk tier
(`infer_seam::prefix_content_keys`), the radix already dedups by content, and
a snapshot keyed by content is valid for as long as any page holding that
content is. The cost is a hash per published page. The re-keying pass and
`restored_len` would both disappear.
