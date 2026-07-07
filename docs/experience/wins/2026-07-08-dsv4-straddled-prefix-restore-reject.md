# DSv4 Straddled Prefix-Restore — Reject Instead of Unsafe Truncate

> Status: Shipped
> Date: 2026-07-08
> Env: 4×H20 (GPUs 2/3/4/5), TP4, DeepSeek-V4-Flash-FP8, prefix cache ON (default)

## Context

Closes suspect #2 from
[`docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md`](../errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md),
"Full persistent-buffer enumeration audit": `Dsv4SlotState::truncate`
(`crates/infer-cuda/src/executor.rs`, the only call site inside
`restore_cached_prefix`, DSv4's position-0 prefix-cache restore path) restored
a stored image at its own captured length (`image_len`) and then truncated the
bookkeeping counters down to the shorter consensus `matched_len` whenever
`image_len > matched_len` — a genuine multi-turn/agentic scenario (a stored
snapshot covers more than the current request's matched prefix). This is a
DIFFERENT bug from the same-day
[prefix-cache-wrong-seed-token-fix](2026-07-08-prefix-cache-wrong-seed-token-fix.md)
(that one was `image_len == matched_len`, an exact-match repeat-prompt corrupt;
this one is `image_len > matched_len`, a straddled partial restore).

## Root Cause

`truncate_decode_len` (`crates/infer-cuda/src/attention/dsa.rs`) only clamps
two counters (`compressed.seq_len`, `dsa_official.packed_rows`) — it never
touches the compressor/indexer's `pending_kv`/`pending_score`/
`prev_overlap_kv`/`prev_overlap_score`, or any layer's `sw_window_cache`, all
of which `swap_in_image` (called immediately before) just set to content
reflecting the LONGER `image_len` history.

Traced both consumers at the kernel level
(`crates/cuda-kernels/csrc/misc/dsv4_attention.cu`):

- `dsv4_compressor_update_body` reads `pending_kv`/`prev_overlap_kv` for
  historical positions on the assumption they hold the block immediately
  preceding the current position. `prev_overlap` specifically is a
  **single-slot "most recently completed block" register** — it has no second
  copy of the block before it, so once ANY compress-ratio block boundary is
  crossed between `matched_len` and `image_len`, there is no way to
  re-derive its true content without a real from-position-0 recompute (the
  raw per-token content that fed it was a transient forward-pass projection,
  never persisted anywhere else).
- `dsv4_swa_key_value` reads `sw_window_cache[pos % sliding_window]` for
  historical positions. The ring is restored reflecting `[image_len -
  sliding_window, image_len)`; the truncated position needs `[matched_len -
  sliding_window, matched_len)`. For any `d = image_len - matched_len > 0`,
  exactly `min(d, sliding_window)` ring slots hold content from a position
  that is now itself truncated away, not the true earlier position at that
  residue.

Both are architecturally the same class of defect as the already-fixed
2026-06-06 DSv4 EAGLE rollback bug (CLAUDE.md §0.1's own anchor) — a
carry/ring buffer's CONTENT not rewound to match a counter that was truncated
— but bit-correct repair here (unlike EAGLE's rollback, which has a
just-captured, position-exact snapshot to restore from) requires data that is
provably not available: `prev_overlap`'s "block before" and the SW ring's
evicted residues were never captured at any position other than `image_len`.

## Fix

`crates/infer-cuda/src/executor.rs`, `restore_cached_prefix`: restore ONLY on
an EXACT `image_len == matched_len` match; reject (return `Err`) otherwise.
The caller (`crates/infer-core/src/prefix.rs`, `attach_cached_prefix`) already
treats a restore failure as "free the slot, fall back to full re-prefill" —
an independently-proven-correct path, unaffected by this defect class.
`lookup_covering`'s own "prefer the longest covering entry" selection (needed
for TP-consensus: a peer rank may only have a longer entry) is left untouched;
rejecting one level up keeps that logic simple and just refuses to consume an
unsafe result.

`Dsv4SlotState::truncate`/`truncate_decode_len` are NOT deleted — they have a
second, safe call site (`crates/infer-cuda/src/executor/spec_decode.rs`, MTP
draft-reject rollback), which is always immediately paired with
`restore_spec_ring_tail` (a just-captured, position-exact ring/FP8-slot
snapshot taken right before the draft ran) — a mechanism that closes exactly
this gap for that one narrow, small-depth case. That snapshot-based repair
does not generalize to a stored prefix image, which can be arbitrarily far
ahead of the requested `matched_len` and was never captured at that position.

## Verification (H20 pod, 2026-07-08)

Build: `cargo build --release --features cuda,nccl --bin arle` — `BUILD_EXIT=0`.
TP=4, GPUs 2/3/4/5, `ARLE_DSV4_MOE_BACKEND=allreduce ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_EXPERT_BACKEND=deepgemm`, `--max-total-tokens 2048`.

**Mechanism confirmation (not just outcome).** A temporary, diagnostic-only
env hook (`ARLE_DSV4_FORCE_STRADDLE=<N>`, reverted after use) forced a deeper
straddle than any single-rank scenario naturally produces (under TP=1,
`matched_len` is always an exact-available length except the existing
same-day "-1 shim" for a full-prompt match — see the wrong-seed-token fix).
With `N=150` (crossing many `compress_ratio=4` blocks and the
`sliding_window=128` boundary on this checkpoint), re-sending
`trace_probe.py`'s TRACKED prompt a 2nd time:

- **Before this fix** (executor.rs reverted, hook still active): `truncate()`
  ran (confirmed via `/v1/stats` `prefix_cache.hits` incrementing and
  `hit_tokens` matching the forced `matched_len` exactly), output stayed
  correct on 6/6 trials for this specific query — an honest empirical null
  result. The data hazard is real and kernel-level-proven (the wrong bytes
  ARE read), but does not reliably flip this easy needle-recall task's
  argmax; per this doc's own case-as-fact discipline, that does not license
  trusting the read is safe in general (case-as-fact prevented over-claiming
  "corrupts every time," it did not license under-claiming the defect away).
- **After this fix**: `restore_cached_prefix` rejects every time (`/v1/stats`
  `kv_system.fallback_recompute` incremented once per straddled call,
  `prefix_cache.hits` stayed 0), falling back to full re-prefill. Output
  correct on 6/6 trials — via the independently-proven-safe path, not the
  removed one.

**Regression check 1 — exact-match repeat-prompt (must stay 100% correct,
same repro as the wrong-seed-token fix).** `trace_probe.py` solo (n=1), 12
reps of the byte-identical TRACKED prompt, clean production boot (no
diagnostic env vars): **12/12 exact `738291`.** Mechanism: the existing "-1
shim" (full-prompt match trims one token) makes every repeat call ALSO an
`image_len != matched_len` case (`image_len - matched_len == 1`), so this fix
now also routes it through reject+full-reprefill — `kv_system.fallback_recompute`
went from 0 to 11 across the 11 repeats, `prefix_cache.hits` stayed 0 (was
previously non-zero, since the pre-fix code accepted this 1-token straddle via
`truncate()`). Cost: this ONE narrow scenario (byte-identical full-prompt
resend, no new content) loses its prefix-cache reuse benefit and always pays
full re-prefill; a genuine multi-turn continuation (new content appended
beyond the cached image) is unaffected — `cached_prefix_match_len`'s own
invariant (`PrefixIndex::match_len`, `crates/infer-cuda/src/executor.rs`)
guarantees `matched_len` is always an exact-available stored length under
single-rank operation, so no straddle exists to reject there.

**Regression check 2 — DeepSeek-V4-Flash-FP8 TP=4 boot.** Clean:
`[multiproc-coord] all 4 worker engines ready; opening HTTP`, no admission
errors.

All diagnostic instrumentation (`ARLE_DSV4_FORCE_STRADDLE` hook in
`crates/infer-core/src/prefix.rs`) reverted after use — `git diff` clean on
both local and pod trees, confirmed via `scripts/pod.sh sync`.

## Rule

- A ring/carry buffer with no absolute-position tag (a single-slot "last
  completed block" register, a modulo-indexed window) cannot be correctly
  rewound by adjusting a LENGTH COUNTER alone — the buffer's own CONTENT must
  either be re-derived from a source that still has the true data, or the
  reuse must be rejected. When re-derivation needs a transient forward-pass
  value that was never persisted beyond the buffer itself (the DSv4 compressor
  score, or the fresh K/V projection feeding `sw_window_cache`), there is no
  safe partial-reuse path — accept exact matches only, and cost the miss as a
  full re-prefill.
- The MTP `capture_spec_rings`/`restore_spec_ring_tail` pattern (snapshot the
  one ring slot right before a small, known-depth perturbation, restore it
  after) is a genuine fix for a SMALL, JUST-CAPTURED rollback — it does not
  generalize to an arbitrarily-old stored image, which was never captured at
  the position being rolled back to.
- An empirical null result (no observed corruption at a given straddle depth,
  on one easy query) does not license trusting a kernel-level-proven wrong
  data read as safe — case-as-fact means decoding the actual failure signal
  when one exists, not treating its absence on one test as proof of
  correctness (`docs/experience/errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`
  is the inverse lesson: don't over-claim either; the derivation itself is
  the evidence here, not the one test's silence).
