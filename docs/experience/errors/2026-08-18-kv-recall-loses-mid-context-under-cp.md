# --kv-recall is broken three ways on Qwen3.6-27B — 2026-08-18

## Context

Verifying that the L2 (host DRAM) and L3 (SSD) KV tiers work under TP=2 CP=2 with
`--kv-recall`, on ThinkingCap-Qwen3.6-27B-FP8. The tier plumbing works. Selection,
the non-CP decode page table, and multi-row prefill do not.

Builds: `cpkv7` (`2ae47a16`) for the survey, then a build of `5d2ccc2eb` (which
carries the fix in `1bf969aa9`) for the follow-up. Every arm below was run against
a same-binary control.

## Phenomenon

Needle position decides the outcome under CP. 16K prompt, 16 concurrent rows, a
distinct needle per row:

| Needle depth | Region | Recall ON | Recall OFF |
|---|---|---|---|
| ~0% | sink (`n_init` = 32 tok) | 48/48 exact | 48/48 exact |
| 50% | middle | **0/48 exact** | 48/48 exact |
| 99% | local window (`n_local` = 256 tok) | 4/4 exact | — |

Without CP the same flag aborts the process, and after that abort is fixed it
emits `!!!!!!!!` above a length threshold.

| config | recall | depth-50 result |
|---|---|---|
| TP=1 CP=1 | off | PASS at 4K, 8K, 16K |
| TP=1 CP=1 | on | abort (pre-fix) / `!!!!` above the prefill chunk size (post-fix) |
| TP=2 CP=2 | off | PASS 48/48 |
| TP=2 CP=2 | on | 0/48 exact, fluent wrong answers, no abort |

## Bug 1 — the scoring query is never captured under CP

`--kv-recall` with `attn_cp>1` is forced onto the 2D path by its own guard
(`executor/qwen35.rs:1076-1087`), so `build_prefill_geometry` always returns
`Some(cp)` (`:1180, :1224`) and `full_attention_paged` takes the ring branch
(`qwen35_attention.rs:472`). `ring_prefill_full_attention` (`:1205-1215`) has no
`layer0_query` parameter. The only writer of `layer0_query` sits at `:1083`,
inside the `} else {` opened at `:483` and closed at `:1107` — unreachable under
CP.

`layer0_query: Some(Vec::new())` (`executor/qwen35.rs:2188`) therefore arrives at
the scorer as an **empty vec**, not a panic. The guard at `recall.rs:246-247`
(`query_layer0.len() >= num_q_heads * head_dim`) is false, `q` stays all zeros,
and every block scores exactly `0.0` (`recall.rs:280-289`). `plan_recall`'s
tie-break is `partial_cmp(...).then(a.cmp(&b))` (`infer-core/src/recall.rs:84-91`),
so an all-equal vector collapses to ascending index: **middle blocks 0..7,
forever, for every request.**

The arithmetic reproduces the measured residency exactly. For `cache_len =
34,926`, `n_init=32`, `n_local=256`, `l_bs=32`, `top_k=8`, `page_size=16`:
working set `[0,288)` ∪ `[34656, 34926)` → global pages `0..=17` and
`2166..=2182` = 35; block-cyclic CP=2 gives rank 0 the even members, 9 + 9 =
**18 local pages**. Measured: 18. Everything else follows — needle at 0% is in
the pinned sink, at 99% in the local window, at 50% in the 97.1% never selected.

## Bug 2 — recall decode sized FA3 from the full context (fixed)

`PageMeta::for_recall_decode` branched on `shard.size`:

```rust
let (kv_lens, last_page_len, write_kv) = if shard.size > 1 {
    (local_token_count, local_last_fill, i32::from(owns_last))
} else {
    let g = total_len % pool.page_size;
    (total_len, if g == 0 { pool.page_size } else { g }, 1)  // full context
};
```

Without CP it passed `total_len` as `seqlen_k` while handing FA3 a page table
holding only the working set (34 pages / 544 tokens against 19,312). FA3 sizes
its split-KV work from `seqlen_k`, so the combine kernel indexed splits with no
pages behind them: `CUTLASS error
(flash-attention/hopper/flash_fwd_combine_launch_template.h:52): Error Internal`,
deterministic at concurrency 2. The CP branch already derived the length from the
table — which is why CP=2 survived and TP=1 aborted.

Fixed in `1bf969aa9`: a 1-wide shard owns every page, so both branches collapse
to the CP arithmetic. Byte-identical whenever `num_pages == global_pages`, i.e.
every non-recall decode. **Verified**: the abort is gone, the server survives the
same workload.

## Bug 3 — recall corrupts multi-row prefill

With the abort gone, `--kv-recall` at TP=1 returns `!!!!` (token 0, degenerate
logits) above a length threshold. `max_tokens=1` already returns `!`, and the
first token comes out of prefill — so the damage is in prefill, not decode.

Fresh server per data point (one request each, no cross-request state):

| ctx | result |
|---|---|
| 1,803 | correct |
| 2,256 | `!!!` |
| 3,600 | correct (with `--chunked-prefill-size` raised) |
| 7,434 | `!!!` |
| 14,929 | `!!!` |

The threshold is the prefill chunk size. `--chunked-prefill-size` is clamped to
`[128, 4096]` (`loaded.rs:2142`), so a prompt above ~4096 tokens takes two or
more prefill rows; row 2 runs `prefill_row_recall` against a page table where
row 1 already evict-dropped pages. Same binary with recall off is correct at
4K, 8K and 16K, so this is recall-specific and not a property of the build.

Once a long request breaks, later short requests on the same server break too.

**Under 2D this is unreachable**, which is why CP fails politely instead of
loudly: the planner forces the whole prompt into one row —
`let mut chunk = if two_d { remaining } else { … }` (`planner.rs:74, 85-89`).
The same fact explains why the recall tier is write-only in every measurement:
`prefetch_logical_pages` only returns pages already holding `EVICTED_PAGE`
(`recall.rs:396-400`), the prefetch loop runs *before* the evict loop
(`executor/qwen35.rs:2250` vs `:2279`), and with one row per request nothing is
evicted yet when prefetch is computed. `useful_write_bytes` reached 5.4 GB while
`useful_read_bytes` and `reuse_hit_*` stayed at exactly 0.

## Selection quality is a separate, unresolved problem

At the lengths where recall is fully wired *and* single-row (TP=1, ctx < 4096),
the selector still loses the needle about half the time: ctx 1,000 correct,
1,500 wrong, 1,800 wrong, 2,000 correct — at 1,500 tokens it keeps 8 of ~33
middle blocks, roughly a quarter of the context, and still misses. So fixing
Bug 1 is necessary but may not be sufficient.

What the selector is, all code-anchored: one scalar per block, GQA-mean-pooled
over all heads (`recall.rs:248-259`); computed at the **first** full-attention
layer only and applied to every layer (`qwen35_attention.rs:1084`); block
relevance is the dot product with the **arithmetic mean of 32 post-RoPE keys**
(`recall.rs:171-201`) — averaging post-RoPE keys rotates each by a different
angle, so the high-frequency dimensions cancel; the plan is **frozen at prefill**
for the whole generation (`executor/qwen35.rs:2161-2164`); and `top_k=8` is
hardcoded with no CLI knob (`recall.rs:48-55`), keeping 0.74% of the middle at
35K context. Quest and InfLLM select per head, per layer, every decode step,
and use per-channel min/max (an admissible upper bound on `max q·k`) rather than
a mean.

## Rule

A front-of-prompt needle cannot gate `--kv-recall`. Position 0 lands inside the
pinned sink window, so the gate passes without the recall cycle ever retrieving
anything — `needle_concurrent.py` gave 48/48 on a configuration that loses 100%
of mid-context content. Gate recall with the needle in the middle:
`needle_concurrent.py <port> <conc> <tokens> <rounds> 50` (added in `bb4e362ea`).

Second: a tier whose read counters sit at exactly 0 while its write counters
climb is a one-way evict path, and that is visible without any correctness test.

Third: an empty vector is not a missing value. `layer0_query: Some(Vec::new())`
survived `.expect()`, survived a length guard, and produced a plausible-looking
plan. A zero query is never legitimate — `recall.rs:246` must `ensure!` on it.

## Follow-up

Ordered, with the gate for each. Bug 3 comes first because it blocks measurement
of everything else.

1. **Bug 3** — make `prefill_row_recall` correct across rows, or refuse recall
   when a request needs more than one prefill row. Gate: TP=1 CP=1,
   `needle_concurrent.py <port> 8 16000 2 0` must stop returning `!`.
2. **Real Gate 0** — with Bug 3 fixed, TP=1 CP=1 `needle_concurrent.py <port> 16
   16000 3 50`. This measures the selector alone, with no CP confound. 48/48
   means Bug 1 is the whole remaining gap; a low score means the selector needs
   the redesign in the section above and wiring it up buys nothing.
3. **Bug 1** — pass `layer0_query` into `ring_prefill_full_attention`
   (`qwen35_attention.rs:1205-1215`), capture after the q_ring prep, and
   broadcast from the tail cp rank. Note `q_ring` is head-major
   (`ring_prefill.cu:54-55`) while the non-CP `q_prepped` is row-major — copying
   the non-CP indexing verbatim produces garbage. Then all-reduce the block
   scores over `attn_cp` and `attn_tp` so every rank plans identically, sizing
   the score vector from `(cache_len - n_init - n_local) / l_bs` rather than
   `block_reps.len()` (which is not rank-invariant, and would deadlock the
   collective). Gate: TP=2 CP=2 depth-50 48/48, depth-0 unchanged at 48/48,
   plus `needle_gate.py`.
4. **`ensure!` on the empty query** (`recall.rs:246`) and a zero-variance warning
   in `plan_recall` (`infer-core/src/recall.rs:83`). Announce the behaviour
   change: recall at CP>1 errors instead of answering wrongly until (3) lands.

Until (1) and (2), `--kv-recall` should not be treated as usable for retrieval
workloads on any parallelism, CP included.

## Method notes

- `/metrics` under multiproc returns a zeroed snapshot while the engine thread is
  busy — `kv_free_pages` and `active_requests` both read 0 mid-request, and the
  scrape itself blocks ~2.3 s. A 0 is not a reading. Counters read before/after a
  run are unaffected.
- `arle_kv_tier_io_useful_read_bytes_total` is disk-only; a host-L2 hit never
  touches it. It is not a valid "did prefetch fire" signal on its own.
- `--chunked-prefill-size` is clamped to `[128, 4096]`, so it cannot be used to
  force a single-row prefill on a long prompt.
- Tier mechanics under CP are correct and measured: per-rank L3 stores
  (`arle-kv-recall-st-<epoch>-format-1-world-4-rank-{0,1,2,3}-page-524288`),
  1.2 GB spilled per rank × 4 ranks, and L1 residency at 18 local pages for a
  34,926-token context.
