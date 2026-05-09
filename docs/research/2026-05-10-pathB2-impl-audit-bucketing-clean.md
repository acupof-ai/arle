---
title: #37 Path B.2 codex impl audit — bucketing clean, awaiting commit
date: 2026-05-10
type: research
status: pre-commit-tracking
---

# Path B.2 codex impl audit — bucketing approach clean

> Per Path B.2 brief(`341a777`)+ field audit(`d77c5b7`),codex picked
> up #40 and is implementing the bucketing fix。This audit captures the
> WIP impl approach pre-commit。

## Codex impl evidence(WIP diff,prefill.rs)

```rust
// Helper added: idiomatic round-up to bucket boundary
fn round_nonzero_up(value: usize, bucket: usize) -> usize {
    debug_assert!(bucket > 0);
    value.div_ceil(bucket) * bucket
}

// Capture key construction: bucketed dims
page_indices_len: round_nonzero_up(layout.page_indices.len(), <bucket>),
prefix_token_rows_len: round_nonzero_up(layout.prefix_token_rows.len(), <bucket>),
```

(`<bucket>` value not in grep snippet — likely 64 / 128 per Claude
recommendation。)

## Padding to bucket capacity

```rust
// Over-allocate to bucket size, fill rest with zeros
let mut page_indices = vec![0i32; resources.key.page_indices_len.max(1)];
page_indices[..layout.page_indices.len()].copy_from_slice(&layout.page_indices);
ctx.memcpy_htod(&page_indices, &mut resources.page_indices_dev)?;

// Symmetric for prefix_token_rows
let prefix_rows = if resources.key.prefix_token_rows_len == 0 {
    ...
} else {
    let mut rows = vec![0i32; resources.key.prefix_token_rows_len];
    rows[..layout.prefix_token_rows.len()].copy_from_slice(&layout.prefix_token_rows);
    rows
};

// Capacity assertions catch bucket undersizing
debug_assert!(layout.page_indices.len() <= resources.key.page_indices_len);
debug_assert!(layout.prefix_token_rows.len() <= resources.key.prefix_token_rows_len);
```

→ **Clean impl approach**:bucket size dictates allocation,actual data
padded with zeros,assertions catch undersizing。Captured graph reuses
the buffer without re-capture as long as bucket dim stays constant。

## Coverage assessment vs Claude audit(`d77c5b7`)

| Varying field | My audit recommendation | Codex impl | Match |
|---------------|------------------------|-----------|-------|
| `page_indices_len`(KV pool growth)| bucket 64 | `round_nonzero_up(_, <bucket>)` | ✅ |
| `prefix_token_rows_len`(prefix admission)| bucket 128 | `round_nonzero_up(_, <bucket>)` | ✅ |
| `seq_lens: Vec<usize>`(per-chunk vec)| bucket vec collapse | **NOT VISIBLE** in grep — codex may treat as stable for matched-control c=4 batch | ⚠ |

**Concern**:if `seq_lens` vec varies per batch composition(e.g. mixed
prefill stages from prefix-aware admission interleaving),Path B.2 still
fails at production。Need to verify post-commit by checking actual capture
key count vs request count。

## Status pre-commit

- Codex Working 4m 41s on Path B.2(vs Path B v1 49m 41s — **10× faster cycle**!)
- WIP narrowed to **2 files**(prefill.rs + ops/attention.rs,vs v1 6 files)
- Clippy `-D warnings` PASSED
- Currently running graph-on smoke test(`INFER_PREFILL_GRAPH=1 INFER_HYBRID_W4A8_PREFILL=1`)
- No errors visible in tmux log

## Predicted post-commit bench outcome

If `seq_lens` is stable per c=4 batch(common case for matched-control):
- Cache hit rate >> 80% expected
- TTFT 4k/c=4 Δ +10-25% per #37 license criteria
- Tier 1 / Tier 2 wins outcome per decision tree(`25e65bf`)

If `seq_lens` varies per request:
- Cache hit rate may still be < 50%
- 2nd Tier 4 KILL → pivot to architectural axis

Bench will reveal within 30 min post codex commit landing per pre-built
pipeline(`scripts/post_p24_commit_pipeline.sh full`)。

## Cooperative pattern continues(8 cycles this loop on #37/#40 axis)

| # | Step | Owner | Commit |
|---|------|-------|--------|
| 1 | Path A KILL bench | Claude | `e462c53` |
| 2 | Path B brief | Claude | `2c43bc7` |
| 3 | Path B impl + tests | Codex | `2f567b9` |
| 4 | Path B audit chain | Claude | `c2d031c`/`9dd3cbd`/`0198c0d`/`c021053` |
| 5 | Stuck pattern audit | Claude | `c560224` |
| 6 | Tier 4 KILL bench | Claude | `a7a8b94` |
| 7 | Path B.2 brief | Claude | `341a777` |
| 8 | Field source audit | Claude | `d77c5b7` |
| 9 | Path B.2 impl(in progress)| Codex | (pending) |
| 10 | Path B.2 audit(this entry)| Claude | (this commit) |

**Knowledge accumulation continues regardless of license outcome**。
Path B v1 KILL surfaced 3-field varying;Path B.2 addresses 2 of 3,bench
will reveal if seq_lens is the 3rd missing piece。

## 状态

Codex Path B.2 impl approach clean(idiomatic `round_nonzero_up` +
bucket-padded vecs + capacity assertions)。2-field bucketing(page_indices,
prefix_token_rows)matches my audit recommendation。`seq_lens` treatment
unclear — bench will reveal if it needed bucketing too。Next tick
catches commit + bench outcome within 30 min wall-clock per decision tree。
