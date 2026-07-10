---
title: #37 Path B capture key field semantics — source audit for B.2 bucketing
date: 2026-05-10
type: research
status: helper-for-codex-pathB2
---

# Path B capture key field semantics — what each field varies on

> Per Path B v1 KILL evidence(`a7a8b94`),capture key still produces 1
> capture per request at 4k production。This audit clarifies which fields
> in `Qwen3PrefillGraphKey`(post `2f567b9`)actually vary per request,
> to inform codex's Path B.2 bucketing fix(`#40`)。

## Current key structure(`infer/src/model/qwen3/prefill.rs:133-153`)

```rust
struct Qwen3PrefillGraphKey {
    total_tokens: usize,           // ← VARIES per request(prompt token count)
    page_size: usize,              // ← FIXED const(16)
    page_indices_len: usize,       // ← VARIES as KV pool grows
    prefix_token_rows_len: usize,  // ← VARIES via prefix-aware admission
    batch_size: usize,             // ← FIXED for fixed concurrency(c=4)
    seq_lens: Vec<usize>,          // ← VARIES per request layout
}
```

## Per-request variability analysis(c=4 4k workload)

| Field | Variability source | 4k bench frequency |
|-------|-------------------|--------------------|
| `total_tokens` | Prompt length per request | **Should be 4096 fixed** for c=4 4k workload(prompt_tokens=4096)|
| `page_size` | Const | Always 16 |
| `page_indices_len` | KV pool fill state(monotonic growth)| **Varies monotonically across requests** |
| `prefix_token_rows_len` | Prefix-aware admission match length per request | **Varies based on prefix cache hits** |
| `batch_size` | Scheduler batch composition | **Fixed at c=4** for matched-control bench |
| `seq_lens: Vec<usize>` | Per-sequence prefill chunk lengths | **Vec content varies per request layout** |

→ **Path B.2 bucketing must address all 3 varying fields**:
1. `page_indices_len`(monotonic growth)
2. `prefix_token_rows_len`(admission-policy-driven)
3. `seq_lens: Vec<usize>`(per-request chunk layout)

## Bucketing recommendation per field

### `page_indices_len`(simplest)

```rust
// Round to 64-page bucket
fn bucket_page_indices_len(n: usize) -> usize {
    (n + 63) / 64 * 64
}
```

For 4k workload:`page_indices` ≈ ceil(4096 / 16) = 256 pages per request,
plus累积 cache pages → typical range 256-2048 → 5-32 buckets。8-key LRU
covers heavy-tail。

### `prefix_token_rows_len`(admission-policy dependent)

```rust
// Round to 128-token bucket
fn bucket_prefix_token_rows(n: usize) -> usize {
    (n + 127) / 128 * 128
}
```

For 4k workload:prefix match length 0-4096 → 32 buckets at 128-bin。
**Higher bucket count risk** — may need 256 or 512 bin for fewer buckets。

### `seq_lens: Vec<usize>`(most complex)

This is a **vector**,not a scalar。Bucketing needs:

```rust
// Hash + bucket the vec by sum + length
fn bucket_seq_lens(v: &[usize]) -> (usize, usize) {
    let total = v.iter().sum::<usize>();
    let bucketed_total = (total + 127) / 128 * 128;
    (v.len(), bucketed_total)
}
```

For c=4 4k:每 request typically `seq_lens = [4096]`(single 4k chunk per
sequence)→ `(1, 4096)` bucket。Should be **stable per request** UNLESS
chunked prefill splits 4097 → [2048, 2048, 1]、 or admission interleaves。

## Risk assessment

**Best case**:`seq_lens` is actually stable for c=4 4k (always
`[4096, 4096, 4096, 4096]` across batch),and bucketing
`page_indices_len` + `prefix_token_rows_len` alone fixes Path B.2 →
80%+ reuse → Δ +10-25% TTFT improvement。

**Worst case**:`seq_lens` varies per request(prefix-aware admission
admits requests at different prefill stages → mixed seq_lens),then
bucketing the vec is needed too(more complex)。

## Empirical signal from Path B v1 KILL bench

bench-output trace summary showed `Plan labels: prefill=774` for 388
requests = **2 prefill plans per request average**。This suggests:
- Either chunked prefill is splitting requests(4096 / 2048 = 2 chunks?)
- OR scheduler is re-running prefill due to graph cache miss + re-capture overhead

If chunked split,`seq_lens` per request is `[2048]` then `[2048]` then `[1]`
(matching codex Phase 0 KILL evidence pattern 2048+2048+1)→ vector
content varies per chunk position even for same prompt → vec bucketing
needed。

## Recommendation for codex Path B.2

```rust
impl Qwen3PrefillGraphKey {
    fn new(layout: &Qwen3PagedPrefillLayout, page_size: usize) -> Self {
        let raw_seq_lens: Vec<usize> = layout.sequences.iter().map(|seq| seq.seq_len).collect();
        let (seq_count, seq_total_bucketed) = bucket_seq_lens(&raw_seq_lens);
        Self {
            total_tokens: bucket_total_tokens(layout.prefill_token_rows.len()),  // round 64?
            page_size,
            page_indices_len: bucket_page_indices_len(layout.page_indices.len()),  // round 64
            prefix_token_rows_len: bucket_prefix_token_rows(layout.prefix_token_rows.len()),  // round 128
            batch_size: layout.sequences.len(),  // fixed for matched-control
            seq_lens: vec![seq_total_bucketed; seq_count],  // collapse vec to (count, bucketed_total)
        }
    }
}
```

Per-key allocation should use **bucketed dim**(over-allocate to bucket
size,not exact dim)to avoid graph re-capture on dim mismatch。

## Predicted bucket count for 4k production workload

| Field | Bin size | Distinct buckets |
|-------|---------:|-----------------:|
| `total_tokens` | 64(if not pure 4096)| 1-2 |
| `page_indices_len` | 64 | 5-10 |
| `prefix_token_rows_len` | 128 | 5-15(depends on prefix cache hit pattern)|
| `seq_lens`(bucketed) | 128 | 1-3 |
| **Combined unique tuples** | | **~10-50 distinct keys** |

**8-key LRU may not be enough** — codex may need to bump LRU size to
**16-32** to cover the combined bucket space。Alternative:tighter bin
sizes(seq_lens 64,prefix_token_rows 256)to reduce bucket count。

## Cross-references

- Path B v1 KILL evidence:`docs/experience/errors/2026-05-10-37-pathB-bench-tier4-kill-cache-miss-at-4k.md`(`a7a8b94`)
- Path B v1 commit:`2f567b9`
- Path B.2 brief:tmux paste-buffer this tick + `docs/research/2026-05-10-pathB2-brief-status.md`(`341a777`)
- Source:`infer/src/model/qwen3/prefill.rs:133-153`(`Qwen3PrefillGraphKey`)

## 状态

Path B v1 capture key has 4 per-request varying fields(total_tokens,
page_indices_len,prefix_token_rows_len,seq_lens vec)。Path B.2 must
bucket all 4 to achieve real cache reuse。Codex picking up #40,this
audit informs implementation choices(bin sizes + LRU size + vec
bucketing strategy)。
