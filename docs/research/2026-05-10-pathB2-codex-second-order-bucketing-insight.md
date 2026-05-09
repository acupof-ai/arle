---
title: #40 Path B.2 — codex's second-order bucketing insight beyond Claude brief
date: 2026-05-10
type: research
status: pre-commit-evidence
---

# Path B.2 — codex's second-order bucketing insight(beyond Claude brief)

> Per codex's #40 Path B.2 draft wins entry
> `docs/experience/wins/2026-05-10-bench-40-pathb2-bucketed-prefill-graph-key.md`
> (untracked,7m wall-clock impl + tests + draft)。

## Codex implementation summary

| Field | Bucket size | Match my brief(`d77c5b7`)|
|-------|------------|--------------------------|
| `page_indices_len` | **64-entry** | ✅(my recommendation:64)|
| `prefix_token_rows_len` | **128-row** | ✅(my recommendation:128)|
| `seq_lens: Vec<usize>` | NOT bucketed | ⚠ codex chose stable-per-batch hypothesis |
| `total_tokens` | NOT bucketed | (not flagged in my brief — likely stable for fixed prompt size)|

→ **2 of 3 fields bucketed**(per my audit's 3-field finding)。Codex
implicitly assumed `seq_lens` stable for matched-control c=4 batch
composition,which is correct empirical guess pending bench validation。

## Codex's second-order insight beyond my brief

```
"Bucketed graph keys must also bucket the captured scalar launch
parameters. A cache hit with stale scalar `total_pages` or
`prefix_token_count` is still a semantic miss."
```

This is **second-order bucketing**:
- First-order:bucket the **key tuple fields**(my brief)→ achieves cache reuse
- Second-order:bucket the **captured scalar launch parameters**(codex's catch)
  → ensures replay uses bucket capacity,not first-capture's exact dim
  → prevents semantic miscompute when reused capture sees larger actual data

Without second-order:cache hit but kernel processes only first-capture's
`total_pages = 8`(say)despite needing 16 pages in current request → wrong
output。

→ **Path B.2 needs BOTH key bucketing AND scalar-capture bucketing** to be
fully correct。Codex caught this via greedy_consistency test or impl
review,not from my brief。

**Anti-pattern for skill v1.7.0 catalog**:
- "**Bucketing without scalar capture sync = semantic cache miss disguised
  as functional cache hit**"
- Detection:functional gates(e2e,greedy_consistency)PASS but production
  output silently wrong vs eager baseline
- Fix:every captured scalar derived from bucketed dim must use bucket
  capacity,not exact dim from first capture

## Smoke evidence(per codex draft)

```
Qwen3 prefill graph capture key: tokens=4/3/8/1 batch=1 pages=64 prefix_rows=0 marlin_scratch=false
```

**Pages bucketed to 64** in smoke log evidence(was exact `pages=1` in
Path B v1 logs)。Prefix_rows=0 because smoke prompts had no prefix matches。

For 4k production:
- `pages = ceil(4096/16) = 256` → bucketed to `256` exactly(divisible by 64)
- Plus growth across requests:up to `512`,`768` etc,all bucketed to next 64-multiple
- Prefix matches:typically 0-4096 → bucketed 0,128,256,...,4096 = max **33 buckets**
- Combined:5-10 distinct keys for 4k production(consistent with my prediction)

## Cooperative pattern continues

| # | Step | Owner | Commit |
|---|------|-------|--------|
| 1-9 | (per `db8091d` chain)| | |
| 10 | Path B.2 impl draft +tests | Codex | (pending) |
| 11 | Codex's second-order insight + Path B.2 audit(this entry)| Claude | (this commit)|

Codex impl 7m wall-clock(vs Path B v1 49m,**7× faster**)。Path B.2
total chain may close at ~10 cooperative cycles when bench A/B completes。

## Predicted Path B.2 bench outcome

If `seq_lens` stable per c=4 batch(codex's implicit assumption):
- 5-10 distinct capture keys for 60s 4k bench(vs Path B v1's 388)
- 8-key LRU covers most → 80%+ reuse
- TTFT 4k/c=4 Δ +10-25%
- **Tier 1 / Tier 2 wins** per decision tree(`25e65bf`)

If `seq_lens` varies(c=4 admission interleaves stages):
- Still > 1 capture per request → 2nd Tier 4 KILL
- → pivot architectural axis

Empirical verification within 30 min wall-clock post codex commit。

## Cross-references

- Codex draft:`docs/experience/wins/2026-05-10-bench-40-pathb2-bucketed-prefill-graph-key.md`(untracked)
- Field source audit:`docs/research/2026-05-10-pathB-key-fields-source-audit.md`(`d77c5b7`)
- Path B.2 brief:`docs/research/2026-05-10-pathB2-brief-status.md`(`341a777`)
- Path B v1 KILL precedent:`docs/experience/errors/2026-05-10-37-pathB-bench-tier4-kill-cache-miss-at-4k.md`(`a7a8b94`)
- Decision tree:`docs/plans/2026-05-10-post-37-license-decision-tree.md`(`25e65bf`)

## 状态

Codex Path B.2 impl draft 完成(7m wall-clock,68 LOC across 2 files,
all functional gates PASS,smoke shows page bucketing 64)。Codex
contributed **second-order bucketing insight**(captured scalar launch
parameters must use bucket capacity,not exact dim)beyond my brief。
Bench A/B verification within 30 min post commit。
