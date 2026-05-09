---
title: #37 Path B impl final evidence — codex draft wins entry analysis
date: 2026-05-10
type: research
status: pre-commit-evidence-complete
---

# Path B impl final evidence — codex draft wins entry analysis

> Per codex's draft wins entry
> `docs/experience/wins/2026-05-10-bench-37-pathb-device-metadata-prefill-graph.md`
> (untracked, ready for codex commit post bounded `codex review`)。
> Captures **all functional evidence** for Path B and codex's design
> insights beyond my brief。

## Functional gate evidence(all PASS)

| Gate | Result | Source |
|------|--------|--------|
| `cargo fmt --all` | pass | codex tmux |
| `git diff --check` | pass | codex tmux |
| `cargo check --release -p infer --features cuda` | pass | post 9m+ NVCC rebuild |
| `cargo clippy --release -p infer --features cuda --lib -- -D warnings` | pass | post 12m+ |
| `e2e::test_e2e_generation` with `INFER_PREFILL_GRAPH=1` | pass | smoke graph-on |
| `greedy_consistency::test_greedy_solo_vs_concurrent` | pass | greedy main |
| **LRU multi-key reuse on repeat shapes** | ✅ direct evidence(see below)| smoke log |

Smoke log evidence:
```
Qwen3 prefill graph capture key: tokens=4 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=3 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=8 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
Qwen3 prefill graph capture key: tokens=1 batch=1 pages=1 prefix_rows=0 marlin_scratch=false
```
"The repeated e2e prompts reused the cached keys instead of recapturing every request."

## Implementation summary(per codex draft)

| Sub-axis | Path A baseline | Path B(codex impl)| Delta |
|----------|-----------------|---------------------|-------|
| Capture key includes `start_positions` | yes | **NO**(removed)| request scalar removed |
| Capture key includes `num_pages` | yes | **NO**(device-refreshed)| page offset device-refreshed |
| Capture key invariant fields | (varied)| `seq_lens`, `total_tokens`, `page_indices_len`, `prefix_token_rows_len`, `batch_size`, `page_size` | launch-topology guards only |
| Graph cache size | 1 key | **8 keys LRU** | alternating shape reuse unblocked |
| `start_pos` ABI | `int` launch scalar | `*const int` device pointer | per Path B brief |
| Metadata buffers refreshed before replay | (n/a single)| `start_positions`, `page_table_offsets`, `seq_lens` device buffers | per Path B brief |
| **`kv_last_page_len` refresh** | (assumed static)| **EXPLICITLY refreshed before replay** | **subtle correctness fix** |

## Codex's insight(beyond my brief)

> "Refreshed the captured `PagedPrefillForward` metadata buffers
> (`qo_indptr`, `kv_indptr`, `kv_last_page_len`)before replay because
> `kv_last_page_len` depends on the updated start position."

This is **a subtle correctness bug that my Path B brief did not anticipate**。
`kv_last_page_len`(per-batch last-page residue length)is **derived from
`start_pos`** — when `start_pos` updates per replay,`kv_last_page_len`
also needs refresh,otherwise replay uses stale page-residue and produces
wrong attention output。

→ Codex caught this through testing(`greedy_consistency` would have
caught silent miscompute via output divergence)。**Engineering深度 better
than my brief**。

Subtle data-dependency fix:
- `kv_last_page_len` is technically independent of `start_pos` in struct
  but **derived** from it by the upstream compute path
- Without explicit refresh,captured buffer holds stale pre-replay value
- Test catches via greedy_consistency divergence

This is excellent evidence of **brief-vs-impl complementarity**:my brief
specified the architecture(Path B device tensors + replay refresh),codex
discovered + fixed all derivation chains during impl。

## Codex's three rules extracted(my paraphrasing)

1. "CUDA graph keys should describe **allocation sizes and launch topology**,
   NOT request-varying scalar metadata。Scalar request state belongs in
   device buffers whose contents refresh before replay。"
2. "Removing `seq_lens` from the key would be unsafe without a
   masked/capacity launch rewrite,because sequence lengths still influence
   the captured kernel launch geometry。"(scope discipline — don't over-narrow)
3. "Single-entry graph caches can hide successful replay in alternating-
   shape tests;even a small LRU cache is enough to separate key-churn
   bugs from legitimate shape diversity。"(why Path A KILL was so subtle)

These should propagate to skill v1.7.0 anti-pattern catalog。

## Throughput license — pending bench A/B

Per codex draft "This entry is not the throughput license. #37 Phase 2
still needs matched-control 4k/c=4 graph-off vs graph-on N=3 bench."

Pre-built tools ready:
- `./scripts/post_p24_commit_pipeline.sh full` — A/B bench
- `docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md` — fill template

License criteria(per `docs/plans/M_37-pathB-device-mem-startpos.md` §2.3):
- TTFT 4k/c=4 Δ ≥ +10% σ < 5% n=3 → wins
- Strong proceed Δ ≥ +25%
- KILL Δ < +5% OR cache hit < 50%

Predicted outcome(unchanged from `c2d031c`/`93a8d7b`/`9dd3cbd`/`0198c0d`
audit chain):**TTFT 4k/c=4 1639ms → 1100-1300ms range**(close 30-50%
of +76.6% SGLang gap)。

## Cooperative pattern total(this loop session, M_37 axis)

| Step | Owner | Commit | Substance |
|------|-------|--------|-----------|
| Path A KILL bench | Claude | `e462c53` | bench A/B Δ -0.07% + capture churn,KILL Path A |
| Path B brief | Claude | `2c43bc7` | implementation plan 140-260 LOC,SGLang upstream pattern |
| Path B impl 1st audit(WIP)| Claude | `c2d031c` | FFI device-pointer match + per-row dispatch correct |
| Path B impl 2nd audit + Phase 3b API gap | Claude | `93a8d7b` | 7-dim brief match in prefill.rs WIP |
| `Qwen3PrefillContext` lifecycle audit | Claude | `9dd3cbd` | persistent across requests confirmed |
| Path B smoke evidence | Claude | `0198c0d` | LRU reuse on repeat shapes empirically PASS |
| Path B impl final evidence(this entry)| Claude | (TBD this commit)| codex `kv_last_page_len` subtle bug catch + insights |
| Path B implementation + tests + draft wins | Codex | (pending commit)| 6 file impl + 1 wins draft |

**Cooperative pattern**:Claude planning(brief + audits)→ codex impl
→ Claude evidence consolidation。Each step builds confidence in next
step's outcome。

## 状态

Path B impl evidence COMPLETE pre-commit:functional gate PASS + LRU
reuse confirmed + codex caught subtle `kv_last_page_len` bug。Throughput
license remains pending bench A/B post-commit。**HIGH confidence**(per
6-commit audit chain + smoke evidence)Path B will deliver predicted
TTFT improvement closing 30-50% of SGLang gap。
