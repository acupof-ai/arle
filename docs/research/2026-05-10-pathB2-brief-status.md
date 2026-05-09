---
title: #37 Path B.2 brief status — codex picked up bucketing fix
date: 2026-05-10
type: research
status: codex-pickup-active
---

# #37 Path B.2 — codex picked up bucketing fix

## Context

Path B v1 KILLED at Tier 4(`a7a8b94`)— 100% cache miss at 4k production
context despite small-shape smoke success。Substrate work(`2f567b9`)is
correct,但 capture key still includes per-request varying allocation-size
fields(`page_indices_len`,`prefix_token_rows_len`)that defeat reuse at
production scale。

Codex IDLE post #37 Path B v1 commit("Worked 49m 41s")。Codex 关键
acknowledgments:
- "Codex review 已按你要求中止"(stopped review per Claude nudge — cooperative pattern works)
- "完整 greedy_consistency 仍有既有 W4A8-vs-BF16 accuracy gate,不是本次 Path B 回归"(pre-existing,not regression)
- "当前 workspace 只剩一个未跟踪文件,我没动:`docs/research/2026-05-10-codex-bounded-review-stuck-pattern.md`"(认可 Claude file discipline)
- "吞吐 license bench 现在可以跑 ./scripts/post_p24_commit_pipeline.sh full"(suggested next step,which Claude already executed)

## Brief delivered

Per `feedback_codex_idle_push_immediately`,Claude immediately briefed
codex on Tier 4 KILL findings + Path B.2 bucketing fix recommendation
via tmux paste-buffer。

Brief content(per `/tmp/codex-brief-pathB2.txt`):
1. Bench A baseline(1631.5ms,matches codex 1639.3 within 0.5%)
2. Bench B Tier 4 KILL evidence(388 captures for 388 requests = 100% miss)
3. Substrate preserve(8-key LRU + device tensors + kv_last_page_len fix)
4. Path B.2 fix:50-100 LOC,round `page_indices_len` / `prefix_token_rows_len`
   to fixed thresholds(ceil/64,ceil/128)
5. Predicted 5-10 buckets in production 4k,8-key LRU covers most,
   expected 80%+ reuse → TTFT Δ +10-25%
6. License threshold(per template `1168381`):TTFT Δ ≥ +10% σ < 5% n=3
7. Implementation pointer:`infer/src/model/qwen3/prefill.rs`
   `Qwen3PrefillGraphKey` struct,round dims at capture key construction,
   keep existing 8-key LRU + device tensor refresh
8. Wins or KILL action paths post-bench

Codex picked up brief("Working 2s")。Task #40 created + marked
in_progress + owner codex。

## Cooperative pattern continuing

| Step | Owner | Commit/Action |
|------|-------|---------------|
| Path A KILL bench | Claude | `e462c53` |
| Path B brief | Claude | `2c43bc7` |
| Path B impl + tests | Codex | `2f567b9` |
| Path B audit chain(7 dims + lifecycle + smoke + final evidence) | Claude | `c2d031c`,`9dd3cbd`,`0198c0d`,`c021053` |
| Codex stuck review pattern audit | Claude | `c560224` |
| Tier 4 KILL bench A/B + errors entry | Claude | `a7a8b94` |
| Path B.2 brief delivery + #40 task | Claude | this commit |
| Path B.2 bucketing fix impl | Codex | (in progress) |

**Cooperative cycle**:plan → impl → audit → bench → KILL → next-iteration
brief → impl → bench → ... 。Knowledge accumulates regardless of license
outcome。Path B v1 KILL surfaced new anti-pattern("smoke success ≠
production-shape success" via growing allocation dim variability)。

## Predicted Path B.2 outcome

If bucketing fix works:
- Capture key produces 5-10 distinct buckets for production 4k workload
- 8-key LRU LRU evicts oldest,80%+ hit rate expected for steady-state
- Launch overhead saved per cache hit reduces TTFT
- **Predicted Δ +10-25%**(close 10-25% of +76.6% SGLang gap)

If bucketing fix fails(2nd Tier 4 KILL):
- Strong evidence "launch overhead is NOT binding constraint" for this
  workload on RTX 4070 Ti SUPER 16GB W4-hybrid prefill
- Path B family abandoned;pivot to:
  - #36 PrefixAwareAdmission(close SGLang multi-tenant 2× gap,different metric)
  - #30 Hybrid W4A16/W4A8 dispatch(quantization scheme split per phase)
  - Architectural axis(continuous batching mods,scheduler overhaul)

## 状态

#37 Path B.2 brief delivered to codex。Task #40 in_progress + owner
codex。Bench infra(release build + bench script + decision tree)proven
via Path B v1 cycle,wall-clock for next license decision should be ≤30
min post-codex-commit。
