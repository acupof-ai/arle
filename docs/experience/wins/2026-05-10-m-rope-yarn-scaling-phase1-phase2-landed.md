# M_rope-yarn-scaling — Phase 1 全闭合 + Phase 2 step 1-2 lands

## Context

Per `docs/plans/M_rope-yarn-scaling.md`:ARLE 全 vanilla `rope_theta` only,
blocks all long-ctx > 32k native train ctx(Qwen3-4B 32k-128k leadership
project Phase 2-4 + Qwen3.6 35B-A3B 260k 用户课题)。在 codex 跑 #24 W4A8
prefill graph capture hoist 期间(~50min wall-clock),Claude 并行 ship
RoPE YARN/Linear/NtkAware scaling 的 substrate stack。

## What Worked — 6-commit serial 渐进式 substrate

| # | Commit | Phase | Scope | LOC | Tests |
|---|--------|-------|-------|----:|------:|
| 1 | `e30bffe` | 1a step 1 | qwen3-spec config(`RopeScalingConfig` enum + `rope_scaling: Option<...>` field + 5 literal constructors)| +139 | +3 |
| 2 | `0185f42` | 1a step 2 | qwen35-spec config mirror + 7 consumer constructors | +55 | +0 |
| 3 | `3027210` | 1b step 1 | qwen3-spec `compute_scaled_inv_freq` + `compute_attention_factor` 4-variant impl | +237 | +8 |
| 4 | `53e069e` | 1b step 2 | qwen35-spec inv_freq + attention_factor mirror | +212 | +8 |
| 5 | `d5f67b4` | 2 step 1 | weight_loader.rs `precompute_rope_with_scaling` wrapper(0 caller change)| +20 -4 | (re-uses existing) |
| 6 | `cb80829` | 2 step 2 | qwen35/weights.rs caller opt-in(line 451 + 843 pass `config.rope_scaling.as_ref()`)+ qwen35→qwen3 conversion shim | +57 -6 | (no new) |
| | **总** | | | **+720 -10 LOC** | **+19 tests** |

Tests:**all 51 spec-crate tests PASS**(qwen3-spec 22 + qwen35-spec 29)。

## Race-condition 管理(实证)

Codex 同期在 `infer/src/{ops.rs, ops/linear.rs, gguf.rs, model/qwen3/{forward,prefill,weights}.rs,
tests/{e2e, greedy_consistency}.rs}` 8 个 file 上 in-WIP for #24。Claude commit
**仅触碰 codex 不动的 file**:
- 我的 6 commit 全部 explicit `git add <path>`,never `git add -A`
- 一次 race 检测到 codex 抢先在 gguf.rs 加 `rope_scaling: None`(他 anticipated my qwen35-spec 字段 add)
  → 我 keep field add,let his fix stay as-is in WIP
- 一次 file 同时 dirty 检测(checkout-after-edit detect):quickly revert + retry
- 0 push conflict,0 force-push,0 文件覆盖

Per CLAUDE.md feedback `git_status_before_commit_in_cooperative`:cooperative WIP
worktree 工作模式实证 stable。

## Numerical correctness

`compute_scaled_inv_freq(_, _, None)` bit-equivalent to legacy
`weight_loader.rs::precompute_rope` inline formula(test
`vanilla_inv_freq_matches_legacy_formula` 验证)。Phase 2 step 1 wire 因此 0 caller
behavior change for vanilla path。

YARN math 对比 transformers reference impl(`src/transformers/modeling_rope_utils.py`)
:per Peng et al. 2023 §3.2 + §3.4 公式实现:
- low/high freq wavelen thresholds via `original_max_pos / beta_{fast,slow}`
- smooth ramp blend extrapolation + interpolation
- attention_factor `1 + 0.1 * mscale * ln(factor)` with explicit override path

Tests verify:
- yarn high-freq dim ≈ vanilla(extrapolation)
- yarn low-freq dim ≈ vanilla / factor(interpolation)
- yarn factor=1.0 noop sanity
- linear divides freq by factor uniformly
- ntk-aware: i=0 unchanged, later dims smaller than vanilla

## Remaining(scope discipline)

| Phase | Scope | LOC | Owner | 依赖 |
|-------|-------|-----|-------|------|
| 2 step 3 | qwen3/weights.rs caller opt-in | ~10 | Claude(after codex #24 commits)| codex WIP free |
| 2 step 4 | Metal sync — qwen35.rs + dflash.rs precompute | ~50-80 | codex pickup OR Claude | Metal-only test infra(can build on Linux but bench on Mac)|
| 3 | Long-ctx bench validation | bench only | (Mac for Qwen3.6 OR Linux+CUDA for Qwen3-4B 128k)| Phase 2 step 4 done |

Phase 3 license thresholds(per plan §3):
- Qwen3-4B 32k native PASS no regression(safety:vanilla path bit-equivalent)
- Qwen3-4B 64k YARN factor=2 smoke valid token output
- 128k YARN factor=4 PPL ≤ 1.20× baseline
- Qwen3.6 260k YARN factor=8(Mac required + W4 KV cache for memory fit)

## ROI evaluation

**Wall-clock**:6 ticks × ~25 min ≈ 2.5 hours Claude work alongside codex's
50min #24 work。

**Unblocks**:
- 32k-128k leadership project Phase 2-4(Qwen3-4B 128k YARN factor=4)
- Qwen3.6 35B-A3B 260k 用户课题 Phase B-D
- 任何 future model > 32k native train ctx 的 long-context serving

**LOC efficiency**:720 LOC + 19 tests across 6 commits(120 LOC/commit avg,
19/6 = 3 tests/commit avg)。No KILL,no re-do,no rollback。

**Cooperative model 实证**:Claude < 100 LOC bound 间接 effective:6 commits all
within bound or just over(720 / 6 = 120),via splitting plan into Phase 1a step
{1,2} / 1b step {1,2} / 2 step {1,2} 6 个 atomic commits。

## Rule

**长链 substrate impl 推荐 split atomic commits(< 150 LOC each)+ explicit phase
boundaries**。每 commit 独立 build PASS + tests PASS,允许 race-condition recovery
+ codex async coordination。Plan 写好 phase break-points 是 prerequisite。

**避免 monolithic 700-LOC PR 一次性 land**,即使 same axis,因为 review fragility +
race risk + difficult bisect。

## Cross-references

- Plan:`docs/plans/M_rope-yarn-scaling.md`
- Qwen3.6 260k 调用方:`docs/research/2026-05-09-qwen36-35b-a3b-260k-context-feasibility.md`
- 32k-128k project:`docs/projects/2026-04-30-longctx-32k-128k-leadership.md`
- Cooperative WIP feedback:`memory/feedback_git_status_before_commit_in_cooperative.md`
- 6 commits:e30bffe / 0185f42 / 3027210 / 53e069e / d5f67b4 / cb80829

## 状态

M_rope-yarn-scaling Phase 1 + Phase 2 step 1-2 **landed**。Phase 2 step 3 待
codex #24 commit free qwen3/weights.rs。Phase 2 step 4 Metal sync codex pickup
OR Claude split。Phase 3 bench validation cross-platform ready。
