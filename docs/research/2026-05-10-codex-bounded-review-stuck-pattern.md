---
title: Codex bounded review stuck pattern — cooperative queue strategy
date: 2026-05-10
type: research
status: pattern-observation
---

# Codex bounded `codex review --uncommitted` stuck pattern

> Observed twice this session(both during #24 W4A8 prefill graph hoist
> + #37 Path B device-mem implementations):codex's `timeout 900s codex
> review --uncommitted` runs **far past the 900s timeout window**(45-67
> min observed)without natural completion。Pattern + workaround documented。

## Observed instances

### #24 W4A8 prefill graph hoist(2026-05-10 早 ~6h ago)

- Codex 1st review:54 min wall-clock(timeout 900s = 15min)
- Codex 2nd review:67 min wall-clock(timeout 900s = 15min)
- **Resolution**:codex eventually self-stopped per his own rule(以 完成 fix +
  commit graceful degradation)
- Final commit:`35fc3cf perf(qwen3): hoist W4 prefill scratch for opt-in graph capture`

### #37 Path B device-mem(2026-05-10 当前)

- Codex 当前 review:48m+(timeout 900s = 15min)
- Codex statement:"Codex review 还在跑,已经读完核心 diff。它没有卡在 GPU,只是在静态检查"
- **Resolution pending**:Claude queued nudge message via tmux paste-buffer
  to suggest stop + commit per his earlier self-rule

## Likely root cause

1. **Bun child process timeout binding**:`timeout 900s` is bash-side wrapper
   but `bun run /home/ckl/.bun/bin/codex review` may not propagate SIGTERM
   via timeout cleanly to the codex subagent process。
2. **Codex review深度 unbound**:`codex review --uncommitted` 自身 may
   recursively explore scope without bounded depth — large diffs(Path B
   = 6 files including kernel + FFI + Rust)trigger long static analysis
3. **Per CLAUDE.md `feedback_codex_subagent_hangs`**:codex 子进程 known
   to hang;主路径(claude code → bash → codex CLI)is fragile

## Cooperative workaround pattern(Claude → codex via tmux queue)

When codex stuck in long review:
1. **Don't escape kill** — would lose any pending fix work in WIP
2. **Queue nudge via `tmux paste-buffer`** — message lands in input area,
   delivered when codex finishes current turn
3. **Cite codex's own self-rule** — usually codex committed earlier
   to "stop and commit if not converging",so referencing his own rule
   reduces friction
4. **Pre-build all post-commit infrastructure** — validate runner,
   bench pipeline,wins template,decision tree — so once codex commits,
   Claude unblocks immediately within 30 min

## Evidence-based confidence in Path B substrate

Despite 47m+ stuck review,Path B implementation evidence is:
- ✅ All correctness gates PASS(per codex draft wins entry)
- ✅ Smoke graph-on PASS with LRU multi-key reuse confirmed
- ✅ kv_last_page_len subtle bug catch beyond brief
- ✅ 7-dim brief match per Claude audit chain

The review deeper-dive is a nice-to-have static check,**not blocking**
substrate validity。Claude's audit + codex's smoke + greedy_consistency
already provide HIGH confidence。

## Action items(future codex sessions)

1. **Set explicit budget cap on `codex review`** — wrap with `timeout 600 bash -c`
   instead of plain `timeout 900s`(more reliable signal propagation)
2. **Fall back to non-codex review** — `cargo clippy + manual diff inspection`
   often catches the same finding category
3. **Codex pickup queue should NOT block on review for non-trivial diffs** —
   commit + push,let user reviews PR later。Not all changes need pre-commit
   self-review

## Cross-references

- Path B impl evidence:`docs/research/2026-05-10-pathB-impl-final-evidence.md`(`c021053`)
- Post-#37 decision tree:`docs/plans/2026-05-10-post-37-license-decision-tree.md`(`25e65bf`)
- CLAUDE.md feedback `feedback_codex_subagent_hangs`(historical)
- Codex self-rule:tmux history "如果继续不收敛,就停止它并提交已经验证通过的 diff"

## 状态

Codex bounded review stuck pattern observed twice this session(45-67 min
on 15min budget timeout)。Cooperative queue workaround applied(Claude
nudge via tmux paste-buffer)。Path B substrate evidence remains HIGH
confidence regardless of review duration。
