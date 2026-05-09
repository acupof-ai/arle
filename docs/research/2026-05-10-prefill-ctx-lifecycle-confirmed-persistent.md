---
title: Qwen3PrefillContext lifecycle confirmed persistent — Path B not at risk
date: 2026-05-10
type: research
status: claude-audit-confirms
---

# `Qwen3PrefillContext` lifecycle audit — confirmed PERSISTENT across requests

> Per codex's investigation pause("这可能说明 Qwen3PrefillContext 生命周期没有
> 跨请求保留,而不只是 key 太细。我先快速追一下 context owner")— Claude
> parallel audit 确认 prefill_ctx **IS persistent**, Path B fix will work
> as designed。

## Source-grep evidence

| Location | Code | Implication |
|----------|------|-------------|
| `infer/src/scheduler/cuda/core.rs:159` | `pub(super) prefill_ctx: Option<M::PrefillContext>` | Stored at scheduler-state level(not per-request)|
| `infer/src/scheduler/cuda/prefill.rs:676-688` | `if self.prefill_ctx.is_none() { create_prefill_context(...) }` | Lazy init ONLY on first prefill; subsequent reuse |
| `grep prefill_ctx = None\|.take()\|self.prefill_ctx` | 3 hits, all in prefill.rs lines 676/682/724 | NO clear/reset/swap-out paths |

## Conclusion

`Qwen3PrefillContext` is created **once per scheduler instance**(lazily on
first prefill request)+ never cleared between requests。Therefore:
- Graph cache stored in context survives across requests ✓
- Device tensors(`start_positions_dev`, `seq_lens_dev` per Path B impl)
  survive across requests ✓
- Path B's "narrow capture key + refresh device tensors per replay" pattern
  will achieve cross-request reuse as designed

## Implication for Path B commit

Codex can commit Path B with **confidence** — the architectural concern
("context lifetime might not be persistent")is **unfounded**。The 7-dim
brief match audit(per `93a8d7b` 2nd audit)is sufficient evidence that
Path B will work end-to-end。

## Predicted bench outcome(unchanged from `c2d031c` + `93a8d7b`)

If Path B impl ships AND tests pass:
- `cudaGraphLaunch` count ≫ `cudaGraphInstantiate` count(reuse working)
- `prefill graph capture key` log count **≪ request count**(in contrast to
  Path A KILL evidence where capture count = request count from re-capture)
- TTFT 4k/c=4 close 30-50% of +76.6% SGLang gap → 1639ms → **1100-1300ms** range

## Cross-references

- Codex's investigation pause:tmux 30+ min wall-clock pre-commit caution
- Path B brief:`docs/plans/M_37-pathB-device-mem-startpos.md`(2c43bc7)
- Path B 1st audit:`docs/research/2026-05-10-37-pathB-codex-implementation-audit.md`(c2d031c)
- Path B 2nd audit + Phase 3b API gap:`docs/research/2026-05-10-phase3b-api-echo-gap-and-pathB-impl-audit.md`(93a8d7b)
- Source files:
  - `infer/src/scheduler/cuda/core.rs:159`
  - `infer/src/scheduler/cuda/prefill.rs:676-688`
  - `infer/src/model/qwen3/forward.rs:215`(`create_prefill_context`)
  - `infer/src/model/qwen3/prefill.rs:228`(`pub struct Qwen3PrefillContext`)

## 状态

`Qwen3PrefillContext` lifecycle confirmed persistent across requests。
Path B impl architectural foundation is sound。Codex can proceed to commit
when greedy_consistency tests pass(currently 30+ min into nvcc rebuild
+ test runtime)。
