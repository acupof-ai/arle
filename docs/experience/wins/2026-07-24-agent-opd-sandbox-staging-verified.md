# Sandbox staging outside the repo: −9.7K prompt tokens/request, rollouts −40% — but CC's floor is ~21K

## Context

The GO-measurement run found every CC request carried ~31K prompt tokens
because sandboxes staged under the ARLE checkout and `claude -p` ingested the
repo `CLAUDE.md`
([wins](2026-07-24-agent-opd-gpu-busy-frac-measured-go.md)). Fix landed
`6bd40d663`+`b0a29443e` (WORK_ROOT default `/tmp/agent-opd`, warn-once guard)
with acceptance deferred to this pod run.

Run: `sandboxfix-s4`, 1×H20 GPU 0, ThinkingCap-Qwen3.6-27B-FP8,
`SMOKE=1 SPEC=off SAMPLES=4 SERVE_PORT=8001`, private tree
`/host/aopd-sandboxfix/` (shared `/host/arle-build` was being mutated by a
concurrent lane mid-run — script reverted + untracked sweep; results at
`/host/aopd-sandboxfix/runs/agent-opd-sandboxfix-s4/`). RUN_EXIT=0; hard gate
pass (completion_tokens > 0 all groups, 0 teardowns, 42/44 CC exit 0 — one
base-eval fail, one cc-timeout, both session-level).

## What Worked

- **Staging verified**: sandboxes at `/tmp/agent-opd/sandboxfix-s4/work/`;
  0 of 185 request dumps carry repo-contract markers; 0 `[sandbox] WARN`.
- **prompt_tokens: 30.9–31.1K → min 21.1 / median 21.4 / max 22.3K** —
  exactly the −9.7K CLAUDE.md payload removed.
- **The "few K" expectation was WRONG**: CC's intrinsic floor is ~21K
  (system prompt + tool schemas + conversation; dump `system` field is only
  6.4K chars — the bulk is tools). Consequence inverted lever ranking again:
  the ~21K floor is *identical across all requests*, so **hybrid prefix reuse
  is back to being the top residual lever**, not a small win.
- **Rollout wall −40%**: 204–256 s/group vs baseline ~410 s (excl. one 600 s
  cc-timeout group). Turn-1 median 139.6 s (baseline 178–245 s).
- **KV**: 8 × 21K ≈ 168K < 250K pool — SAMPLES=8 fits again (baseline
  8 × 31K ≈ 248K exhausted it).
- gpu_busy_frac 0.28–0.44 (baseline 0.30–0.34): busy secs fell 135 → 58–113
  with the wall. Idle-bound verdict (mega-rollout GO) unchanged. Caveat: with
  `--staleness 1` group busy-windows overlap, so per-group frac attribution
  is skewed (the timeout group read 0.0 busy despite 366 completion tokens);
  within-run comparison only.
- Rewards: 19/20 train samples pass; evals base 7/8, round-0 8/8, round-1
  8/8. Plot-guard (`031c8c3f8`) engaged: exit 0 with the plot script absent.

## Rule

Stage CC sandboxes outside any repo tree (now the script default; the
warn-once guard trips otherwise). Don't estimate a prompt floor from one
visible component — measure the dump: CC's own preamble is ~21K before any
CLAUDE.md. Shared-prefix work sizes against that floor. On the shared pod
tree, run acceptance from a private copy when another lane is active — the
shared tree mutated three times mid-run here.
