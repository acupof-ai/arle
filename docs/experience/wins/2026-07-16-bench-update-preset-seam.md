# UpdatePreset seam + in-process serve plumbing — bench pending-remote

> Status: pending-remote — CUDA runtime change, no local GPU. Covered by plan
> gates F.1-F.5 of
> 2026-07-16-agent-rl-unified-infra.md
> (P4 pod validation): correct-inference needle gate across re-merge, reward
> parity audit, wall-clock A/B vs the bash loop.

## Context

P0 (`b5f0f406`, serve ownership plumbing + codex P1/P2 lifecycle fixes) and P1
(`a0c7ed9ae` + `99a72ed19`, UpdatePreset: 8 algorithm presets over one update
path) of the agent-RL unified-infra plan. Behavior contract: the three shipped
strategies (rejection-ce / sao-dis / sao-value) byte-identical through their
presets; default CLI unchanged.

P2 orchestrator tranches (`1a74d192d` cc-harness driver, `16d9fdb05` simplify
pass, `a46ab3388` serve + per-group round loop + always-on metrics) ride this
entry: their wall-clock license is the F.3 A/B vs the bash loop; correctness is
F.1 (needle across re-merge) + F.2 (reward parity, py→Rust denominator change
expected and documented there).

## What must be measured (pod, H20)

1. Preset behavior parity: one replay round each for rejection-ce / sao-dis /
   sao-value on a collected records set — losses match pre-refactor values.
2. F.3 wall-clock A/B once P2 (orchestrator) lands — this entry's runtime diff
   is licensed as part of that A/B, not separately.

## Rule

Runtime-change tranches of a multi-phase refactor may share one bench entry,
but the entry must exist from the first tranche (this stub) and be filled by
the phase gate that produces the numbers.
