# Verification harness: Metal gate + standalone check + temp arm integrated

2026-08-23 · tooling · harness improvement

## Context

The correctness gate (`needle_gate.py` + `lever_gate.sh`) was CUDA-pod-only,
had no pass/fail exit code on its own, and the temp coherence arm was a
separate invocation. Local Metal development — the primary dev path on Mac —
had no correctness gate at all; the smoke test only checks CLI help text.
The harness existed but was underutilized because it couldn't run locally
and couldn't gate anything without a manual baseline.

## What changed

1. **`needle_gate.py --check`** — standalone gate mode: exits 0 if every
   length has ≥ `--min-exact` (default 1) exact hits, exit 1 otherwise.
   No baseline log required. The absolute threshold is less precise than a
   baseline envelope but removes the baseline-management friction for
   one-off checks.

2. **`lever_gate.sh GATE_PROFILE=metal`** — boots a Metal serve instead of
   CUDA. Default model is the canonical `mlx-community/Qwen3.6-35B-A3B-4bit`;
   override with `MODEL=`. Skips the DSv4 multi-GPU setup entirely.

3. **Temp arm integrated into `lever_gate.sh`** — runs `needle_gate.py temp`
   after the needle ladder. The greedy-only gate misses argmax-preserving
   distortions; the temp arm catches them. Skip with
   `LEVER_GATE_SKIP_TEMP=1`.

## Verified

Metal gate on Qwen3.5-0.8B-MLX-4bit, lengths 115/300/446, 1 run each:

```
len=115 exact=1 partial=0 miss=0
len=300 exact=1 partial=0 miss=0
len=446 exact=1 partial=0 miss=0
TEMP-ARM PASS tokens=200/200 glued=None
[gate] correctness PASS: summaries=3
```

## One-command local gate

```bash
GATE_PROFILE=metal \
  MODEL=models/Qwen3.5-0.8B-MLX-4bit \
  LENGTHS=115,300,446 RUNS=1 \
  LEVER_GATE_ALLOW_NO_BASELINE=1 \
  bash scripts/lever_gate.sh <label>
```

For a standalone check without lever_gate.sh (serve already running):

```bash
PORT=18189 python3 scripts/needle_gate.py 115,300,446 1 --check
```

## Rule

A correctness gate that can't run on the dev machine doesn't gate anything.
The harness must be runnable locally with one command, with a pass/fail exit
code, before it can be wired into any automated workflow.
