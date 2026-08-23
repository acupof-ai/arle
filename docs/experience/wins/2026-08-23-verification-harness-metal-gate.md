# Verification harness: Metal gate, CI needle gate, concurrent arm, bench_compare fix

2026-08-23 · tooling · harness improvement

## Context

The correctness gate (`needle_gate.py` + `lever_gate.sh`) was CUDA-pod-only,
had no pass/fail exit code on its own, and the temp coherence arm was a
separate invocation. Local Metal development — the primary dev path on Mac —
had no correctness gate at all; the smoke test only checks CLI help text.
The harness existed but was underutilized because it couldn't run locally,
couldn't gate anything without a manual baseline, and had no CI coverage.

## What changed

1. **`needle_gate.py --check`** — standalone gate mode: exits 0 if every
   length has ≥ `--min-exact` (default 1) exact hits, exit 1 otherwise.
   No baseline log required.

2. **`lever_gate.sh GATE_PROFILE=metal`** — boots a Metal serve instead of
   CUDA. Default model is the canonical `mlx-community/Qwen3.6-35B-A3B-4bit`;
   override with `MODEL=`. Skips the DSv4 multi-GPU setup entirely.

3. **Temp arm integrated into `lever_gate.sh`** — runs `needle_gate.py temp`
   after the needle ladder. Skip with `LEVER_GATE_SKIP_TEMP=1`.

4. **Concurrent needle arm integrated** — `needle_concurrent.py` (N in-flight
   requests, distinct needle per row) now runs inside `lever_gate.sh` after
   the temp arm. Catches cross-row state mix-up in batched decode that a
   single-request ladder cannot see. Skip with `LEVER_GATE_SKIP_CONCURRENT=1`.
   Tunables: `CONCURRENT_CONC` (default 4), `CONCURRENT_TOKENS` (2000),
   `CONCURRENT_ROUNDS` (1), `CONCURRENT_DEPTH` (0; set 50 under `--kv-recall`).
   Temp and concurrent arms run in parallel.

5. **Metal CI needle gate** — `.github/workflows/metal-ci.yml` now runs the
   needle ladder on `mlx-community/Qwen3.5-0.8B-MLX-4bit` after building the
   arle binary. Same config as the local pre-push gate. Gate scripts added
   to the workflow paths filter.

6. **`bench_compare.py` rewritten for v1 format** — the old tool expected a
   pre-v1 snapshot format and could not parse `arle.bench_throughput.v1`
   output from `bench_throughput.py`. Now keys on concurrency, supports
   decode/ttft_p50/itl_p50/rps metrics, refuses cross-workload comparison
   (dataset_sha256 or max_tokens mismatch), and flags `complete==0` as
   COLLAPSE instead of silently skipping.

7. **Stale cleanup** — deleted one-off B2 CP-decode scripts (`gate_arm.sh`,
   `serve_arm.sh`); fixed `infer/models/` → `models/` paths in 10 scripts
   (the `infer/` crate was deleted 2026-06-04).

## Verified

Metal gate on Qwen3.5-0.8B-MLX-4bit, lengths 115/300/446, 1 run each:

```
len=115 exact=1 partial=0 miss=0
len=300 exact=1 partial=0 miss=0
len=446 exact=1 partial=0 miss=0
TEMP-ARM PASS tokens=200/200 glued=None
[gate] correctness PASS: summaries=3
```

`bench_compare.py` tested with synthetic v1 snapshots: regression detection,
COLLAPSE on zero-complete, dataset/max_tokens mismatch refusal.

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
code, before it can be wired into any automated workflow. CI coverage is the
last step, not the first — the gate must prove itself locally first.
