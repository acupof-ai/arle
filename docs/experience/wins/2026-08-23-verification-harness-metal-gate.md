# Verification harness: Metal gate, CI needle gate, concurrent arm, bench_compare fix

2026-08-23 · tooling · harness improvement

## Context

The correctness gate (`needle_gate.py` + `lever_gate.sh`) was CUDA-pod-only,
had no pass/fail exit code without a manual baseline, and ran the temp
coherence arm as a separate invocation. Local Metal development — the primary
Mac dev path — had no gate at all; the smoke test only checks CLI help text.

## What changed

1. **`needle_gate.py --check`** — standalone gate mode: exits 0 if every
   length has ≥ `--min-exact` (default 1) exact hits, exit 1 otherwise. No
   baseline log required.
2. **`lever_gate.sh GATE_PROFILE=metal`** — boots a Metal serve (default
   `mlx-community/Qwen3.6-35B-A3B-4bit`, override with `MODEL=`), skipping the
   DSv4 multi-GPU setup.
3. **Temp + concurrent arms integrated into `lever_gate.sh`** — run in
   parallel after the needle ladder; skip with `LEVER_GATE_SKIP_TEMP=1` /
   `LEVER_GATE_SKIP_CONCURRENT=1`. The concurrent arm (`needle_concurrent.py`,
   N in-flight requests, distinct needle per row) catches cross-row state
   mix-up in batched decode that a single-request ladder cannot see. Tunables
   in `docs/environment.md`.
4. **Metal CI needle gate** — `.github/workflows/metal-ci.yml` runs the needle
   ladder on `mlx-community/Qwen3.5-0.8B-MLX-4bit` after building arle; gate
   scripts added to the paths filter. CI resource overrides:
   `ARLE_METAL_AVAILABLE_RESERVE_MB=1024`, `ARLE_METAL_RUNTIME_HEADROOM_MB=128`,
   `SERVE_FLAGS="--system-reserve-bytes 1G --memory-budget-bytes 5G --allow-swap"`
   (7 GiB GitHub runners cannot fit the 6 GiB reserve + 4 GiB headroom defaults).
5. **`bench_compare.py` rewritten for v1 format** — the old tool could not
   parse `arle.bench_throughput.v1` output. Now keys on concurrency, refuses
   cross-workload comparison (dataset_sha256 or max_tokens mismatch), and
   flags `complete==0` as COLLAPSE.
6. **Stale cleanup** — deleted one-off B2 CP-decode scripts (`gate_arm.sh`,
   `serve_arm.sh`); fixed `infer/models/` → `models/` paths in 10 scripts.

Local one-command gate:

```bash
GATE_PROFILE=metal MODEL=models/Qwen3.5-0.8B-MLX-4bit \
  LENGTHS=115,300,446 RUNS=1 LEVER_GATE_ALLOW_NO_BASELINE=1 \
  bash scripts/lever_gate.sh <label>
```

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

## Rule

A correctness gate that can't run on the dev machine doesn't gate anything.
The harness must prove itself locally — one command, pass/fail exit code —
before it is wired into CI.
