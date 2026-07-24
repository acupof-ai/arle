# Sweet-spot corpus → first real-gradient agent-OPD run (#173)

**Goal: a non-zero dapo gradient and a held-out pass-rate delta on real repo
tasks.** Everything upstream already exists — fetch → clean → stage produced
`data/opd-corpora/staged-sweetspot3/` (609 instances: 449 train / 160 eval,
5 repos, `manifest.json scan="clean"`, built 2026-07-12) — the gap is one
env-var of wiring plus the profile → band → train sequence on the pod.

## Security constraint (hard, non-negotiable)

Only corpora whose manifest records `scan="clean"` ship to the pod. The scan is
`scripts/opd_security_filter.py` (60+ rules: CVE/exploit/malware/credential/
injection/… over every text field AND every staged file), applied three times in
the existing pipeline: repo blocklist at selection, tree pre-scan at staging,
final `--scan` sweep before tarball. Any subset re-derived from a clean root
(the comfort-band output) gets its own final `--scan` before shipping. A flagged
instance is dropped, never edited; the filter is never relaxed.

## Phase 0 — wiring (local, ~15 LOC, no GPU)

`scripts/agent_opd_curve.sh` hardwires `gen_agent_opd_tasks.py` (line 118) and
the `tasks_train.jsonl`/`tasks_eval.jsonl` names (lines 163/180/183). Add
`CORPUS_ROOT` env: when set, skip generation, set `CORPUS=$CORPUS_ROOT`, and
resolve dataset names via `TRAIN_JSONL`/`EVAL_JSONL` vars (pre-staged roots use
`train.jsonl`/`eval.jsonl`; synthetic keeps `tasks_*.jsonl`). The comfort-band
block is already generic over `$CORPUS`. Schema needs no work: `train agent-opd`
loads SweTask (`swe_dataset.rs:85`) and sweetspot3 rows are SweTask + extras.
Verify: `bash -n` + an arg-echo dry run.

## Phase 1 — 27B profile round (pod, ~4 h)

Ship `staged-sweetspot3.tgz` (already built) + rebuilt binary. Run the existing
comfort-band profile leg (curve script step 1b) on the real corpus:
`CORPUS_ROOT=… TASK_LIMIT=32 SAMPLES=8 SPEC=off ROUNDS=1`, task-selection off.
32 groups × ~7 min (smoke-measured group wall) ≈ 3.7 h. Round-robin the 5 repos
so no repo dominates. Output: `cb_profile/metrics.jsonl` per-task pass rates.
Launch nohup + watchdog (this session's `run-stagger.sh` pattern survives agent
session limits).

## Phase 2 — band filter (minutes)

`comfort_band.py --pass-lo 0.2 --pass-hi 0.8 --max-seq 22000 --min-tests 2` →
`corpus-band`. `--max-seq 22000` is load-bearing: the writeback VRAM wall skips
seq > 23 000 (`update_strategy.rs:650`; errors/2026-07-22 — length, not missing
variance, caused the last null gradient). Then `opd_security_filter --scan` over
the band root (per the constraint above). Expected band occupancy 20–50% of the
32 profiled; if < 8 tasks, widen to 0.1–0.9 as the single variable and re-cut.

## Phase 3 — train + temp A/B (pod, ~1 day)

Stage B on the banded corpus: `dapo`, `SAMPLES=8`, `ROUNDS=4`, two arms
`ROLLOUT_TEMPERATURE=0.3` vs `1.0` (#173 item 2: panel observed 1.0
degenerates; the A/B licenses 0.3), same GPU/port, sequential. Acceptance
gates, all decoded from logs not inferred:

1. Zero `SKIP … > max_update_seq` lines (the band filter worked).
2. `mean_loss > 0` on ≥1 round with `zero_variance_groups < groups` (gradient
   actually reached the backward).
3. Held-out Δpass vs the run's own `BASE_REPEATS` baseline envelope; claims
   < 5 pp on n ≤ 200 need multi-seed ≥ 5 before any wins entry states a gain.

## Phase 4 — verdict

Accept-or-reject line in CHANGELOG the same day; wins/ (or errors/) entry with
the decoded numbers; close #173 (item 1 by phases 1–2, item 2 by the temp A/B).

## Out of scope

Mega-rollout width (>1 concurrent group) — separate lever, premise (GPU
busy-frac GO, wins/2026-07-24) already measured, sequenced after this so the
first real-gradient run stays single-variable.
