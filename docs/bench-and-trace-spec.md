# Bench & Trace Specification

> Process rule for running, recording, and **iterating** on benchmarks and
> traces. Linked from [`AGENTS.md`](../AGENTS.md) and
> [`CLAUDE.md`](../CLAUDE.md) §Benchmarks. The fill-in
> skeleton is [`TEMPLATE-bench-guidellm.md`](experience/wins/TEMPLATE-bench-guidellm.md);
> this doc governs the process, not the skeleton.

**The four things that matter** (28 原则 — these carry 80% of the value):

1. **Every run has a written hypothesis** — the only defence against measurement-bug wins.
2. **Auto-iterate on information, not schedule** — stop when numbers converge, loop when they don't (§5).
3. **Profile pairs with bench** — a profile without a bench anchor is rejected (§6).
4. **Wins log is immutable history** — never overwrite; deltas cite prior (§7).

Everything else is support. If a rule below doesn't serve one of those four, it's optional.

Scope: canonical guidellm throughput / latency sweeps, supporting
component-level helper benches, and every trace (nsys, ncu, Metal capture,
MLX instruments, `tracing` spans). Helper scripts may inform diagnosis, but
they do not replace `scripts/bench_guidellm.sh` as the canonical throughput /
latency truth source.

---

## 1. Required report sections

Every run produces `docs/experience/wins/YYYY-MM-DD-<kind>-<label>.md` with
all of these. Missing one → the run doesn't count. The canonical
`guidellm` win template mirrors this order so the report and template stay
aligned.

| # | Section | Content |
|---|---------|---------|
| 1 | **Goal** (+ type) | One sentence. Type ∈ {baseline, regression, optimization, diagnosis, ceiling}. |
| 2 | **Hypothesis** | Expected outcome *before* the run. Enables §5 to judge "was this surprising?". |
| 3 | **Command** | Exact CLI + env vars + seed. Copy-pasteable. |
| 4 | **Environment** | GPU/SoC model + VRAM, CUDA/Metal version, commit sha (never dirty tree), feature set, model + weights path. |
| 5 | **Results** | Raw table first (TTFT p50/p99, ITL p50/p99, tok/s, req/s actual). Then add the service-side KV / scheduler headline counters from the captured stats surface: `peak active`, `peak waiting`, `peak prefill_queue`, `peak kv_util`, `prefix_hit_rate`, `prefix_skip_rate`, `kv_fetch_q`, `kv_fetch_waiters`, `kv_store_q`, `kv_store`, `kv_bp`, plus `tier_recall` / `tier_src` / `tier_promoted` / `tier_fallback` when the workload exercised tier recall. Also include completed vs incomplete input/output token accounting from GuideLLM when available. Add VRAM peak / kernel counts only when the wrapper or a paired trace actually produces them. No summaries replacing numbers. Link raw artefacts, including service trace snapshots (`service_stats_before.txt`, `service_stats_trace.jsonl`, `service_stats_after.txt`, summary). |
| 6 | **Problems** | Anything that degraded, crashed, or deviated from §4 watch-list. Include smallest reproducer. |
| 7 | **Learnings** | Generalizable rules, not run-specific facts. Each actionable: "X bound by Y → tune Z first". |
| 8 | **Δ vs baseline** | Link prior entry + Δ% row. "First run" if none exists. |

Acid test: if a reviewer can't answer "would I get the same numbers if I
reran this?" from §3+§4 alone, the entry is incomplete.

---

## 2. Tools

- **`scripts/bench_guidellm.sh <label>`** — canonical throughput / latency sweep wrapper. Params locked in [`plans/guidellm-integration.md`](plans/guidellm-integration.md) §3; changing them is a deliberate commit. Wrapper enforces one-at-a-time (serial) runs and captures `/v1/stats` service trace before/during/after each run.
- **`scripts/bench_throughput.py`** — legacy helper for narrower synthetic / diagnostic checks; keep it for historical reproducibility only. `bench_kv_cache*.py` remains component-level / internal-only.
- **`scripts/profile_nsys_guidellm.sh <label>`** — preferred infer-side
  `Nsight Systems` wrapper; attaches to a running server, drives a short
  bench-anchored load, and exports `.nsys-rep` + `.sqlite` + stats + summary.
- **`scripts/profile_ncu_guidellm.sh <label> --family <name>`** — preferred
  infer-side `Nsight Compute` wrapper; attaches to a running server, drives a
  bench-anchored load, and exports `.ncu-rep` + summary.
- **`nsys profile` / `ncu --set full`** — raw CUDA profiler CLIs; still valid
  for one-off work, but repo-owned captures should prefer the wrappers above.
- **Xcode Metal capture / MLX instruments** — Metal trace → `.gputrace`.

---

## 3. Goal types → iteration policy

| Type | Success = | When to stop |
|------|-----------|--------------|
| baseline | Data captured | One clean run |
| regression | Δ within noise band | One run if within; else diagnosis loop |
| optimization | Beats noise band AND hypothesis held | §5 stopping rules |
| diagnosis | Root cause named + reproducer | Root cause in §6 |
| ceiling | Saturation + bottleneck named | Saturation reached |

---

## 4. Watch-list during a run (top-5 — the 80%)

Confirm each before trusting §5 results. Deviation → §6 entry.

1. **Warmup.** Discard first 3–5s. Cold caches skew TTFT p50.
2. **Launches per token.** If launches are roughly equal to generated tokens, the bottleneck is dispatch rather than compute; do not claim a compute ceiling from that run shape.
3. **Determinism.** Same seed twice → TTFT p50 within ±2%. Higher variance = investigate first.
4. **Thermal + background noise.** Check `nvidia-smi dmon` / `powermetrics` for throttling; no other GPU processes.
5. **Prefix-cache state + tokenizer.** Declare cold/warm in §3; verify `prompt_tokens` matches model tokenizer.

Long tail (memory pressure, client-side saturation, slot misconfig) → note
in §6 if encountered, not a pre-run gate.

---

## 5. Auto-iteration

**Iterate when** any holds:

| Signal | Action |
|--------|--------|
| Variance >5% across repeats | Longer `--max-seconds`, pin clocks; don't trust until <2%. |
| Result beats hypothesis by >20% | Rerun once — too-good wins are usually measurement bugs. |
| Result misses hypothesis by >20% | Switch to **diagnosis** goal (profile) before further tuning. |
| Saturation not reached (req/s still climbing) | Raise `--rate` ceiling. |
| §4 watch-item deviated | Fix, rerun. Never publish compromised numbers. |

**Stop when all hold:**

1. Variance <2% across last 2 runs.
2. Hypothesis confirmed or falsified with a named reason.
3. §4 watch-list clean.
4. Δ% vs prior baseline recorded.

A clean falsification is a successful run. Don't grind for false precision.

**Triggers outside a single task:**

- Any optimization commit touching `infer/src/ops/`, `crates/cuda-kernels/csrc/`, `crates/mlx-sys/src/` → regression run vs latest baseline. No exceptions.
- Diagnosis entry without a follow-up fix entry within 14 days = debt item → open in `docs/plans/`.

---

## 6. Profile document format

**Profile** = trace-driven investigation (nsys/ncu/Xcode/MLX). Bench asks
"how fast?"; profile asks "why?". Lives in `wins/` (or `errors/` on bug
discovery).

Filename: `YYYY-MM-DD-profile-<backend>-<model>-<what>.md`

Required sections (§1 plus these):

- **Capture params** — tool + command + window (e.g. "200ms steady-state, slot 4")
- **Bench anchor** — link to the bench entry this profile explains, same commit + workload. Orphan profile = rejected.
- **Top-N kernels** — table: kernel | calls | total µs | avg µs | % of frame
- **Launches per token** — mandatory for decode
- **Roofline** — achieved TFLOPs or mem-GB/s vs theoretical peak
- **Findings** — each = bottleneck + evidence line + proposed fix

Rules:

- **One profile, one question.** Split if answering two.
- **Scope the capture** to ≤1s of steady state. Full-run `.nsys-rep` is unreadable.
- **Never commit raw captures** (`.nsys-rep`, `.ncu-rep`, `.gputrace`) — hundreds of MB. Keep under `bench-output/`, cite sha256.
- Small annotated timeline PNGs (<500 KB) under `experience/wins/assets/<date>-<slug>/` are encouraged.

---

## 7. Folder layout

```
ARLE/
├── AGENTS.md / CLAUDE.md            ← link this spec
├── docs/
│   ├── bench-and-trace-spec.md      ← THIS FILE (process)
│   ├── perf-and-correctness-gates.md← pass/fail thresholds (what)
│   ├── plans/guidellm-integration.md← canonical params
│   └── experience/
│       ├── wins/                    ← bench + profile entries; immutable history
│       │   ├── TEMPLATE-bench-guidellm.md
│       │   ├── YYYY-MM-DD-bench-<label>.md
│       │   ├── YYYY-MM-DD-profile-<backend>-<model>-<what>.md
│       │   └── assets/<date>-<slug>/← small PNGs only
│       └── errors/                  ← bench that surfaced a bug
├── bench-output/                    ← gitignored; raw .json/.csv/.html/.nsys-rep/.gputrace
├── benchmarks/                      ← committed baseline JSONs (small)
├── scripts/bench_guidellm.sh         ← canonical throughput / latency wrapper
├── scripts/profile_nsys_guidellm.sh  ← canonical infer Nsight Systems wrapper
├── scripts/profile_ncu_guidellm.sh   ← canonical infer Nsight Compute wrapper
└── scripts/bench_throughput.py       ← legacy helper / deprecation banner
```

**Three locations, three rules:**

1. `docs/experience/wins/` — **immutable**, one file per run. Superseded findings = new dated entry citing the old.
2. `bench-output/` — **raw, ephemeral, gitignored**. Large artefacts go to shared storage; cite URL + sha256.
3. `benchmarks/*.json` — **committed baselines** (small). Update = deliberate commit.

---

## 8. Handshake with the rest of docs/

Intent vs reality vs numbers:

| Kind | Role | Handshake |
|------|------|-----------|
| **Intent** — `projects/`, `plans/`, `research/`, `reviews/` | Describe *what we want* | **Cite** wins entries as evidence; never duplicate numbers. Plan acceptance gates name a specific wins entry. |
| **Reality** — `experience/wins/`, `experience/errors/` | Record *what happened* | Implement §1 + §6. Errors/ for regressions; wins/ otherwise. |
| **Thresholds** — `perf-and-correctness-gates.md` | Define pass/fail | This spec defines **how** to measure them. |
| **Params** — `plans/guidellm-integration.md` §3 | Lock canonical flags | This spec forbids per-run override. |
| **Numbers** — `bench-output/`, `benchmarks/*.json` | Hold data | Wins entries link into them. |

One-line: **intent describes, experience records, artefacts hold, this spec governs how reality becomes a trustworthy record.**

---

## 9. PR checklist

```
- [ ] Goal stated (type: baseline/regression/opt/diagnosis/ceiling)
- [ ] Hypothesis recorded before the run
- [ ] §1 wins entry committed (profile? also §6 skeleton + bench anchor)
- [ ] Env pinned: GPU, driver, commit sha, features, weights
- [ ] Raw artefacts in bench-output/<date>-<label>/; sha256 cited
- [ ] §4 watch-list reviewed
- [ ] §5 stopping rules satisfied (or iteration rationale stated)
- [ ] Δ% vs prior baseline
- [ ] Cross-link: project/plan/review that commissioned the run
```
