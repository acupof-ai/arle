# Bench & Trace Specification

> Canonical rules for running, recording, and **iterating** on benchmarks and
> traces in agent-infer. Linked from [`CLAUDE.md`](../CLAUDE.md) §Benchmarks.
> The wins-entry skeleton lives in
> [`TEMPLATE-bench-guidellm.md`](experience/wins/TEMPLATE-bench-guidellm.md);
> this doc governs **process** around that template.

Scope: every benchmark run (guidellm sweep, microbench, nsys/ncu/Xcode capture,
Python scripts under `scripts/bench_*.py`) and every trace (nsys, ncu, Metal
capture, MLX instruments, tokio-console, `tracing` spans).

---

## 1. Mandatory report sections

Every bench/trace **must** produce a dated entry under
`docs/experience/wins/YYYY-MM-DD-<kind>-<label>.md` with these sections filled.
Missing any section → the run doesn't count.

| # | Section | What goes in it |
|---|---------|-----------------|
| 1 | **Goal** | One sentence. What question does this run answer? ("Does int8 KV halve decode VRAM without hurting ITL p99?"). Not "run guidellm". |
| 2 | **Hypothesis** | The expected outcome *before* running. Needed so §7 can decide whether the result was surprising. |
| 3 | **Parameters** | Every non-default flag, env var, model, dataset, seed, profile, rate schedule, num_slots, batch size, prompt/output token budget. Copy-paste the exact command. |
| 4 | **Environment** | Hardware (GPU model + VRAM / SoC + unified RAM), driver (CUDA + nvcc / macOS + Metal version), commit sha, feature set, server launch command, guidellm/tool version. |
| 5 | **Results** | Raw table first (TTFT p50/p99, ITL p50/p99, tok/s, req/s actual, VRAM peak, kernel counts). **No summaries replacing raw data.** Link to `benchmarks.{json,csv,html}` or `nsys-rep` / `.gputrace`. |
| 6 | **Problems observed** | Anything that degraded, crashed, OOM'd, stalled, or looked anomalous. Include the *smallest reproducer* if one exists. |
| 7 | **Learnings** | Rules generalizable to future work. Each learning is actionable: "X is bound by Y, so tune Z first". Lessons that only apply to this one run don't belong here — they go in §6. |

Structure is a **contract**, not a suggestion. CI-style: if a reviewer can't
answer "would I get the same numbers if I reran this?" from §3+§4 alone, the
entry is incomplete.

---

## 2. Goal taxonomy

Every run declares exactly one goal type. This drives §7 iteration logic.

| Type | Purpose | Success = |
|------|---------|-----------|
| **baseline** | Establish a reference point | Data captured; no claim about good/bad |
| **regression-check** | Did a commit harm perf? | Δ vs prior snapshot within noise band |
| **optimization** | Did a change improve perf? | Δ vs prior snapshot beats noise band AND hypothesis held |
| **diagnosis** | Explain an anomaly | Root cause named; reproducer captured |
| **ceiling** | Find max achievable under constraint | Saturation point + bottleneck identified |

---

## 3. Canonical tools

| Kind | Tool | Where |
|------|------|-------|
| End-to-end throughput/latency | `scripts/bench_guidellm.sh <label>` | Canonical; params locked in [`docs/plans/guidellm-integration.md`](plans/guidellm-integration.md) §3 |
| Prefill/decode micro | `scripts/bench_throughput.py`, `scripts/bench_kv_cache*.py` | For component-level isolation |
| CUDA kernel trace | `nsys profile` / `ncu --set full` | Save `.nsys-rep` / `.ncu-rep` alongside wins entry |
| Metal kernel trace | Xcode Metal capture / MLX instruments | Save `.gputrace` alongside wins entry |
| Runtime span trace | `tracing` + `tracing-subscriber` JSON | Pipe to file; link from §5 |

Changing canonical params is a **deliberate commit**, never a per-run flag flip.

---

## 4. Environment pinning

Record enough to reproduce:

- GPU: exact model + VRAM (e.g. `A100 80GB SXM4`, not "A100")
- Driver/toolkit: `CUDA 12.4 / nvcc 12.4.131` or `macOS 14.5 / Metal 3.1`
- Commit: short sha of `main` at run time (never run from a dirty tree — if
  dirty, note it explicitly and treat the run as diagnosis-only)
- Feature set: full `cargo build --release …` invocation
- Model weights: path + sha256 of config.json if weights were regenerated
- Server: exact launch command + env vars (`INFER_CUDA_SM`, `INFER_TRITON_PYTHON`, …)

---

## 5. Raw output policy

- **Always commit raw artefacts** under `bench-output/<date>-<label>/` and link
  from §5. Summaries in the wins entry are pointers, not replacements.
- Tables use the TEMPLATE columns; don't drop columns because "they weren't
  interesting" — the next reader may be looking for exactly that.
- Per-rate rows for sweeps: synchronous → saturation. No binning.

---

## 6. Things to watch during a bench run

Before reporting results, confirm each of these — noting any deviation in §6:

1. **Warmup.** Discard first 3–5s; cold caches skew TTFT p50.
2. **Thermal/throttling.** For long runs, check `nvidia-smi dmon` / `sudo powermetrics` for clock droop; if present, shorten run or pin clocks.
3. **Kernel-launch count per token.** If launches ≈ tokens, the bottleneck is dispatch, not compute — don't claim a compute ceiling. (See [`feedback_measure_batching_before_ceiling.md`](../.claude/projects/-Users-user-code-agent-infer/memory/feedback_measure_batching_before_ceiling.md).)
4. **Memory pressure.** VRAM peak must be <95% capacity, else numbers reflect allocator behaviour, not model.
5. **Client-side saturation.** If guidellm's client CPU is pegged, req/s reported is a *client* ceiling. Run client on a separate machine for decisive ceiling runs.
6. **Background noise.** No other GPU processes; `nvidia-smi` must show only the server. Same for `mlx` on Metal.
7. **Determinism.** Same seed, same params, two runs: TTFT p50 should be within ±2%. Higher variance → investigate before trusting §5.
8. **Prefix-cache state.** Cold vs warm cache changes TTFT by 10×. Declare which regime in §3.
9. **Num_slots actual vs configured.** Log the scheduler's effective slot count; misconfig silently caps throughput.
10. **Tokenizer mismatches.** `prompt_tokens=1024` in guidellm ≠ 1024 model tokens if tokenizers differ — verify.

Any "yes" to "did this deviate?" becomes a §6 entry.

---

## 7. Auto-iteration rules

Whether a run **loops into another run** is decided by information content,
not a fixed schedule. Iterate if and only if one of these triggers fires:

### 7.1 Information-volume triggers

| Signal | Action |
|--------|--------|
| Variance >5% across repeat runs | Iterate with longer `--max-seconds` and/or fixed clocks; don't trust §5 until variance <2%. |
| Result within noise band of hypothesis | **Stop.** One run suffices; don't grind for false precision. |
| Result beats hypothesis by >20% | Iterate **once** to confirm — regressions that look like wins are usually measurement bugs. |
| Result misses hypothesis by >20% | Iterate with **diagnosis goal** (nsys/ncu/Metal capture) before tuning further. |
| Sweep saturation point not reached (req/s still climbing at last rate) | Iterate with higher ceiling in `--rate`. |
| Any §6 watch-item deviated | Fix the deviation and rerun — don't publish compromised numbers. |

### 7.2 Bench-process triggers

- **After any optimization commit** touching hot paths (`infer/src/ops/`, `crates/infer-cuda-kernels/csrc/`, `crates/mlx-sys/src/`): run a regression-check against the most recent baseline for the affected backend+model. No exceptions.
- **Before merging a kernel change**: attach before/after wins entries in the commit message.
- **Diagnosis → fix → verify** is one loop: a diagnosis entry without a follow-up fix entry within 14 days is a debt item; log it in `docs/plans/` or open an issue.

### 7.3 Stopping rules

Stop iterating when **all** hold:

1. Variance <2% across last 2 runs.
2. Hypothesis confirmed or falsified with a root cause in §6.
3. §6 watch-list has zero deviations.
4. Delta vs prior baseline is recorded with a Δ% column.

Stop is as important as start — endless re-running burns GPU hours and
pollutes the wins log. A clean falsification is a successful run.

---

## 8. Trace-specific addenda

Traces (nsys / ncu / Metal capture) follow §1–§7 with these extras:

- **Scope the capture.** Full-run nsys files are unreadable. Trigger start/stop around ≤1s of steady state (`cudaProfilerStart/Stop` or `MTLCaptureManager`).
- **Report the top-5 kernels by total time** in §5. Link the raw trace file.
- **Always compute launches-per-token** for decode traces — §6 watch-item #3.
- **Compare like-for-like.** Same prompt, same slot count, same seed as the bench run being diagnosed.

---

## 9. Profile document format

A **profile** = a dedicated trace-driven investigation (nsys/ncu/Xcode/MLX
instruments). Different from a bench: a bench answers "how fast?", a profile
answers "why?". Profiles are still recorded under `docs/experience/wins/`
(or `errors/` if they surfaced a bug), but with a stricter skeleton:

Filename: `YYYY-MM-DD-profile-<backend>-<model>-<what>.md`
(e.g. `2026-04-18-profile-cuda-qwen35-decode-launches.md`)

Required sections (superset of §1):

```markdown
# <title> — profile, <backend>, <YYYY-MM-DD>

## Goal                  ← §1.1: question the profile answers
## Hypothesis            ← §1.2: suspected bottleneck before capture
## Environment           ← §4: full pinning
## Capture params        ← tool + exact command + capture window (e.g. "200ms steady-state, slot 4")
## Bench anchor          ← link to the bench wins entry this profile explains; same seed/prompt/slots
## Top-N kernels         ← table: kernel | calls | total µs | avg µs | % of frame
## Launches per token    ← decode only; MUST be present
## Roofline / utilisation← achieved TFLOPs or mem-GB/s vs theoretical peak
## Annotated timeline    ← screenshot or ASCII sketch of the critical ≤100ms window
## Findings              ← each finding: bottleneck + evidence line(s) + fix proposal
## Learnings             ← §1.7: generalizable rules
## Raw artefacts         ← paths to .nsys-rep / .ncu-rep / .gputrace (in bench-output/, not git)
```

Rules:

- **One profile, one question.** Multi-purpose profiles dilute evidence.
  Split into separate entries if two questions are being answered.
- **Pair with a bench.** Every profile cites the bench wins entry it
  explains (same commit, same workload). An orphan profile with no bench
  anchor is rejected.
- **Never commit raw capture files** (`.nsys-rep`, `.ncu-rep`, `.gputrace`)
  — they are hundreds of MB. Keep them under `bench-output/<date>-<label>/`
  (gitignored) and link from §Raw artefacts with a sha256.
- **Screenshots OK.** A 200-KB annotated timeline PNG under
  `docs/experience/wins/assets/<date>-<slug>/` is worth 1k words of prose.

---

## 10. Folder layout — where everything lives

```
agent-infer/
├── CLAUDE.md                              ← project contract; links this spec
├── docs/
│   ├── index.md                           ← PARA index (this spec listed under Governance)
│   ├── bench-and-trace-spec.md            ← THIS FILE — the process rule
│   ├── perf-and-correctness-gates.md      ← minimum-pass gates; bench is one input to them
│   ├── gpu-benchmark-a100.md              ← A100-specific methodology reference (resource-style)
│   ├── plans/
│   │   └── guidellm-integration.md        ← §3 locks canonical guidellm params
│   ├── projects/                          ← time-bound efforts; link bench entries as evidence
│   ├── plans/                             ← in-flight design+execution; cite bench deltas
│   ├── research/                          ← feasibility; may precede a bench
│   ├── reviews/                           ← audits (e.g. cuda-kernel-six-principles); bench-driven
│   ├── resources/                         ← user-facing refs (e.g. metal-dflash-params)
│   ├── areas/                             ← long-running concerns (precision, observability)
│   ├── archives/                          ← inactive
│   └── experience/
│       ├── wins/                          ← bench + profile entries (this spec §1 + §9)
│       │   ├── TEMPLATE-bench-guidellm.md ← fill-in skeleton for bench entries
│       │   ├── YYYY-MM-DD-bench-<label>.md
│       │   ├── YYYY-MM-DD-profile-<backend>-<model>-<what>.md
│       │   └── assets/<date>-<slug>/      ← small PNG timelines, flame graphs (<500 KB ea.)
│       ├── errors/                        ← bench that surfaced a bug → goes here, not wins/
│       └── reviews/                       ← code-review findings; bench-adjacent
├── bench-output/                          ← gitignored; raw artefacts (.json/.csv/.html/.nsys-rep/.gputrace)
│   └── <YYYY-MM-DD>-<label>/
├── benchmarks/                            ← committed baseline JSONs (small, reference numbers)
│   └── <backend>_baseline.json
└── scripts/
    ├── bench_guidellm.sh                  ← canonical e2e sweep
    ├── bench_throughput.py                ← component-level
    ├── bench_kv_cache*.py                 ← KV-specific
    └── bench_agent_trace.py               ← agent-workload trace
```

**The three locations, three rules:**

1. `docs/experience/wins/` — **immutable history**, one file per run, small.
   Never overwrite; superseded findings get a new dated entry that cites the
   old one.
2. `bench-output/` — **raw, ephemeral, gitignored**. Exists on the machine
   that ran the bench. Upload large artefacts to shared storage (separate
   bucket) and cite the URL + sha256 from the wins entry.
3. `benchmarks/` — **committed baselines**, only the small JSON numbers a
   regression check needs to load. Update = deliberate commit.

---

## 11. Relationship to the rest of docs/

This spec is **governance** (process rule). Every other docs/ folder has a
defined handshake with it:

| Folder / doc | Direction | Handshake |
|--------------|-----------|-----------|
| [`CLAUDE.md`](../CLAUDE.md) §Benchmarks | → this spec | Links here as read-first. |
| [`perf-and-correctness-gates.md`](perf-and-correctness-gates.md) | → this spec | Defines pass/fail thresholds; this spec defines **how** to measure against them. |
| [`docs/plans/guidellm-integration.md`](plans/guidellm-integration.md) §3 | → this spec | Canonical params; this spec forbids per-run overrides. |
| [`docs/experience/wins/TEMPLATE-bench-guidellm.md`](experience/wins/TEMPLATE-bench-guidellm.md) | implements §1 | Skeleton for the bench entry type. |
| [`docs/experience/wins/`](experience/wins/) | implements §1, §9 | Every bench/profile lives here. |
| [`docs/experience/errors/`](experience/errors/) | implements §1 + rule | If the bench uncovered a bug, the entry goes to `errors/` instead; link both ways. |
| [`docs/experience/reviews/`](experience/reviews/) | adjacent | Code-review findings; may reference a bench entry as evidence. |
| [`docs/projects/`](projects/) | cites wins | Project docs **cite** wins entries as milestone evidence; they do not duplicate numbers. |
| [`docs/plans/`](plans/) | cites wins | Plan acceptance gates cite concrete wins entries ("accepted per `wins/2026-04-15-tiered-kv-m2b-remote.md`"). |
| [`docs/research/`](research/) | precedes bench | Feasibility may motivate a bench; the bench result then updates or closes the research doc. |
| [`docs/reviews/`](reviews/) | drives bench | Standalone audits (e.g. cuda-kernel-six-principles) commission bench/profile runs; findings land back here. |
| [`docs/resources/`](resources/) | independent | User-facing tool docs (DFlash params, KV quant). Bench/profile may reference them for params. |
| [`docs/areas/`](areas/) | long-running | Continuous concerns (precision, throughput drift); bench entries accumulate as evidence over time. |
| [`docs/archives/`](archives/) | terminal | Retired bench entries move here **only** if the whole area is deprecated; otherwise never archive wins. |
| [`docs/architecture.md`](architecture.md), [`codebase-map.md`](codebase-map.md), [`support-matrix.md`](support-matrix.md) | reference | Tell the profile/bench **where** to measure (which backend, which model tier). |
| [`docs/gpu-benchmark-a100.md`](gpu-benchmark-a100.md) | specialization | A100-specific methodology notes; this spec is the general rule. |
| `benchmarks/*.json` | artefact | Committed baseline numbers; regression checks diff against these. |
| `bench-output/` | artefact | Raw, gitignored; wins entries link here. |
| `scripts/bench_*.{sh,py}` | tool | Must be invoked via the canonical command listed in the wins entry §3. |

**One-line summary:** projects/plans/research/reviews **describe intent**;
experience/wins and experience/errors **record reality**; bench-output and
benchmarks **hold numbers**; this spec **governs how reality becomes a
trustworthy record**.

---

## 12. Checklist (copy into PR description when landing perf work)

```
- [ ] Goal stated (type: baseline/regression/opt/diagnosis/ceiling)
- [ ] Hypothesis recorded before the run
- [ ] Canonical tool + exact command in §3
- [ ] Environment pinned (§4): GPU, driver, commit sha, features, weights
- [ ] Raw artefacts under bench-output/<date>-<label>/ (gitignored); sha256 cited
- [ ] Wins entry at docs/experience/wins/YYYY-MM-DD-<kind>-<label>.md
- [ ] §6 watch-list reviewed; deviations noted
- [ ] §7 stopping rules satisfied (or iteration rationale stated)
- [ ] If profile: §9 skeleton followed; bench anchor linked
- [ ] Δ vs prior baseline computed
- [ ] Learnings (§1.7) are generalizable rules, not run-specific facts
- [ ] Cross-links: project/plan/review that commissioned this run
```
