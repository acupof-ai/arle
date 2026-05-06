# Bench & Trace Specification

> Process rule for running, recording, and **iterating** on benchmarks and
> traces. Linked from [`AGENTS.md`](../AGENTS.md) and [`CLAUDE.md`](../CLAUDE.md)
> §Benchmarks. The fill-in skeleton is
> [`TEMPLATE-bench-guidellm.md`](experience/wins/TEMPLATE-bench-guidellm.md);
> this doc governs the process, not the skeleton.

**The four things that matter** (28 原则 — these carry 80% of the value):

1. **Every run has a written hypothesis** — the only defence against measurement-bug wins.
2. **Auto-iterate on information, not schedule** — stop when numbers converge, loop when they don't (§6).
3. **Profile pairs with bench** — a profile without a bench anchor is rejected (§8).
4. **Wins log is immutable history** — never overwrite; deltas cite prior (§9).

Everything else is support. If a rule below doesn't serve one of those four, it's optional.

Scope: canonical guidellm sweeps, supporting component-level helper benches,
and every trace (nsys, ncu, Metal capture, MLX instruments, `tracing` spans).
Helpers may inform diagnosis but never replace `scripts/bench_guidellm.sh` as
the throughput / latency truth source.

---

## 1. Required report sections

Every run produces `docs/experience/wins/YYYY-MM-DD-<kind>-<label>.md` with
all sections filled. Missing one → the run doesn't count. The `guidellm`
template mirrors this order so report and template stay aligned.

| # | Section | Content |
|---|---------|---------|
| 1 | **Goal** (+ type) | One sentence. Type ∈ {baseline, regression, optimization, diagnosis, ceiling}. |
| 2 | **Hypothesis** | Expected outcome *before* the run. Enables §6 to judge "was this surprising?". |
| 3 | **Command** | Exact CLI + env vars + seed. Copy-pasteable. |
| 4 | **Environment** | GPU/SoC + VRAM, CUDA/Metal version, commit sha (never dirty), feature set, model + weights path. |
| 5 | **Results** | Raw client-side table first (TTFT p50/p99, ITL p50/p99, tok/s, req/s actual). Then the §3 internal-sources headline counters that the workload exercised. No summaries replacing numbers. Link raw artefacts including service-stats snapshots. |
| 6 | **Problems** | Anything that degraded, crashed, or deviated from §5 watch-list. Smallest reproducer. |
| 7 | **Learnings** | Generalizable rules, not run-specific facts. Each actionable: "X bound by Y → tune Z first". |
| 8 | **Δ vs baseline** | Link prior entry + Δ% row. "First run" if none exists. |

Acid test: a reviewer should be able to answer "would I get the same numbers
if I reran this?" from §3 + §4 alone.

---

## 2. Tools

Grouped by purpose; prefer the wrapper over the raw CLI so captures land in
the canonical layout automatically.

**Throughput / latency (canonical):**
- **`scripts/bench_guidellm.sh <label>`** — params locked in [`plans/guidellm-integration.md`](plans/guidellm-integration.md) §3. Enforces serial runs via `bench-output/.bench_guidellm.lock`; captures `/v1/stats` before/during/after; emits K6 silent-OOM warnings. Variant: `--concurrencies 1,4,16,64 --max-seconds 120` for fixed-c reference comparisons.

**Profile (preferred wrappers, attach-mode):**
- **`scripts/profile_nsys_guidellm.sh <label>`** — Nsight Systems → `.nsys-rep` + `.sqlite` + stats + summary.
- **`scripts/profile_ncu_guidellm.sh <label> --family <name>`** — Nsight Compute → `.ncu-rep` + summary.

**Component / fallback:**
- **`scripts/bench_throughput.py`** — legacy synthetic helper; historical reproducibility only.
- **`scripts/bench_kv_cache*.py`** — internal component checks.
- **`nsys profile` / `ncu --set full`** — raw CUDA CLIs; wrappers preferred.
- **Xcode Metal capture / MLX instruments** — Metal → `.gputrace`.

---

## 3. Internal information sources

External tools (guidellm, nsys, ncu) report **client-side** views: latency,
throughput, kernel timeline. They miss the **server-side** state that
explains *why* those numbers came out. Every wins entry whose workload
touched a layer below MUST cite the matching counters — that's what makes a
delta attributable.

### 3.1 `/v1/stats` service trace

Captured automatically by `bench_guidellm.sh` as
`service_stats_before.txt`, `service_stats_trace.jsonl`,
`service_stats_after.txt`, plus a summary. Headline counters:

| Layer | Counters | Reads as |
|-------|----------|----------|
| Scheduler | `peak active`, `peak waiting`, `peak prefill_queue` | Slot pressure + queueing |
| KV memory | `peak kv_util` | Did we run hot on KV? |
| Prefix cache | `prefix_hit_rate`, `prefix_skip_rate` | Was the run cold or warm? |
| KV transport | `kv_fetch_q`, `kv_fetch_waiters`, `kv_store_q`, `kv_store`, `kv_bp` | Tier I/O backpressure |
| Tier recall | `tier_recall`, `tier_src`, `tier_promoted`, `tier_fallback` | Multi-tier hit path |

Cite only the counters the workload actually exercised — listing
zero-valued counters dilutes the report. If `kv_util ≈ 1.0` or
`prefill_queue` is non-trivial throughout, that fact belongs in §1 row 5,
not §6.

### 3.2 Scheduling envelope log

`infer/src/backend/cuda/bootstrap.rs` emits a
`Scheduling envelope (resolved | SGLang-equiv)` line at server boot. **Paste
it verbatim** into the wins entry whenever the run is compared against an
external reference (SGLang, vLLM, prior commit). Silent param drift
(e.g. `max_prefill_tokens=2048` vs reference `16384`) is the single most
common 5× regression — the envelope log is the contract that makes drift
visible.

### 3.3 Token accounting

GuideLLM reports completed vs incomplete input/output tokens; both go in §1
row 5 when available. Mismatch between requested and accepted tokens is a
silent indicator of stream truncation or tokenizer mismatch.

### 3.4 K6 silent-OOM detector

`bench_guidellm.sh:emit_oom_warnings` flags HTTP-200 responses with empty
output (the K6 failure mode). Useful but **insufficient on its own** — see
§7.1: the e2e correctness gate also catches HTTP-200 + degenerate-text
failures (`!!!!!` regression, `47bad713`).

---

## 4. Goal types → iteration policy

| Type | Success = | Stop when |
|------|-----------|-----------|
| baseline | Data captured | One clean run |
| regression | Δ within noise band | Within band → done; else diagnosis loop |
| optimization | Beats noise band AND hypothesis held | §6 stopping rules |
| diagnosis | Root cause named + reproducer | Root cause in §6 |
| ceiling | Saturation + bottleneck named | Saturation reached |

---

## 5. Watch-list during a run (top-5 — the 80%)

Confirm each before trusting §1 row 5. Deviation → §6 entry.

1. **Warmup.** Discard first 3–5s; cold caches skew TTFT p50.
2. **Launches per token.** If launches ≈ generated tokens, dispatch is the bottleneck — don't claim a compute ceiling from that shape.
3. **Determinism.** Same seed twice → TTFT p50 within ±2%. Higher = investigate before publishing.
4. **Thermal + background noise.** `nvidia-smi dmon` / `powermetrics` for throttling; no other GPU processes.
5. **Prefix-cache state + tokenizer.** Declare cold/warm in §3 of the entry; verify `prompt_tokens` matches the model tokenizer.

Long tail (memory pressure, client-side saturation, slot misconfig) → §6 if
encountered, not a pre-run gate.

---

## 6. Auto-iteration

**Iterate when** any holds:

| Signal | Action |
|--------|--------|
| Variance >5% across repeats | Longer `--max-seconds`, pin clocks; don't trust until <2%. |
| Result beats hypothesis by >20% | Rerun once — too-good wins are usually measurement bugs. |
| Result misses hypothesis by >20% | Switch to **diagnosis** goal (profile) before further tuning. |
| Saturation not reached (req/s still climbing) | Raise `--rate` ceiling. |
| §5 watch-item deviated | Fix, rerun. Never publish compromised numbers. |

**Stop when all hold:** variance <2% across last 2 runs; hypothesis confirmed
or falsified with a named reason; §5 clean; Δ% vs prior baseline recorded. A
clean falsification is a successful run — don't grind for false precision.

**Triggers outside a single task:**

- Optimization commit touching `infer/src/ops/`, `crates/cuda-kernels/csrc/`, `crates/mlx-sys/src/` → regression run vs latest baseline. No exceptions.
- Diagnosis entry without a follow-up fix entry within 14 days = debt → open in `docs/plans/`.

---

## 7. Hard-won protocol rules

Codified from 2026-04-28→29 lessons. Each rule fixes a specific, observed
failure mode of "the bench technically ran but the number was a lie".

### 7.1 Correctness gate before perf reporting

Tok/s is meaningless if the model emits garbage. The TileLang `clear=False`
regression (`47bad713`) shipped headline numbers while 4-token prompts
returned `"!!!!!"`. **Gate** every wins entry with one of:

- a passing `cargo test --release -p infer --test e2e --features cuda`, or
- a curl smoke: 4-token prompt → non-empty, first 5 chars not all identical.

K6 (§3.4) catches "200 + empty"; the e2e test catches "200 + degenerate
text". Both failure modes have shipped before — don't trust K6 alone.

### 7.2 Sweep ≠ fixed-concurrency

`guidellm --profile sweep` auto-picks 10 strategies linspaced between sync
and measured throughput. On a 24 GB L4 at 4096-in/256-out, the realised
sweep is `sync (0.10 r/s) → throughput (0.27 r/s)` — concurrency in flight
≈ 1×–3× sync. **It does not cover `c=16`,** which is what SGLang/vLLM
headline numbers usually report. Comparing our sweep "throughput" tok/s to
their "c=16" tok/s is apples-to-oranges (throughput-mode = unbounded
saturation, TTFT 13 s; c=16 = bounded in-flight, TTFT 1–2 s).

**Rule:** match the reference's concurrency model.
- `bench_guidellm.sh <label> --concurrencies 1,4,16,64 --max-seconds 120` — fixed-c vs reference.
- `bench_guidellm.sh <label>` (canonical sweep) — load-curve characterization.

Both valid; they answer different questions. The wins entry must state
which.

### 7.3 Duration adequacy

Single-run tok/s variance scales with √(samples). At c=16 a 30 s run admits
~10–15 requests; 60 s → ~30; 120 s → ~60. At `--fast` (30 s) we measured
σ ≈ 50 tok/s on the headline metric — too noisy for delta attribution.

| Purpose | Min duration |
|---------|--------------|
| Iteration / smoke | `--fast` (30 s c=16) |
| Wins-entry headline | 60 s (sweep default) |
| Fixed-c vs reference | 120 s + n=3 if variance |
| Long-context (prompt ≥ 8k) | 180 s |
| Decode-bound (small prompt, long output) | 60 s + n=3 |

Variance >10% across n=3 → double the duration before publishing.

### 7.4 Param alignment with reference

Every run compared against an external reference must paste the §3.2
scheduling envelope log verbatim. Drift like `max_prefill_tokens=2048` vs
reference `16384` is silent and costs ~5× TTFT — the F4 fix (`8f6965c3`)
was driven by exactly this.

### 7.5 Server lifecycle hygiene

- Stale lock after a killed run: `rm -f bench-output/.bench_guidellm.lock`.
- Slot leak when client disconnects mid-stream (K7 — see [`projects/2026-04-29-perf-bug-roundup.md`](projects/2026-04-29-perf-bug-roundup.md)). Restart server between sessions when status is uncertain; verify `/v1/stats` shows `active=0 waiting=0` before re-running.

---

## 8. Profile document format

Profile = trace-driven investigation (nsys/ncu/Xcode/MLX). Bench asks "how
fast?", profile asks "why?". Lives in `wins/` (or `errors/` on bug
discovery).

Filename: `YYYY-MM-DD-profile-<backend>-<model>-<what>.md`

Required sections (§1 plus these):

- **Capture params** — tool + command + window (e.g. "200 ms steady-state, slot 4")
- **Bench anchor** — link to the bench entry this profile explains, same commit + workload. Orphan profile = rejected.
- **Top-N kernels** — table: kernel | calls | total µs | avg µs | % of frame
- **Launches per token** — mandatory for decode profiles
- **Roofline** — achieved TFLOPs or mem-GB/s vs theoretical peak
- **Findings** — each = bottleneck + evidence line + proposed fix

Rules: one profile, one question; scope capture to ≤1 s of steady state;
never commit raw `.nsys-rep`/`.ncu-rep`/`.gputrace` (hundreds of MB — keep
under `bench-output/`, cite sha256); small annotated timeline PNGs (<500 KB)
under `experience/wins/assets/<date>-<slug>/` are encouraged.

---

## 9. Folder layout

```
ARLE/
├── AGENTS.md / CLAUDE.md            ← link this spec
├── docs/
│   ├── bench-and-trace-spec.md      ← THIS FILE (process)
│   ├── perf-and-correctness-gates.md← pass/fail thresholds (what)
│   ├── plans/guidellm-integration.md← canonical params
│   └── experience/
│       ├── wins/                    ← bench + profile entries; immutable
│       │   ├── TEMPLATE-bench-guidellm.md
│       │   ├── YYYY-MM-DD-bench-<label>.md
│       │   ├── YYYY-MM-DD-profile-<backend>-<model>-<what>.md
│       │   └── assets/<date>-<slug>/← small PNGs only
│       └── errors/                  ← bench that surfaced a bug
├── bench-output/                    ← gitignored; raw + service_stats_*
├── benchmarks/                      ← committed baseline JSONs (small)
└── scripts/
    ├── bench_guidellm.sh            ← canonical throughput / latency
    ├── profile_nsys_guidellm.sh     ← Nsight Systems wrapper
    ├── profile_ncu_guidellm.sh      ← Nsight Compute wrapper
    └── bench_throughput.py          ← legacy helper
```

**Three locations, three rules:**

1. `docs/experience/wins/` — **immutable**, one file per run. Superseded findings = new dated entry citing the old.
2. `bench-output/` — **raw, ephemeral, gitignored**. Large artefacts → shared storage; cite URL + sha256.
3. `benchmarks/*.json` — **committed baselines** (small). Update = deliberate commit.

---

## 10. Handshake with the rest of docs/

| Kind | Role | Handshake |
|------|------|-----------|
| **Intent** — `projects/`, `plans/`, `research/`, `reviews/` | Describe *what we want* | **Cite** wins entries as evidence; never duplicate numbers. Plan acceptance gates name a specific wins entry. |
| **Reality** — `experience/wins/`, `experience/errors/` | Record *what happened* | Implement §1 + §8. Errors/ for regressions; wins/ otherwise. |
| **Thresholds** — `perf-and-correctness-gates.md` | Define pass/fail | This spec defines **how** to measure them. |
| **Params** — `plans/guidellm-integration.md` §3 | Lock canonical flags | This spec forbids per-run override. |
| **Numbers** — `bench-output/`, `benchmarks/*.json` | Hold data | Wins entries link into them. |

One-line: **intent describes, experience records, artefacts hold, this spec
governs how reality becomes a trustworthy record.**

---

## 11. PR checklist

```
- [ ] Goal stated (type: baseline/regression/opt/diagnosis/ceiling)
- [ ] Hypothesis recorded before the run
- [ ] §1 wins entry committed (profile? also §8 skeleton + bench anchor)
- [ ] Env pinned: GPU, driver, commit sha, features, weights
- [ ] §3 internal sources cited (service trace, envelope log if vs reference)
- [ ] Raw artefacts in bench-output/<date>-<label>/; sha256 cited
- [ ] §5 watch-list reviewed
- [ ] §6 stopping rules satisfied (or iteration rationale stated)
- [ ] §7 protocol respected (correctness gate, sweep-vs-c, duration, envelope, lifecycle)
- [ ] Δ% vs prior baseline
- [ ] Cross-link: project/plan/review that commissioned the run
```
