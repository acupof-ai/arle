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

## 9. Checklist (copy into PR description when landing perf work)

```
- [ ] Goal stated (type: baseline/regression/opt/diagnosis/ceiling)
- [ ] Hypothesis recorded before the run
- [ ] Canonical tool + exact command in §3
- [ ] Environment pinned (§4): GPU, driver, commit sha, features, weights
- [ ] Raw artefacts committed under bench-output/<date>-<label>/
- [ ] §6 watch-list reviewed; deviations noted
- [ ] §7 stopping rules satisfied (or iteration rationale stated)
- [ ] Δ vs prior baseline computed
- [ ] Learnings (§7) are generalizable rules, not run-specific facts
```
