# GuideLLM integration — canonical bench truth source

**Status:** canonical (wrapper live; 2026-04-20).
**Owner:** ckl. **Drives:** documents `scripts/bench_guidellm.sh` as the
project's **canonical throughput/latency measurement tool** and keeps the
remaining guardrails in one place.
**Trigger:** 2026-04-16 discussion — the house-grown sweep script
overlaps 1:1 with a well-maintained upstream tool ([vllm-project/guidellm](https://github.com/vllm-project/guidellm)),
and we're about to start cross-referencing numbers with vLLM/SGLang where
guidellm is already the reference point. Pick the tool everyone else picks,
stop hand-rolling.

---

## 1 · Why guidellm becomes the truth source

- **LLM-native metrics**: TTFT / ITL / tok-s / request-rate distributions
  with p50/p90/p95/p99, not generic HTTP RPS. The legacy throughput helper
  only emits mean + stddev.
- **sweep profile** auto-scans from `synchronous` to saturation — one
  command replaces our manual concurrency grid.
- **HTML report** is shareable; JSON is machine-readable for diffing across
  wins entries.
- **vLLM-official**, actively maintained, `pip install guidellm`. The nearest
  alternatives are either archived ([llmperf](https://github.com/ray-project/llmperf)
  — archived 2025-12-17) or being replaced ([genai-perf → AIPerf](https://github.com/ai-dynamo/aiperf)).
- Our `/v1/completions` + `/v1/chat/completions` (streaming) are OpenAI
  compatible, so guidellm attaches with zero server-side changes.

### What this does NOT replace

- `bench_kv_cache*.py`, `bench_offload*.py`, `bench_agent*.py`, `bench_long_agent.py`
  — those measure **internal behaviour** (prefix-cache hit rate, offload
  paths, agent-trace shapes) that guidellm can't observe from outside.
- PPL / quality evals — out of scope; guidellm is a performance tool only.

---

## 2 · Decisions (locked 2026-04-16)

| # | Decision | Rationale |
|---|---|---|
| 1 | **guidellm = sole truth** for throughput / TTFT / ITL wins. `bench_throughput.py` is retained only for historical / narrow helper runs; new wins use the guidellm wrapper. | Single canonical tool; no duplicate scripts. |
| 2 | **Wrapper script assumes the server is already running.** | Avoids "my bench crashed the server" failure mode; keeps concerns orthogonal. |
| 3 | **Same canonical config for CUDA and Metal backends.** | One `profile=sweep`, one dataset shape — makes cross-backend comparison a pure hardware delta, not a dataset delta. |

---

## 3 · Canonical bench parameters (the "truth" definition)

Write **once** into `scripts/bench_guidellm.sh` and
`docs/experience/wins/TEMPLATE-bench-guidellm.md`. The wrapper is the public
contract; `guidellm benchmark run` is the internal implementation detail.
Changing these values is a deliberate act, not a flag flip — any change
lands in a commit whose subject says so, and new wins reference the date of
the change.

```
--profile sweep
--data   prompt_tokens=4096,output_tokens=256
--max-seconds 60
--outputs json --outputs csv --outputs html
--random-seed 20260416
```

**Why these specific numbers:**
- `prompt_tokens=4096,output_tokens=256` — matches the shipped wrapper and
  the long-context / parity sweeps already used as canonical truth points.
  Shorter probes such as `1024/256` remain useful for focused smoke checks,
  but they are not the baseline.
- `--max-seconds 60` — sweep visits ~6 rate points, so total ~6–10 min per
  run; short enough to run before lunch, long enough for percentiles to
  stabilise. Do NOT drop below 30s — p99 noise explodes.
- `--random-seed 20260416` — frozen initially; bumped only if the prompt
  distribution becomes the limiting factor (unlikely — sweep pads with
  synthetic tokens).
- `sweep` profile — guidellm auto-stepping from synchronous to saturation
  matches how we'd think about the curve manually, and the HTML report
  visualises it.

---

## 4 · Target topology

```
agent-infer/
├── pyproject.toml
│   └── [project.optional-dependencies]
│       └── bench = ["httpx==0.28.1", "guidellm[recommended]>=0.3"]   # ← NEW
├── requirements-bench.txt                                             # ← + guidellm
├── scripts/
│   ├── bench_guidellm.sh                                              # ← canonical wrapper
│   └── bench_throughput.py                                            # ← legacy helper / deprecation banner
├── bench-output/                                                      # ← gitignored, raw guidellm JSON/HTML
├── docs/
│   ├── experience/wins/
│   │   └── TEMPLATE-bench-guidellm.md                                 # ← NEW skeleton
│   └── plans/
│       └── guidellm-integration.md                                    # ← this doc
├── CLAUDE.md                                                          # ← §Benchmarks updated
├── infer/src/http_server/AGENTS.md                                    # ← points at guidellm for perf verify
└── .gitignore                                                         # ← + bench-output/
```

**File count:** 8 touches (7 edits + 1 new dir). Above the ≥3-file
threshold → approach-first → this doc exists → approved → OK to proceed.

---

## 5 · Wrapper contract (`scripts/bench_guidellm.sh`)

Shell, not Python: we don't want another venv-bootstrap path, and the
canonical parameters fit in one heredoc.

```
Usage:
  scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH] [--trace-interval-ms N]

Required:
  <backend-label>      e.g. cuda-h100, cuda-a100, metal-m3max
                       used to name the output directory and wins file

Defaults (override with flags):
  --target   http://localhost:8000
  --model    Qwen/Qwen3-4B        (matches default HTTP server startup)
  --processor models/Qwen3-4B     (tokenizer path / HF id)
  --trace-interval-ms 1000        (/v1/stats polling cadence)

Behaviour:
  1. Acquire a global lock (`bench-output/.bench_guidellm.lock`) so only one
     bench run executes at a time.
  2. Check `guidellm` is on PATH. If not → print `pip install -e .[bench]`
     hint and exit 2.
  3. Check target responds to `/v1/models`. If not → exit 2 with
     "server not running at <target>, start it with
     scripts/start_infer.sh first".
  4. Capture service-side trace around the run:
        `service_stats_before.txt` (snapshot)
        `service_stats_trace.jsonl` (/v1/stats polling during run)
        `service_stats_after.txt` (snapshot)
        `service_stats_trace_summary.md` (peak waiting/active/kv summary)
  5. Invoke:
        guidellm benchmark run \
            --target "$TARGET" --model "$MODEL" --processor "$PROCESSOR" \
            --profile sweep \
            --data "prompt_tokens=4096,output_tokens=256" \
            --max-seconds 60 \
            --random-seed 20260416 \
            --output-dir "bench-output/$(date +%Y-%m-%d)-$LABEL/" \
            --outputs json --outputs csv --outputs html \
            --backend openai_http \
            --backend-kwargs '{"validate_backend": "/v1/models"}'
  6. Extract headline metrics from benchmarks.json:
        sweep rate points (req/s)
        TTFT p50 / p99 (ms)
        ITL  p50 / p99 (ms)
        output tok/s per rate point
  7. Print them as a markdown table on stdout, and write the same table
     plus the filesystem path of the HTML report into a new
     `docs/experience/wins/YYYY-MM-DD-bench-guidellm-<label>.md`
     file, seeded from `TEMPLATE-bench-guidellm.md`, with service-trace artefacts listed.

Exit codes:
  0   bench completed, wins stub written
  2   environment not ready (guidellm missing, server down, lock conflict)
  3   guidellm exited non-zero
  4   invalid/short-circuited benchmark result shape
```

Metric extraction is **jq** (already on both dev hosts) — no extra Python.
The wrapper's only hard dependency is `guidellm` itself + `jq` + `curl`.

---

## 6 · Wins entry template

`docs/experience/wins/TEMPLATE-bench-guidellm.md`:

```markdown
# <short title> — guidellm sweep, <backend-label>, <date>

## Goal
- <one sentence describing the benchmark goal and goal type>

## Hypothesis
- <expected outcome before the run>

## Command
- `scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME] [--processor PATH]`

## Environment
- Backend: <cuda|metal> · model: <Qwen/Qwen3-4B | ...>
- Hardware: <GPU/SoC model, VRAM, CUDA/Metal version>
- Commit: <short sha>
- Feature set: <cargo features>
- Non-default flags: <env vars, server flags>

## Canonical params
- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <label>`

## Results — sweep headline table
| rate (req/s) | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | out tok/s |
|---|---|---|---|---|---|
| ... sweep points ... |

## Problems
- <anything that degraded, crashed, or deviated from the watch-list>

## Learnings
- <generalizable rule or tuning takeaway>

## Δ vs baseline
- Baseline: [`<date>-bench-guidellm-<label>.md`](<relative path>)
- % change per column if prior snapshot exists.

## Artefacts
- Raw: `bench-output/<date>-<label>/benchmarks.{json,csv,html}`
- HTML report: `bench-output/<date>-<label>/benchmarks.html`
```

CLAUDE.md rule "Never overwrite before-snapshots" carries over verbatim —
the `<date>-<label>` naming enforces it at the filesystem level.

---

## 7 · Active doc wording

Active docs should say:

> - **Canonical tool: `scripts/bench_guidellm.sh <label>`** (thin wrapper
>   around [`vllm-project/guidellm`](https://github.com/vllm-project/guidellm)).
>   Parameters are locked in `docs/plans/guidellm-integration.md` §3;
>   changing them is a deliberate commit, not a flag flip.
> - `scripts/bench_throughput.py` is **deprecated**; keep it only for
>   historical reproducibility and component-level diagnostics. New wins MUST
>   use the guidellm wrapper.

---

## 8 · Acceptance criteria

Codex-executable gates. Implementation is complete when all hold:

1. `pip install -e .[bench]` on a clean venv installs `guidellm` and
   `guidellm --version` prints ≥0.3.
2. `scripts/bench_guidellm.sh cuda-local` (with a local server running)
   runs end-to-end and exits 0.
3. `scripts/bench_guidellm.sh nonexistent-backend` with **no server**
   running exits 2 with the "server not running" message.
4. `scripts/bench_guidellm.sh cuda-local` with `guidellm` uninstalled exits
   2 with the pip hint.
5. A new `docs/experience/wins/<today>-bench-guidellm-cuda-local.md` gets
   created and is populated with a real metric table (not just the template).
6. `bench-output/` is ignored by git (not in `git status --porcelain`
   after a run).
7. `scripts/bench_throughput.py --help` still works AND prints a
   `DEPRECATED: use scripts/bench_guidellm.sh instead` notice to stderr
   before the normal help.
8. `CLAUDE.md` §Benchmarks points at the canonical `scripts/bench_guidellm.sh`
   wrapper.
9. Historical wins may still reference the old sweep script, but active docs
   must use the canonical wrapper.

No Rust touched. No nvcc touched. This is purely shell + Python deps +
docs.

---

## 9 · Trip wires — when to stop and re-plan

- guidellm CLI flag names change across the pinned version range (run
  `guidellm benchmark --help` and diff against §3).
- Server's `/v1/models` endpoint changes shape or goes away → step 2 of
  the wrapper breaks.
- `bench-output/` accidentally gets committed because someone overrode
  the gitignore → wrapper should refuse to run if the output dir is
  tracked.
- **Some Metal hosts still cannot complete the canonical sweep reliably.**
  The canonical truth surface remains `scripts/bench_guidellm.sh` with the
  §3 params for every backend, but when a local Apple box or hosted runner
  still hits the MLX allocator resource-limit panic during sweep's
  throughput-burst stage, do not silently redefine the benchmark. Record a
  `pending-remote` wins stub for the canonical run, then use exploration
  mode (`--quick` or explicit overrides) only for local diagnosis.
  Root cause + fix plan:
  [`docs/experience/errors/2026-04-15-metal-allocator-resource-limit-panic.md`](../experience/errors/2026-04-15-metal-allocator-resource-limit-panic.md).
  CUDA is unaffected.

---

## 10 · Execution

**Claude owns**: this doc, the CLAUDE.md edit, the AGENTS.md pointer, the
wins template (§6). Reason: planning + docs per CLAUDE.md delegation rule.

**Codex owns**: `scripts/bench_guidellm.sh` (shell + jq), `pyproject.toml`
bench extra edit, `requirements-bench.txt` edit, `.gitignore` edit, the
`bench_throughput.py` deprecation banner, and running the acceptance
criteria §8 gates on a reachable server.

**Hand-off order:**
1. Claude writes this doc (done).
2. Claude writes `docs/experience/wins/TEMPLATE-bench-guidellm.md`, the
   CLAUDE.md §Benchmarks edit, and the `infer/src/http_server/AGENTS.md`
   pointer.
3. Codex implements the wrapper + dep edits + deprecation banner and runs
   §8 gates 1, 3, 4, 6, 7, 8, 9 (the ones that don't need a GPU server).
4. ckl (or Claude on the CUDA host) runs §8 gates 2, 5 end-to-end with a
   real server up.
5. Wins entry for the first real run gets authored by Claude (it's a
   docs product), citing the bench-output/ artefacts.
