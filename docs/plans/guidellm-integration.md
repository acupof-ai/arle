# GuideLLM integration — canonical bench truth source

**Status:** 🟡 planned (2026-04-16) — execution to be delegated to Codex.
**Owner:** ckl. **Drives:** replaces `scripts/bench_throughput_sweep.py`
as the project's **canonical throughput/latency measurement tool**.
**Trigger:** 2026-04-16 discussion — the house-grown sweep script
overlaps 1:1 with a well-maintained upstream tool ([vllm-project/guidellm](https://github.com/vllm-project/guidellm)),
and we're about to start cross-referencing numbers with vLLM/SGLang where
guidellm is already the reference point. Pick the tool everyone else picks,
stop hand-rolling.

---

## 1 · Why guidellm becomes the truth source

- **LLM-native metrics**: TTFT / ITL / tok-s / request-rate distributions
  with p50/p90/p95/p99, not generic HTTP RPS. Our existing
  `bench_throughput_sweep.py` only emits mean + stddev.
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
- PPL / quality evals — out of scope; guidellm is a performance tool.
- `scripts/bench_throughput_sweep.py` — kept in-tree with a deprecation
  banner so existing wins entries stay reproducible, but new wins MUST cite
  guidellm. See §4.

---

## 2 · Decisions (locked 2026-04-16)

| # | Decision | Rationale |
|---|---|---|
| 1 | **guidellm = sole truth** for throughput / TTFT / ITL wins. `bench_throughput_sweep.py` gets a deprecation banner but stays. | Don't delete reproducibility for past wins; don't duplicate future wins. |
| 2 | **Wrapper script assumes the server is already running.** | Avoids "my bench crashed the server" failure mode; keeps concerns orthogonal. |
| 3 | **Same canonical config for CUDA and Metal backends.** | One `profile=sweep`, one dataset shape — makes cross-backend comparison a pure hardware delta, not a dataset delta. |

---

## 3 · Canonical bench parameters (the "truth" definition)

Write **once** into `scripts/bench_guidellm.sh` and
`docs/experience/wins/TEMPLATE-bench-guidellm.md`. Changing these values is
a deliberate act, not a flag flip — any change lands in a commit whose
subject says so, and new wins reference the date of the change.

```
--profile sweep
--data   prompt_tokens=1024,output_tokens=256
--max-seconds 60
--outputs json,csv,html
--random-seed 20260416
```

**Why these specific numbers:**
- `prompt_tokens=1024,output_tokens=256` — midpoint of the shapes already
  appearing in `docs/experience/wins/` (tiered-KV entries hover around
  1K prompts; Metal Qwen3.5 decode wins use 512–2048).
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
│   ├── bench_guidellm.sh                                              # ← NEW canonical wrapper
│   └── bench_throughput_sweep.py                                      # ← deprecation banner
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
  scripts/bench_guidellm.sh <backend-label> [--target URL] [--model NAME]

Required:
  <backend-label>      e.g. cuda-h100, cuda-a100, metal-m3max
                       used to name the output directory and wins file

Defaults (override with flags):
  --target   http://localhost:8000
  --model    Qwen/Qwen3-4B        (matches default HTTP server startup)

Behaviour:
  1. Check `guidellm` is on PATH. If not → print `pip install -e .[bench]`
     hint and exit 2.
  2. Check target responds to `/v1/models`. If not → exit 2 with
     "server not running at <target>, start it with
     scripts/start_pegainfer.sh first".
  3. Invoke:
        guidellm benchmark \
            --target "$TARGET" --model "$MODEL" \
            --profile sweep \
            --data "prompt_tokens=1024,output_tokens=256" \
            --max-seconds 60 \
            --random-seed 20260416 \
            --output-dir "bench-output/$(date +%Y-%m-%d)-$LABEL/" \
            --outputs json,csv,html
  4. Extract headline metrics from benchmarks.json:
        sweep rate points (req/s)
        TTFT p50 / p99 (ms)
        ITL  p50 / p99 (ms)
        output tok/s per rate point
  5. Print them as a markdown table on stdout, and write the same table
     plus the filesystem path of the HTML report into a new
     `docs/experience/wins/YYYY-MM-DD-bench-guidellm-<label>.md`
     file, seeded from `TEMPLATE-bench-guidellm.md`.

Exit codes:
  0   bench completed, wins stub written
  2   environment not ready (guidellm missing, server down)
  3   guidellm exited non-zero
```

Metric extraction is **jq** (already on both dev hosts) — no extra Python.
The wrapper's only hard dependency is `guidellm` itself + `jq` + `curl`.

---

## 6 · Wins entry template

`docs/experience/wins/TEMPLATE-bench-guidellm.md`:

```markdown
# <short title> — guidellm sweep, <backend-label>, <date>

## Context
- Backend: <cuda|metal> · model: <Qwen/Qwen3-4B | ...>
- Hardware: <GPU/SoC model, VRAM, CUDA/Metal version>
- Commit: <short sha>
- Feature set: <cargo features>
- Non-default flags: <env vars, server flags>

## Canonical params
- `--profile sweep  --data prompt_tokens=1024,output_tokens=256  --max-seconds 60  --random-seed 20260416`
- Wrapper: `scripts/bench_guidellm.sh <label>`

## Results (headline table)
| rate (req/s) | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | out tok/s |
|---|---|---|---|---|---|
| ... sweep points ... |

## Artefacts
- Raw: `bench-output/<date>-<label>/benchmarks.{json,csv,html}`
- HTML report: `bench-output/<date>-<label>/benchmarks.html`

## Delta vs previous
- Baseline: [`<date>-bench-guidellm-<label>.md`](<relative path>)
- % change per column if prior snapshot exists.
```

CLAUDE.md rule "Never overwrite before-snapshots" carries over verbatim —
the `<date>-<label>` naming enforces it at the filesystem level.

---

## 7 · CLAUDE.md edit (§Benchmarks)

Replace:

> - Tool: `scripts/bench_throughput_sweep.py --label <name>`.

With:

> - **Canonical tool: `scripts/bench_guidellm.sh <label>`** (thin wrapper
>   around [`vllm-project/guidellm`](https://github.com/vllm-project/guidellm)).
>   Parameters are locked in `docs/plans/guidellm-integration.md` §3;
>   changing them is a deliberate commit, not a flag flip.
> - `scripts/bench_throughput_sweep.py` is **deprecated**; kept only so
>   historical wins remain reproducible. New wins MUST use the guidellm
>   wrapper.

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
7. `scripts/bench_throughput_sweep.py --help` still works AND prints a
   `DEPRECATED: use scripts/bench_guidellm.sh instead` notice to stderr
   before the normal help.
8. `CLAUDE.md` §Benchmarks no longer mentions `bench_throughput_sweep.py`
   as the canonical tool.
9. `grep -R "bench_throughput_sweep" docs/ | wc -l` drops only where it's
   called out as deprecated — historical wins must still reference it.

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

---

## 10 · Execution

**Claude owns**: this doc, the CLAUDE.md edit, the AGENTS.md pointer, the
wins template (§6). Reason: planning + docs per CLAUDE.md delegation rule.

**Codex owns**: `scripts/bench_guidellm.sh` (shell + jq), `pyproject.toml`
bench extra edit, `requirements-bench.txt` edit, `.gitignore` edit, the
`bench_throughput_sweep.py` deprecation banner, and running the
acceptance criteria §8 gates on a reachable server.

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
