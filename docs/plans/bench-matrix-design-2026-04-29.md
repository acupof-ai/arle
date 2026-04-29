# Bench matrix redesign — 2026-04-29

Captures gaps identified comparing our current canonical bench
(`scripts/bench_guidellm.sh`) against SGLang's `bench_serving.py` and
vLLM's `vllm bench serve`. Output of a research subagent pass on
2026-04-29; the matrix below is proposed-but-not-yet-implemented.

---

## Current state (canonical, today)

`scripts/bench_guidellm.sh <label>` runs:

- **Profile**: guidellm 0.6.0 `--profile sweep` (auto-picks 10 strategies)
- **Data**: clamped synthetic, `prompt_tokens=4096`, `output_tokens=256`
- **Duration**: `--max-seconds 60` per strategy
- **Strategy expansion**: sync → 8 async-constant linspaced rates → throughput
  - On L4 / Qwen3-4B FP8: sync=0.10 r/s, throughput=0.27 r/s, intermediates
    `0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.27` r/s.
- **Backend hit**: `/v1/completions` over OpenAI HTTP
- **Headline metrics**: TTFT p50/p99, ITL p50/p99, out tok/s, req/s actual

Exploration escapes: `--fast` (30 s c=16), `--quick` (~4 min c=1,2,4,8).

## Gaps vs SGLang / vLLM canonical

1. **No realistic dataset.** SGLang / vLLM default to ShareGPT (real
   chat prompts/responses) with `--num-prompts 1000`. We emit only
   clamped-synthetic random — masks chat-shape variance.
2. **No long-context coverage.** 4096-in is short for KV-quant /
   KV-tier ROI bench. SGLang/vLLM run 8k-32k for prefill scaling.
3. **Sweep linspace too narrow.** L4 sweep produces 0.10-0.27 r/s
   (1× to 2.7× sync). SGLang/vLLM use Poisson at fixed rates OR
   `--max-concurrency` lists `{1, 2, 4, 8, 16, 32, 64}` — log-spaced
   concurrency with steady-state percentiles.
4. **No Poisson arrivals.** SGLang defaults to Poisson via
   `np.random.exponential(1/rate)`; vLLM defaults Poisson with
   `--burstiness 1.0`. Our sweep uses `AsyncConstantStrategy`
   (deterministic) — masks burst-handling behaviour.
5. **Missing metrics in headline.** SGLang reports
   `mean/median/std/p99 TTFT`, `TPOT` (distinct from ITL), `ITL
   p95/p99/max`, `e2e p90/p99`, `total_throughput`,
   `peak_concurrency`, `max_output_tokens_per_s`. Our headline drops
   std-dev, TPOT, p95 ITL, e2e percentiles, peak concurrency, input
   throughput.
6. **No isolation / decode-bound case.** Pure-decode (small prompt,
   long output, c=1) is the canonical ITL ceiling probe — we lack
   one.

## Proposed bench matrix

| Workload         | Dataset        | Prompt        | Output | Concurrency        | Duration  | Headline metric                  |
|------------------|----------------|---------------|--------|--------------------|-----------|----------------------------------|
| short-chat       | ShareGPT       | sampled 50-500 | 80-300 | sweep + Poisson @ {1,4,16,64} | 90 s/pt   | sustained tok/s at TTFT p95      |
| max-throughput   | random clamped | 4096          | 256    | `throughput` unbounded         | 120 s     | peak output tok/s (current)      |
| decode-isolation | random clamped | 1024          | 1024   | sync (c=1)                     | 60 s      | ITL p50/p99 ceiling              |
| long-context-8k  | random clamped | 8192          | 256    | concurrent {1, 4}              | 120 s     | TTFT p50, peak `kv_util`         |
| long-context-32k | random clamped | 32768         | 256    | concurrent {1}                 | 180 s     | KV-tier recall, prefill TFLOPs   |
| prefix-cache     | shared-prefix synth | 2048 prefix + 256 unique | 128 | concurrent {16}    | 90 s      | `prefix_hit_rate`, TTFT delta    |

The current 4096-in/256-out clamped-synthetic stays as the cross-backend
**max-throughput** reference. The other rows are additive.

## Concrete script changes (not yet implemented)

1. **`--workload {sharegpt|maxthroughput|decode-iso|longctx-8k|longctx-32k|prefix-cache}`** flag in `scripts/bench_guidellm.sh`. Each
   maps to a frozen `(PROFILE, DATA, RATE_OVERRIDE, MAX_SECONDS)`
   tuple. Workload-pinned values still produce wins entries
   (canonical mode); ad-hoc overrides remain exploration-only.

2. **`--strategy {constant|poisson}`** flag — flip guidellm's
   `--profile sweep` async leg (sweep accepts `strategy_type` per
   `profiles.py:589`).

3. **Extend `extract_rows` jq filter** at lines 680-714 to add
   `ttft_mean`, `ttft_std`, `tpot_p50`, `itl_p95`, `e2e_p99`,
   `peak_concurrency`, `input_tok/s` columns to match SGLang.

4. **`bench-output/datasets/sharegpt.jsonl`** + a `--data file:...`
   form (guidellm 0.6.0 supports custom JSONL via
   `--data prompt_field=...`). ShareGPT lives outside git
   (gitignore).

5. **Update `docs/plans/guidellm-integration.md` §3** to document the
   workload matrix as additive: 4096/256 synthetic stays the
   cross-backend reference; the others are layered.

## Open questions (pre-implementation)

- **ShareGPT availability.** No copy under repo (`find` returned
  nothing). Decide: pull `anon8231489123/ShareGPT_Vicuna_unfiltered`
  from HF, or use SGLang's loader (`bench_serving.py` downloads to
  `~/.cache`) and symlink to `bench-output/datasets/`.
- **Custom-dataset format in guidellm 0.6.0.** `profiles.py` only
  handles strategy generation; the dataset path is
  `guidellm/preprocess/` (not opened by the research pass).
  Need to confirm whether `--data` accepts a JSONL file directly or
  requires the `synthetic_text` deserialiser fork.
- **Long-context vs current `--max-seq-len`.** Server's KV pool
  sizing on L4 (22 GB) caps real long-ctx capacity; verify 32k × 1
  concurrent fits with current pool (148k tokens at fp8) before
  committing the `longctx-32k` row.
- **Poisson sweep stability.** Variance budget in
  `bench-and-trace-spec.md` §5 is 2% across repeats — Poisson at
  low rates may bust this on a 60 s window. May need 120 s for
  `longctx-32k`.

## Source paths / references

- `scripts/bench_guidellm.sh` (lines 57-84, 601-622, 680-714)
- `docs/plans/guidellm-integration.md` §3
- `docs/bench-and-trace-spec.md` §1, §5
- `docs/experience/wins/2026-04-29-bench-guidellm-canonsweep-v3-fp8.md`
- `.venv/lib/python3.12/site-packages/sglang/bench_serving.py`
  (lines 875-910 metrics, 943-950 Poisson, 1929-2058 args)
- `.venv/lib/python3.12/site-packages/guidellm/benchmark/profiles.py`
  (lines 574-720 sweep)
- `.venv/lib/python3.12/site-packages/guidellm/settings.py:87`
  (`default_sweep_number = 10`)
- `https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/benchmarks/serve.py`
  (datasets list, `--burstiness`, `--ramp-up-strategy`,
  `--num-prompts=1000`, `--request-rate=inf`)

---

This file is a **plan**. It will be split into individual
implementation tickets and PRs. The sweep matrix above will land as
incremental commits; the wins-entry contract gets one canonical row
per workload, dated and diff'd against the prior.
