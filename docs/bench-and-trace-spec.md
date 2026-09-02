# Benchmark and trace specification

> Status: Active

This is the reporting and evidence contract for benchmarks, profilers, and
performance claims. The canonical runner is `scripts/bench_throughput.py`; the
report skeleton is [`TEMPLATE-bench.md`](experience/wins/TEMPLATE-bench.md).

## 1. Required report

Every run records these sections:

1. **Goal** — one user-visible wall-clock metric and workload.
2. **Hypothesis** — one mechanism, one treatment, one predicted delta.
3. **Parameters** — exact command, workload, seed, concurrency, duration or
   request count, prompt/output token distribution, and server flags.
4. **Environment** — commit and binary hash, host, GPU, driver, CUDA/Metal,
   model, dtype, TP/EP, slots, and KV configuration.
5. **Results** — raw fixed-concurrency rows, completed/incomplete/error counts,
   decode tok/s (`1000 / ITL mean`), req/s, TTFT p50/p99, ITL p50/p99, and
   A/B delta.
6. **Problems** — correctness failures, OOM, retries, cache contamination,
   profiler gaps, and other confounders.
7. **Learnings** — PASS, KILL, or pending-remote; measured number; next wall.

Missing one section means the run does not count.

## 2. Goals

Use exactly one primary goal per run:

- **Latency:** TTFT or ITL at a fixed concurrency and request shape.
- **Throughput:** decode tok/s at fixed concurrency, or saturation throughput
  under an explicit SLO.
- **Memory:** peak device/host bytes for a fixed workload.
- **Correctness:** decoded outputs and request status for the failing slice.
- **Diagnosis:** wall-clock attribution of a named stage.

A component kernel speedup is diagnosis, not an end-to-end serving win.

## 3. Canonical benchmark

`bench_throughput.py` is the ONLY sanctioned runner. guidellm was removed
2026-07-16 (silently dropped `max_tokens` → 16-token outputs; synthetic-data
failures on the DSv4 config; c32 accounting drift) — do not reintroduce it.

Published results use a checked JSONL workload:

```bash
python3 scripts/gen_bench_prompts.py bench-agent-32k-64.jsonl 64 32768 256
python3 scripts/bench_throughput.py \
  --url http://127.0.0.1:8000 \
  --model <model> \
  --prompts-jsonl bench-agent-32k-64.jsonl \
  --concurrency-grid 1,4,8,16 \
  --requests-per-concurrency 16 \
  --max-tokens 256 \
  --seed 20260416 \
  --timeout-seconds 900 \
  --output bench-output/<label>/bench
```

Synthetic prompts are smoke-only. They may repeat prefixes and therefore cannot
license cache-sensitive changes.

The JSON artifact is the source of truth. CSV is a view. Preserve both.

The runner always sends `ignore_eos=true`: this is a fixed-output performance
workload, so every request must generate exactly `max_tokens`. Empty or invalid
decoded output still fails the correctness gate.

The warmup prepends a dedicated marker to the first workload prompt. This keeps
the production-length warmup while preventing it from priming a measured
session's prefix cache.

### 3.3 Workload — long agent sequences, one shape for everything

**Every performance claim runs on `gen_bench_prompts.py`'s 32k-token agent
contexts.** That is the shape ARLE actually serves: a coding agent replays its
whole transcript — system, tool schemas, tool outputs — on every turn. Short
prompts are a different machine. At ~3.4k tokens the run is decode-dominated
and the prefill, KV-residency, page-table, and long-context attention paths are
barely exercised; a treatment can post a large short-prompt delta and be a wash
or a regression on the real workload. Do not publish a short-prompt number and
call it a serving result.

**The workload is multi-turn, and prefix reuse is part of the machine under
test.** A real agent's turn k+1 replays turn k's transcript verbatim and appends
to it, so the served prefill is the ~1k-token delta, not the whole context —
only turn 0 of a session pays full price. A dataset of one-shot unique contexts
measures a permanent cache miss and reports a prefill-bound machine that nobody
runs. `gen_bench_prompts.py` emits `sessions × turns` prompts where turn k's
text is a strict prefix of turn k+1's, laid out turn-major so the in-flight set
at concurrency C is C distinct sessions and every reused prefix belongs to a
turn that already completed.

Shape defaults come from the coding-agent trace medians in TraceLab
(arXiv:2606.30560; 4,265 real Claude Code / Codex sessions, 350k LLM steps):
**119K prefix tokens, 875 append tokens, 214 output tokens per step, 8.8 steps
per request, 95.7% global prefix-cache hit rate.** Deviating is allowed and
sometimes forced (KV budget), but the deviation is a stated parameter of the
run, not a silent default.

Rules:

- `sessions` ≥ the highest benched concurrency, and
  `--requests-per-concurrency` a multiple of `sessions` — otherwise the tail
  turns never run and the point silently measures cold prefill only.
- Sessions are mutually unique (header + per-round indices), so reuse comes
  only from a session's own history. Report `prefix_hits` / `hit_tokens` from
  `/v1/stats` and compare against the 95.7% reference. A cold turn 1 means
  reuse is broken (eviction, KV pressure, or a cache-defeating treatment) —
  that is a finding, not a number to average through.
- Report the cold and warm slices separately; a single figure across both hides
  which of the two a treatment moved, and their ratio depends entirely on
  `turns`.
- **Confirm the length.** The generator estimates tokens at 3.6 chars/token —
  a ratio, not a tokenizer. Record `usage.prompt_tokens` p50 from the run and
  state it next to the target. Outside ±10%, re-generate with a corrected
  `target_tokens` before reporting; a run whose real p50 is 24k is not a 32k
  run.
- Prefer `--requests-per-concurrency` over `--seconds-per-concurrency` here:
  one 32k request is seconds of work, so a duration budget silently changes the
  completed-request count across arms.
- A shorter context is legitimate only as an explicit **context-scaling sweep**
  (e.g. 4k/8k/16k/32k against the same treatment), never as the single reported
  point.
- KV budget is the usual failure: 32k × concurrency must fit, or the run
  measures preemption. Record the slot line and the queue/preempt counters.

### 3.4 Per-turn TTFT — the agent-loop measurement

`scripts/bench_multiturn_ttft.py` measures what a coding-agent session feels:
one conversation, `--turns` (default 12) turns, a `--system-words` system
prompt (default ~4.8K tokens) and `--tool-words` of synthetic tool output per
turn (default ~350 tokens), every turn re-sending the whole history over
`/v1/chat/completions`. Content and the assistant replies are fixed, so two
servers receive byte-identical requests. Report turn 1 (cold), the median of
turns 2..N, and turn N; always `--warmup` so turn 1 is prefill, not model
load. A cross-server row needs the same weights on the same machine in the
same hour (§3.1), and the prefix-restore path must be proven engaged
(`prefix-attach` count in the serve log). Snapshots go to
`benchmarks/snapshots/`, the row to `benchmarks/README.md`.

### 3.0 Rolling baseline — the default iteration path

[`docs/baselines.md`](baselines.md) holds the champion row per config
fingerprint (model, TP/EP, GPU set, serve flags, slot line, dataset, build
sha). The default iteration is ONE arm: bench the candidate, compare against
the champion row. No second arm, no revert-rebuild, no env-flip arms —
every accepted binary is archived on the pod, so any later two-arm run
reuses archives.

- Candidate Δ clears the fingerprint's drift band (measured ±3%,
  2026-07-16; use >2× band): accept on the rolling comparison, update the
  champion row, archive the binary.
- Δ within the band: **do not kill — every stable positive gain is kept.**
  Escalate to §3.1 matched A/B against the archived champion binary
  (≥3 trials per arm, median + range) to resolve sign.
- Fingerprint changed (flags, GPU set, slots, dataset, driver): re-anchor
  the champion first; the stored row is invalid for comparison.
- Every ~5 accepted updates and before any default flip: one anchor A/B
  vs the oldest archived binary to bound accumulated drift.

### 3.1 Matched A/B — the disambiguation tool

Keep fixed:

- checkout, binary, machine, GPU clocks, and server lifecycle;
- model, dtype, TP/EP, slots, KV settings, and all unrelated flags;
- JSONL workload, request order, concurrency, duration, output cap, and seed.

Change one treatment variable. Run baseline and treatment from the same shell,
side by side. If the observed delta is below 5% or overlaps run variance, use
at least three trials per arm and report median plus range.

### 3.2 Scheduling envelope

For every point record:

- offered concurrency and completed request count;
- prompt and completion token min/p50/max;
- active, queued, retry, prefix-hit, and KV residency counters from `/v1/stats`;
- incomplete, errored, timed-out, or empty responses.

If these differ materially across A/B, the comparison is invalid until
explained.

## 4. Correctness gate

Before timing:

1. run one non-degenerate prompt through the same endpoint and flags;
2. require non-empty, completed, non-repeating output;
3. inspect decoded outputs on every failing slice;
4. confirm request errors and timeouts are not counted as valid samples.

For a new kernel, quantization path, rollback path, or speculative decoder, also
run the model-specific correctness gate. Token identity against another kernel
is not required when MoE non-determinism applies; coherent autoregressive output
and the model-specific gate are required.

### 4.1 Numerical acceptance for kernel optimizations

Measure numerical error before timing. Use the same inputs for the accepted
kernel and the candidate, and preserve the raw outputs or a reproducible seed.
The reference is FP64 or FP32 as required to make its error negligible relative
to the output dtype. A comparison against the old kernel alone cannot establish
accuracy.

Classify the operation before choosing the gate:

| operation | required gate |
|---|---|
| copy, split, index, reshape, mask, or address-only change | output bits match exactly; any mismatch is a correctness failure |
| pointwise math stored as BF16/FP16 | zero finite/non-finite or sign-class mismatches and at most 1 output-dtype ULP from the reference |
| GEMM, reduction, attention, recurrent update, or changed accumulation order | report `max_abs`, `p99_abs`, RMSE, max relative error with a stated near-zero floor, and cosine; the candidate may not worsen the accepted kernel's reference-error metrics by more than 5% |
| quantization or format conversion | report saturation count, zero/non-finite handling, round-trip error, and the format-specific bound; cosine alone is insufficient |

The 5% value is a regression budget relative to the accepted implementation.
It does not define absolute model accuracy. If the accepted error is zero, the
default budget is zero. A different bound must be derived and written into the
hypothesis before the run; observing the candidate first and then choosing a
bound invalidates the comparison.

Exercise every production shape touched by the change plus boundary cases:
unaligned tails, smallest and largest supported dimensions, zeros, signed small
values, and the finite extrema expected at that call site. Use at least three
seeds for generated inputs. A production trace supplies the shapes; the
numerical comparison remains required.

Then run the model gate on the exact candidate binary:

1. `scripts/lever_gate.sh` against a same-config baseline envelope, with the
   needle ladder repeated three times;
2. `python3 scripts/needle_gate.py temp` for math, quantization, reduction, or
   kernel-route changes that can preserve greedy argmax while corrupting the
   distribution;
3. require zero request errors, zero empty outputs, zero new miss classes, and
   zero looping outputs.

MoE non-determinism relaxes cross-binary token identity only at the model layer.
The bit-exact operator contract and operator reference-error bound remain. A
rollback passes when the restored source, rebuilt binary identity, and
model gate all match the named pre-change baseline; source equality alone is
insufficient.

## 5. Server and cache hygiene

- Run one benchmark process at a time.
- Start each arm from the declared cache state. For cache-independent claims,
  use unique prompts or restart and prove zero prefix hits.
- Do not reuse a server after OOM, assertion, NCCL failure, or allocator
  corruption.
- Capture server logs and `/v1/stats` before and after every point.
- A 200 response with empty output, missing usage, incomplete stream, or retry
  exhaustion is a failure, not throughput.
- ITL is valid only when each non-empty SSE event carries one completion token;
  the runner rejects event/token count mismatches instead of reporting chunk latency.
- Cold-load time is separate from request TTFT. Report it independently.

## 6. Duration and stop rules

On the long-agent workload use a fixed request count (§3.3); 120 seconds per
concurrency is the fallback for short-prompt smoke only. A run counts when it
completes enough work for stable medians and the report states the budget.
Extend or repeat when:

- the first and second halves differ by more than 5%;
- queue depth, prefix-hit rate, or memory residency is still moving;
- the treatment delta is within normal variance;
- fewer than 20 completed requests contribute to a reported point.

Stop and fix the harness or server when any request is incomplete, errored, or
empty. Do not average through failures. Stop exploring a treatment after a
controlled wall-clock KILL unless a new measurement changes the mechanism.

## 7. Tracing

Use traces to attribute an end-to-end result:

```bash
scripts/profile_nsys_bench.sh <label> --model <model> \
  --prompts-jsonl <workload.jsonl> --concurrency-grid 4 \
  --seconds-per-concurrency 60
scripts/profile_ncu_bench.sh <label> --family <kernel-family> --model <model> \
  --prompts-jsonl <workload.jsonl> --concurrency-grid 4 \
  --seconds-per-concurrency 60
```

Every trace reports both:

- time inside the selected NVTX or kernel window;
- that time divided by per-request or whole-run wall clock.

Host/GPU overlap claims require timestamps on the same timeline: request start,
host enqueue begin/end, CUDA event or kernel begin/end, synchronization, and
response completion. A visual overlap alone is not proof. Export the timeline
or SQLite rows used for the calculation.

Nsight Compute is a kernel microbenchmark. Pair it with the fixed-concurrency
native benchmark before claiming serving impact.

## 8. Acceptance

A performance change is built in when:

1. correctness passes;
2. the matched A/B scheduling envelope aligns;
3. the named wall-clock metric improves on the target workload;
4. no TTFT, ITL, throughput, memory, or failure-rate regression violates the
   stated SLO;
5. raw JSON/CSV, logs, and any profiler artifacts are preserved;
6. a dated `wins/` or `errors/` entry links the baseline and treatment.

There is no minimum gain threshold. Any positive target wall-clock median over
at least three matched trials is enough. If the sign is within observed
run-to-run noise, keep measuring until it resolves. A correctness failure stays
non-default regardless of speed; retain measured work for repair instead of
deleting it.

Use the conservative framing when component and wall-clock percentages disagree.
The per-request result wins.

## 9. Repository lifecycle

Runtime and benchmark-parameter changes require a dated report under
`docs/experience/wins/` or `docs/experience/errors/`. If hardware is unavailable,
commit a `pending-remote` report naming the exact remote gate. Documentation and
dev-only tooling are exempt; state the exemption in the commit body.

Never overwrite a report. An after report links its baseline and reports the
delta. Update CHANGELOG on a phase exit, default flip, or license-or-kill verdict.
