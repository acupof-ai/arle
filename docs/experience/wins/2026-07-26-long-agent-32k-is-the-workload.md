# The bench workload is now 32k agent contexts — the short one was measuring a different machine

## Context

Every serving number in this repo came from ~130-token to ~3.4k-token prompts.
ARLE serves a coding agent: each turn replays the whole transcript — system,
tool schemas, tool outputs — so the real prompt is tens of thousands of tokens.
The spec-decode work of 2026-07-25/26 exposed the mismatch: a phase timer said
the decode step was the whole story, while the aggregate throughput refused to
move in proportion.

## What Worked

`gen_bench_prompts.py` now emits the canonical workload: 64 agent contexts of
32k tokens each — system + tool schemas, then repeated
(ask → tool_call → tool output → summary) rounds, then the live question. Unique
header and per-round indices per doc, so prefix reuse cannot mask prefill cost.
The bench spec (§3.3), `docs/baselines.md` (rule 5), `TEMPLATE-bench.md`,
`agent-method.md`, and `run_dsv4_bench.sh` all point at it; the short-prompt
champion rows are marked retired rather than deleted.

First anchor (ThinkingCap-Qwen3.6-27B-FP8, 1×H20, no spec, 8 req/point,
max_tokens 256, measured `prompt_tokens` 33000):

| c | out tok/s | total tok/s | wall s |
|---|----------:|------------:|-------:|
| 1 | 3.4       | 486.0       | 547.0  |
| 4 | 12.9      | 1748.9      | 152.0  |
| 8 | 14.3      | 1906.7      | 139.5  |

**68.4 s per request at c=1, of which ~7 s is the 256-token decode — ~89% is the
33k prefill, running at ~540 tok/s.** The same serve on ~130-token prompts
reports 38 out tok/s. The 11× gap is entirely prompt length, and it re-scopes
the whole decode-side effort: speculative decoding accelerates the ~10% slice.
Prefill throughput also degrades with length (~1270 tok/s at ~5k prompt tokens
vs ~540 at 33k), which is the next thing to probe.

Two harness defects surfaced on the first run, both invisible at short lengths:

- The generator's chars/token constant was 3.6 (a prose ratio). Agent content —
  code, JSON, log lines — measures **2.80**, so a "32k" dataset was really 42.6k.
  The spec's ±10% confirm-against-`usage.prompt_tokens` rule caught it on run one.
- `bench_throughput.py` strips the prompt and posts it to `/v1/completions`. The
  dataset ended on a full stop, so greedy decoding answered EOS: 1 completion
  token, zero output events, warmup refused the run. The dataset now ends on the
  `assistant:` cue. The runner also only counted `delta.content`, so a thinking
  model streaming `reasoning_content` reported zero events on a short cap.

## Rule

A benchmark shape is a claim about the machine under test. At ~3.4k tokens the
run is decode-bound and the prefill, KV-residency, page-table, and long-context
attention paths are barely exercised — a treatment can post a large delta there
and move nothing that a user waits on. Measure the shape you serve, and confirm
the prompt length from the server's own `usage`, not from a chars/token ratio:
this dataset was 30% longer than its name on the first attempt.
