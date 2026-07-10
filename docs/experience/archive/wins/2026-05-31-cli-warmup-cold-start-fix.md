# In-process CLI cold-start fix: warmup at load (9 → 64 tok/s first turn) + REPL leading-whitespace

## Context

User on the `arle` agent REPL saw decode at **9–14 tok/s** (not the ~80 they
expected from benches) and "多一行空格" (extra blank lines before the reply /
first tool line).

## Root cause (measured, clean single-instance — confounded measurements with two
concurrent `arle` were discarded)

The in-process CLI path (`LoadedInferenceEngine::load`) set the wired *limit* but
ran **no startup warmup**, unlike `metal_serve` (which does `run_startup_warmup`).
So the 19 GB of weights were mmap'd-lazy, not resident — the first turns paged
them in from disk. Clean 3-turn ramp on Qwen3.6 (35B-A3B MoE, only 3 B active per
token, so experts fault in gradually):

| Turn | TTFT | decode | process RSS |
|---|---|---|---|
| 1 (cold) | 4.9 s | **8.9 tok/s** | (climbing) |
| 2 | 405 ms | 24.9 | — |
| 3 (warm) | 137 ms | **75.1 tok/s** | **18.5 GB resident** |

The blank lines: the model streams leading `\n\n` (thinking) which was printed
verbatim, then `on_trace_event` added another `println!()` before the tool line.

## What Worked

- **`loaded.rs::warmup()`** — a warmup generation at load, the in-process analogue
  of `metal_serve`'s startup warmup. Key: a **long, lexically-diverse prompt**
  prefills ~120 tokens in one batched pass, each routing to its own MoE experts,
  so it faults+wires a broad expert set cheaply (prefill ~300 tok/s) rather than
  via slow token-by-token decode. + a visible `warming up model weights… done`
  on stderr. Best-effort (never blocks startup). Sync `complete()` is safe — the
  CLI `run()` is not under a tokio runtime, so `blocking_recv` doesn't panic.
- **`repl.rs` leading-whitespace trim** — `on_text_chunk` trims `trim_start()`
  while no visible text has streamed yet (per turn), so the reply/tool line begins
  right after the prompt; internal whitespace preserved once real text starts.

## Results (clean, single instance, Qwen3.6 Metal, M4 Pro)

| | before | after |
|---|---|---|
| **Turn-1 decode** | 8.9 tok/s | **63.7 tok/s** (7.2×) |
| Turn-1 TTFT | 4.9 s | 296 ms |
| Load wall (incl. warmup) | 1.8 s | 5.6 s |
| Leading blank lines | 2–3 | 0 |

Turn-1 63.7 is near the in-process warm ceiling (~75; `metal_serve` ~88 — the gap
is the agent layer: 2 channel hops + 200 µs poll + per-chunk tps metering). Turn 2+
is fully warm.

## Rule

A wired *limit* is not residency — `set_wired_limit` permits wiring, it doesn't
fault the pages in. Any in-process model path that wants warm first-token latency
must run a warmup forward (mirror `metal_serve`), and for a sparse MoE the warmup
should be a **diverse prompt** (prefill-routes many experts) not just N greedy
decode tokens. Measure cold-start with ONE instance — two `arle` each wiring 20 GB
on a 48 GB box thrash and produce fantasy (3–4 tok/s) numbers.
