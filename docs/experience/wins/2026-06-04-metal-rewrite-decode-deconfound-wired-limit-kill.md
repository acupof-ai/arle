# Metal rewrite "33.5% decode regression" re-localized: real decode gap is ~−18%, wired-limit auto-pin KILLED as the cause

## Context

Continuing the `infer-metal` (rewrite, `Engine<MetalExecutor, MetalKvPool>`,
branch `arch/ideal-inference-engine`) perf-regression hunt. A prior workflow
confirmed the regression by same-session matched A/B and then KILLED the
per-token `publish_slot` / `drain_session` cadence hypothesis
([`errors/2026-06-04-metal-rewrite-publish-drain-cadence-kill.md`](../errors/2026-06-04-metal-rewrite-publish-drain-cadence-kill.md)).
The headline "rewrite is 33.5% slower than legacy" compared the rewrite
agent-workflow **turn-wall tok/s** against legacy `metal_bench`
`generation_tps` = **pure steady decode** (prefill + warmup excluded). That is
not apples-to-apples. This entry resolves two ranked levers:

- **Lever 1** — de-confound the framing: get the rewrite's PURE decode tok/s.
- **Lever 2** — wired-limit auto-pin (unconditional `mlx::set_wired_limit`,
  ~2× residency) as a decode-throughput cause, via a real same-binary toggle
  (the prior workflow falsified it by source survey only, never an A/B).

All runs: canonical `mlx-community/Qwen3.6-35B-A3B-4bit` MoE (+ Qwen3.5-0.8B
sanity), c=1, `agent-bench` `synthetic(256, 3, 32, 48)`, M-series Mac (48 GB),
warmup = turn-0 cold prefill (reported separately). A `std::sync::Once` probe
printed the live wired-limit state on every run
([`feedback_path_probe_before_perf_claim`]).

## What Worked

### Lever 1 — framing de-confounded (the real verdict)

Instrumented `agent-bench::run_agent_workflow_with_probe` to split each
`engine.step()` wall into **prefill-phase** vs **decode-phase**, keyed on the
existing `TtftObserver` first-token boundary (a step is prefill-phase iff no
first token had been committed when it began). New `TurnMetric.prefill_wall`
/ `decode_wall` + `WorkflowMetrics::{pure_decode_tok_s, turn_wall_tok_s,
total_prefill_wall, total_decode_wall}`.

Qwen3.6 canonical (auto-pin ON = HEAD default), per-turn decode is flat and the
turn-wall framing is dominated by the turn-0 cold prefill:

| framing                  | value      |
|--------------------------|------------|
| turn_wall tok/s (confounded) | **18.1–18.4** |
| **PURE decode tok/s**    | **66.3 / 69.2 / 69.2** (3 runs) |
| turn-0 prefill_wall      | 5.4 s (cold: graph build + first MoE encode) |
| turn-1/2 prefill_wall    | ~0.20 s (radix prefix reuse fires) |
| per-turn decode tok/s    | 64.5 / 68.3 / 66.3 (stable) |

**Verdict:** the rewrite PURE decode is **~69 tok/s vs legacy 84.3 = −18%** — a
REAL decode regression, but materially smaller than the −33.5% turn-wall number.
The extra gap in the turn-wall framing is the **turn-0 cold prefill (5.4 s over
a 3-turn / 8 s workflow)**, not the decode hot path. The fix locus is split:
(a) a ~−18% steady-decode regression in the MoE forward / step round-trip, and
(b) a large cold-prefill / graph-build cost amortized over only 3 turns.
Qwen3.5-0.8B sanity: PURE decode 242.5 vs legacy 282.5 = −14% (same direction,
same magnitude band).

Correctness gate held in every run: TTFT ticks 6 → 3 across turns (radix prefix
reuse confirmed).

### Lever 2 — wired-limit auto-pin: KILLED by matched A/B

Added an `INFER_METAL_WIRED_LIMIT=0` opt-out to the rewrite executor, then ran
auto-pin ON (HEAD) vs OFF on the SAME freshly-built binary, probe-confirmed:

| state (Qwen3.6)          | PURE decode tok/s | peak RSS | turn-0 prefill_wall |
|--------------------------|-------------------|----------|---------------------|
| A: auto-pin ON (HEAD)    | 66.3, **69.2** (rerun) | 19.55 GB | 5.39 s |
| B: auto-pin OFF          | **69.2**          | 10.22 GB | **9.13 s** |

- **Decode: no reproducible effect.** A reruns at 69.2 = identical to B's 69.2;
  the first A=66.3 was within-state variance. The +4.4% A→B in the first pair
  did not survive a rerun → noise, below the ≥10% bar. Qwen3.5-0.8B: A 242.5 vs
  B 234.1 (auto-pin OFF *slower*, wash). **Lever 2 KILLED.**
- **Mechanism confirmed but irrelevant to throughput.** RSS 19.55 GB → 10.22 GB
  (matches legacy 8.9–11.1 GB band) proves the pin is real, but it buys no
  decode speed on an unpressured 48 GB host.
- **Counter-finding — auto-pin OFF HURTS cold prefill.** turn-0 prefill 5.39 s
  → 9.13 s (+69%): without pinning, the first MoE encode pages weights in under
  graph-build pressure. Auto-pin ON is the correct default; the +9 GB RSS it
  costs is the documented p99-ITL-under-pressure insurance, and removing it is a
  net loss.

executor.rs was reverted to HEAD after the A/B (clean kill). The Lever 1
prefill/decode-split instrumentation is KEPT in `agent-bench` — it is the
diagnostic that produced the verdict, not a fix.

## Files changed

- `crates/agent-bench/src/lib.rs` — `TurnMetric.{prefill_wall,decode_wall}`;
  `WorkflowMetrics::{total_prefill_wall,total_decode_wall,turn_wall_tok_s,
  pure_decode_tok_s}`; per-step prefill/decode wall split in
  `run_agent_workflow_with_probe`; both Metal benches now print the split +
  pure-decode tok/s + peak RSS GB. (KEPT — measurement infra.)
- `crates/infer-metal/src/executor.rs` — Lever-2 `INFER_METAL_WIRED_LIMIT=0`
  opt-out, used for the A/B then REVERTED. (tree clean at HEAD.)

## Rule

- A rewrite-vs-legacy throughput number that compares **turn-wall (incl. cold
  prefill + scheduler ticks)** against legacy **pure `generation_tps`** is a
  framing confounder. Split per-step wall on the first-token boundary BEFORE
  attributing the gap to decode: here −33.5% turn-wall decomposed into −18% real
  decode + a turn-0 cold-prefill tail.
- "Source survey falsified it" ≠ "a matched A/B killed it." The wired-limit
  lever needed a real same-binary toggle: it confirmed the 2× RSS mechanism yet
  showed ZERO reproducible decode effect AND a +69% cold-prefill regression when
  disabled. Pin stays on.
- Rerun the winning state of any A/B before crediting a single-pair delta:
  A=66.3 vs B=69.2 looked like +4.4% until A reran at 69.2 = noise.

## Next-ranked lever (both above resolved; decode gap remains)

The real ~−18% steady-decode regression is now isolated to the MoE forward /
step path. Next lever: the rewrite `step()` **two-phase submit→poll round-trip**
— `submit_decode` does `async_eval(logits)` then `argmax` + a SECOND
`async_eval(sampled)`, and `poll` does a THIRD `mlx::eval(sampled)` on the next
tick; legacy `run_cpp_step` resolves the token in a single eval. Per-token that
is one extra device round-trip + one extra scheduler tick of latency on the
critical path. A/B: collapse the greedy poll into the submit eval (resolve the
argmax scalar inside `submit_decode` before returning `Ready`) vs HEAD
two-phase, same matched protocol.
