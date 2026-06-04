# TOP Metal perf fix KILLED: per-token publish + C++ session teardown are NOT the rewrite-vs-legacy decode gap

## Context

Rewrite `infer-metal` (`Engine<MetalExecutor, MetalKvPool>`, branch
`arch/ideal-inference-engine`) measured **33.5% slower** than the legacy
`infer/src/backend/metal` backend on canonical Qwen3.6-35B-A3B-4bit c=1 (56.1 vs
84.3 tok/s; Qwen3.5-0.8B 26%: 209 vs 282.5) by a clean same-session matched A/B
(`metal_bench` legacy reference, in-tree at HEAD). Ranked root causes were:
(1) per-token `drain_session`/`begin_session` churn in `submit_decode`
(`executor.rs:327`), and (2) per-token full-history `publish_slot` re-slicing all
pages from page 0. Both projected to recover most of the gap toward 84.3 and drop
RSS from 19.55 GB to the legacy 8.9-11.1 GB band.

## Root Cause (of the FALSE hypothesis)

Both causes were tested as **isolated, env-gated, single-variable, same-binary,
side-by-side A/Bs** on canonical Qwen3.6 MoE (`bench_agent_workflow_metal_qwen36_canonical`,
c=1, warmup excluded, wall-clock per-turn steady framing). A `std::sync::Once`
probe confirmed each toggle was live on the bench path
(`feedback_path_probe_before_perf_claim`).

- **Var1** `INFER_METAL_DEFER_PUBLISH` (slice only newly-completed pages vs all
  from 0): A=54.2/56.3, B=55.6/55.9 → +2.6% then −0.7%, sign-swinging noise.
  Qwen3.5-0.8B 212.8 vs 212.6 (0%).
- **Var2** `INFER_METAL_RESIDENT_SESSION` (keep C++ session resident, drain+publish
  only on page boundary): A=51.8/56.5, B=40.6/57.5 → −21.6% then +1.8%, huge
  within-state variance, no reproducible recovery.
- **RSS pinned at 19.55 GB in EVERY state** — the page-buffer-pinning mechanism
  predicted RSS would fall once pages stopped accumulating; it did not.
  Independently falsifies that mechanism.

The estimate assumed ~19 pages/token of re-slicing; the probe proved the steady
bench context is only **4 full pages** (page_size 16) → ~80 `mlx::slice` nodes/token
is negligible vs the MoE forward's ~600-1000 primitives. The publish/drain decode
cadence is simply not the bottleneck at this shape.

## Fix

Reverted both env-gated changes; tree left clean. The real gap is elsewhere — most
likely (a) the agent-workflow **TURN-WALL framing confounder** (rewrite 56 folds in
per-turn suffix re-prefill at `chunked_prefill_size=64` + scheduler/poll ticks;
legacy 84.3 is pure `generation_tps`), and/or (b) the **wired-limit auto-pin**
(`executor.rs:106-115`, rewrite pins ~2× residency = 19.55 GB and was FALSIFIED by
source-survey only, never by a matched A/B toggle on the rewrite binary), and/or
(c) the rewrite `step()` two-phase submit→poll round-trip (a second `mlx::eval`).

## Rule

- A step-time-decomposition / launch-count source hypothesis for a decode
  regression is **not evidence** — confirm the assumed scaling (here: pages/token)
  with a runtime probe BEFORE crediting it, and isolate each variable with a
  same-binary env-gated A/B.
- When a rewrite-vs-legacy number compares **turn-wall (incl. prefill+scheduler)**
  against legacy **pure decode `generation_tps`**, the framing itself is a
  confounder: decompose prefill-wall vs decode-wall before attributing the gap to
  the decode hot path.
- "Source survey falsified it" is not the same as "a matched A/B killed it" — the
  wired-limit lever was dismissed by reading code; it still needs a real toggle.
