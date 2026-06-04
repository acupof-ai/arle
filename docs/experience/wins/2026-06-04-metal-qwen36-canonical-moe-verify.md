# Rewrite Metal path runs the canonical Qwen3.6-35B-A3B-4bit MoE end-to-end

> **⚠️ CORRECTNESS PASS, NOT A PERF WIN.** The tok/s below were initially reported
> as a win without a Δ% vs baseline — ckl flagged them as a sizable REGRESSION vs the
> prior Metal path (2026-06-04). This entry stands as the *correctness* + end-to-end
> run record only; the perf regression is being quantified + fixed by workflow
> `metal-perf-regression-fix` (prime suspect: the rewrite dropped legacy's
> `set_wired_limit` auto-pin, which gave −82% c=1). See
> [[feedback_bench_delta_vs_baseline_not_raw]].

**Status:** correctness PASS (runs end-to-end, correct output, prefix reuse holds);
**perf = REGRESSED vs baseline, fix in flight.**
**Track:** R3-Metal (`crates/infer-metal` + `agent-bench`), branch `arch/ideal-inference-engine`.
**SKU:** Apple Silicon (this Mac), MLX, `mlx-community/Qwen3.6-35B-A3B-4bit` (~19.5 GB, MoE).

## Context

Prior Metal verification (#1/#2/#8) used Qwen3.5-0.8B (dense). Per CLAUDE.md the
canonical Metal target is the Qwen3.6-35B-A3B-4bit MoE — benching the production
shape catches MoE-specific issues a dense 0.8B can't. The rewrite Metal executor
(`infer-metal`, which parses Qwen3.5/3.6 MoE config via `MetalQwen35MoeConfig`)
had not been exercised end-to-end on the canonical MoE through the rewrite engine.

## What Worked

Added `agent-bench` test `bench_agent_workflow_metal_qwen36_canonical` (mirrors the
0.8B workflow bench, canonical model). Drives the rewrite
`Engine<MetalExecutor, MetalKvPool>` over a 3-turn growing-context agent workflow.

```
[agent-workflow METAL Qwen3.6-35B-A3B-4bit] turns=3 total_gen=144 total_wall=14.8s tok_per_s=9.7 peak_rss=19.5 GB
  turn 0 prompt_len=288 gen=48 ttft_ticks=6 wall=12.63s   <- cold: model load + warmup + first prefill
  turn 1 prompt_len=368 gen=48 ttft_ticks=3 wall=1.23s    <- prefix reuse (ttft 6->3)
  turn 2 prompt_len=448 gen=48 ttft_ticks=3 wall=0.93s    <- prefix reuse
```

Steady-state decode (turns 1-2): 48 tokens in ~1 s ≈ **48 tok/s**. The 9.7 tok/s
aggregate is dragged down by turn 0's one-time cold cost (19.5 GB MoE residency +
warmup). Prefix reuse cuts ttft 6→3 ticks and turn wall 12.6s→~1s. Forward is
correct end-to-end (no bail, coherent multi-turn generation, peak RSS = model size).

## Rule

- **Verify Metal on the canonical MoE, not just a dense small model.** The rewrite
  Metal MoE path (`MetalQwen35MoeConfig`, `is_moe_layer`) only counts as verified
  once the 35B-A3B-4bit production shape runs end-to-end — dense 0.8B doesn't
  exercise expert routing/residency.
- **Report steady-state decode separately from the cold turn.** Turn 0 folds in
  one-time model load + warmup; the aggregate tok/s understates real throughput.
  Quote both (cold turn-0 wall, steady turns-1-2 tok/s).
