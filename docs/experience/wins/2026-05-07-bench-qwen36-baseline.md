# Bench — Qwen3.6 35B-A3B Metal canonical baseline — 2026-05-07

## Goal

Establish a baseline for the new Metal canonical model
(`mlx-community/Qwen3.6-35B-A3B-4bit` MoE, 35B total / 3B active per
token, per AGENTS.md "Metal canonical model" landed commit 97db09e),
and validate the c=8→c=10 step-jump hypothesis from
[`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md)
technique #2 (`MLX_MAX_OPS_PER_BUFFER=200` + `MLX_MAX_MB_PER_BUFFER=200`).

## Hypothesis

The MLX command-buffer commit cliff that bites at c=8→c=10 on Qwen3.5
will close with the boosted env tunes on Qwen3.6 too — same MLX
allocator path, larger model just means more ops per buffer.

## Params

- Binary: `target/release/metal_serve` rebuilt with oMLX-C v3
  (default-on, no env flag) at the canonicalize-Qwen3.6 commit
  (`97db09e`).
- Model: `mlx-community/Qwen3.6-35B-A3B-4bit` (cached in HF hub,
  ~19 GB, 4 safetensors shards).
- `--max-running-requests 16`.
- Workload: `/tmp/cN_smoke.sh <N>` — N concurrent
  /v1/chat/completions, max_tokens=64, temperature=0.0. Same workload
  as
  [`2026-05-07-bench-c-scaling-omlx-c-default.md`](2026-05-07-bench-c-scaling-omlx-c-default.md)
  for shape-comparability against Qwen3.5.
- A/B: env vars unset (baseline) vs
  `MLX_MAX_OPS_PER_BUFFER=200 MLX_MAX_MB_PER_BUFFER=200` (cmdbuf200).

## Results — total step μs at c=1/4/8/16

| batch | baseline avg | baseline p50 | cmdbuf200 avg | cmdbuf200 p50 | Δ p50 |
|------:|-------------:|-------------:|--------------:|--------------:|------:|
| c=1   | 12706 | 11625 | 12228 | 11357 | −2.3% |
| c=2   | 18322 | 16585 | 14782 | 15138 | −8.7% |
| c=3   | 19911 | 20314 | 28750 | 21651 | +6.6% |
| c=4   | 24311 | 24230 | 27199 | 23853 | −1.6% |
| c=8   | 41448 | 41510 | 42064 | 41534 | +0.1% |
| c=16  | 73229 | 67811 | 75692 | 70308 | +3.7% |

→ **Hypothesis falsified for Qwen3.6.** The MLX cmdbuf env tune is
essentially noise (max ±9% on small samples, no consistent direction)
on Qwen3.6 MoE. The c=8→c=10 cliff predicted on Qwen3.5 dense doesn't
materialize the same way on a 35B MoE — the bottleneck has shifted.

## Phase breakdown (oMLX-C pipelined, baseline) — the new bottleneck

| phase | c=4 p50 μs | c=8 p50 μs | c=16 p50 μs |
|-------|-----------:|-----------:|------------:|
| `prep` | 11 (0.0%) | 13 (0.0%) | 12 (0.0%) |
| `build_graph` | 1147 (4.7%) | 1388 (3.3%) | 2423 (3.6%) |
| **`async_kick`** | **22985 (94.9%)** | **40045 (96.5%)** | **65344 (96.4%)** |
| `sample` | 10 (0.0%) | 12 (0.0%) | 13 (0.0%) |
| `pool_dw` | 0 | 0 | 0 |
| total | 24230 | 41510 | 67811 |

→ **`async_eval` itself is now 95% of step time.** That's not
"kickoff" — MLX is doing the MoE forward encode + commit
synchronously inside the `mx::async_eval` call. With Qwen3.5 dense
this was 1.5 ms / 23.7%; with Qwen3.6 MoE it's 23 ms at c=4,
65 ms at c=16. The host-side pipeline overlap (oMLX-C) still works
(`sample` collapsed to ~10 μs, confirming `eval(prev_sampled)` is
nearly free) but the wall-clock budget is dominated by graph encoding
elsewhere.

## Throughput (tok/s = batch * 1e6 / total_us p50)

| batch | tok/s p50 |
|------:|----------:|
| c=1   | 86 |
| c=4   | 165 |
| c=8   | 193 |
| c=16  | 236 |

Aggregate throughput climbs 2.7× from c=1 to c=16, but per-row ITL
also goes from 12 ms to 68 ms — the model size (35B) puts ARLE in a
fundamentally different regime than the 0.8B Qwen3.5 baseline (5.3 ms
at c=4, 1133 tok/s at c=16).

## Problems / observations

1. **The MLX cmdbuf env hypothesis was Qwen3.5-specific.** Don't promote
   it as a default for Qwen3.6 — pull the `MLX_MAX_OPS_PER_BUFFER=200`
   recommendation back from being a global default. The
   `docs/environment.md` section will note "verified on dense models;
   neutral on MoE".
2. **`async_eval` is no longer async on MoE**. The 23 ms at c=4 is the
   MLX dispatcher synchronously encoding the forward graph for a 35B
   MoE model. This is the new ceiling for oMLX-C-style pipelining —
   the host work-overlap window is shrunk to the graph-encode time
   itself.
3. **The next levers are MoE-specific** — expert routing fusion, batched
   expert matmul, mxfp4 metadata caching. Research subagent dispatched
   in parallel to this bench tick; results land in next entry.
4. **Pipelining still works** — sample phase remains essentially zero.
   The 15.3% v3 Qwen3.5 win pattern (sample-phase elimination) holds
   on Qwen3.6 too; it's just dwarfed by the 23 ms async_kick.

## Learnings

1. **Re-baseline after canonical-model changes.** The Qwen3.5 phase
   profile (sample dominates) doesn't hold on a 35B MoE. Bench
   numbers and bottleneck rankings are model-shape-specific.
2. **Don't generalize a model-specific finding into a global default.**
   The MLX cmdbuf=200 was a clear win on Qwen3.5; on Qwen3.6 it's
   wash-or-loss. Today's commit at AGENTS.md / environment.md
   recommended it for c≥8 — that needs an erratum.
3. **Phase timing transfers; phase rank doesn't.** The
   `INFER_PHASE_TIMING` infrastructure shipped commit 7a54050 still
   localizes the bottleneck in one bench cycle, and that's how this
   tick found "async_kick = 95%" in 30 seconds. The path probe
   (`metal_path_probe: oMLX-C pipelined step FIRED`) confirmed
   oMLX-C is engaged on the new model.

## Errata to AGENTS.md / environment.md

Today's commit 97db09e recommended `MLX_MAX_OPS_PER_BUFFER=200
MLX_MAX_MB_PER_BUFFER=200` "for any Metal bench at c≥8" — that's
overstated. The recommendation should be **"verified on Qwen3.5 dense
at c≥8 (where the cmdbuf cliff actually hits); neutral on Qwen3.6
MoE."** Will land an errata commit citing this wins entry.

## Rule

When the canonical model changes (e.g. dense→MoE, small→large),
re-bench from scratch BEFORE recommending any tuning that came from
the prior model's profile. Phase rank changes; cmdbuf env tunes that
worked for one shape may be wash-or-loss for another.

## What worked

- The oMLX-C v3 pipelining still works (sample phase 10 μs at all
  batch sizes — same as Qwen3.5 v3).
- Phase timing infrastructure carried over with zero changes.
- Path probes confirmed correct dispatch.

## Next

1. **Wait for MoE research subagent** (dispatched this tick) — top 3
   ROI MoE-specific Metal patterns. If anything is host-side or
   env-tunable, try it next tick.
2. **Erratum to AGENTS.md** narrowing the cmdbuf=200 recommendation.
3. **Investigate the 23ms async_kick at c=4** with the C++-side
   profiler hooks already in `mlx_qwen35_model.cpp` (the
   `profile_generate` instrumentation around forward(...)) to see if
   it's MoE routing graph build, expert kernel encoding, or commit
   serialization.

## References

- Predecessor (Qwen3.5 dense scaling):
  [`2026-05-07-bench-c-scaling-omlx-c-default.md`](2026-05-07-bench-c-scaling-omlx-c-default.md)
- Hypothesis source:
  [`docs/research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md`](../../research/2026-05-07-mlx-ecosystem-survey-c4-itl-gap.md)
  technique #2
- Canonicalization commit: `97db09e docs(metal): canonicalize
  Qwen3.6-35B-A3B-4bit as the global Metal model`
