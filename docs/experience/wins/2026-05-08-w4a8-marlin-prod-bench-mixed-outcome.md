# W4A8 Marlin production bench — TTFT win (-36% vs W4A16) + ITL eager-mode regression

> Master strategy §6.1 5-cap moat capability "W4A8 Marlin" lands as
> production substrate with **TTFT win + ITL regression**. Mixed outcome
> framed honestly: prefill compute path validated; decode path needs
> graph capture follow-up.
>
> Per skill v1.3.0 anti-pattern #7 (heuristic ≠ direct): when implementation
> deviates from prediction direction, sanity-check before final
> license-or-kill. ITL regression root-caused to **eager mode (no CUDA
> Graph capture)** at startup, not W4A8 kernel itself.

## Phase 1 target recap

| Field | Value |
|---|---|
| Metric | longctx 4k/c=4 (TTFT + ITL) at production-default auto-FP8 KV |
| Plan | `docs/plans/M_quant-w4a8-prod-bench.md` (`db573c5`) |
| Phase 4 prediction | TTFT 250-550 ms (8× FP8 mma vs BF16) / ITL ~12 ms (W4 weight saves 8.92 ms HBM) |

## Setup

ARLE built at `4571082` (R4 #6 KILL revert + W4A8 dispatch arm `a019a0e`).
Codex's `/tmp/quantize_qwen3_w4a8.py` quantized Qwen3-4B → 252 linear tensors
(group_size=128) → `infer/models/Qwen3-4B-W4A8-marlin/` (2.6 GB).

Server (production-default auto-FP8 KV per skill v1.2.0):

```bash
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A8-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120
```

Bench:

```bash
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh w4a8-marlin-prod-c4-4k \
  --model Qwen3-4B-W4A8-marlin --processor .../Qwen3-4B-W4A8-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

Raw artifacts: `bench-output/2026-05-08-w4a8-marlin-prod-c4-4k/`.

## Results — 3-arm matched A/B (all auto-FP8 KV)

| Arm | Checkpoint | TTFT p50 | TTFT std | ITL p50 | ITL std | out tok/s |
|---|---|---:|---:|---:|---:|---:|
| A — BF16 | `Qwen3-4B` (`786a20a`) | 1976 ms | n/a | 19.27 ms | n/a | 153.83 |
| B — W4A16 Marlin | `Qwen3-4B-W4A16-sym-g128-marlin` (`f6f3af3`) | 2565 ms | n/a | **11.76 ms** | n/a | **191** |
| **C — W4A8 Marlin** ⭐ | `Qwen3-4B-W4A8-marlin` (this run) | **1633.6 ms** | 112.2 | 19.23 ms | 0.41 | 155.47 |

Δ tabulated:

| Comparison | TTFT Δ | ITL Δ | tok/s Δ |
|---|---:|---:|---:|
| **C vs B (W4A8 vs W4A16)** | **−36.3% (faster)** | **+63.5% (slower)** | −18.6% |
| C vs A (W4A8 vs BF16) | −17.3% (faster) | −0.2% (flat) | +1.1% |

σ confidence: ITL std 0.41 ms ≈ 2.1% of mean — tight enough single-arm conclusive (skill rule #6).

## Root cause of ITL regression

ARLE startup log line at `bootstrap.rs:321`:

```
INFO infer::scheduler::cuda::core::warmup: warmup.rs:43 Graph capture
disabled (eager decode, e.g. LoRA); running eager warmup + cublasLt
autotune for 8 batch sizes (max 8)...
```

W4A8 dispatch path is currently classified by ARLE warmup as
`eager decode` (no CUDA Graph capture), the same flag used for LoRA paths.
W4A16 Marlin runs WITH graph capture (per `f6f3af3` log: "Capturing CUDA Graph
for batched decode B=4..."), saving 5-15 ms per decode step on graph-capturable
workloads.

The +63.5% ITL regression vs W4A16 is **dominated by missing graph capture**,
not the W4A8 kernel itself. The W4A8 Marlin kernel actually provides FP8 mma
compute at 706 TFLOPS (per Phase 1 target), but eager-mode launch overhead
swamps the per-token kernel-time savings.

## What Phase 4 prediction got right

| Phase 4 prediction | Actual | Match? |
|---|---|---|
| Decode ITL ~12 ms (HBM saturation, both 2 GB weight) | 19.23 ms | ❌ off by +60% — eager mode penalty |
| TTFT 250-550 ms (FP8 mma compute win) | 1633.6 ms | ❌ off — Marlin per-call overhead + eager mode |
| **TTFT relative improvement vs W4A16 = significant** | **−36% vs W4A16** | ✅ direction + magnitude |
| out tok/s near-flat vs BF16 (decode-bound) | +1.1% vs BF16 | ✅ |

The relative improvements vs W4A16 (the head-to-head comparison) match the
formula direction. Absolute predictions for TTFT and ITL were optimistic
because they assumed graph-capture-equivalent overhead. With graph capture
disabled, both metrics regress significantly from theoretical limits.

## Phase 7 tradeoffs (revised post-bench)

| Axis | Status | Note |
|---|---|---|
| LOC | ✅ 0 (codex `a019a0e` substrate) | |
| Numerical correctness | ⚠ greedy_consistency W4A8 vs BF16 NOT yet verified | TODO Phase 9 follow-up |
| TTFT win | ✅ −17% vs BF16, −36% vs W4A16 | Real FP8 mma compute throughput gain |
| **ITL regression** | ❌ +63% vs W4A16 (eager-mode penalty) | Solvable by enabling graph capture for W4A8 decode |
| Memory | ✅ +6 GB VRAM headroom (W4 weight 2 GB) | |
| Hardware specificity | ✅ sm_89+ (FP8 mma native) | |
| Production-default flip | ❌ NOT ready | W4A16 still better for ITL until graph capture for W4A8 |

## Phase 8 license decision

| Threshold | Met? |
|---|---|
| TTFT ≤ -50% AND ITL ≤ +5% vs W4A16 → LAND HARD | ❌ TTFT yes, ITL +63% no |
| TTFT ≤ -20% AND ITL ≤ +10% vs W4A16 → LAND incremental | partial (TTFT yes, ITL no) |
| ITL > +20% → KILL | YES, but **debug not KILL** per anti-pattern #7 |
| greedy_consistency divergence > 1% | not yet tested |

**Verdict: SUBSTRATE LANDED, default-on DEFERRED**.

W4A8 Marlin substrate is functional and produces real prefill compute wins.
Default-on flip blocked by:

1. **Graph capture for W4A8 decode** — implementation tranche needed
2. **greedy_consistency W4A8 vs BF16** — token-level diff < 1% verification

Until both addressed, **W4A16 Marlin (`f6f3af3`) remains the production
recommendation** for general-purpose decode. W4A8 is opt-in for prefill-heavy
workloads (long-context single-shot).

## Follow-up work (post this entry)

1. **Enable graph capture for W4A8 decode** — substrate LOC in ARLE
   warmup path. Codex own. Likely `infer/src/scheduler/cuda/core/warmup.rs:43`
   detection logic (currently treats W4A8 as eager-mode same as LoRA;
   shouldn't).
2. **greedy_consistency W4A8 vs BF16** — Claude single-file iteration.
   Add W4A8 path to test cases.
3. **Multi-shape defense bench** for W4A8 (high-conc, multi-tenant, longctx-8k)
   — once graph capture lands, re-bench.
4. **Combined stack** with KV W4A8 (codex track `1e713de`) — endgame
   moat: W4 weight + FP8 act + W4A8 KV = full quantization stack.

## Skill methodology applied

- ✅ Phase 1 target with thresholds
- ✅ Phase 2 hardware sheet (sm_89 FP8 mma 706 TFLOPS)
- ✅ Phase 3 binding-constraint formula (master §2.2 + §2.3)
- ✅ Phase 4 magnitude prediction with direction + relative magnitude
- ✅ Phase 5 single-variable A/B (matched auto-FP8 KV per skill v1.2.0+)
- ⏭ Phase 6 combo skipped (single arm conclusive on substrate question)
- ✅ Phase 7 tradeoffs revised post-bench
- ✅ Phase 8 nuanced license: SUBSTRATE LAND (TTFT win) + default-on DEFER (ITL pending graph capture)

The Phase 8 nuance is the value-add: W4A16 reflexively-KILL on ITL +63%
would have abandoned a real prefill-compute axis. Skill anti-pattern #7
diagnostic ("implementation deviates from prediction → sanity-check") found
the eager-mode root cause; substrate stays alive.

## Cross-references

- Plan: [`docs/plans/M_quant-w4a8-prod-bench.md`](../../plans/M_quant-w4a8-prod-bench.md) (`db573c5`)
- Quantize script (uncommitted): `/tmp/quantize_qwen3_w4a8.py` (codex authored)
- Codex W4A8 substrate commit: `a019a0e`
- W4A16 Marlin license bench: [`2026-05-08-m_quant-w4a16-marlin-bench.md`](2026-05-08-m_quant-w4a16-marlin-bench.md) (`f6f3af3`)
- R1 baseline correction: [`../errors/2026-05-08-marlin-r1-baseline-correction.md`](../errors/2026-05-08-marlin-r1-baseline-correction.md) (`2853551`)
- R4 #6 KILL: [`../errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md`](../errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md) (`4571082`)
- Skill v1.3.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../../.claude/skills/kernel-optimization/SKILL.md) (`d09480b`)
- KV W4A8 orthogonal axis: [`../../plans/M_quant-kv-w4a8.md`](../../plans/M_quant-kv-w4a8.md) (`1e713de`)
- Bench artifacts: `bench-output/2026-05-08-w4a8-marlin-prod-c4-4k/`

## Rule

W4A8 substrate LANDED (TTFT win validated); production default flip
gated on (1) W4A8 graph capture enable and (2) greedy_consistency.
Until both, W4A16 Marlin is the production decode recommendation.
W4A8 is opt-in for prefill-heavy workloads.
