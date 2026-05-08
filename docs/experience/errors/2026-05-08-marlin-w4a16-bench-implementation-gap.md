# Marlin W4A16 decode bench shows 1.06× ITL — ARLE implementation gap, not hardware ceiling

> Skill applied: `kernel-optimization` (Phase 1-5 + 7-8 walked).
> First skill demonstration. Verifies methodology catches an
> implementation gap that pure formula-prediction would have falsely
> credited as 1.86× win.

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | decode ITL p50 (Qwen3-4B 4k longctx c=4, BF16 KV) |
| Baseline | 19.27 ms (ARLE pre-Phase 0 BF16, `786a20a`) |
| License threshold | ≥ 1.5× decode (ITL ≤ 12.85 ms) per `M_quant` §9.2 |
| Kill threshold | ≤ 1.0× → debug; not direct KILL per skill anti-pattern #7 |

## Phase 2 — Hardware

sm_89 RTX 4070 Ti SUPER · HBM 672 GB/s · 100 KB smem/SM · 88.5 BF16 / 706 FP8 TFLOPS · Marlin sm_80+ native.

## Phase 3 — Binding constraint (formula-grounded)

Decode ITL formula (`M_quant` §2.1):

```
ITL = weight_HBM + KV_read + sample + overhead
BF16: 8 GB / 672 GB/s = 11.9 ms (62% of 19.27 ms ITL)
Remaining 7.37 ms = KV + sample + schedule (unchanged across quant)
```

Binding = weight HBM bandwidth on read. Ground-truth via formula
(skipping ncu profile because `scripts/profile_ncu_guidellm.sh` carries
ncu 2026.1.1.0 incompatible `--attach-pid`; formula evidence sufficient
for memory-bandwidth case where measured 62% utilization is the direct
upper bound).

## Phase 4 — Formula prediction

```
W4 weight = 4B × 0.5 byte = 2 GB
ITL_lower(W4) = 2 / 672 + 7.37 = 2.98 + 7.37 = 10.35 ms
predicted speedup (theoretical 100% util) = 19.27 / 10.35 = 1.86×
predicted speedup (70% util reasonable) = 19.27 / ~14 = 1.4×
license band = 1.5× (M_quant §9.2)
```

## Phase 5 — Single-variable A/B (matched controls)

Setup:
- Both arms use **same model class** (Qwen3-4B, hidden=2560, 36 layers)
- Same KV dtype: `--kv-cache-dtype bf16` (matched per skill checklist)
- Same `--num-slots 8 --max-seq-len 5120`
- Same workload: `prompt_tokens=4096, output_tokens=256, c=4, max-seconds=120, warmup=10`
- Single GPU (no contention; codex 0:0 was idle during run)

```bash
# Marlin treatment
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-GPTQ-Int4-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120 --kv-cache-dtype bf16

PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh marlin-w4a16-c4-4k \
  --model Qwen3-4B-GPTQ-Int4-marlin \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B-GPTQ-Int4-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_min=256,output_tokens_max=256'
```

### Results

| Metric | ARLE BF16 (`786a20a`) | ARLE Marlin W4A16 | Δ | Phase 4 predicted |
|---|---:|---:|---:|---:|
| **ITL p50** | 19.27 ms | **18.13 ms** | **−5.9% (1.06×)** | 1.86× theoretical / 1.4× practical |
| ITL std | n/a | **0.02 ms** | extremely tight σ | — |
| TTFT p50 | 1976 ms | **2331.8 ms** | **+18.0% (regression)** | should not regress |
| TTFT std | n/a | 7.7 ms | tight σ | — |
| out tok/s | 153.83 | 150.37 | −2.3% | +154% |
| TPOT mean | n/a | 27.16 ms | — | — |

Raw artifacts: `bench-output/2026-05-08-marlin-w4a16-c4-4k/`.

ARLE startup logs confirm Marlin checkpoint loaded
(`marlin_repacked: true` in `quantize_config.json`), CUDA Graph
warmup completed for B=1..8, no kernel errors during bench.

## Phase 7 — Tradeoffs explicit (per skill, user 2026-05-08 directive)

| Axis | Status | Note |
|---|---|---|
| LOC complexity | ✅ 0 LOC | Marlin path already in repo (`marlin_kernel.cu` + dispatch in `quant.rs`); bench is pure verification |
| Hardware specificity | ✅ sm_80+ | Works on Ada |
| Compiler/runtime version | ✅ no TileLang dep | Native CUDA C |
| Maintainability | ⚠ workflow | Per-model GPTQ checkpoint required; 5 variants already on disk for Qwen3-4B |
| **Numerical correctness** | ❌ **NOT verified** | greedy_consistency BF16-vs-Marlin not run this round — gap; literature claim ≤ 0.5 PPL but per ARLE rule must verify |
| Generality | ⚠ single shape | Only 4k longctx c=4; not yet bench at high-conc 1k/256/c=64 (must defend +30% lead) or multi-tenant (defend +80%) |
| Memory budget | ✅ +6 GB VRAM | W4 weight 2GB vs BF16 8GB → KV pool can be larger |
| Scheduling impact | ✅ none | No envelope or admission change |
| **Implementation gap** | ❌ **Major** | Predicted 1.86× theoretical / 1.4× practical; actual 1.06× = **73% of predicted gain missing** |

**No-tradeoff axes**: scheduling, LOC, HW. **Real tradeoffs**: implementation gap (severe), numerical-not-verified, multi-shape-not-verified, workflow burden.

Per skill rule "no tradeoff = not at extremes" — here the major tradeoff IS the implementation gap, indicating ARLE Marlin path is not extreme. The fix is implementation, not hardware.

## Phase 8 — License or kill decision

| Threshold | Met? |
|---|---|
| ≥ 1.5× ITL → ✅ proceed | ❌ 1.06× |
| 1.0-1.5× → debug per anti-pattern #7 | ✅ |
| Direct KILL on hardware grounds | ❌ — too early; implementation suspected |

**Verdict: DEBUG, not KILL**. The result does not refute the W4A16 axis on sm_89 hardware (formula bandwidth math is independent of implementation). It refutes ARLE's *current* Marlin engagement at the cited workload.

## Root cause hypotheses (ranked by likelihood)

1. **Marlin only routes for select GEMM ops, leaving substantial weight HBM in BF16 path.**
   ITL saved 1.14 ms vs predicted 8.92 ms → 13% engagement.
   8GB BF16 weight = 0.768 GB embeddings/lm_head + 7.2 GB GEMM weights. If only 13% of GEMM weights are dispatched to Marlin (or partial layer coverage), saving matches the observed 1.14 ms.
   Verify with: nsys profile + count Marlin kernel launches per step vs BF16 GEMM launches.

2. **Marlin used in prefill via fallback path that adds dequant overhead.**
   TTFT +18% regression argues prefill is going Marlin → dequant → BF16 GEMM.
   Marlin is for HBM-bound *decode* (single-token Q × full weight); for compute-bound *prefill* (batched Q × weight, weight reuse high), Marlin's W4-unpack overhead per call exceeds the bandwidth saving since the weight is read once and computed against many Q rows.
   Fix: route prefill to BF16 path (load BF16 weights too, OR Marlin-decode-only dispatch).

3. **Marlin kernel itself is not at peak utilization on sm_89.**
   Marlin paper benched on A100 (sm_80). sm_89 Ada has different L2 / register / scheduler behavior. Marlin may need re-tune.
   Counter-evidence: literature reports Marlin ≥ 80% HBM utilization on sm_80 Ampere; sm_89 should be similar or better.

4. **Activation in BF16 going through Marlin's FP16 path.**
   Marlin original is W4 + FP16 input, not BF16. ARLE may be doing extra dtype conversion or running a fallback FP16 path. Worth verifying via grep.

## Recommended next steps (deferred — not committed beyond this entry)

1. **Verify Marlin engagement via nsys** — count `Marlin*` vs `gemm_*_bf16` kernel launches per step. If Marlin launches < 50% of weight GEMMs, hypothesis #1 confirmed.
2. **Try the W4A16-sym-g128-marlin variant** — `infer/models/Qwen3-4B-W4A16-sym-g128-marlin`. May have different group size or symmetric quant; quick A/B against this run's numbers.
3. **Greedy consistency BF16-vs-Marlin** — `cargo test --test greedy_consistency` with both checkpoints; verify token-level diff < 1%.
4. **Inspect ARLE Marlin dispatch** — `infer/src/ops/linear.rs` (or equivalent) — is Marlin used for ALL weight GEMMs in prefill+decode, or only decode?
5. **High-conc + multi-tenant bench at Marlin** — defend the +30% / +80% leads under quant.

If hypothesis #1 confirms, fixing dispatch coverage moves ITL from 1.06× → 1.4-1.6× (closer to formula). Worth a Phase 0v2-style implementation push.

## Skill rule application (per `kernel-optimization`)

- ✅ Phase 1: target stated explicitly with metric + baseline + threshold
- ✅ Phase 2: hardware sheet referenced (sm_89 + HBM constants)
- ✅ Phase 3: binding constraint named (HBM bandwidth on weight) via formula evidence (skill permits formula when measured utilization is direct upper bound)
- ✅ Phase 4: formula prediction with magnitude (1.86× theoretical / 1.4× practical)
- ✅ Phase 5: matched A/B (same model class + KV dtype + workload + slots)
- ⏭ Phase 6: combo A/B not run (single var sufficient — kept for variant sweep follow-up)
- ✅ Phase 7: tradeoffs enumerated; major axis = implementation gap
- ✅ Phase 8: KILL withheld in favor of debug per anti-pattern #7 (implementation < expected → verify before declaring hardware ceiling)

## Anti-pattern caught

This entry illustrates skill anti-pattern #7 ("cuBLASLt heuristic ≠ cutlass direct mma" generalized): when measured Δ% < formula-predicted by ≥ 50%, the failure is implementation, not hardware. Direct-KILL would have abandoned a real bandwidth axis with debug-able root cause.

Skill methodology cost: 1 bench run + 1 errors entry. Without methodology, the natural reading "1.06× → KILL W4A16" would have rejected the M_quant W4A8 path (master strategy combined target = W4 weight + FP8 activation = 7.9× prefill + 1.86× decode). The errors entry preserves the axis for later debug.

## Round 2 — alloc_zeros → alloc fix attempted, NULL result (2026-05-08)

**Diagnosis update**: read `infer/src/ops/linear.rs:660-739` (`run_marlin_w4_gemm`).
Each Marlin call issues 6 kernel launches:

1. `alloc_zeros x_fp16` → cudaMemsetAsync (predicted ~7us)
2. `bf16_to_fp16_cuda` elementwise
3. `alloc_zeros y_fp16` → cudaMemsetAsync (predicted ~7us)
4. `alloc_zeros workspace` → cudaMemsetAsync (Marlin atomic accum, must keep)
5. `marlin_gemm_cuda`
6. `fp16_to_bf16_cuda` elementwise

**Hypothesis**: 4-5 launches per Marlin call (vs 1 cuBLAS BF16 GEMM call) drives the
+18% TTFT regression and limits decode ITL gain. cuBLAS GEMM = 252 launches per
chunk; Marlin = 252 × 6 = 1512 launches per chunk → 6× launch density.

**Phase 4 prediction (Round 2)**: skip `alloc_zeros` zero-init for `x_fp16` and
`y_fp16` (both fully overwritten by conversion / GEMM) → save 2 × 252 cudaMemsetAsync
launches per chunk = ~3.5 ms/token decode and ~7 ms/req prefill.

**Single-variable A/B (Phase 5)**: replaced `alloc_zeros` with `unsafe alloc` on
`x_fp16` and `y_fp16`. Workspace kept zeroed (Marlin atomic accumulation needs).

| Metric | Marlin baseline | Marlin alloc-skip | Δ |
|---|---:|---:|---:|
| TTFT p50 | 2331.8 ms | 2334.9 ms | +0.13% (within σ) |
| ITL p50 | 18.13 ms | 18.14 ms | +0.06% (within σ) |
| ITL std | 0.02 ms | 0.02 ms | flat |

**Verdict — NULL result**. Skipping `alloc_zeros` produced no measurable Δ. Two
explanations consistent with observation:

- cudarc's CUDA pool returns already-zeroed memory (cudaMemsetAsync internally
  elided when caller holds a freshly-pooled buffer).
- The cudaMemsetAsync launch overhead is < 1 us in practice on Ada — far below
  the 7 us per-launch estimate, so 504 saved launches contribute < 0.5 ms.

**Tradeoff** (Phase 7 revisit): the change adds uninitialized-memory risk for
zero net benefit → reverted. Phase 8 says NULL + new risk = no land.

**Remaining hypotheses (Round 3+ candidates)**:

| # | Hypothesis | How to test | Cost |
|---|---|---|---|
| 4 | BF16↔FP16 elementwise conversion launches dominate | nsys count `bf16_to_fp16` vs `marlin_gemm` time | 30 min |
| 5 | Marlin kernel sub-peak utilization on sm_89 (paper benched on A100/sm_80) | ncu single-launch profile of `marlin_gemm_cuda` | 30 min, blocked on wrapper fix |
| 6 | `W4A16BatchGemv` (ARLE-native, used when marlin_aligned fails) may be BF16-native + faster — bypass Marlin | edit dispatch `(_, W4A16) => W4A16BatchGemv` for `batch>1` | ~10 LOC, 1 bench |
| 7 | The other variant `Qwen3-4B-W4A16-sym-g128-marlin` may have different group_size or symmetry quant — quick A/B vs current `Qwen3-4B-GPTQ-Int4-marlin` | swap `--model-path` + bench | 5 min |

Round 3 should start with #7 (cheapest) → #6 (low LOC + immediate impact if
W4A16BatchGemv is truly BF16-native and competitive) → #4/#5 (need profiler).

**Round 2 cost**: 1 build + 1 bench + 1 file edit + revert = ~5 min wall-clock.
Per skill methodology: NULL results are accumulation-of-knowledge (rule #6:
"License-or-kill with σ < 5%"); they narrow the hypothesis space without burning
multi-day implementation. The alloc hypothesis is now eliminated; remaining
hypotheses target the actual binding mechanism (kernel utilization or kernel
choice).

## Cross-references

- Skill: `.claude/skills/kernel-optimization/SKILL.md` (`faffcb0`)
- M_quant plan: [`docs/plans/M_quant-fp8-w4-magnitude-path.md`](../../plans/M_quant-fp8-w4-magnitude-path.md) §2.3 + §9.2
- Master strategy §0.1 + §3.3: [`docs/projects/2026-05-07-arle-master-strategy.md`](../../projects/2026-05-07-arle-master-strategy.md)
- ARLE Marlin path: `crates/cuda-kernels/csrc/gemm/marlin_kernel.cu` + `marlin_repack.cu`
- ARLE quant dispatch: `infer/src/quant.rs` (`QuantFormat::Gptq` + `GptqKernel::Marlin`)
- Bench artifacts: `bench-output/2026-05-08-marlin-w4a16-c4-4k/`
- Marlin paper: <https://arxiv.org/abs/2408.11743>
