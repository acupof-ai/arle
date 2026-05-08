# M_quant W4A8 production bench — 3-arm A/B at auto-FP8 KV (post-quantize)

> **Status**: GATED on W4A8 quantize completion (background `bk84vqk81`,
> `/tmp/quantize_qwen3_w4a8.py --src Qwen3-4B --dst Qwen3-4B-W4A8-marlin`,
> ETA 30-60 min).
> Master strategy §6.1 5-cap moat: Marlin tensor-core (✓) + W4A8 weight (in-flight)
> + KV W4A8 (codex own).
> Skill v1.3.0 anti-pattern #12 hardened (`d09480b`) — tensor-core dominance
> assumption applies; no hybrid dispatch needed.

## Phase 1 — Target (skill v1.3.0)

| Field | Value |
|---|---|
| Metric | longctx 4k/c=4 ITL + TTFT (Qwen3-4B-W4A8-marlin, auto-FP8 KV) |
| Baseline A | BF16 19.27 ms ITL / 1976 ms TTFT (`786a20a`) |
| Baseline B | W4A16 Marlin 11.76 ms ITL / 2565 ms TTFT (`f6f3af3`) |
| **License — full M_quant Phase 1** | ITL ≤ **10.35 ms** (1.86× vs BF16 per master §2.3) AND TTFT ≤ 250 ms (7.9× vs BF16, theoretical limit) |
| License — incremental (mix Marlin) | ITL within 0-10% of W4A16 (≤ 12.94 ms), TTFT within 0-15% of W4A16 — accept as substrate landing |
| Kill | ITL > +20% vs W4A16 (≥ 14.1 ms) — W4A8 implementation regression |
| Wall-clock budget | 5-7 min (3 arms × 120s + start/kill ARLE) |

## Phase 2 — Hardware

sm_89 RTX 4070 Ti SUPER · same as M_quant master plan §0:
- HBM 672 GB/s
- BF16 mma 88.5 TFLOPS
- **FP8 mma 706 TFLOPS** (8× BF16, sm_89 native — KEY for W4A8)
- 100 KB smem/SM, 64 K reg/SM

## Phase 3 — Binding constraint (formula-grounded)

W4A8 attacks BOTH axes:
- **Decode** (memory-bandwidth bound): W4 weight = 2 GB vs BF16 8 GB → 4× HBM saving on weight read
- **Prefill** (compute bound): FP8 mma 706 TFLOPS vs BF16 88.5 TFLOPS → 8× compute throughput

W4A16 Marlin (`f6f3af3`):
- Decode: ITL 11.76 ms = 1.64× faster than BF16 (W4 weight bandwidth saving)
- Prefill: TTFT 2565 ms = +30% slower than BF16 (Marlin per-call launch overhead)

W4A8 Marlin **predicted improvement over W4A16 Marlin**:
- Decode: marginal (W4 weight already ~2 GB)
- **Prefill: significant** (FP8 mma vs BF16 mma; same launch overhead amortized over 8× faster compute)

## Phase 4 — Formula prediction

```
W4A8 decode ITL_lower = 2 GB / 672 GB/s + 7.37 ms overhead = 10.35 ms
  (same as W4A16 since both have 2 GB weight; 7.37 ms is fixed)
W4A8 decode utilization (per W4A16 Marlin's 84% from 2853551 correction):
  ~12 ms ITL practical — **slightly worse than W4A16's 11.76 ms** because
  W4A8 adds activation quantization step

W4A8 prefill TTFT formula (M_quant §2.2):
  per-layer FLOPS @ M=8192 (4k chunked × 2 chunks × c=4) = 3.22 TFLOPS
  × 36 layers = 116 TFLOPS total prefill compute
  BF16 @ 88.5 TFLOPS @ 66% utilization = 1976 ms (master baseline — exact match)
  W4A8 @ 706 TFLOPS @ 66% utilization = **249 ms theoretical**
  W4A8 @ 706 TFLOPS @ 30% utilization (Marlin overhead) = **548 ms practical**

Predicted W4A8 vs W4A16 Marlin baseline:
- ITL: ~12 ms (slight regression vs 11.76 — activation quant cost)
- TTFT: ~250-550 ms (vs W4A16's 2565 ms = -78% to -90% IMPROVEMENT)
- out tok/s: tied or slightly worse than W4A16 (decode-bound)
- **TTFT is the winning metric**; ITL approximately tied
```

The big W4A8 win is **TTFT collapse** at long-context prefill (FP8 tensor cores
8× compute), NOT decode ITL (which is HBM-bandwidth-bound, both W4 weight = 2 GB).

## Phase 5 — Single-variable A/B (matched controls)

3 arms at production-default auto-FP8 KV (no `--kv-cache-dtype` override
per skill v1.2.0+):

| Arm | Checkpoint | Expected ITL | Expected TTFT |
|---|---|---|---|
| A — BF16 baseline | `Qwen3-4B` | 19.27 ms (`786a20a`) | 1976 ms (`786a20a`) |
| B — W4A16 Marlin | `Qwen3-4B-W4A16-sym-g128-marlin` | 11.76 ms (`f6f3af3`) | 2565 ms (`f6f3af3`) |
| **C — W4A8 Marlin** ⭐ | `Qwen3-4B-W4A8-marlin` (post-quantize) | ~12 ms predicted | **~250-550 ms predicted** |

### Bench command (after quantize completes)

```bash
# Verify W4A8 checkpoint complete
ls -la infer/models/Qwen3-4B-W4A8-marlin/
test -f infer/models/Qwen3-4B-W4A8-marlin/model.safetensors.index.json && echo OK

# Start ARLE auto-FP8 KV (no override)
CUDA_HOME=/opt/cuda TORCH_CUDA_ARCH_LIST=8.9 \
  ./target/release/infer --model-path infer/models/Qwen3-4B-W4A8-marlin \
  --port 8000 --num-slots 8 --max-seq-len 5120 \
  > /tmp/arle-w4a8.log 2>&1 &
sleep 35
curl -sS http://localhost:8000/v1/models  # sanity

# Bench Arm C
PATH=/home/ckl/projects/arle/.venv/bin:$PATH \
  scripts/bench_guidellm.sh w4a8-marlin-prod-c4-4k \
  --model Qwen3-4B-W4A8-marlin \
  --processor /home/ckl/projects/arle/infer/models/Qwen3-4B-W4A8-marlin \
  --concurrencies 4 --max-seconds 120 --warmup 10 \
  --data 'prompt_tokens=4096,prompt_tokens_stdev=1,prompt_tokens_min=4096,prompt_tokens_max=4096,output_tokens=256,output_tokens_stdev=1,output_tokens_min=256,output_tokens_max=256'

kill %1
```

### Matched controls (skill v1.3.0 checklist)

- [ ] All 3 arms use auto-FP8 KV (production default)
- [ ] Same `--num-slots 8 --max-seq-len 5120 --port 8000`
- [ ] Same data spec (4096 in / 256 out, c=4, max-seconds=120, warmup=10)
- [ ] greedy_consistency W4A8 vs BF16 token-level diff < 1% (correctness gate)
- [ ] σ < 5% across n=3 (W4A8 has new code path; n=3 mandatory not n=1)
- [ ] No other GPU process (bench ≠ codex test ≠ Edge browser GPU)

## Phase 6 — Combinational A/B (post-license)

If Phase 5 LANDs (full or incremental), combine with KV W4A8 (codex track,
`docs/plans/M_quant-kv-w4a8.md`):

| Arm | Weight | KV |
|---|---|---|
| C₁ | W4A8 Marlin | auto-FP8 (default) |
| C₂ | W4A8 Marlin | KV W4A8 (`--kv-cache-dtype w4a8` post codex KV W4A8 land) |

C₂ is the master strategy §6.2 5-cap moat **endgame stack** — 7.9× prefill
compute + 1.86× decode bandwidth + 4× KV pool capacity for long-context.

## Phase 7 — Tradeoffs (skill v1.3.0)

| Axis | Status | Note |
|---|---|---|
| LOC | ✅ 0 (codex W4A8 substrate at `a019a0e`, +1605 LOC) | Just bench |
| Hardware specificity | ✅ sm_89+ | FP8 mma is sm_89 native |
| Compiler/runtime | ⚠ cuda 13.2 (verified by codex W4A8 `a019a0e` review) | |
| Maintainability | ⚠ workflow | W4A8 quantize is offline (`/tmp/quantize_qwen3_w4a8.py`); per-model checkpoint required |
| **Numerical correctness** | ❌ **must verify** | greedy_consistency W4A8 vs BF16; literature claims ≤1% PPL but verify locally |
| Generality | ⚠ multi-shape required | high-conc + multi-tenant + longctx-8k must NOT regress |
| Memory budget | ✅ +6 GB VRAM headroom (W4 weight 2GB vs BF16 8GB; +KV pool capacity) | |
| Scheduling impact | ✅ none | Just dispatch via `MarlinW4A8` enum |
| **Activation quant overhead** | ⚠ unknown | `quantize_bf16_rows_to_int8_cuda` + dequant — bench measures net |
| **Sm_89 FP8 mma utilization** | ⚠ unknown | NVIDIA spec 706 TFLOPS; cuBLASLt FP8 hit 24% (`§9` M_quant); cutlass FP8 (codex track) may differ from custom Marlin W4A8 |

## Phase 8 — License decision

| Δ vs Arm B (W4A16 Marlin) | Action |
|---|---|
| TTFT ≤ -50% (≤ 1283 ms) AND ITL within +5% of W4A16 (≤ 12.35 ms) | **LAND HARD** — full M_quant Phase 1 license |
| TTFT ≤ -20% (≤ 2052 ms) AND ITL within +10% (≤ 12.94 ms) | LAND incremental — substrate validated |
| TTFT 2052-2700 ms (NULL band) AND ITL within ±10% | LAND with note + Phase 6 combined sweep |
| TTFT > +5% regression OR ITL > +20% (≥ 14.11 ms) | **KILL** — W4A8 implementation issue, debug |
| greedy_consistency divergence > 1% token | KILL — accuracy unacceptable |

## Pre-execution checklist

- [ ] Background quantize `bk84vqk81` completed (poll `/tmp/w4a8-quantize.log`)
- [ ] `infer/models/Qwen3-4B-W4A8-marlin/` checkpoint complete (model.safetensors.index.json + tokenizer files)
- [ ] ARLE built with `MarlinW4A8` dispatch enabled (already on main `a019a0e`)
- [ ] greedy_consistency runs against BF16 baseline (token diff < 1%)

## Cross-references

- W4A16 license bench: [`docs/experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md`](../experience/wins/2026-05-08-m_quant-w4a16-marlin-bench.md) (`f6f3af3`)
- R4 #6 KILL (hybrid dispatch refuted): [`docs/experience/errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md`](../experience/errors/2026-05-08-r4-hybrid-dispatch-killed-batch4-decode-regression.md) (`4571082`)
- Codex W4A8 substrate: `crates/cuda-kernels/csrc/gemm/marlin_w4a8_kernel.cu` (`a019a0e`)
- KV W4A8 orthogonal axis: [`M_quant-kv-w4a8.md`](M_quant-kv-w4a8.md) (`1e713de`)
- M_quant master plan: [`M_quant-fp8-w4-magnitude-path.md`](M_quant-fp8-w4-magnitude-path.md) §2.2 + §3
- Skill v1.3.0: [`.claude/skills/kernel-optimization/SKILL.md`](../../.claude/skills/kernel-optimization/SKILL.md) (`d09480b`) — anti-pattern #12 hardened

## Rule (per skill v1.3.0 + R4 #6 evidence)

- **Don't add hybrid dispatch threshold to W4A8 path**. R4 #6 KILL evidence
  shows tensor-core dominance for Marlin W4 GEMM at batch ≥ 2 on sm_89.
  Same applies to W4A8 (FP8 mma is even faster than W4A16's FP16 mma).
- **Greedy gate first**, bench second. W4A8 introduces activation quant —
  numerical drift higher than W4A16; verify before benching.
- **TTFT is the magnitude metric** for W4A8. ITL near-tied with W4A16 is the
  expected outcome (both 2 GB weight); the win is FP8 mma compute throughput.
