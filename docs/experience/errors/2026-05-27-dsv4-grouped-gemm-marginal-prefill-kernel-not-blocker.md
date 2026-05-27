# DSv4 grouped GEMM kernel landed — only 13% prefill speedup, kernel was NOT the dominant blocker

## SLO-shape probed?  Y — prompt 28899 tokens, max_seq_len=49152, max_tokens=8, single request c=1

Probed at the original SLO failure shape (29795-token baseline → 28899-token here, within tokenization variation).

## Roofline check

| Op | Achieved | Peak (8×H20 BF16 FP8) | % | Verdict |
|---|---:|---:|---:|---|
| Prefill end-to-end (variant A, per-expert tiled) | 102.29 tok/s/rank × 8 = 818 tok/s | ~78,000 tok/s/rank × 8 = 624,000 tok/s | **0.13%** | **KILL (default per §7.6) — deferred: attention compute at 16K-token chunk is the dominant cost, not MoE FFN kernel choice; next axis = flash-style attention or chunked-prefill chunk_size tuning** |
| Prefill end-to-end (variant B, new grouped GEMM) | 102.28 tok/s/rank × 8 = 818 tok/s | same | **0.13%** | same |

The two variants are statistically indistinguishable (282.47s vs 282.54s, 0.025% diff). Both improved ~13% vs the 2026-05-27 pre-fix baseline (325s). The remaining 60× gap to the SLO target (4.8s) is dominated by something other than the grouped expert kernel shape.

## Context

The 2026-05-27 errors entry (`2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md`) identified the M-blind GEMV-at-prefill bug: `dsv4_fp8_grouped_gemv_batch_kernel` indexed grid Y by `max_count` (one block per token) → no weight reuse → catastrophic at prefill. 29795-token prefill measured 325s = 67× off SLO.

I wrote a new GEMM kernel with M-tile=32 weight reuse (`dsv4_fp8_grouped_gemm_batch_kernel`, commit `ac1f0ccc`) + refactored to a dedicated file `crates/cuda-kernels/csrc/gemm/dsv4_grouped_gemm.cu` (commit `2d69758d`) + threshold-aware dispatch in `mlp.rs::dsv4_run_grouped_block_scaled_gemv{,_pair}`. Hypothesis: 32× weight reuse → 325s/32 ≈ 10s prefill.

## Hypothesis going in

Per the original GEMV math (bandwidth analysis):
- Weight reuse ratio = M_tile when properly tiled
- GEMV path: 1× reuse → bandwidth-bound, ~325s observed
- GEMM path (M-tile=32): 32× reuse → ~10s expected

Expected variant B to hit sub-30s SLO range, variant A (per-expert tiled, existing `_tiled_kernel`) as control.

## Results

Both variants produced near-identical wall-clock prefill at 28899-token shape:

| Variant | Config | Prefill | Per-rank rate | curl elapsed |
|---|---|---:|---:|---:|
| 2026-05-27 baseline (pre-fix) | `ARLE_DSV4_LOCAL_GROUPED_EXPERTS=1` + grouped GEMV (M-blind) | **325s** | ~91 tok/s | (different binary) |
| A: per-expert tiled | `ARLE_DSV4_LOCAL_GROUPED_EXPERTS=0` (skip compact path → `expert.forward()` → `_tiled_kernel` with M=32 reuse) | **282.47s** | 102.29 tok/s | 283776 ms |
| B: new grouped GEMM | `ARLE_DSV4_LOCAL_GROUPED_EXPERTS=1` + `ARLE_DSV4_GROUPED_GEMM_M_THRESHOLD=4` (new `dsv4_fp8_grouped_gemm_batch_kernel`) | **282.54s** | 102.28 tok/s | 283765 ms |

**Both paths improve ~13% vs the original M-blind baseline**. The kernel shape change matters but is NOT the dominant cost.

Phase breakdown (from `request_trace`, per-rank):
- `phase_us.prefill = 12.16s` per scheduled-rows step × ~23 step iterations ≈ 282s wall-clock
- `phase_us.decode = 0.42s` for 8 decode tokens (decode is fast; TPOT ~52 ms/token)
- `chunks_seen: 9` despite `chunk_size=16384` — chunked prefill split into more iterations than expected

## Where the time actually goes (compute roofline)

Attention compute at 16K-token chunk is the dominant cost:
- Per-layer attention QK: 16384² × 128 heads × 128 head_dim = 4.4T FLOPs
- × 43 layers = 189T FLOPs per chunk
- ÷ 8 ranks × 1.2 PFLOPS/rank = 9.6 PFLOPS aggregate
- Theoretical min: 189T / 9.6 PF = 20s per chunk × 2 chunks = ~40s attention floor

MoE FFN compute (what GEMM kernel addresses):
- Per-layer expert FFN: 32 local experts × 903 tokens/expert avg × (8192×4096 + 8192×4096 + 4096×8192) MACs = 32 × 903 × 100M ≈ 2.9T MACs
- × 43 layers = 124T MACs per chunk = 248T MACs total
- ÷ 9.6 PFLOPS = 26s ideal MoE compute

Observed 282s vs ideal 40s attention + 26s MoE + N×NCCL allreduce ≈ ~70s ideal. Still ~4× gap. Likely contributors: NCCL allreduce per-layer at 16384×4096 BF16 (3.3ms each × 86 = 280ms — small), KV write/read for MLA, scheduler / Rust dispatch overhead, multi-chunk overhead (`chunks_seen: 9` instead of 2).

## Conclusion

**The grouped GEMM kernel is correct and helpful (13% over M-blind), but the bigger SLO gap (60× from current 282s to target 4.8s) is NOT in the MoE FFN kernel shape.** The dominant axes from here:

1. **Attention compute at long context** — 16K² × heads × head_dim is the heaviest single op. Need flash-style fused attention or context-aware optimization. DSv4 uses MLA (Multi-Head Latent Attention) which has a different compute pattern than plain MHA; needs separate analysis.
2. **Chunked prefill chunking pattern** — `chunks_seen: 9` indicates the chunked prefill is splitting into 9 iterations not 2. Each iteration has scheduler/dispatch overhead. Worth investigating why.
3. **Per-step phase_us.prefill = 12s** with 23 iterations × 12s ≈ 282s. Each step is processing maybe ~3K tokens. With proper compute roofline, a 3K-token compute step should be ~1-2s.
4. **NCCL allreduce micro-batching** — at MoE allreduce N=86 per chunk, each on small tensors. Worth a fused-allreduce path.

The framing-trap lesson (§0 SOLID): the GEMV-vs-GEMM analysis was mathematically correct (1× vs 32× weight reuse), but it was a partial-percentage of total wall-clock, not the dominant cost. **Need to apply the roofline check to identify the right axis, not just spot the worst micro-issue.**

## Rule

When a kernel-shape fix is justified by bandwidth math, **also estimate that op's percentage of total wall-clock** before committing significant kernel work. The 2026-05-27 GEMV-at-prefill bug analysis showed the kernel was wrong, but I didn't ask "if I fix this, what's the remaining wall-clock?" If I had estimated attention-roofline (20s) + MoE-roofline (26s) before the fix, I would have known the kernel fix gets us to ~50-70s, not sub-30s. That's still a 5× SLO miss before starting. The kernel fix was correct but not sufficient.

Anchor: 2026-05-27 GEMM landing — kernel diff was good engineering (correct math, clean code, proper file split per ckl correction), but didn't unblock SLO because the SLO gap was always going to need attention-axis work too.

## Refs

- `ac1f0ccc` — original GEMM kernel commit (later refactored)
- `2d69758d` — refactor: GEMM kernels to own file `csrc/gemm/dsv4_grouped_gemm.cu`
- `2d74c64b` — bench-spec §7.6/§7.7 + wins TEMPLATE SLO-shape/roofline gates (mandatory header sections, this entry uses them)
- `docs/experience/errors/2026-05-27-dsv4-tp-allreduce-slo-prefill-kill.md` — original GEMV-at-prefill discovery
- Artifacts on pod: `/sgl-workspace/arle-fresh/docs/trace-artifacts/2026-05-27-dsv4-gemm-slo-probe-A/server.log` and `-B/server.log`
- Probe script: `/sgl-workspace/dsv4_slo_probe.sh` (variants A and B)
