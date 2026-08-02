# T1 gate+up row-fusion: −2.1% decode ITL, and the Marlin fixed-grid correction — CUDA, 2026-08-03

> Status: **Shipped, default path** (`3e383c082`, #196 T1). c=1 W8A16 decode
> ITL p50 **26.88 → 26.31 ms (−0.6 ms)**. Smaller than predicted — and the
> reason re-ranks the whole program: Marlin launches a FIXED `sms × 1` grid
> and iterates tiles internally, so wider N does NOT lift occupancy. Fusion
> only recovers launch overhead. T3 (micro-GEMV fusion, ~4 ms) is now the top
> busy-time lever; T4 (whole-step graph, ~5 ms) unchanged.

## What shipped

`mlp.gate_proj`+`up_proj` load as ONE row-fused `[2*inter, hidden]` matrix:
loader pair-fuse (W8A16 concats INT8+scales BEFORE the Marlin repack → fused
matrix repacks and frees INT8 once; bf16/FP8 fuse on device via
`DeviceMatrix::fuse_rows`). `dense_mlp` = one fused GEMM + `silu_mul_fused`
(kernel existed since the split_qkv drop, previously unwired) + down GEMM: 3
launches → 2 per layer, −64 marlin launches/step. LoRA merge stays correct
through row-window addressing (`lora_row_offset`, one canonical pristine base
per fused buffer, window-scoped merge/restore) — OPD bf16 students unaffected.

## Measured (H20 GPU 6, same 32k c=1 protocol as the SGLang matched A/B)

| arm | ITL p50 | ITL p99 |
|---|---:|---:|
| unfused (same day, `f2c07d0cf`-era binary) | 26.88 ms | 27.46 ms |
| **T1 fused** | **26.31 ms** | 26.99 ms |

TTFT p50 30.5→24.5 s is NOT T1's — this binary also carries the ctx-bind +
FA3-batch1 prefill fixes the unfused arm lacked.

Correctness: greedy 120-tok and 60-tok completions **byte-identical** to the
unfused binary (md5-equal) despite the changed GEMM tiling. Needle harness
scored 0/9 on BOTH arms with identical instant-EOS (`out=''`) — the
abliterated iso-tc-huihui checkpoint EOSes raw completions immediately, a
harness cliff, not a regression (baseline-arm reproduction per
[[feedback_validate_comparison_inputs_before_bug]]).

## Learnings

**Marlin's m=1 grid is `blocks = sms × blocks_per_sm(=1)` regardless of N** —
the kernel walks tiles internally. Fusing projections widens the tile walk,
not the wave, so the occupancy story I predicted (~1.5–2 ms) was wrong; the
~0.6 ms that materialized is 64 launches of overhead. Corollary: SGLang's
marlin busy time equals ours; their 17.1 ms step wins on (a) no in_proj_a/b
micro-GEMVs (fused into their qkvz/ba merged projections, ~4 ms here) and (b)
whole-step graph (~5 ms idle here). That is the measured T3 → T4 order.
