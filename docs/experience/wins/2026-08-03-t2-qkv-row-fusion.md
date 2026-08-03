# T2 qkv + qkvz row-fusion: −2.5% decode ITL — CUDA, 2026-08-03

> Status: **Shipped, default path** (#196 T2). c=1 W8A16 decode ITL p50
> **23.80 → 23.21 ms**; cumulative vs pre-#196 baseline **26.88 → 23.21
> (−13.7%)**. Same 32k c=1 protocol (H20 GPU 6, 16×256 tok, seed 20260416).

## What shipped

Two fusion groups, both via the T1/T3 machinery generalized to N parts
(`SafetensorLoader::load_matrices_row_fused`, per-part row-shard specs so
column-parallel and head-sharded TP layouts ride one loader):

- **Full attention**: `q_proj`/`k_proj`/`v_proj` load as one
  `[q_gated + 2*kv, hidden]` `qkv_proj`; one GEMM + `split_qkv` into the
  existing q/k/v buffers — rope/prep/cache consumers untouched.
- **Linear attention**: `in_proj_qkv`/`in_proj_z` load as one
  `[qkv + z, hidden]` `in_proj_qkvz`; one GEMM + `split2` (the generalized
  unequal-halves split that replaced `split_halves`).

Removes ~80 marlin launches/step (16 full layers × 2 + 48 linear layers × 1).
LoRA windows generalized: `lora_row_window` returns `(offset, rows)` per
projection inside its fused matrix (FullK at `[q_gated, kv)`, LinearZ at
`[qkv_dim, z_dim)`, …), one pristine base per fused buffer.

## Numbers

| arm | ITL p50 | ITL p99 |
|---|---:|---:|
| T5 (previous) | 23.80 | 24.37 |
| **T2 (this)** | **23.21** | **23.84** |
| SGLang, same kernel + same weights | 17.07 | 18.67 |

−0.59 ms from ~80 launches removed ≈ 7 µs/launch (gap + marlin fixed-grid
prologue), consistent with the module-ledger estimate of 0.9 ms (the marlin
busy share of SGLang's remaining launch-count lead).

Correctness: greedy reasoning-channel output byte-identical to the T5 binary
on a 120-token thinking trajectory (marlin row-fusion preserves per-row
accumulation order); bench 16/16 complete.

## Learnings

- The pair-fusion loader generalizes to N parts and per-part shard specs with
  no new formats: head-sharded TP q/k/v and column-shard pairs are the same
  code path, and `fuse_rows` chains for triples at load time.
- The marlin launch-count delta vs SGLang (357 vs 270) is now closed; what
  remains of the GEMM gap is per-launch prologue on identical launch counts.
