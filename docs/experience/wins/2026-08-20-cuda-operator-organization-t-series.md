# CUDA operator organization — T-series structural phase exits

## Context

Parent plan `docs/plans/2026-08-20-cuda-operator-organization.md`. One session
executed T0–T4, T6, T7, and the T8 registry prep; T5 (MoE split)
and the T8 MoE registry binding landed 2026-08-21 once the peer released
`infer-cuda/src/moe.rs`.

Commits: T0 · 1B // ·
T1 + · T2 verdict (ACCEPT) ·
Tranche 2 + · T2L ·
T3 //// ·
T4 · T6 + · T7 ·
T8 prep · clippy.

## What Worked

- ~90 typed launchers across `cuda-kernels/src/{quant_linear,tensor_ops,
  attention,ring_attention,recurrent,sampling,comm}.rs`; `infer-cuda` raw FFI
  reduced from 217 production call sites to zero (remaining: the plan-permitted
  GDR fq fn-pointer table and peer-held `moe.rs`).
- `loader.rs` 6,722 → 2,859 common + `qwen35_load.rs` 3,273 + `dsv4/load.rs`
  2,302 + qwen3-dense assembly in `model.rs` (pure mechanical move).
- Load-time quant storage validation (`validate_quant_linear_storage`) closes
  the historical freed-source/no-route defect class; 19 table-driven
  invalid-state cases.
- `infer-cuda/src/moe.rs` 4,604 → facade 154 + `moe/qwen.rs` 1,672 +
  `moe/dsv4.rs` 2,364 + `moe/dsv4_deepep.rs` 434 (pure move; `crate::moe::*`
  paths unchanged; `cargo check --features cuda,nccl,deepep` on H20).
- Registry: 15 semantic operators, 45 implementations (MoE: qwen35/dsv4
  experts + dsv4 transport, legality read from the route owners), legality read from the
  real route owners; autograd NVRTC family catalog + identity tuple; bf16
  module warmup moved to `set_tape_dtype`.
- Remote receipt (H20, `arle-tranche1-eval` vs `arle-tseries`
 Qwen3.8-27B-NVFP4): route counters identical
  (fp8_per_channel_deepgemm 288, fp4.marlin 336, fp4.widen 224, fp8.marlin
  437, fp8_gemv absent), completion text identical, needle 12/12 exact DET at
  512/4096/16384/32768 on both. Wall-clock A/B not rerun for the wrapper
  series: zero kernel-work change, counter identity is the receipt; the T2
  A/B (c=1/4/8/16 in noise) covers the dispatch consolidation.
- Eval harness on the HEAD binary: VERDICT PASS 3/3 (prefix_reuse,
  token_reuse, multiturn_concurrent).
- Multi-model receipt (2026-08-21, same binary pair, GPU-isolated serial runs):
  ThinkingCap-Qwen3.6-27B-FP8 and Qwen3.6-27B-FP8 each show identical route
  counters, identical greedy text, and needle 12/12 exact DET at
  512/4096/16384/32768 on both binaries. DSv4-Flash-FP8 requires TP4; only
  three GPUs were free — pending GPU availability.

## Rule

A launcher-boundary migration is one family per commit, launch receipt
identical, raw call deleted in the same commit; the batched remote receipt is
counter identity + identical greedy text + the needle ladder on the exact
candidate binary.
