# TP decode graph capture: industry path exists — CUDA, 2026-08-22

> Status: Research, implementation started

## Context

NVFP4-27B TP8 decode on 8×H20 showed TP barely scaling (TP8 c=1 66.9 tok/s ≈ TP1;
aggregate ceiling 127 tok/s, ~0.6% of the bandwidth roofline). Two code comments
blocked the obvious fixes: "NCCL all-reduce is not graph-capturable on this stack"
(`executor/qwen35.rs:791`, hard-disables decode graphs under TP) and a 1024-route
floor on the DeepGEMM grouped-GEMM MoE path ("JIT not capture-safe" at small route
counts). Industry sweep (NCCL docs/issues, vLLM, SGLang, DeepGEMM, TRT-LLM) to
check both premises.

## Findings

1. **NCCL collectives are graph-capturable since NCCL 2.9** (CUDA ≥11.3). vLLM,
   SGLang, TRT-LLM all capture TP all-reduce inside decode graphs by default.
   Pattern: issue `ncclAllReduce` directly on the capture stream (vLLM PyNccl
   ctypes; torch.distributed is the non-capturable part), one graph per
   batch-size bucket, fixed buffers. One GPU per process — ARLE's model already.
   Real caveats: NCCL 2.19/2.20 capture VRAM blowup (`NCCL_CUMEM_ENABLE=0`,
   nccl#1234); 2.26 perf regression, fixed in 2.27.5 (nccl#1692); cross-node
   host-staged transport without GPUDirect RDMA breaks capture (vllm#46253,
   irrelevant — single node NVLink); conditional nodes incompatible (nccl#1986).
2. **DeepGEMM has no route-count floor.** Masked grouped GEMM exists exactly for
   decode-under-graph (`m_indices=-1` padding skips rows); SGLang/vLLM run it at
   all decode shapes. Capture hazards (PR #113 -1-index illegal access, from_blob
   UAF) are upstream-fixed. JIT must finish before capture (SGLang precompiles
   M=1..16384). Padding TMA waste addressed by PR #380 (skip padding I/O,
   BLOCK_M=32 → M=1 gate_up 30.6→8.3 µs, 98-99% of pure-read DRAM peak).
3. **DeepGEMM FP4 grouped GEMM is Blackwell-only.** H20 (sm_90) has no FP4
   tensor cores; FP4 MoE on Hopper is off the industry fast path — ARLE's
   `moe_fp4_e2m1_grouped_gemv` is frontier work, and the 1024 floor gates the
   FP8 path (DSv4), not NVFP4.
4. **Industry numbers:** Qwen3-235B FP8 on H20 c=1 = 71.65 tok/s (derived ~peak
   HBM BW); DeepSeek-R1 on 16×H20 EP16 = 675 tok/s/GPU at BS=32; vLLM 2.2k
   tok/s/H200 (Wide-EP). SGLang ablation: graph capture worth 1.6× at c=1.
   Architecture everywhere: graph-captured masked/padded grouped GEMM + DeepEP
   low-latency a2a (capturable; normal/high-throughput mode is not — implicit CPU
   wait). Per-token GEMV is nobody's decode MoE strategy.

## Rule

"NCCL not graph-capturable" is false on NVLink + modern NCCL — the fix is to call
collectives on the capture stream with fixed per-shape buffers, not to disable the
graph. For MoE a2a under graph: DeepEP low-latency mode only.
