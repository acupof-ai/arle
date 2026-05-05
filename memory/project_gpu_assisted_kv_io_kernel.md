---
name: GPU SM-assisted KV I/O kernel — alternative to cudaMemcpyAsync DMA for fragmented KV transfers
description: For many-small/non-contiguous KV copies (e.g., paged-block scatter/gather between T0 GPU HBM and T1 host pinned DRAM), dedicated CUDA SM kernel beats cudaMemcpyAsync PCIe DMA by ~3× (LMSYS official). Relevant to A6 retention promote/demote bandwidth + future tier I/O optimization.
type: project
originSessionId: d301e8fb-4674-4ac9-a73e-639200c55d56
---
**Why:** User-shared technique (2026-05-03). Walks through industry KV-transfer optimization beyond default DMA path.

**The problem with cudaMemcpyAsync for paged KV:**
- PCIe DMA engine is optimized for **few large contiguous transfers**
- ARLE paged KV is **many small (16-token-block) non-contiguous segments** when promoting/demoting between T0 and T1
- DMA scheduling overhead per segment dominates → low effective bandwidth utilization
- Symptom: T1 promote/demote latency proportional to segment count, not byte count

**The fix — GPU SM-driven kernel:**
- Custom CUDA kernel launches many threads on GPU SMs
- Each thread does strided copy of a fragment
- Many SMs in parallel = aggregate scattered blocks into effectively-large transfer
- Bypasses PCIe DMA scheduler overhead
- LMSYS official benchmark: **3× throughput vs cudaMemcpyAsync**

**Relevance to current ARLE work:**
- A6 (commits `7b9ffb2f`/`f83a8e05`, post-rewrite SHAs differ): T1 host-pinned retention. Uses `HostPinnedPool` (kv-native-sys arena). Promote/demote currently goes through standard CUDA APIs (likely `cudaMemcpyAsync`).
- (A) SessionSlot eviction policy in flight: doesn't need this yet (eviction is bookkeeping; the actual block freeing is policy not bandwidth).
- **Likely deferred deliverable**: A8 GPU-assisted KV transfer kernel — to be considered AFTER (A) closes W4 mission gate. If (A) succeeds and W4 baseline shows promote/demote latency dominating tail TTFT, A8 becomes the next world-first lever.

**Where to look in code:**
- `crates/cuda-kernels/csrc/kv/` — CUDA kernel home for KV ops
- `infer/src/kv_tier/transport/local_cuda.rs` — current LocalCudaTransport plumbing (uses cudarc / cudaMemcpyAsync analogs)
- `infer/src/kv_tier/coordinator.rs` — async promote/demote command/event channel; bandwidth-relevant call site
- `infer/src/kv_tier/host_pool.rs` — HostPinnedPool (kv-native-sys arena), the destination buffer for demote
- `infer/src/kv_tier/transport.rs` — KVTransport trait (where the kernel-vs-DMA choice would land)

**Industry references** (when implementing):
- LMSYS / SGLang KV transport (search for `kv_send` / `kv_recv` kernels in their cuda src)
- Mooncake (uses NIXL for cross-node, but its T1↔T0 path may have similar pattern)
- vLLM 0.20.0+ has prefix cache cross-tier transfer — check `vllm/v1/core/kv_cache_manager.py` and `csrc/cache_kernels.cu`

**Acceptance signal** (when A8 is on the table):
- Bench W4 (or any high-promote-rate workload) against current cudaMemcpyAsync path → measure promote_ms_p99
- Implement SM kernel → re-bench
- Target: ≥ 2× reduction in promote_ms_p99 OR ≥ 50% reduction in T1 fetch_wait_p99

**Composes with:**
- `project_a6_session_kv_retention_2026-05-02.md` — A6 retention substrate uses these transfers
- `kv_tier/AGENTS.md` — "KVTransport trait" is the right interface boundary for swapping the impl
- `memory/project_cow_kv_cache_insight.md` — KV is WORM, transfers are pure copy (no consistency concerns), good fit for kernel-driven path
