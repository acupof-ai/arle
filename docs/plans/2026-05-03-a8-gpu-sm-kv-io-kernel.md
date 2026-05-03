# A8 — GPU SM-Assisted KV I/O Kernel

**Status:** `pending — 待落地`. Gated on (A) session-keyed lookup closing the W4 mission gate (`docs/plans/2026-05-02-agent-load-bench-spec.md` §4.3). Do NOT start implementation until W4 canonical run shows the bottleneck is promote/demote bandwidth, not lookup correctness.

**Owner:** unassigned (will be codex once gate condition met).

**Created:** 2026-05-03.

## 1. Goal

Replace `cudaMemcpyAsync` PCIe DMA path with a dedicated CUDA SM kernel for paged KV scatter/gather between T0 (GPU HBM) and T1 (host pinned DRAM). Target: ≥ 2× reduction in promote_ms_p99 OR ≥ 50% reduction in T1 fetch_wait_p99 on a session-resume workload.

## 2. Why

`cudaMemcpyAsync` is optimized for **few large contiguous transfers**. ARLE paged KV is the opposite shape: many small (16-token-block) **non-contiguous** segments scattered across the page pool. The PCIe DMA scheduler pays per-segment setup overhead, so effective bandwidth scales with segment count not byte count.

The fix used by SGLang/LMSYS: a custom CUDA kernel that launches many threads on GPU SMs. Each thread does a strided copy of one fragment; many SMs in parallel aggregate the scattered blocks into a single effectively-large transfer. LMSYS official benchmark reports ~3× throughput vs `cudaMemcpyAsync` for this access pattern.

Applied to ARLE this means: A6 retention promote/demote (currently bandwidth-bound under heavy session resume) gets a 2-3× tail-latency reduction, which removes the dominant tail-TTFT contribution on W4-class workloads once (A) lookup is correct.

## 3. Why Pending — Acceptance Gate Before Starting

A8 should NOT start until:

1. **(A) closes:** `docs/experience/wins/...session-keyed-w4-...md` shows `matched_prefix_tokens > 7000` mean and `avoided_prefill > 50%` aggregate on W4 canonical.
2. **Bandwidth is the binding constraint:** the same wins entry must show `promote_ms_p99 > 30%` of resume TTFT p99, OR `kv_fetch_wait_samples > 50%` of resume turns. If lookup is closed but tail TTFT is decode-bound or compute-bound, A8 is the wrong lever.

If both (1) and (2) hold, A8 becomes the next world-first lever.

## 4. Scope (when started)

- **In:** dedicated CUDA kernel for T0↔T1 strided block copy, swappable behind the `KVTransport` trait.
- **Out:** T2 (NVMe) and T3 (NIXL) transports — those are I/O-bound differently, not the same fix.
- **Out:** changing the paged-KV layout itself — kernel must work with current `TokenKVPool` block shape.

## 5. Touch Points

| Path | Role |
| --- | --- |
| `crates/cuda-kernels/csrc/kv/` | Where the new kernel lives. Match existing kernel-prelude discipline per `crates/cuda-kernels/AGENTS.md`. |
| `infer/src/kv_tier/transport.rs` | `KVTransport` trait — the boundary where kernel-vs-DMA dispatch happens. |
| `infer/src/kv_tier/transport/local_cuda.rs` | Existing `LocalCudaTransport` (cudarc / `cudaMemcpyAsync`); the new kernel path lives next to it as a sibling impl, not a replacement. |
| `infer/src/kv_tier/coordinator.rs` | Async promote/demote command/event channel; bandwidth path runs through here. |
| `infer/src/kv_tier/host_pool.rs` | `HostPinnedPool` Zig-backed arena — the destination/source on the T1 side. Pinned memory is required for SM-driven copy correctness. |

## 6. Industry References

| Engine | File path / pattern | Notes |
| --- | --- | --- |
| SGLang | `python/sglang/srt/layers/...` and `csrc/...` (search `kv_send`/`kv_recv`) | LMSYS originator; their kernel is the closest analog. |
| Mooncake | NIXL-backed but T0↔T1 has similar pattern | Useful for descriptor-style fragmented transfer design. |
| vLLM | `csrc/cache_kernels.cu` and `vllm/v1/core/kv_cache_manager.py` | Has `cache_kernels` for prefix cross-tier transfer; check 0.20.0+ for the SM-driven path. |

## 7. Risks

- **Kernel correctness:** misaligned fragments, page-fault-on-demand pinned memory, race with active kernels on the same SM. Need careful CUDA stream + event synchronization with the active inference stream.
- **Pinned memory pressure:** `HostPinnedPool` capacity bounds the kernel's effective queue depth. If host pool is small, kernel saturates and falls back to small batches — losing the parallelism win.
- **Wrong-binding-constraint risk:** if A8 ships but A's W4 is decode-bound, no measurable win — wasted cycle. Acceptance gate (§3) protects against this.

## 8. Acceptance Criteria

- New kernel under `crates/cuda-kernels/csrc/kv/` with unit tests against the cudaMemcpyAsync reference output (byte-equal).
- `KVTransport` impl swap is feature-flag gated (default off) initially, then default on after A/B bench.
- W4 canonical re-bench (same trace as the (A)-closing run) shows: promote_ms_p99 reduced ≥ 2× OR resume TTFT p99 reduced ≥ 30%.
- Wins entry under `docs/experience/wins/` with the exact LMSYS comparison table (DMA vs SM kernel) on the same hardware.

## 9. Dependencies

- (A) session-keyed lookup must close W4 first — without it, the bandwidth path isn't the binding constraint and A8 is paperwork.
- `crates/cuda-kernels/AGENTS.md` discipline for kernel layout.
- Memory: `project_gpu_assisted_kv_io_kernel.md` (technical reference + ARLE call site map).

## 10. Composition

- Stacks ON TOP of A6 (`f83a8e05` retention) and A1+A2+A3 (session-affinity admission + stats + lookup).
- Does NOT replace any existing tier — adds a faster transport for the existing T0↔T1 path.
- Independent of (B) chat-lib (`cd5fbb90`); both are orthogonal optimizations.
- May unblock a future A9 — RDMA-class T0↔T3 (NIXL) — by sharing the same fragment-aggregation primitive.
