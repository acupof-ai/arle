# KV-Tier HiCache-Borrowed Improvements (P0/P1/P2 Roadmap)

**Created:** 2026-05-04.
**Reference:** [`docs/research/2026-05-04-sglang-hicache-guide.md`](../research/2026-05-04-sglang-hicache-guide.md) — full deep-dive on SGLang HiCache.
**Owner:** unassigned (P0 staged for next session).
**Status:** P0 prep landing in this commit; P0/P1/P2 follow-ups pending bench gating.

## 1. Goal

Map the 7 HiCache optimizations onto ARLE's existing `kv_tier` skeleton, identify the real gaps (vs perceived gaps), and stage a sequence of small, complete tranches that close them without introducing half-states. Final outcome: ARLE matches HiCache's TTFT/throughput envelope on prefix-heavy session-resume workloads (the "TTFT 降 56–84%" curve in the LMSYS 2025-09-10 blog).

## 2. Scope

**In scope:** persistence policy (write-through variants + write-back), prefetch policy + metadata-only query API, layer-wise compute/transfer overlap, cross-tier KV layout abstraction, MLA write-back deduplication, NIXL/Mooncake real backend.

**Out of scope:** new tier introduction (we keep T0/T1/T2/T3), new model architectures (DSv3 / sparse attention readiness lives in its own track — see `docs/plans/2026-05-01-deepseek-v4-readiness.md`), prefix matching algorithm changes (RadixCache stays as-is).

## 3. Current State Mapping (HiCache 7 innovations → ARLE)

| HiCache innovation | ARLE current | Gap |
|---|---|---|
| 6.1 GPU SM-assisted I/O kernel | A8 plan exists (`docs/plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md`), gated on W4 closure | Identified, gate is correct |
| 6.2 Layer-wise compute-transfer overlap | T1↔T0 readmission is batch-level | **Missing** |
| 6.3 layer_first vs page_first layout | `TokenKVPool` paged layout, no cross-tier layout abstraction | **Missing**, blocks T3 RDMA efficiency |
| 6.4 Three write policies | `WriteBackMode::{WriteThrough, WriteThroughSelective}` exists at `infer/src/scheduler/cuda/policy.rs:14-18`; **`WriteBack` (defer-until-evict) variant missing**, and enum is in `scheduler/cuda` not `kv_tier/` | Variant + relocation |
| 6.5 Three prefetch policies + metadata query | `PrefetchMode::{BestEffort, WaitComplete}` exists at `policy.rs:8-11` (WaitComplete is dead code); **`Timeout` variant missing**; KVBackend has only `exists()` not `query()` | Variant + new trait method |
| 6.6 MLA write-back dedup | Qwen3/3.5 are GQA, not MLA; **DSv3 path needs this when it lands** | Roadmap-bound |
| 6.7 Zero-copy + RDMA | `nixl.rs` is stub (`rdma-nixl-real` feature optional), `SharedFsStore` is real but FS-backed | Milestone-bound |

**Key empirical findings from current-state exploration (2026-05-04):**

- `prefix_cache::Node.hit_count: u32` already exists at `infer/src/prefix_cache.rs:137` and is incremented on lookup at lines 485-487. **No new field needed for `WriteThroughSelective` hot-detection.**
- `Coordinator` already exposes `submit_prefetch_plan()` (via `CoordinatorHandle`); the entry point exists, the async queue + policy doesn't.
- `WriteBackMode` despite its name has **no `WriteBack` variant**. Naming is misleading vs HiCache convention.

## 4. Differential Plan — Three Phases

### P0 — Foundation (this commit + 1-2 follow-ups)

**P0.0 (LANDED 2026-05-04):** Pure refactor — moved `PrefetchMode` / `WriteBackMode` from `infer/src/scheduler/cuda/policy.rs` to `infer/src/kv_tier/policy.rs`, renamed to `PrefetchPolicy` / `WritePolicy` (HiCache-aligned names), visibility bumped to `pub`. Zero behavior change; foundation for P0.1 and P0.2. All 536 lib tests + 65 kv_tier tests pass. Re-exported from `kv_tier.rs` root.

**P0.1 (follow-up):** Add `WritePolicy::WriteBack` variant.
- Coordinator suppresses `submit_store` calls in `WriteBack` mode at produce time; instead, registers a deferred-persist hook on the block.
- When `prefix_cache::RadixCache::evict()` is about to drop a block whose `WritePolicy == WriteBack` and `host_value` exists, coordinator enqueues a persist op before the slot is freed.
- Acceptance: a unit test where a block is produced, never re-hit, then evicted, ends up in T2/T3; same trace under `WriteThroughSelective` does not (because `hit_count == 0 < threshold`).
- Touchpoints: `kv_tier/coordinator.rs` (new evict-hook path), `prefix_cache.rs` (call hook before dropping `Node`), `kv_tier/policy.rs` (add variant).

**P0.2 (follow-up):** Add `PrefetchPolicy::Timeout { base_ms: u32, per_ki_token_ms: u32 }` variant + metadata-only `KVBackend::query()`.
- New `KVBackend::query(handle: &KVHandle) -> Result<KVQueryResult, TransportError>` returning `KVQueryResult { exists: bool, byte_len: u64, content_hash: Option<BlockFingerprint> }`. Default impl falls back to `exists()` for backends that don't support richer metadata.
- `Coordinator` gains a `prefetch_pending: VecDeque<PrefetchTicket>` queue. `submit_prefetch_plan` enqueues; a worker loop drives `query` first, then `fetch` for hits.
- `PrefetchPolicy::Timeout` computes deadline as `base_ms + per_ki_token_ms × N / 1024` per HiCache (Part VI §6.5); on expiry, prefetch ticket is cancelled and prefill batch starts without that segment (it gets recomputed on GPU).
- Default flips to `Timeout { base_ms: 50, per_ki_token_ms: 10 }` after bench validates.
- Touchpoints: `kv_tier/backend.rs`, `kv_tier/policy.rs`, `kv_tier/coordinator.rs`, all `KVBackend` impls (`shared_fs.rs`, `nixl.rs` stub returns `unsupported`), scheduler call site.

### P1 — Throughput multipliers (after P0 closes)

**P1.1: Layer-wise compute-transfer overlap (HiCache 6.2).**
- Per-layer `LayerReady(layer_idx, slot_id)` events emitted by `LocalCudaTransport` as each layer's KV finishes copying T1→T0.
- Backend's prefill loop awaits the event before forward layer N, but does not block subsequent-layer transfer (producer/consumer with bounded queue).
- Metal skipped: unified memory makes this a no-op.
- Stacks multiplicatively with A8 (smaller per-layer transfer × better hide ratio).
- Touchpoints: `kv_tier/transport/local_cuda.rs`, `backend/cuda/forward.rs` (or equivalent), event channel in `kv_tier/coordinator.rs`.
- Acceptance: 80-layer model (Qwen3-32B class) readmission TTFT drops ≥ `min(transfer_ms, compute_ms)` per layer × N layers; targeting 50–70% of pure-transfer baseline hidden.
- **Viability confirmed by 2026-05-04 bench**: `docs/experience/wins/2026-05-04-bench-kv-tier-copy-throughput.md` measures host-memcpy ceiling at 12 GiB/s (medium blocks), giving per-layer T1→T0 transfer ≈ 1.3 ms vs ≈ 5–15 ms layer compute on Qwen3-32B class — overlap hides essentially all transfer time. Same arithmetic on T1↔T2 disk gives 36 ms per layer (far larger than compute); confirms the design choice that **layer-wise overlap applies to T1↔T0 only, not T1↔T2** — matches HiCache's split between L1↔L2 layer-wise overlap and L2↔L3 prefetch+timeout.

**P1.2: Cross-tier KV layout abstraction (HiCache 6.3).**
- `KVPayload` gains `layout: KVLayout::{LayerFirst, PageFirst}`.
- T1 `HostPinnedPool` defaults to `LayerFirst` (GPU-friendly); T2/T3 writers can request `PageFirst` for zero-copy RDMA.
- Selection rule: no remote backend → `LayerFirst`; NIXL/Mooncake real backend enabled → `PageFirst`.
- **Do not start until** NIXL real backend is functional — early layout abstraction is hypothetical-future-requirement design (per CLAUDE.md guidance).
- Touchpoints: `kv_tier/io.rs`, `kv_tier/transport/nixl.rs` (real path), maybe a transposing kernel under `crates/cuda-kernels/csrc/kv/`.

### P2 — Roadmap-gated

**P2.1: MLA write-back dedup (HiCache 6.6).**
- Trigger: DSv3 model spec lands (`crates/dsv3-spec` or similar, see `docs/plans/2026-05-01-deepseek-v4-readiness.md`).
- Coordinator gates `submit_store` on `is_rank_zero()` when model spec reports MLA; skips on TP ranks > 0.
- Saves N× write traffic on TP=N MLA models — meaningful only once a real MLA model runs.

**P2.2: NIXL real + Mooncake (HiCache 6.7 + Part VII).**
- Trigger: RDMA cluster available + `rdma-nixl-real` feature stable.
- Zero-copy: pass `HostPinnedPool` region addresses directly to NIXL; no intermediate buffer.
- Multi-NIC: env-var-driven device list (Mooncake-style `MOONCAKE_DEVICE` mapping).
- Bench against the same trace used for P0/P1 to quantify cross-instance hit benefit.

## 5. Why this ordering

```
                          ┌── A8 GPU SM kernel (gated on W4) ──┐
                          ▼                                     │
[NOW] P0.0 refactor ─► P0.1 WriteBack ─► P0.2 Prefetch+query ─┐ │
                                                               ▼ ▼
                                                        P1.1 Layer-wise overlap
                                                               │
        P1.2 Layout abstraction ◄── NIXL real cluster
                                                               │
        P2.1 MLA dedup ◄── DSv3 readiness                      │
        P2.2 NIXL/Mooncake real ◄── RDMA cluster ──────────────┘
```

**Hard ordering constraints:**

1. **P0.2 before P1.1.** Layer-wise overlap only matters when there's an async transfer to hide; P0.2 creates the prefetch queue that P1.1 hides.
2. **P0.0 before P0.1 / P0.2.** Co-locating policies in `kv_tier/` is the foundation; adding variants in `scheduler/cuda/` would entrench the wrong ownership.
3. **A8 + P1.1 multiply, don't add.** A8 reduces per-layer transfer time; P1.1 hides what's left. Doing only one captures only one factor of the speedup.
4. **P1.2 not before NIXL real.** Designing a layout abstraction without a remote backend that benefits from it is hypothetical-future-requirements work.

## 6. Acceptance Gates

Each tranche must produce an entry under `docs/experience/wins/` (or `errors/` on regression). For tranches that touch only Rust types and don't move bytes (e.g., P0.0 pure rename), `bench-exempt: pure refactor` in the commit body is acceptable per CLAUDE.md `§Benchmarks` exemption rules.

For P0.1 onward, every tranche needs:
- One `scripts/bench_guidellm.sh` run on the most recent baseline for the affected backend (CUDA on Linux, Metal on Mac), with a Δ% row.
- Wins entry citing the before-snapshot.
- If bench can't run locally (Mac for CUDA changes), open `wins/` entry as `pending-remote` and cite the remote ticket.

Specific quantitative gates:

| Tranche | Quantitative gate |
|---|---|
| P0.1 WriteBack | `evicted_then_remiss_count` (block evicted then needed again, must recompute) drops ≥ 30% vs `WriteThroughSelective` on a session-resume trace |
| P0.2 Prefetch+query | `fetch_wait_p99` for T2/T3 hits drops to ≤ `base_ms + per_ki × N / 1024` under `Timeout`; prefill batch admission no longer blocks on T2 sync read |
| P1.1 Layer-wise overlap | 80-layer Qwen3-32B readmission TTFT drops ≥ 30% vs P0.2 baseline |
| P1.2 PageFirst | NIXL transfer of N tokens × M layers achieves ≥ 70% of single-large-block RDMA peak |
| P2.1 MLA dedup | TP=8 DSv3 write traffic to T2/T3 drops ≥ 7× (theoretical 8×, allowing for non-MLA tensors) |
| P2.2 NIXL real | Cross-instance hit added on top of single-instance baseline reduces TTFT ≥ 50% on shared-prompt traces |

## 7. Touch points (canonical reference)

```
infer/src/kv_tier/
  policy.rs           [NEW in P0.0]   PrefetchPolicy / WritePolicy enums
  coordinator.rs      [P0.1, P0.2]    evict hook + prefetch queue
  backend.rs          [P0.2]          query() trait method
  io.rs               [P1.2]          KVLayout enum
  transport/
    local_cuda.rs     [P1.1]          LayerReady events
    nixl.rs           [P2.2]          real RDMA + zero-copy
    shared_fs.rs      [P0.2]          query() impl

infer/src/scheduler/cuda/
  policy.rs           [P0.0]          re-export from kv_tier::policy
  core.rs             [P0.1, P0.2]    consume new policy variants

infer/src/prefix_cache.rs   [P0.1]    evict hook callback
infer/src/backend/cuda/     [P1.1]    forward awaits LayerReady
crates/cuda-kernels/csrc/kv/ [P1.2]   transpose kernel for PageFirst
```

## 8. Risks

- **Refactor churn.** P0.0 is a rename; downstream crates that re-export `WriteBackMode` (none today, verified) would break. Verified by `cargo check --workspace` before commit.
- **Evict hook race (P0.1).** If eviction runs on the scheduler thread and `WriteBack` enqueues a persist mid-eviction, ordering matters. Same constraint as the existing M3 coordinator-locking invariant (see `kv_tier/AGENTS.md` §Invariants 5).
- **Default flip risk (P0.2).** Changing default `PrefetchPolicy` to `Timeout` after validation is a behavior change; do it as a separate commit with its own bench, not folded into the variant-introduction commit.
- **`pending-remote` accumulation.** P0.1+ will accumulate stub wins entries for CUDA-only changes verified on a Mac. Each must cite the remote-machine ticket clearly to prevent silent skips.

## 9. Cross-references

- HiCache deep-dive: [`docs/research/2026-05-04-sglang-hicache-guide.md`](../research/2026-05-04-sglang-hicache-guide.md)
- Live tier design: [`docs/projects/tiered-kv-cache.md`](../projects/tiered-kv-cache.md)
- Active readmission plan: [`docs/plans/tiered-kv-hicache-readmission.md`](tiered-kv-hicache-readmission.md)
- A8 GPU SM kernel: [`docs/plans/2026-05-03-a8-gpu-sm-kv-io-kernel.md`](2026-05-03-a8-gpu-sm-kv-io-kernel.md)
- Module guide: [`infer/src/kv_tier/AGENTS.md`](../../infer/src/kv_tier/AGENTS.md)
- Bench/trace spec: [`docs/bench-and-trace-spec.md`](../bench-and-trace-spec.md)
