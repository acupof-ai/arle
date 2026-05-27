# Multi-tier KV cache — current state (2026-05-28)

User question: "多层 kv 现在是否好用" — is multi-layer KV actually working?

Short answer: **the local CUDA T0/T1/T2 path is live and observable**;
**T1 is dormant on OPD-shaped workloads** (16-512 tok prompts) because the
default `t1_host_pinned_min_prompt_tokens = 4096` gate marks short-prompt
blocks `host_swap_eligible = false`. T1 fires on long-context SERVE
(>=4096 tok) and is validated end-to-end against the new T4a/T4b metrics.
T3 RDMA-class transports remain skeletal. Shared-fs cluster L3 is wired.

Date: 2026-05-28. Source commit: HEAD `f82415a5` (this branch).
Authoritative reads: `infer/src/kv_tier/AGENTS.md`,
`docs/projects/tiered-kv-cache.md`,
`docs/plans/tiered-kv-hicache-readmission.md`,
`docs/experience/wins/2026-05-25-kv-tier-observability-{code-patch,serve-baseline}.md`.

---

## (a) Tier status table

| Tier | Module file(s) | Status | Evidence | OPD (16-512 tok) | SERVE (>=4k tok) |
|---|---|---|---|---|---|
| T0 GPU HBM | `paged_kv` in `crates/cuda-kernels/src/paged_kv.rs` (owner) + `infer/src/scheduler/cuda/core.rs:340-375` (`record_sealed_gpu_blocks`) + `infer/src/scheduler/cuda/runtime/admission.rs:1175-1196` (staged promote) | **live** | evidence: `cargo check -p infer --no-default-features --features no-cuda` clean; T4b run reports T0 hit rate 0.25 after staged replay (`docs/experience/wins/2026-05-25-kv-tier-observability-serve-baseline.md` line 126). | fires (prefix cache attaches sealed prompts) | fires |
| T1 host pinned DRAM | `infer/src/kv_tier/host_pool.rs` (arena) + `infer/src/scheduler/cuda/core.rs:1400-1482` (`demote_block_to_host`) + `1521-1620` (`spill_host_blocks_if_pressured`) + `1649-1750` (`evict_prefix_cache_if_pressured`); trigger `infer/src/scheduler/cuda/runtime/scheduler_loop.rs:215` and `:425` | **live on CUDA, dormant on OPD prompts** | evidence: `core.rs:1385` and `:1182` gate sets `host_swap_eligible = session_id.is_some() && prompt_tokens.len() >= t1_host_pinned_min_prompt_tokens`; default `4096` at `scheduler/types.rs:446`. `prefix_cache.rs:1383` short-circuits eviction candidates that fail the flag. T4b reports 176 demote observations / 214 MB on a 4k-token session replay (`2026-05-25-kv-tier-observability-serve-baseline.md` line 122-128). | dormant (host_swap_eligible=false, demote skipped at `core.rs:1407`) | fires for session-scoped, prompt>=4096-token requests; GuideLLM without session_id does not exercise T1 (line 91-103 of the T4b doc) |
| T2 NVMe disk | `infer/src/kv_tier/transport/disk.rs` (DiskStore) + `infer/src/kv_tier/coordinator.rs` (queued put/get) + `scheduler/cuda/core.rs:1526` gate (`t2_disk_tier_enabled` + cluster_shared_backend) + `scheduler/cuda/core.rs:1571` (store-target disk gate) | **live but flag-gated, off by default** | evidence: `scheduler/types.rs:395` field declared `pub t2_disk_tier_enabled: bool`, default `false` at line 448; CLI flag wires via `main.rs:2160` (`--disk-store-root` turns it on). T4b session-protected blocks → T1 retained, T1→T2 store did **not** fire (line 149-151 of T4b doc). disk_store wiring code exists in `kv_tier/transport/disk.rs:872` (self-heal test). | inert (T2 off by default) | available; not observed in T4b because session refs protected host blocks from store drain |
| T3 cluster shared / remote | `infer/src/kv_tier/transport/shared_fs.rs` (POSIX), `infer/src/kv_tier/transport/nixl.rs` (M5 RDMA stub), `infer/src/kv_tier/backend.rs` (ClusterSharedBackend enum) | **shared-fs live (skeletal), RDMA dormant** | evidence: `kv_tier/transport/nixl.rs:74-110` declares `NixlTransport { name }` with no real ops; comment "M5-stub transport"; `kv_tier/AGENTS.md:108-110` "M5 RDMA-class remote transports … remain skeletal — design-ready, blocked on M4 stabilization". `cluster_shared_backend` config field exists in `scheduler/types.rs:398` and is consumed in `scheduler/cuda/core.rs:1526`. | inert | available when configured; no current OPD/SERVE bench exercises it |
| T0↔T1 local CUDA transport scaffold | `infer/src/kv_tier/transport/local_cuda.rs` | **dead-skeleton, design-frozen** | evidence: only callers are its own tests; `transport.rs:30 pub use local_cuda::LocalCudaTransport;` is consumed nowhere. AGENTS.md:49 documents it as "future P0' NVLink peer hop". Real T0↔T1 byte movement runs through `paged_kv_pool.copy_pages_to_host()` in `core.rs:1425`, not this trait. | unused | unused |

**Apple Silicon (Metal) skips T1** per `kv_tier/AGENTS.md:31` (unified memory).
Metal joins at M4 for T2 disk only (`backend/metal/runtime.rs:543`
constructs `MetalTierAdapter::new(disk_store)`; T1 stays at 0 pressure
via `with_paged_pool_pressure(0.0)`).

---

## (b) Auto tier-down current state

### Trigger location

`evict_prefix_cache_if_pressured()` at
`infer/src/scheduler/cuda/core.rs:1649` is the **T0→T1 spill entry**. It
is called from the scheduler loop at two sites:

- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs:215` (admission
  step, only when `self.waiting` is non-empty — evidence at lines 212-215)
- `infer/src/scheduler/cuda/runtime/scheduler_loop.rs:425` (cleanup step
  at end of tick, comment cross-refs `core::Scheduler::evict_prefix_cache_if_pressured`)

A second, **synchronous fallback** path:
`evict_prefix_cache_for_allocation()` at `core.rs:1756`, called from
`core.rs:985` and `admission.rs:1066` when paged-pool allocation fails.

### Does the T1 4096-tok gate actually fire?

**Evidence (source citation):** the gate is *not* in
`demote_block_to_host` itself — it is set **at block publish time**, two
call sites:

1. `core.rs:1384-1385` (initial publish): `session_id.is_some() &&
   prompt_tokens.len() >= self.config.t1_host_pinned_min_prompt_tokens`
   is passed as `host_swap_eligible` to `record_sealed_gpu_blocks`.
2. `admission.rs:1181-1182` (staged-prefix promote path, same predicate).

`record_sealed_gpu_blocks` (`core.rs:341-375`) writes the flag into
`BlockMetadata` via `BlockMetadataUpdate { host_swap_eligible: Some(...) }`.

The downstream filter is in `prefix_cache.rs:1360,1383` —
`select_blocks_with_policy(..., require_host_swap_eligible: bool, ...)`.
At `core.rs:1675`, eviction-with-demote calls
`select_blocks_with_policy(..., true)`, so candidates with
`host_swap_eligible=false` are skipped before they ever reach
`demote_block_to_host()`. Even if they did, `core.rs:1407` re-checks
the flag and short-circuits to `Ok(0)`.

**OPD verdict (16-512 tok prompts):** `prompt_tokens.len() < 4096` →
`host_swap_eligible = false` → **demote never fires**. T1 is dormant.

**SERVE verdict (>=4k tok prompts, session-scoped):** flag set true,
demote fires under T0 pressure. T4b run measured 176 demote events
on a 4096-token controlled replay (T4b doc line 124).

**Non-session prompts (any length):** `session_id.is_none()` → flag
false regardless of prompt length. GuideLLM default behaviour
(no session_id) → T1 stays dormant. T4b documents this trap on line
93-95: "GuideLLM does not send `session_id`, so it correctly exercised
long prefill serving latency but did not mark blocks `host_swap_eligible`."

### What's missing to unlock T1 on SERVE long-context

Today, T1 is already unlocked on SERVE long-context for the
session-aware HTTP replay path. The actual gap is **not in the
scheduler** — it is in **client workloads**:

- GuideLLM does not emit `session_id`. Fix lives in the bench harness,
  not in `infer/src/`.
- For OPD-style training (16-512 tok), the 4096-tok floor is a
  deliberate choice. Lowering it costs registered-host-memory budget
  per block — see license-or-kill below.

**Smallest concrete patch to validate T1 on SERVE under a normal
GuideLLM workload** (if that's the goal): add `--session-id-mode
per-request` (or similar) to `scripts/bench_guidellm.sh` to emit a
unique session_id per request. Zero `infer/src/` change. Currently
out of scope per this audit's task.

---

## (c) Code-quality audit (deletion-style)

### Audited files

- `infer/src/kv_tier/` — all 14 files including transport/, coordinator/
- `infer/src/scheduler/cuda/` — `core.rs`, `core/construction.rs`,
  `runtime/{admission,scheduler_loop,fetch,helpers,tests}.rs`,
  `policy.rs`

### Findings

1. **`infer/src/kv_tier/coordinator/bench.rs:56-58,306-311`** — `fmt_avg_us`
   helper called only via `let _ = (...)` sink. Zero callers, no
   functional effect. **DELETED** in commit f82415a5 (this audit's first
   commit). Evidence: `grep -rn fmt_avg_us` shows definition + 3 throwaway
   calls only.

2. **`infer/src/kv_tier/transport/local_cuda.rs`** —
   `LocalCudaTransport` struct + `KVTransport` impl. evidence:
   `grep -rn LocalCudaTransport` returns the file itself,
   `transport.rs:30 pub use`, and one AGENTS.md doc reference; **no
   consumer ever instantiates or imports it**. The actual T0↔T1 copy
   runs through `paged_kv_pool.copy_pages_to_host()` (`core.rs:1425`).
   **Status: design-frozen skeleton**, kept by AGENTS.md:49 with the
   note "future P0' NVLink peer hop". Removal would be an
   architectural change (>3 files: transport.rs re-export, AGENTS.md,
   docs/projects/tiered-kv-cache.md §6 milestones, plus the file
   itself); **not deleted** per CLAUDE.md "Approach-first for >3
   files or architectural decisions — outline and wait." Candidate
   for future approach-first review.

3. **`infer/src/scheduler/cuda/policy.rs:70-82`** —
   `impl KvTierAdapter for TieredKvPolicy` provides three no-op
   methods (`paged_pool_pressure → 0.0`, `submit_demote → Ok(())`,
   `submit_promote → Ok(())`). evidence:
   `grep -rn "dyn KvTierAdapter\|: KvTierAdapter"` returns only the
   two `impl` declarations. The trait is **never consumed
   polymorphically**. The CUDA scheduler only calls inherent methods
   (`allow_prefetch`, `choose_store_target`). The `KvTierAdapter`
   trait surface is real on Metal (`backend/metal/runtime.rs:393`)
   but stale on CUDA. **Status: dead trait impl**. Removal touches
   `scheduler/cuda/policy.rs` + `kv_tier.rs` (trait def + re-export)
   + AGENTS.md (mentions the trait), which crosses 3 files → defer
   to approach-first review; **not deleted in this audit**.

4. **`infer/src/kv_tier/chunk.rs`** — `KVBlock`, `KVSpan`, `KVHandle`,
   `KVSpanId`, `SpanTaskKey`, `LayerRange`, `TokenRange` are pub
   re-exported (`kv_tier.rs:167-168`) but **no consumer outside the
   kv_tier module**. They are the planned canonical chunk
   abstraction from `docs/plans/tiered-kv-hicache-readmission.md`
   §4. **Status: design-ready, not consumed**. Removal kills a
   documented future API surface. **Not deleted.**

5. **No half-states / parallel old+new paths found** in the audited
   files. The 2026-04-15 `directory.rs` deletion (per AGENTS.md:55)
   is clean — `TierDirectory` and `BlockDescriptor` are gone, their
   fields live on `RadixCache::Node`. No shadow `dashmap` or
   `sharded map` paths remain (kv_tier.rs:128 retires that note).

6. **No `#[allow(dead_code)]` markers in kv_tier/** beyond
   `kv_tier/transport/nixl.rs:132` (which is the M5 skeleton). The
   two markers in `scheduler/cuda/spec_path.rs:10,659` are
   **out-of-scope** for this audit and **stale**:
   `build_sparse_draft_views` IS reachable (called at
   `spec_path.rs:404`). Left for parent (spec-decode track).

7. **Pre-existing build break (not in audit scope, not touched):**
   `infer/src/scheduler/types.rs:556` —
   `mixed_prefill_token_budget(decode_rows: usize)` signature
   changed in commit `a32ef68d` (NCCL B-1 C.4.2) but caller in
   `infer/src/scheduler/cuda/execution.rs:510` still passes zero
   args. Breaks `cargo check -p infer --no-default-features
   --features cuda,no-cuda` (Mac CUDA-Rust hybrid typecheck).
   `cargo check -p infer --no-default-features --features no-cuda`
   alone is clean. **Reported, not fixed** — out of kv-tier scope
   and "Leave other people's dirty paths in place" applies.

### Net deletion

10 LOC removed in `kv_tier/coordinator/bench.rs` (commit `f82415a5`,
this audit). Other candidates documented above are architectural —
deferred to approach-first review.

---

## (d) License-or-kill decisions

Format: mean / sigma / KILL action.

### Decision 1: Unlock T1 for SERVE long-context

**Status:** already unlocked. T4b SERVE baseline +
session-aware replay validated demote/promote on a 4096-token
session replay (176 demotes, 214 MB demoted bytes,
58.5 ms staged-readmission wait per T4b doc).

**License:** none required — already shipped. Next action is **client
harness**: add session_id emission to GuideLLM bench so the throughput
baseline can also pressure T1.

- mean: GuideLLM-with-session_id would exercise T1 on >=4096-tok
  prompts where the current GuideLLM baseline does not (T4b doc
  line 91 reports 0.0% prefix hit, 0 demote bytes).
- sigma: ROI is observability/coverage, not throughput — adds zero
  measured tok/s; expected delta ±0%.
- KILL: if a session-aware GuideLLM run shows
  `demote_to_host_bytes_total = 0` despite >=4096-tok session prompts
  and 80%+ T0 pressure, that means the eviction path is silently
  bypassing demotion — file an errors entry, kill the gate
  hypothesis, and rerun under `RUST_LOG=infer::scheduler::cuda::core=debug`
  to trace `evict_prefix_cache_if_pressured` candidate selection.

### Decision 2: Lower the 4096-tok gate

**Status:** not licensed today. Current default
`t1_host_pinned_min_prompt_tokens = 4096` (`scheduler/types.rs:446`)
is deliberate.

- mean: lowering to 1024 would let mid-length OPD/agent prompts mark
  blocks host-swap-eligible. Cost: ~4× more blocks compete for the
  bounded host pinned pool (`HostPinnedPool::new` capacity defaults
  to `max_slots * 16 * host_block_bytes` per `core/construction.rs:259-261`,
  clamped to 64 MiB minimum). Benefit unmeasured.
- sigma: ROI hypothesis only — no per-shape bench. The 4096 floor
  was set without published numbers (no wins entry cites it). On
  W4 long-session workloads the cap-bytes flag exists
  (`--t1-host-pinned-capacity-mb`).
- KILL: drop the gate to 1024, bench
  `scripts/bench_guidellm.sh w4-session-1k-vs-4k` per the spec, and
  kill the change if (a) host_pool_high_pressure_ticks rises >10×
  vs 4096 baseline, OR (b) demote latency p99 jumps >2×, OR (c)
  the spilled-then-evicted "host pool dropped GPU blocks" warn
  (`core.rs:1724`) fires >1% of admission ticks. Defer until a real
  workload demands it — current OPD/SERVE traffic doesn't.

### Decision 3: Move kv_tier to next milestone (M5 RDMA / NIXL)

**Status:** still gated on M4 stabilization. Evidence:
`infer/src/kv_tier/AGENTS.md:108-110` —
"M5 RDMA-class remote transports (`transport/nixl.rs`) remain skeletal —
design-ready, blocked on M4 stabilization."

- mean: M4 (cluster-shared T3 over shared-fs + observability) is
  the current shipping focus. T4a/T4b metrics landed
  2026-05-25 (`docs/experience/wins/2026-05-25-kv-tier-observability-*.md`).
  T4b confirms T1 demote + staged readmission counters fire
  end-to-end. T2 store under session refs has the documented
  "live path protects session-owned host blocks" gap (T4b doc
  line 149-151) — a behaviour finding, not a regression.
- sigma: no measured demand for RDMA today. Single-node CUDA bench
  envelope is the working surface.
- KILL: do not start M5 RDMA implementation until at least one of:
  (a) multi-node SERVE deployment is on the roadmap with a
  concrete cluster topology, OR (b) `shared_fs` transport hits its
  documented bandwidth ceiling on a >=2-host bench, OR (c) a P5+
  project doc lists M5 as the next licensed milestone. The
  `NixlTransport` stub stays the way it is until then.

---

## Hypotheses vs evidence summary

| Claim | Type |
|---|---|
| T0→T1 trigger lives at `evict_prefix_cache_if_pressured` | **evidence** (grep + source citation, file:line) |
| OPD prompts (16-512 tok) leave T1 dormant | **evidence** (gate at `core.rs:1385`, `prefix_cache.rs:1383` filter, T4b doc 4096-tok session replay confirms the active path) |
| SERVE 4k+ session-aware requests exercise T1 | **evidence** (T4b 176 demote observations, 214 MB demoted bytes) |
| GuideLLM-without-session-id never fires T1 | **evidence** (T4b SERVE baseline: 0 demote bytes, 0.0% prefix hit) |
| Lowering the 4096-tok gate is net-positive | **hypothesis** (no bench data — license-or-kill blocked on a real workload) |
| `LocalCudaTransport` and `KvTierAdapter for TieredKvPolicy` are dead | **evidence** (grep, no callers) — but architectural removal deferred |
| M5 RDMA blocked on M4 stabilization | **evidence** (AGENTS.md cite) |

---

## Files cited

- `/home/ckl/projects/arle/infer/src/kv_tier/AGENTS.md`
- `/home/ckl/projects/arle/infer/src/kv_tier/host_pool.rs`
- `/home/ckl/projects/arle/infer/src/kv_tier/transport/local_cuda.rs`
- `/home/ckl/projects/arle/infer/src/kv_tier/transport/nixl.rs`
- `/home/ckl/projects/arle/infer/src/kv_tier/coordinator/bench.rs`
- `/home/ckl/projects/arle/infer/src/kv_tier/coordinator.rs`
- `/home/ckl/projects/arle/infer/src/scheduler/cuda/core.rs`
- `/home/ckl/projects/arle/infer/src/scheduler/cuda/core/construction.rs`
- `/home/ckl/projects/arle/infer/src/scheduler/cuda/runtime/admission.rs`
- `/home/ckl/projects/arle/infer/src/scheduler/cuda/runtime/scheduler_loop.rs`
- `/home/ckl/projects/arle/infer/src/scheduler/cuda/policy.rs`
- `/home/ckl/projects/arle/infer/src/scheduler/types.rs`
- `/home/ckl/projects/arle/infer/src/main.rs`
- `/home/ckl/projects/arle/infer/src/prefix_cache.rs`
- `/home/ckl/projects/arle/docs/projects/tiered-kv-cache.md`
- `/home/ckl/projects/arle/docs/plans/tiered-kv-hicache-readmission.md`
- `/home/ckl/projects/arle/docs/experience/wins/2026-05-25-kv-tier-observability-code-patch.md`
- `/home/ckl/projects/arle/docs/experience/wins/2026-05-25-kv-tier-observability-serve-baseline.md`
