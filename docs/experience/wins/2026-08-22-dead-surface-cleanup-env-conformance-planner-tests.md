# Dead-surface cleanup + hot-path env conformance + planner tests — 2026-08-22

> Status: Verified locally (infer-core 15/15, lever_gate suite, CUDA lint gate, Metal build). CUDA runtime behavior pending-remote (Mac).

## Context

An architecture health scan (5 parallel readers over seam/engine, CUDA, Metal,
gates, perf evidence) surfaced five candidate weak spots. Each was
adversarially re-verified against the source before fixing; four sub-claims
were rejected as deliberate design (below).

## What Worked

**Dead surface removed.**
- `AdmissionVerdict::ShedTo` had no producer and the engine treated it as
  `Admit`; the variant and its match arm are deleted.
- `host_demoted_pending_inflight` was hardcoded 0 at its only write site and
  plumbed through 6 files (schema, multiproc relay, metrics, CLI). The
  `KvPageTier` trait has no data source for it; the field and all plumbing are
  deleted.
- `evict_slot_page` / `reinstate_slot_page` are KEPT (deliberately retained by
  3f826c204 for the remote-L3 hole-tolerance path), but 6 doc sites referencing
  the deleted `--kv-recall` flag are corrected — including a README feature
  bullet and a support-matrix row.

**Hot-path env conformance.**
- `dsv4_moe_transport()` parsed two env vars per call from the decode/prefill/
  spec hot path, in a file whose header contract is "The statics are the single
  truth — no env reads". Now parsed once into a `OnceLock`.
- 8 per-layer `std::getenv` readers in `mlx_qwen35_model.cpp` (32+ lookups per
  decode step) now use function-local `static const` caching, matching the
  file's own existing op-profile pattern. A follow-up simplify pass
  (`993c3e49b`) collapsed the bool sites into a `parse_env_bool` helper and
  cached the 8th reader, `qwen35_cpp_gdr_threadgroup_y`.

**Gate loudness.**
- `lever_gate.sh` with no `BASELINE_LOG` and `LEVER_GATE_REQUIRE_EXACT=0`
  accepted any miss count — the envelope comparison was silently skipped. It
  now exits 3 with an explicit message unless `LEVER_GATE_ALLOW_NO_BASELINE=1`
  seeds a first baseline. Suite extended with both arms. The simplify pass
  exported that opt-out in `gate_arm.sh`'s seed arm, which the new default
  would otherwise kill.

**Planner tests.**
- The engine's admission/preempt/park repair — the most livelock-prone code in
  the runtime (2026-07-05 TP=4 hang, pod round-5 park ping-pong) — had zero
  unit tests. New host-only tests in `infer-core/src/planner.rs`: a mock
  `BackendExecutor` over the real `HostPagedKvPool`, covering prefill shedding,
  decode preemption with recompute fallback, park success/refuse paths, and
  plan_mode. 15/15 in 0.00s.

**Stale-metallib guard.**
- `mlx-sys/build.rs` stamps the Metal toolchain (compiler path, clang version,
  macOS version) and drops the cmake build dir on a stamp change, so a
  macOS/Xcode update forces a metallib rebuild instead of the runtime "Unable
  to build metal library from source" panic. The simplify pass added matching
  `rerun-if-changed` triggers for the clang path and `SystemVersion.plist` —
  every stamp component now has a trigger that re-runs build.rs to compare it.

## Rejected (deliberate design, not defects)

- **needle_gate.py exits 0 on misses** — deliberate; the verdict is the
  SUMMARY line and lever_gate.sh does the gating.
- **Metal non-greedy sampling downgrades to device greedy** — documented
  tradeoff (host sampling = per-token D2H stall), opt-out via
  `--metal-host-sampling`.
- **MLX op-level FFI panics** — MLX GPU errors are not recoverable at the op
  level; panic → supervisor restart is the standard pattern. Result propagation
  would be a large diff for no recoverability.
- **`INFER_METAL_DFLASH_MAX_ROWS` env read** — load-time once per process, not
  hot path.
- **CP training cp=2/cp=4 dead gate arm** — real, but the fix is a FlashQLA
  kernel build for H=8/Hg=8 (GPU work). Pending-remote; see
  errors/2026-08-21-cp2-131072-stacked-faults-and-the-a2a-core-ceiling.md.

## Rule

Adversarial review of a health-scan finding means re-deriving it from the
source: 4 of 12 sub-claims were stale or deliberate. A dead metric with no
data source gets deleted, not wired with an invented one; a kept primitive
gets docs that match the tree.

## Environment

- Host: Mac (darwin); Metal lane, cpu lane, CUDA lint gate
  (`CUDARC_CUDA_VERSION=12080`, no nvcc)
- infer-core: 15/15 unit tests pass; lever_gate suite passes; CUDA lint gate
  clean; Metal release build clean
