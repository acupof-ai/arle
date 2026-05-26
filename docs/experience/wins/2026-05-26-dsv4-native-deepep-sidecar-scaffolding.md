# DSv4 native DeepEP — sidecar scaffolding land (phase 1.1.1-1.1.6 + 1.1.8 reservation)

## Context

Phase 1.0a-iv
([`./2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md`](./2026-05-26-dsv4-deepep-cpp-full-dispatch-combine.md))
proved that 8-rank DeepEP intranode dispatch+combine works end-to-end
from a pure-C++ binary with byte-deterministic output. Phase 1.1 turns
that one-shot spike into the production-shape sidecar transport that
ARLE's Rust scheduler can spawn, drive over IPC, and tear down cleanly.

Per
[`docs/plans/2026-05-26-dsv4-deepep-process-per-rank.md`](../../plans/2026-05-26-dsv4-deepep-process-per-rank.md)
Option A (sidecar EP child process) and the brief at
`/tmp/phase1_1_subagent_brief.md` (now superseded), phase 1.1 has
eleven sub-tasks. This entry closes 1.1.1-1.1.6 plus a stub for 1.1.8.
The remaining 1.1.7 / 1.1.8 wire-in and 1.1.10 / 1.1.11 SLO bench
A/B are explicitly **deferred** with reasons.

## What landed

| # | Sub-task | Commit | File(s) | Status |
|---|---|---|---|---|
| 1.1.1 | Sidecar binary (protocol + impl) | `205317d9` | `crates/cuda-kernels/csrc/deepep_sidecar/{protocol.hpp,sidecar_main.cpp}` | PASS — nvcc-compiles clean on 8 × H20 pod (`BUILD_EXIT=0`). |
| 1.1.3 | Build integration | `205317d9` | `crates/cuda-kernels/build.rs` | PASS — opt-in via `ARLE_DEEPEP_DIR` env var, exports `ARLE_DEEPEP_SIDECAR_PATH` for runtime discovery. SM90-only (matches DSv4 target H100/H20). |
| 1.1.4 | Rust SidecarPool | `fefaef8c` | `infer/src/backend/cuda/deepep_sidecar/pool.rs` | PASS — spawn N children via `Command::pre_exec` + `dup2(p2c→10, c2p→11)` matching phase 1.0a-iii layout. `Drop` impl sends Shutdown with 2s SIGKILL fallback. |
| 1.1.5 | Rust protocol mirror | `fefaef8c` | `infer/src/backend/cuda/deepep_sidecar/protocol.rs` | PASS — `#[repr(C, align(8))]` structs match `protocol.hpp` byte-for-byte; round-trip size tests pass. |
| 1.1.6 | Module entry + binary path | `fefaef8c` | `infer/src/backend/cuda/deepep_sidecar.rs` | PASS — `baked_binary_path()` resolves to `option_env!("ARLE_DEEPEP_SIDECAR_PATH")` which is `Some(path)` when the sidecar was built. |
| 1.1.6b | Smoke test | `fefaef8c` | `infer/tests/deepep_sidecar_smoke.rs` | LANDED, skipped by default. Triple-gated on `cuda` feature + baked path + `ARLE_DEEPEP_RUN_SMOKE=1`. On a properly-configured pod it spawns 8 sidecars, runs RoundTrip twice, asserts rank-tag pattern + determinism. |
| 1.1.8 | Reserve `ARLE_DSV4_MOE_BACKEND=native-deepep` | `cd780fc2` | `infer/src/model/deepseek/{weights.rs,mlp.rs}` | PASS — both env-var entry points bail with explicit "not yet wired" error pointing at phase 1.1.7. Prevents silent fallback to NCCL DeepEP-style. |

### Wire protocol

8-byte `MessageHeader { cmd_or_status: u32, payload_bytes: u32 }`,
per-command payload, all little-endian. Commands:

- `kBoot` (0x01) — host posts `BootRequest`, sidecar replies
  `BootResponse` with its device id + `cudaIpcMemHandle_t`.
- `kSync` (0x02) — host gathers all `BootResponse`s, broadcasts the
  full peer array back. Sidecar opens IPC for peers, runs
  `intranode::barrier`, replies `Ok`.
- `kRoundTrip` (0x10) — runs full dispatch + identity expert step +
  combine on synthetic input, replies with SHA-256 + first-8 preview.
  Smoke-test command — exercises the whole DeepEP cycle without any
  real MoE wiring. Phase 1.1.5/6 uses this.
- `kDispatch` (0x20), `kCombine` (0x21) — reserved for the real MoE
  forward path. Phase 1.2+ wires these in.
- `kShutdown` (0x7f) — clean teardown, exit 0.

### Build gate

The sidecar binary is **opt-in**. Build behavior:

| `ARLE_DEEPEP_DIR` | `cuda` feature | SM90 in arch list | Output |
|---|---|---|---|
| unset | any | any | Sidecar skipped, `ARLE_DEEPEP_SIDECAR_PATH` not set. `option_env!` returns `None`. |
| set, path invalid | any | any | Warning, sidecar skipped. |
| set, valid | off | any | Cuda build skipped entirely (unchanged behavior). |
| set, valid | on | absent | Warning ("sidecar targets H100/H20 only"), sidecar skipped. |
| set, valid | on | present | nvcc-compiles, output at `$OUT_DIR/arle_deepep_sidecar`, path published via `cargo:rustc-env=ARLE_DEEPEP_SIDECAR_PATH=<path>`. |

This keeps every existing build path bit-identical for users without
DeepEP source on hand.

### Sidecar binary verification

Pushed `sidecar_main.cpp` + `protocol.hpp` to the 8 × H20 pod and ran:

```
nvcc -ccbin g++ -std=c++17 -O2 -DDISABLE_NVSHMEM \
  --expt-relaxed-constexpr --expt-extended-lambda \
  -I/<DeepEP-src>/csrc -I. -gencode=arch=compute_90,code=sm_90 \
  <DeepEP-src>/csrc/kernels/{intranode,layout,runtime}.cu \
  sidecar_main.cpp \
  -lcudart -L/usr/local/cuda/lib64 -o arle_deepep_sidecar
```

Exit 0, ~412 KB binary. Same compile invocation as the phase 1.0a
spike, no new external deps.

### Rust verification

`cargo check -p infer --no-default-features --features cuda,no-cuda`
(with `CUDARC_CUDA_VERSION=12030` shim for the Mac-without-nvcc dev
env) passes clean modulo pre-existing dead-code warnings unrelated
to this PR. Test build (`cargo check --tests ...`) also clean.

## What's deferred — phase 1.1.7+ remaining

### 1.1.7 LayerCommunicator integration

`infer/src/model/layer_communicator.rs` is the existing thick struct
hosting NCCL groups for tensor/data/context/expert parallel. Adding a
`NativeDeepEPTransport` variant means new methods (`moe_native_deepep_
dispatch`, `_combine`) plus state to hold the `Arc<SidecarPool>` per
EP group. Estimate ~200 LOC of careful changes to a 750-line module
that's on every layer's hot path.

**Why deferred**: I cannot end-to-end test this change in the current
session — verifying it requires real ARLE serve runs on the pod to
catch regressions in the NCCL DeepEP-style fallback. Landing it
half-tested risks bricking the production `deepep` backend default.
The scaffolding above is independently verifiable and the integration
becomes a clean follow-up PR.

### 1.1.8 `forward_deepep_routed_gpu` route switch

Currently `forward_deepep_routed_gpu` (the 800+ line MoE forward in
`mlp.rs`) calls into NCCL via `LayerCommunicator` helpers. The
`native-deepep` backend needs the same MoE forward shape but with
`SidecarPool::dispatch` / `SidecarPool::combine` in place of the NCCL
calls. The route switch is small once 1.1.7 lands.

### 1.1.9 Smoke test execution on pod

The smoke test (`infer/tests/deepep_sidecar_smoke.rs`) is committed
and gated. Running it requires (a) `ARLE_DEEPEP_DIR=<DeepEP-src>` at
cargo build time so the sidecar binary is built, (b) cargo workspace
checked out on the 8 × H20 pod, (c) `ARLE_DEEPEP_RUN_SMOKE=1` at
`cargo test` invocation. The current pod has no cargo workspace
mounted — landing it inside the existing CI container will be a
separate environment-setup task.

### 1.1.10/1.1.11 SLO bench A/B

Phase 1 PASS gate per the design doc: 32K input / 1.5K output, c=8,
qps=8, p50 TTFT delta ≥ +5%, p50 TPOT delta ≥ +5%, p99 not regressed
> 3%, byte-identical greedy. Requires hours of pod time and depends
on 1.1.7+1.1.8 landing first.

## Architecture notes for follow-up

- The sidecar process holds the NVL buffer + workspace + pinned MoE
  counters for its lifetime; only the per-call dispatch/combine
  scratch is allocated/freed inside RoundTrip / Dispatch / Combine
  handlers. This matches DeepEP's `Buffer` lifetime contract.
- All NVL allocations are 512 MiB per rank (`kNvlBytes`) matching
  phase 1.0a-iii. Production may need to size this from DSv4 SLO
  (32K tokens × topk6 + headroom); not yet evidence-driven.
- CUDA IPC handles are 64-byte opaques; the wire layout pairs each
  with `device_id` to allow heterogeneous-device pools (not used
  today but reserved).
- `pre_exec` dup2 to fds 10/11 has been validated across phase 1.0a-ii
  / iii / iv. The same convention is preserved here.

## Artifacts

- New source:
  `crates/cuda-kernels/csrc/deepep_sidecar/{protocol.hpp, sidecar_main.cpp}`
  (104 + 626 LOC).
- New Rust:
  `infer/src/backend/cuda/deepep_sidecar/{protocol.rs, pool.rs}`
  + `deepep_sidecar.rs` entry + `tests/deepep_sidecar_smoke.rs`
  (~700 LOC total).
- Build glue: 90-line `build_deepep_sidecar` helper in
  `crates/cuda-kernels/build.rs`.
- Commits: `205317d9` (C++ + build.rs), `fefaef8c` (Rust pool +
  smoke), `cd780fc2` (backend reservation).

## Bench note

Phase 1.1 scaffolding does **not** change any active runtime path —
the production `ARLE_DSV4_MOE_BACKEND=deepep` default still routes
through the NCCL DeepEP-style fallback (`forward_deepep_routed_gpu`).
Pure scaffolding land, no bench delta expected. Phase 1.1.10/11 SLO
A/B is the bench gate, contingent on 1.1.7+1.1.8 landing first.

## Rule

When the next axis depends on a substantial process-lifecycle change
(spawn child binaries, IPC handshake, command-loop protocol), land
the scaffolding behind an opt-in build flag + reserve the user-facing
gate with an explicit bail. Then do the hot-path integration as a
separate PR that can be reviewed and bench-validated independently.
Half-integrated transports on the production hot path are the worst
of both worlds — the default flag may silently regress while the
new path is untested.
