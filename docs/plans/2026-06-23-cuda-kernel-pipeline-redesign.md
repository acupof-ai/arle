# CUDA kernel pipeline — first-principles redesign (registry-driven, prebuilt-release)

Status: design locked, execution pending (pod-verified tranches).
Frame: this is **our own code** — redesign to best-in-class, not a deletion cleanup.
Only dead/vendored cruft (`vendor/tilekernels`) is deleted.

## Problem (evidence, not inference)

The TileLang AOT pipeline has two structural faults, both measured:

1. **N-place lockstep to add one kernel head-config.** Proven: every paged-attn
   FFI shares ONE identical 18-arg signature (`attention.rs` macros
   `tilelang_prefill_hd128_decl` … all byte-identical; the comment itself says
   "HD256 prefill — same FFI shape as HD128, only the cubin's baked head_dim
   differs"). The same `(q_heads, kv_heads)` list is hand-duplicated in 4 places:
   - `tools/tilelang/batch_*_*.py` `SUPPORTED_HEADS`
   - `build.rs` `TILELANG_*_HEAD_CONFIGS` consts (build.rs:327-377)
   - the macro invocation in `src/ffi/attention.rs` (e.g. :461)
   - the dispatch arm in the consumer op
   build.rs and ffi comments literally document the manual triple-sync. Pure
   accidental complexity: there is no per-config logic, only a different symbol name.

2. **28M of committed generated artifacts, 22.6M of it pure redundancy.** Each
   kernel is committed 3×: `.cu` (0.9M total, real source), `.cubin` (3.6M, real
   binary), and `_smXX.c` (22.6M) — which is **the cubin hexdumped to a C byte
   array** (verified byte-for-byte: 34296 cubin bytes == 34296 `0x..` tokens), 6×
   bloat. Plus 31 dispatch `.c` that are a pure SM-switch template (build.rs:273).
   Irreducible real content = 4.5M; the other 23.5M is text-of-binary + regenerable
   template. Structure is 100% uniform (62/62 embed `.c` = exactly 1 cubin + 1
   bespoke launcher; the 2.8M outlier too), so no kernel forces an exception.

## Principle

A kernel is declared **once**. From that one declaration the build derives every
mechanical surface — device binary, C launcher, SM dispatch, Rust FFI extern,
Rust dispatch table, Python codegen inputs. Binaries ship as a reproducible
release, not git text. Building needs **zero required env** (everything auto-detected).

## Target architecture

### 1. Single source of truth — `crates/cuda-kernels/kernels.toml`

```toml
[[kernel]]
family   = "batch_decode_paged"   # batch_decode_paged | batch_prefill_paged | gated_delta_rule | ...
head_dim = 128
q_heads  = 32
kv_heads = 8
kv_dtype = "bf16"                 # bf16 | fp8
phase    = "decode"               # decode | prefill | split_partial | split_merge (omit for non-attn)
abi      = "paged_attn_v1"        # names the shared 18-arg C signature
```

Adding a head-config = **one row**. The Python `SUPPORTED_HEADS` lists are derived
from this (codegen reads the toml), so the 4-place sync collapses to 1.

### 2. Generated from the registry (build.rs → `OUT_DIR` → `include!`)

- `ffi_tilelang_generated.rs`: the `unsafe extern "C"` block (one shared-signature
  macro fed all symbols from the registry) + `fn resolve(family, head_dim, q, kv,
  kv_dtype, phase) -> AttnKernelFn`. Replaces the hand-written macros + per-config
  invocations + the consumer dispatch arm. (bindgen-style `include!` of an
  OUT_DIR `.rs` is standard Rust; no proc-macro needed.)
- The per-kernel C launcher (grid/arg/shmem — bespoke, TileLang-emitted) stays as
  **generated code**, keyed off the registry. Bespoke logic stays as code, never
  encoded as registry data (that is the schema-churn trap we deliberately avoid).
- SM-dispatch `.c`: regenerated from build.rs's existing template — no longer committed.

### 3. Artifact pipeline — reuse the existing prebuilt path, make it first-class

The prebuilt path already exists: `ARLE_CUDA_KERNELS_PREBUILT_DIR` →
`link_prebuilt_cuda_artifacts` (build.rs:2025) links `libkernels_cuda.a` +
`libtilelang_kernels_aot.a` (+ `arle_deepep_sidecar`), skipping nvcc AND TileLang
entirely. Promote it to primary:

- **Consume (default, serving/offline):** auto-detect a conventional path
  (`target/kernel-cache/<kernel-set>-<sm-set>/` or `crates/cuda-kernels/prebuilt/`);
  if the two archives are present, link them. **No env.** Archives are published as
  a versioned release per (kernel-set × SM); offline pods get them via `tn push`
  into the conventional dir — same local-fed path already trusted for deps.
- **Produce (kernel-dev / CI, on a GPU box):** auto-detected TileLang/Python regen
  → nvcc → the two archives → uploaded as the next release. No committed binaries.
- **Result:** `generated/` (28M) leaves git. git carries the *authoring* source
  (`kernels.toml` + `tools/tilelang/*.py` DSL) + the small generated Rust glue.
  A kernel change = a toml row + a release bump, not 200KB of text churn.

Rejected alternatives: runtime JIT (Triton-style) re-adds Python + cold-start on
the serving box — ARLE deliberately chose AOT; Git LFS needs an LFS pull on clone
(network) and is clumsier to feed offline than a plain tarball.

### 4. build.rs modularization

2777 lines doing 5 jobs → focused `#[path]` modules under `build/`: `sm_detect`,
`registry`, `tilelang_codegen`, `ffi_codegen`, `vendored_libs`, `prebuilt`, `nvcc`.

## Env collapse (target: 0 required)

| env | now | after |
|-----|-----|-------|
| `ARLE_TILELANG_REGEN` | force regen | **deleted** — auto-regen when archive/cubin absent or `kernels.toml` row's src-hash mismatches |
| `INFER_TILELANG_PYTHON` | pick python | auto (`find_tilelang_python` exists); optional override only |
| `TORCH_CUDA_ARCH_LIST` / `CMAKE_CUDA_ARCHITECTURES` | pick SM | auto (`sm_targets_from_nvidia_smi` exists); optional override only |
| `ARLE_CUDA_KERNEL_SET` | full/dsv4/opd | **→ cargo feature** (`kernels-dsv4`, `kernels-opd-gdr`) |
| `ARLE_CUDA_KERNELS_PREBUILT_DIR` | prebuilt selector | auto-detect conventional path; optional override only |

Required env after: **0**. The many `ARLE_CUDA_ENABLE/DISABLE_*` kernel toggles
fold into cargo features where they gate inclusion (separate sweep, low priority).

## Verification reality

CUDA-only → cannot build on this Mac. Every tranche verifies on the **H20 pod**:
`cargo build --release --features cuda` green + `scripts/needle_gate.py` + a
`docs/experience/wins/` bench entry (Verify-phase exit). Mac CI typecheck
(`cargo check -p infer-api … --features cuda,no-cuda`) is a smoke pre-check only.

## Tranches (each: small commit + pod-verify)

0. ✅ `vendor/tilekernels` removed (vendored, 0 refs) + README ref cleaned (`d638c134`).
1. ✅ **Registry + generated FFI** (`64dc0b13`). `kernels.toml` (34 rows, 25 ffi)
   drives `build.rs`-emitted `ffi_tilelang_generated.rs`; 9 macros + invocations +
   the consumer match → `include!` + `resolve_*()`. Mac-green; FFI parity proven
   byte-equal (25==25, all 4 ABI sigs); net −905 build.rs lines. **Pod-pending**:
   nvcc/cc + link all 3 KernelSet branches + needle gate ×3 vs bf16 envelope + bench
   (see [wins stub](../experience/wins/2026-06-23-cuda-registry-ffi-codegen-pending-remote.md)).
2. **Prebuilt-release first-class.** Conventional-path auto-detect; a
   `scripts/build_kernel_release.sh` on a GPU box produces the two archives + sidecar.
3. **Drop committed `generated/`** once prebuilt is the fed default; regen-on-demand
   from source for GPU-box builds.
4. **Env collapse** + `ARLE_CUDA_KERNEL_SET` → cargo feature.
5. **build.rs modularization.**

## Decisions taken (defaults, correct if silent)

- Release artifacts host: GitHub Release on origin (internal store if ckl redirects).
- From-source build now requires the TileLang toolchain (industry-normal: flash-attn
  ships prebuilt-or-compile). The old "fully-vendored build needs no Python"
  invariant is replaced by "prebuilt archive needs no Python; from-source needs the
  toolchain" — strictly cleaner (binaries reproducible + hash-pinned, not git text).
- HIP/Vulkan kernels adopt this registry pattern later, when their config matrix
  grows (YAGNI now — their matrices are tiny).
