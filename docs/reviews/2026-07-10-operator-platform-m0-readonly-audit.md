# Operator platform M0 read-only audit

Date: 2026-07-10

## Verdict

The evidence licenses a scope-reduced vertical slice, not a global operator
platform. Static inventory confirms the truth/engagement problem, while current
build and artifact sizes do not justify OCI, a build farm, or an all-at-once
registry migration. Canonical GPU measurements remain incomplete because the
local and pod source trees are not at the same commit.

The first implementation target should be the evidence plumbing and the exact
Qwen FP8 dense-projection cells, not DSv4. DSv4 already combines stateful
composites, JIT, TP/EP, provider fallback, and topology; starting there would
confound the registry, policy, and artifact migrations.

Plan: [`../plans/2026-07-10-operator-artifact-dev-release-system.md`](../plans/2026-07-10-operator-artifact-dev-release-system.md).

## Scope and source state

This audit changed no runtime source and launched no model.

Local source:

```text
commit: 5c6c436c7
host: Apple Silicon, Darwin arm64
rustc/cargo: 1.95.0
nvcc: absent
```

The local worktree changed concurrently during the audit: multiple existing
`cuda-kernels`/`infer-cuda` files and two unrelated experience entries became
dirty or deleted. This audit did not edit them. Static findings describe the
live inspected worktree and must be regenerated from the clean M0 commit before
they become canonical inventory. The local typecheck below completed before
those concurrent CUDA edits appeared.

Pod source during inspection:

```text
commit: f7891c3f, detached and dirty
GPU: 8x NVIDIA H20, sm90, 97,871 MiB each
driver: 535.161.08
nvcc: CUDA 12.9
topology: NV18 GPU-to-GPU, two NUMA domains
```

The pod evidence below proves environment and mechanism, not source-aligned
performance for local commit `5c6c436c7`. A canonical M0d run requires one clean
commit on both hosts plus binary symbol verification.

## Static inventory

### FFI

There are 312 checked-in handwritten CUDA FFI declarations, not 313. The earlier
count included the ordinary Rust helper `ffi/nccl.rs::check`. The generated
TileLang include contributes 39 more declarations, so the declared Rust FFI
surface is 351 for the inspected build shape. Final authority remains the
lane-specific generated file plus archive/link evidence.

| Domain | Declarations |
| --- | ---: |
| gemm | 66 |
| misc | 53 |
| attention | 32 |
| moe | 29 |
| kv | 27 |
| quant | 18 |
| nccl | 18 |
| recurrent | 16 |
| comm | 16 |
| embedding | 12 |
| norm | 9 |
| elementwise | 9 |
| sampling | 8 |

256 have a Rust reference outside their declaration module. Of the remainder:

- one NCCL symbol is consumed by a same-module helper;
- two row-quantize entries are test-only;
- 53 have no Rust runtime caller and are dead-ABI candidates.

No candidate is licensed for deletion. C-to-C calls, dynamic providers,
real/stub symbol replacement, and archive retention still need `nm`, linker-map,
provider marker, and runtime-counter evidence.

Inventory must use independent axes:

```text
role: semantic endpoint | composite stage/helper | provider ABI/control |
      test/probe-only | dead candidate
liveness: default | feature-gated | shape-gated | unreachable
```

This prevents NCCL lifecycle symbols and workspace/preflight helpers from being
misrepresented as semantic operators.

### CUDA translation units

The tree contains 61 checked-in `.cu` files:

| Domain | TUs |
| --- | ---: |
| misc | 19 |
| gemm | 16 |
| attention | 14 |
| kv | 5 |
| quant | 3 |
| moe | 2 |
| comm | 1 |
| kvcacheio | 1 |

This is not the build DAG. `crates/cuda-kernels/build.rs:1977-2041` replaces or
adds FlashMLA sources, `:2081-2112` selects FA3 real/stub sources, and feature
gates change custom all-reduce membership. Vendor TUs are additional nodes.

The inspected sm90 artifact confirms the difference:

```text
libkernels_cuda.a: 70 archive objects, 1,080 unique defined global symbols
libtilelang_kernels_aot.a: 93 archive objects, 93 unique defined globals
```

Real FA3 and FlashMLA marker symbols plus DeepGEMM preflight are present. This is
artifact evidence for that pod build only, not proof for every product lane.

`ARLE_CUDA_KERNEL_SET=dsv4_flash` disables TileLang AOT and emits stubs, but the
recursive native `csrc` collection still compiles. It is not a native
incremental build mode (`build.rs:696-732,1372-1395,1977-1979`).

### TileLang rows and reachability

`crates/cuda-kernels/kernels.toml` contains 48 rows:

```text
attention BF16: 20
split partial/merge: 8
attention FP8: 11
legacy GDR: 6
FlashQLA: 3
```

39 emit generated FFI; nine are build-only.

Static reachability:

- eight HD128/KV8 BF16 rows are reachable by default through
  `infer-cuda/src/attention.rs:874-920`;
- six legacy GDR rows are conditionally called by
  `autograd/src/backend_cuda.rs:3913-4075` under the chunkwise-prefill and short
  sequence gates;
- three FlashQLA rows are reachable only with the build gate, sm90, the default
  false runtime option, and the required recurrent shape
  (`build.rs:1415-1437`, `qwen35.rs:5791-5880`);
- 31 rows have no current runtime caller: 12 BF16 HD64/HD256, eight split, and
  11 FP8 rows.

This is source evidence only. Runtime counters per product lane must prove
engagement or non-engagement before deletion.

The comment in `cuda-kernels/src/ffi/attention.rs:352` says 25 generated paged
attention symbols; the registry currently emits 39 FFI rows. Documentation has
already drifted from generated truth.

## Existing composites

The current runtime already selects call structures larger than one kernel:

1. paged-attention prep -> attention -> quant finalize
   (`infer-cuda/src/attention.rs:714-755`);
2. fused MHC pre plus RMSNorm, replacing two calls and an intermediate
   (`infer-cuda/src/hc.rs:410-413`);
3. Qwen MoE gate+up paired projections across Dense/FP8/FP4 providers
   (`infer-cuda/src/moe.rs:1720-1803`);
4. grouped w13 -> fused SwiGLU+requant -> grouped down DeepGEMM
   (`infer-cuda/src/moe.rs:1379-1429`);
5. fused DSv4 Q RMS/RoPE plus K RoPE preparation
   (`infer-cuda/src/attention.rs:5675-5713`);
6. fused DSA indexer and FP8 paged-MQA cache/layout operations
   (`infer-cuda/src/attention.rs:8242-8270,9257-9345`);
7. FlashMLA SW-ring/current-token/compressed packing and its batched replacement
   (`infer-cuda/src/attention.rs:2769-2788,3094-3150`).

Therefore a one-operator/one-entry registry would encode the wrong performance
unit. Explicit named composites are necessary; a general graph optimizer is not.

## Current dispatch and policy

Dispatch is distributed across backend/model branches:

- Qwen FP8 dense uses hard-coded row floors and legality checks in
  `infer-cuda/src/quant_linear.rs:27-32,221-230,482-503`.
- Qwen MoE selection also depends on load-time grouped weight construction,
  flags, and routed rows (`infer-cuda/src/moe.rs:113-118,527-541`).
- Qwen decode already has a two-launch composite replacing a three-launch
  sequence (`infer-cuda/src/moe.rs:811-930`).
- DSv4 fused WQKV and attention provider decisions live in separate branches
  (`infer-cuda/src/attention.rs:1621-1662,1821-1835,5023-5073`).
- Topology models requested TP/EP axes, while communicator startup may select or
  demote the realized collective backend (`infer-topo/src/topology.rs:121-166`,
  `infer-cuda/src/tp.rs:167-203,304-342`).

Policy identity therefore needs GPU SKU/physical SM count and realized
collective state in addition to SM and requested topology.

## Evidence plumbing gap

`/v1/stats` currently exports scheduling, throughput, prefix/KV, and speculative
decode fields, but no operator, selector, composite, or artifact counters
(`infer-server/src/schema.rs:444-547`).

The existing host-only stats path is reusable:

```text
infer-seam/src/lib.rs:101-152
  -> infer-core/src/lib.rs:765-768
  -> infer-server/src/execution.rs:48-81
  -> infer-server/src/multiproc_relay.rs:251-368
  -> infer-server/src/schema.rs
```

`scripts/bench_guidellm.sh:520-545,586-667` wraps JSON stats but still parses
legacy `key=value` text. Current trace summaries can silently report `n/a` even
when raw stats exist. The M0c self-test must pass before any new canonical
operator benchmark.

## Build and artifact chain

### Cargo-side compilers

`crates/cuda-kernels/build.rs` can run GPU detection, Python/TileLang, nvcc,
host C/C++, ar, nm, and the external DeepEP sidecar compiler. Separately,
`crates/deepep-sys/build.rs` compiles the in-process DeepEP/NVSHMEM archive,
including relocatable device code and device link.

Moving only `cuda-kernels/build.rs` out of Cargo would not satisfy zero GPU
compilation.

### Two DeepEP products

- `cuda-kernels` builds an external process sidecar with NVSHMEM disabled.
- `deepep-sys` builds an in-process archive and may enable NVSHMEM.

The sidecar path is currently embedded through an absolute Cargo OUT_DIR. The
ideal product must find it through its manifest or an executable-relative path.
M0a must license KEEP/MERGE/DELETE before M1 models DeepEP as one provider.

### Current identity gaps

The current prebuilt manifest omits several load-bearing inputs and includes
absolute/non-behavioral values. The TileLang cache uses an FNV64 key over only a
subset of source, ABI, requirements, and nvcc release. It does not fully bind
installed package/patch bytes, CUDA/CUTLASS headers, host compiler, or full
compiler argv. Legacy artifacts without `SRC_HASH` are accepted.

The fast build harvests the newest matching Cargo OUT_DIR by mtime rather than
an exact graph output. A recorded FlashMLA-decode environment variable is not
read by current build behavior, producing a false cache input.

These are counterexamples for the new build-ID schema, not reusable identity.

`strings` on the inspected native archive exposes absolute
`/host/arle-build/.../deepgemm` and CUTLASS include paths. Path normalization and
executable/manifest-relative provider discovery are therefore measured needs,
not speculative cleanup.

### Runtime JIT

DeepGEMM preflight requires headers, CUTLASS, nvcc, and cuobjdump
(`csrc/gemm/deepgemm_native.cu:353-407`). A miss executes nvcc and cuobjdump,
loads the module, and caches it (`:1246-1400`). The current digest covers
generated code and architecture, but not the full compiler/toolchain/provider
identity.

Qwen warmup creates M=2048, M=16, and grouped route-derived shapes
(`infer-cuda/src/qwen35.rs:1965-2157`). Only cold-cache lifecycle tracing can
prove whether warmup covers the first real request.

## Measured development/build evidence

### Local Mac typecheck

Canonical command:

```bash
CUDARC_CUDA_VERSION=12080 cargo check -p infer-api --release \
  --no-default-features --features cuda,no-cuda --lib
```

Results on local commit `5c6c436c7`:

| State | Wall | Result |
| --- | ---: | --- |
| infer-server/infer-api check | 2.60 s | PASS, CUDA/TileLang explicitly skipped |
| exact hot repeat | 0.56 s | PASS, CUDA/TileLang explicitly skipped |

This validates the no-nvcc Mac typecheck path, not the future artifact consumer.

### Existing pod build logs

Recent sm90 logs show:

| Build | Wall | Cargo crates listed |
| --- | ---: | ---: |
| full `arle --features cuda` | 113 s | 11 |
| full `arle --features cuda` | 110 s | 8 |
| `infer-cuda` small-M example | 79 s | 3 |
| `arle --features cuda,nccl` | 59.7 s | 6 |
| separate tree `cuda,nccl` | 59.3 s | 6 |
| exact hot `arle --features cuda,deepep` no-op | 0.408 s | 0 |

The canonical pod environment also records a forced kernel regeneration of 251
seconds versus 53 seconds from its persistent cache. These numbers are workload
signals only: the inspected pod HEAD differs from the local source commit.

The hot no-op emitted cached build-script warnings but no `Compiling` line. It
was not process-traced: the current producer image lacks `strace`. Therefore it
proves Cargo freshness and wall time, not yet the zero-compiler subprocess
contract.

### Runtime dependency closure

`ldd` on the existing pod product reports libc, libstdc++, libcuda,
libcudart.so.12, libcublas.so.12, libcublasLt.so.12, libnccl.so.2, OpenSSL, and
system libraries. It cannot report DeepGEMM's `dlopen("libcuda.so.1")` or runtime
compiler subprocesses. M0b requires both static and cold runtime closure.

## First policy migration

Use `qwen.fp8_dense_projection` first:

- the existing `infer-cuda/examples/fp8_smallm_gemm_probe.rs` already performs
  a same-process M sweep with reused buffers;
- the candidate is activation pack/quantize plus DeepGEMM;
- references are FP8 GEMV on Hopper and dequantize plus BF16 GEMM pre-Hopper;
- the current H20 crossover evidence covers three exact Qwen shapes, not every
  FP8-block-scaled matrix;
- the probe checks non-NaN output but lacks candidate/reference numerical delta.

Add numerical parity, persist only exact measured cells, and leave every other
shape `INSUFFICIENT` or on its existing fallback. Do not migrate DSv4 first.

## Canonical M0 commands

Run from one clean, source-aligned producer after toolchain preflight:

```text
required audit tools: strace, lddtree, readelf, diffoscope, cuobjdump, nm, ar
current pod missing: strace, lddtree, diffoscope
```

```bash
/usr/bin/time -v -o /tmp/<case>.time \
  strace -ff -qq -s 4096 -e trace=process -o /tmp/<case>.proc \
  env CARGO_TARGET_DIR=/tmp/<case>-target \
      TORCH_CUDA_ARCH_LIST=9.0 \
      ARLE_CUDA_KERNEL_CACHE_DIR=/tmp/<case>-cache \
      cargo build --profile release-fast --features cuda,nccl -p arle --bin arle

rg 'execve\(' /tmp/<case>.proc* |
  sed -E 's/.*execve\("([^"]+).*/\1/' | sort | uniq -c
```

Cases: cold/no cache, warm no-op, Rust-only, one native TU, one TileLang source,
ABI/registry, link-only, DeepEP intranode, DeepEP+NVSHMEM, T1, sm70, Blackwell.

Runtime closure:

```bash
readelf -dW <product>
lddtree <product>
LD_DEBUG=libs strace -ff -e trace=openat,execve -o /tmp/runtime <serve-command>
```

Cold DeepGEMM:

```bash
DG_JIT_CACHE_DIR=$(mktemp -d) \
  strace -ff -e trace=process,openat -o /tmp/deepgemm <serve-command>
find "$DG_JIT_CACHE_DIR" -type f -printf '%P %s\n' | sort
```

Reproducibility:

```bash
sha256sum producer{1,2}/{libkernels_cuda.a,libtilelang_kernels_aot.a,arle_deepep_sidecar}
diffoscope producer1 producer2
strings producer*/libkernels_cuda.a | rg '/(workspace|home|tmp|usr/local/cuda)'
```

## Remaining uncertainty

- No CUDA archive was available locally for `nm` or linker-map validation.
- No model ran during this audit; TileLang 8+6+3 reachability is static.
- No cold DeepGEMM cache was traced.
- No source-aligned pod build was run because local and pod commits differ and
  both trees contain unrelated/concurrent work.
- A100/sm80 and V100/sm70 runtime fallback evidence remains external.
- Metal/MLX, HIP, and Vulkan need the same role+liveness inventory.

These are explicit M0 exits, not reasons to delay M0a/M0c work.
