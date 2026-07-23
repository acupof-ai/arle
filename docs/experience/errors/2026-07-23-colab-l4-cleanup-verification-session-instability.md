# Colab L4 cleanup verification blocked by unstable execution sessions

## Context

Commit `834d6d54e4d1584ee5e153546ac04100800e43bc` needed a cold `sm_89` source
build, Qwen3.5-4B inference, needle gate, CUDA MoE test, and matched throughput
A/B.

A real Google Colab L4 was allocated and identified as NVIDIA L4 (`sm_89`,
23,034 MiB), driver `580.82.07`, CUDA `12.8.93`.

## Root Cause

The first build attempt never reached CUDA/FlashMLA: the repo's
`.cargo/config.toml` forces `rustc-wrapper = "sccache"`, and Colab has no
`sccache`. After overriding the wrapper with `CARGO_BUILD_RUSTC_WRAPPER=''`, a
second build ran for about 2m18s, reached native CUDA/FlashMLA compilation, and
then aborted in `cuda-kernels/build.rs` because Colab's system TileLang imported
an unpatched `apache-tvm-ffi` and hit duplicate `TypeAttr __ffi_repr__`
registration.

The repo's `scripts/pod-tilelang-env.sh` is the intended fix, but Colab's
`/usr/bin/python3` lacks `ensurepip`, so its `python3 -m venv` path fails. The
original session disappeared during that repair. Multiple replacement L4
allocations then failed at the CLI transport layer (`WebSocketConnectionClosed
Exception: socket is already closed`) before the first command, and at least one
allocation reached the kernel but failed on a probe token that was not preserved
across reconnects.

## Fix

No code workaround was added. The next stable session should start from the
exact treatment commit, not the current HEAD:

```bash
cd /content/arle
git fetch --depth 1 origin 834d6d54e4d1584ee5e153546ac04100800e43bc
git checkout --detach 834d6d54e4d1584ee5e153546ac04100800e43bc
```

Then create the repo-local TileLang venv and run the canonical bootstrap:

```bash
POD_TREE=/content/arle bash scripts/pod-tilelang-env.sh
```

`scripts/pod-tilelang-env.sh` now falls back to `virtualenv` when `python3 -m
venv` is unavailable, no longer inherits system site packages, and checks the
pinned `tilelang==0.1.11` before skipping dependency installation. Require
`patched-ok`; this sequence has not yet been observed to complete on Colab, so
treat it as the next unverified attempt until `patched-ok` is printed.

Rebuild with a cold, source-only configuration:

```bash
env -u ARLE_CUDA_KERNELS_PREBUILT_DIR \
CARGO_BUILD_RUSTC_WRAPPER='' \
ARLE_CUDA_KERNEL_CACHE=0 \
TORCH_CUDA_ARCH_LIST=8.9 \
CARGO_TARGET_DIR=/tmp/arle-colab-target \
cargo build --release --features cuda -p arle --bin arle
```

`ARLE_CUDA_KERNELS_PREBUILT_DIR` is explicitly unset so the build cannot link
prebuilt artifacts. `ARLE_TILELANG_REGEN` is intentionally omitted: a clean
checkout with cache disabled already regenerates missing artifacts, and `REGEN`
would write generated files back into the source tree. `CMAKE_CUDA_ARCHITECTURES`
is omitted because `TORCH_CUDA_ARCH_LIST=8.9` is the canonical, effective
setting. A dedicated `CARGO_TARGET_DIR` makes the cold-build claim auditable and
prevents reuse of a stale `target/`.

## Verification plan

This L4 gate covers only the single-rank CUDA path. It does **not** close the
DSv4 multi-rank startup gate from
`docs/experience/wins/2026-07-22-cuda-canonical-tp-runtime-pending-remote.md`;
that still requires a multi-GPU host because `--features cuda` does not compile
the `#[cfg(feature = "nccl")]` TP/NCCL branch.

Run, on the L4, against the treatment binary above:

1. Non-degenerate Qwen3.5-4B chat smoke with coherent output.
2. `scripts/lever_gate.sh` / `scripts/needle_gate.py` with the Qwen3.5-4B profile
   at 115/300/446/2000/8000 tokens, three same-config repeats per rung. Decode
   any miss, timeout, or loop before changing code.
3. The CUDA MoE pointer-table test that exercises all five selectors
   (`qweight_u8`, `scale_f32`, `qscale_fp8`, `qweight_i8`, `qscale_bf16`); the
   existing W4A16 test alone covers only `qweight_i8` and `qscale_bf16`.
4. Throughput with `scripts/bench_throughput.py`. Start single-arm against the
   rolling L4 champion per `docs/bench-and-trace-spec.md` §3.0; escalate to a
   matched A/B only if the delta lands inside the fingerprint drift band. There
   is currently no archived L4 champion row in `docs/baselines.md`, so record the
   baseline commit, binary hash, model revision, flags, and raw JSON/CSV before
   comparing.

## Rule

A Colab allocation is not a verification result. The gate remains
`pending-remote` until the cold build, decoded inference, needle matrix, the
full CUDA MoE selector test, and the throughput gate all complete on the pinned
treatment commit. Never convert environment reachability, partial compilation,
or a single-rank L4 pass into a runtime PASS, and never close the DSv4
multi-rank NCCL gate from a single-L4 result.
