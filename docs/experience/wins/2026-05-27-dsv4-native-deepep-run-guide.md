# DSv4 native-deepep — pod-side run guide

## Goal

After B-3.3 ([`./2026-05-27-native-deepep-forward-B33-wired.md`](./2026-05-27-native-deepep-forward-B33-wired.md))
landed the forward path and B-3.3 perf round (commits 09bdb21a +
8837c7cb) trimmed ~2.3 ms/request from the routed FFN, the missing
piece for "completely running DSv4 with native DeepEP" is the **build +
launch glue**. This entry documents the end-to-end run sequence on the
8 × H20 pod.

## Pre-reqs (verified)

| Component | Source | Notes |
|---|---|---|
| ARLE main | `git@github.com:cklxx/arle.git`, branch `main` @ `25d70f54` | After commit `25d70f54 build(cuda): dsv4_toolchain.sh native-deepep build flow`. |
| DSv4 weights | `/sgl-workspace/models/DeepSeek-V4-Flash` (pod convention) | Path passed to `--model-path`. |
| DeepEP source | `/sgl-workspace/DeepEP` (deepseek-ai/DeepEP) | Source tree with `csrc/kernels/api.cuh`. Required for native-deepep. |
| DeepGEMM | `crates/cuda-kernels/vendor/deepgemm` (vendored) | Default; required for `--expert-backend deepgemm`. |
| CUDA | `/usr/local/cuda` SM 9.0 (H20) | `CUDA_HOME` exported by toolchain. |
| NCCL | bundled with CUDA or in `LD_LIBRARY_PATH` | Detected by `detect_nccl`. |

## Build (release, native-deepep linked)

```bash
cd /workspace/arle
./scripts/dsv4_toolchain.sh build \
    --moe-backend native-deepep \
    --deepep-dir /sgl-workspace/DeepEP
```

Validates `CUDA_HOME` / NCCL / DeepGEMM / DeepEP source layout, then:

```bash
TORCH_CUDA_ARCH_LIST=9.0 ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 \
ARLE_DEEPEP_DIR=/sgl-workspace/DeepEP \
cargo build --release -p infer --features cuda,nccl --bin infer
```

Build artifact: `target/release/infer`. The deepep-sys static archive
(`libarle_deepep.a` ~3 MB) is linked into the binary. If `ARLE_DEEPEP_DIR`
is missing, `build.rs` falls back to **deepep_stub** mode — the build
still succeeds, but `ARLE_DSV4_MOE_BACKEND=native-deepep` will bail at
boot time in `NativeDeepEp::boot` with a clear error pointing at the
flag. The toolchain validates this up front so the build never
silently produces a stub binary when native was requested.

## Smoke (single-prompt greedy completion)

```bash
./scripts/dsv4_toolchain.sh smoke \
    --moe-backend native-deepep \
    --deepep-dir /sgl-workspace/DeepEP \
    --model-path /sgl-workspace/models/DeepSeek-V4-Flash \
    --prompt "Compute 137 + 269. Answer with the number only." \
    --max-tokens 32
```

The script launches `infer` with the standard 8 × H20 config (43
distributed layers, fp8 KV, num_slots=1), waits up to 600 s for
`/v1/models` to respond, then sends a single OpenAI-API completion
request and writes:

  - `docs/trace-artifacts/dsv4-toolchain-local/server.log`
  - `docs/trace-artifacts/dsv4-toolchain-local/smoke-response.json`
  - `docs/trace-artifacts/dsv4-toolchain-local/models.json`

## Expected boot-path log lines

At rank 0 with `RUST_LOG=info` (the toolchain default), the
native-deepep boot path produces:

```
[native-deepep] rank 0/8 booted (device_id=0, peer_handles=8)
...
[native-deepep] rank 7/8 booted (device_id=7, peer_handles=8)
```

If you see `deepep-sys was built in stub mode` instead, the build
did not link the native archive — re-run the build with
`--deepep-dir` set. The toolchain's `env-check` subcommand prints
the resolved `ARLE_DEEPEP_DIR` value or `(unset — not native-deepep)`
so you can verify which path is active without re-building.

## Numerical parity check

The cleanest parity smoke is to run the same prompt twice with
different MoE backends and diff the response:

```bash
# A: NCCL DeepEP-style fallback (production today)
./scripts/dsv4_toolchain.sh smoke --moe-backend deepep \
    --artifact-root /tmp/dsv4-A-deepep ...

# B: Native DeepEP intranode
./scripts/dsv4_toolchain.sh smoke --moe-backend native-deepep \
    --artifact-root /tmp/dsv4-B-native-deepep \
    --deepep-dir /sgl-workspace/DeepEP ...

diff /tmp/dsv4-A-deepep/smoke-response.json \
     /tmp/dsv4-B-native-deepep/smoke-response.json
```

Greedy decode at temperature 0 with the same prompt + seed should
produce byte-identical `text` fields. If they diverge, the most
likely root causes (per B-3.3 wins entry's audit):

  1. Weight double-application — the per-expert FFN multiplies by
     routing weight (via `dsv4_scatter_packed_expert_cuda`), and
     `Buffer.combine` may ALSO apply `d_topk_weights`. Verify
     intranode::combine's weight semantics against DeepEP source
     (`/sgl-workspace/DeepEP/csrc/kernels/intranode.cu`).
  2. `num_recv=0` edge case — `expert_out` left uninit, combine
     reads garbage. Pre-zeroing `combined_x` is the safety net.
  3. `is_token_in_rank` boolean — the C ABI uses `bool*` but our
     scratch allocates `u8`. C++ `bool` is platform-defined size;
     verify it's 1 byte on H20 (CUDA convention).

## SLO bench (B-4, follow-up)

Once parity passes:

```bash
./scripts/bench_guidellm.sh dsv4-native-deepep \
    --model /sgl-workspace/models/DeepSeek-V4-Flash \
    --moe-backend native-deepep \
    --expert-backend deepgemm
```

Gate per multiproc-serve pivot doc: TTFT +5%, TPOT +5%, p99 not
regressed >3% vs `dsv4-deepep` (NCCL fallback). Cross-link the
guidellm wins entry to this run guide for traceability.

## Rule

A new GPU library wired through a runtime needs **three artifacts** to
count as "completely running":

  1. The forward path code (B-3.3 chain — landed).
  2. A build-flow helper that validates the library's source +
     compile-time prereqs and exports the right env vars (this
     commit — `dsv4_toolchain.sh` native-deepep validation).
  3. A run-flow helper that launches the server + sends a smoke
     request + captures artifacts (existing `smoke` subcommand —
     no native-deepep-specific work needed, the existing flow
     already drives the path).

Without (2), users get a stub binary at runtime with a runtime error.
Without (3), users have a binary but no canonical "did it work?"
verification. Both are required for "完全跑起来".

The dsv4_toolchain.sh changes are the (2) artifact — they don't
change the runtime behavior, just make the build step impossible to
silently get wrong.
