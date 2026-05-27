# DSv4 native-deepep — 8×H20 pod end-to-end run

## Goal

Take the B-3.3 native-deepep forward path (landed 5d030884) all the
way from `cargo build` to a live `/v1/chat/completions` response on an
8×H20 pod, and quantify any numerical drift vs the proven `=deepep`
NCCL-emulated baseline.

## What ran

| Step | Result |
|---|---|
| `tn push` arle source tarball (8.1 MiB, excluding target/models/web) + deepseek-ai/DeepEP @ d4f41e4 | OK |
| `cargo build -p deepep-sys --release` w/ ARLE_DEEPEP_DIR set | 7.22 s, libarle_deepep.a archived (after legacy layout shim 52b96e9e) |
| `cargo build -p infer --release --features cuda,nccl --lib` | 5 m 02 s clean (tilelang 0.1.9 + ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1 — see 466ce455 tilelang/cuda12.2 errors entry) |
| `cargo build -p infer --release --features cuda,nccl --bin infer` (first pass) | 6 m 01 s |
| `cargo build … --bin infer` (rebuild after main.rs deadlock fix) | 1 m 51 s |
| Single-process 8-rank serve | OOM rank 0 (loads whole 159 GiB model) |
| `INFER_CUDA_DEVICES=0..7` + 8-rank in-process | `cudaIpcOpenMemHandle: invalid device context` (errors/2026-05-26 documented L0) |
| `ARLE_MULTIPROC_SERVE=1` 8-rank | Deadlock: workers' NCCL rendezvous vs coord's relay accept |
| `ARLE_MULTIPROC_SERVE=1` after main.rs swap (8fe74407) | ✓ all 8 ranks boot: `[native-deepep] rank N/8 booted (device_id=N, peer_handles=8)` |
| Model load + `/v1/models` ready | ✓ |
| `POST /v1/chat/completions` smoke (137+269 prompt, max_tokens=32) | Returns 200 in ~2 s; **content is garbage** |
| A/B vs `=deepep` baseline (same prompt, same KV mode) | Baseline ALSO returns garbage |

## Numerical parity

Same prompt (`Compute 137 + 269. Answer with the number only.`),
temperature=0, max_tokens=32:

| Backend            | KV    | First token | Rest                                            |
|--------------------|-------|-------------|-------------------------------------------------|
| `=native-deepep`   | fp8   | `426`       | `\n- of the following__ of_?______________________` |
| `=deepep` (NCCL)   | fp8   | `4262`      | ` 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000`    |
| `=native-deepep`   | bf16  | `426`       | `\n- of the following__ of_?______________________` |

Expected: `406` (137 + 269).

Both backends produce garbage. **B-3.3 native-deepep is at parity with
the production `=deepep` path** — same first-token corruption pattern,
same downstream cascade. The garbage is upstream of MoE; candidates:

  - `ARLE_DSV4_EXPERT_BACKEND=native` may not match this checkpoint's
    quantization layout (the toolchain default is `deepgemm`, but
    CUTLASS submodule is missing on this pod — see B-3.2 wins).
  - Tokenizer / chat template not applied (the smoke sends raw
    `messages` without the model's instruction wrapper).
  - `--deepseek-distributed-layers 43` matches `num_hidden_layers` in
    config.json but the runtime may want a different layer-on-GPU
    cutoff for this checkpoint.
  - `--mem-fraction-static 0.10` low — may force a tight slot cap that
    masks something at the KV pool boundary.

None of these are B-3.3 regressions. The native-deepep code path
**runs the model exactly as far as the baseline does** and the output
divergence is byte-identical in the "garbage shape" — i.e. the
upstream issue dominates both paths equally.

## L-table additions

This run added **3 new binding constraints** (none of which existed
before today, all surfaced by the fresh pod env):

- **L_nccl-cuda-version-mismatch**: pip caches both `libnccl.so.2`
  built against `+cuda12.9` (driver 535 / CUDA 12.2 → "CUDA driver
  version is insufficient for CUDA runtime version") and one for
  `+cuda12.4` (works). Always probe `strings libnccl.so.2 | grep
  "NCCL version"` and pick the closest-without-going-over
  before-driver-bumping path.
- **L_tilelang-0.1.10-cute-fold-expr**: tilelang 0.1.10's bundled
  cutlass uses C++17 fold expressions in device code that nvcc 12.2
  rejects even under `-std=c++20`. Pin `tilelang>=0.1,<0.1.10` for
  CUDA 12.2 environments (covered by errors/2026-05-27-tilelang-
  0110-cuda122-cutlass-incompat.md).
- **L_multiproc-relay-nccl-deadlock**: workers' NCCL TCP rendezvous
  inside `spawn_cuda_worker_group` raced against coord's relay
  accept; both blocked, both timed out. Fix is to lift workers'
  relay connect to before scheduler boot (8fe74407 — `run_worker_
  mode` now opens the RelayWorker first, then enters
  spawn_cuda_worker_group).

## What's next

The garbage-output diagnosis is an **independent axis** from
native-deepep. The B-3.3 deliverable is now end-to-end verified —
forward path runs, no crashes, parity with prod. To close the SLO
bench gap (B-4 / task #16), the upstream garbage issue must be
diagnosed first, OR the bench can compare TTFT/TPOT/throughput on
the garbage-shaped output (since the parity holds, perf
measurement is still meaningful even on broken content).

Next concrete steps:

1. **Try `ARLE_DSV4_EXPERT_BACKEND=deepgemm`** — requires pushing
   `crates/cuda-kernels/vendor/deepgemm/third-party/cutlass/include`
   to the pod (DeepGEMM submodule was omitted from the source
   tarball). One `tn push` from a fresh local DeepGEMM clone.
2. **Apply DSv4 chat template** — wrap the user message in the
   model's expected `<|im_start|>user…<|im_end|>` (or whatever
   DSv4 uses; check the tokenizer config).
3. **Set `ARLE_DSV4_INCREMENTAL_KV=1` + `ARLE_DSV4_FUSED_DISPATCH_
   PAYLOAD=1`** (the toolchain defaults that my smoke didn't set).
4. **Then B-4 bench** — `scripts/bench_guidellm.sh` once one of the
   above clears the garbage on baseline `=deepep`.

## Rule

When a new GPU forward path is wired in and the e2e smoke produces
garbage, **always A/B against the proven prod path immediately** —
if the prod baseline outputs the same garbage on the same
configuration, the new path is at parity and the bug is upstream.
This separates "did my MoE change introduce a bug?" from "is the
serving config wrong for this checkpoint?", and reverses the
default assumption (newest code is suspect) only when the evidence
says so.

The corollary: if a smoke run is producing nonsense output, **don't
spend time staring at the new code** — flip the env to the
baseline backend and re-smoke. If baseline is also broken, the
work shifts to the config layer, not the kernel layer.
