# DSv4 batched spec surface cleanup

## Context

DSv4 batched decode and batched MTP are now the B>1 default paths. Keeping
`--dsv4-batched-decode`, `INFER_DSV4_BATCHED_DECODE`, `ARLE_DSV4_BATCHED_MTP`,
`ARLE_DSV4_BATCHED_MTP_DRAFT`, and `ARLE_DSV4_BATCHED_MTP_COMMIT` implied old
per-row or re-forward fallback lanes that the current code should not expose.

## What Worked

- Removed the `--dsv4-batched-decode` CLI/env bridge.
- Made DSv4 decode dispatch simple: B=1 uses single-row decode; B>1 batches.
- Made B>1 greedy MTP always use `spec_step_batched`.
- Removed the unused MTP batch/draft/commit env gates and the dead batched commit
  helper.
- Narrowed `spec_step_batched` and `mtp_forward_level_batched` so spec code no
  longer accepts an arbitrary `positions` vector. MTP draft positions are derived
  from `start_positions[s] + draft_level`; `positions` remains only on normal
  batched decode sampling.
- Removed the batched verify `fold` parameter; it now always persists
  `spec_normed` for commit fold.

## Verification

Local no-CUDA gates:

```text
rustfmt --edition 2024 --check crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/executor.rs crates/cli/src/args.rs crates/cli/src/serve.rs
PASS

git diff --check -- crates/infer-cuda/src/dsv4.rs crates/infer-cuda/src/executor/spec_decode.rs crates/infer-cuda/src/executor.rs crates/cli/src/args.rs crates/cli/src/serve.rs
PASS

CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib
PASS: 6 passed

CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
PASS

CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS

CUDARC_CUDA_VERSION=12090 cargo clippy -p cli --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
PASS
```

## Bench Status

Remote DSv4 CUDA gate on H20 passed from a local git bundle at
`1c41c4a8`:

```text
pod tree: /data01/arle-gpu-verify-1c41c4a8
git rev-parse --short HEAD: 1c41c4a8
binary: target/release-fast/arle
build: scripts/dsv4_fast_build.sh, release-fast, cuda,nccl
result: PASS in 1m02s

strings target/release-fast/arle | grep removed knobs
result: PASS, old DSv4 batched env/CLI knobs absent

cargo test -p infer-cuda --profile release-fast --features cuda,nccl spec_decode --lib
result: PASS, 6 passed
```

Serve smoke:

```text
model: /data01/models/DeepSeek-V4-Flash
GPUs: 8x NVIDIA H20
serve flags:
  --backend cuda
  --num-slots 16
  --spec-type mtp
  --mtp-draft-tokens 2
  --mtp-draft-topk 2
env:
  ARLE_MULTIPROC_SERVE=1
  INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7
  ARLE_DSV4_MOE_BACKEND=allreduce
  ARLE_DSV4_EXPERT_BACKEND=deepgemm
result: PASS, /v1/models ready after 30s
```

B>1 MTP shape gate:

```text
4 concurrent chat completions, max_tokens=32
result: PASS, 4/4 completed, 0 errored

server.log:
  dsv4-mtp-batched lines: 754
  verify_rows=3 lines: 1040
  verify_rows=7 lines: 0

sample:
  [dsv4-mtp-batched] slot=3 depth=2 topk=2 draft_rows=2 verify_rows=3
```

Short GuideLLM sanity bench:

```text
command:
  GUIDELLM_OUTPUTS="json csv" ./scripts/bench_guidellm.sh \
    dsv4-mtp-d2t2-topk2-1c41c4a8-smoke \
    --target http://127.0.0.1:19041 \
    --model DeepSeek-V4-Flash \
    --processor /data01/models/Qwen3-0.6B \
    --concurrencies 1,4 \
    --data prompt_tokens=128,prompt_tokens_stdev=1,prompt_tokens_min=128,prompt_tokens_max=128,output_tokens=64,output_tokens_stdev=1,output_tokens_min=64,output_tokens_max=64 \
    --max-seconds 30 \
    --warmup 1

artifact:
  /data01/arle-gpu-verify-1c41c4a8/bench-output/2026-06-18-dsv4-mtp-d2t2-topk2-1c41c4a8-smoke-run4

headline:
  conc1: successful=6, incomplete=0, errored=0, out=29.64 tok/s, total=89.54 tok/s, TTFT p50=141.8ms, ITL p50=30.91ms
  conc4: successful=17, incomplete=3, errored=0, out=33.99 tok/s, total=102.39 tok/s, TTFT p50=495.1ms, ITL p50=98.93ms
```

The conc4 incomplete requests are the expected fixed-window cutoff from
`--max-seconds 30`; there were no errored requests. The DSv4 serve session was
stopped after validation and `nvidia-smi` returned all H20s to 0 MiB used.

## Rule

Once a fallback is deleted, delete the public knob and function parameter that
suggests it still exists.
