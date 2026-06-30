# DSv4 compact FP8 MoE decode lane for B>1

## Context

TP4 DSv4 page-attn validation left the decode step with MoE still around 28ms at
batched decode. Phase logs on the pod showed the B>1 allreduce lane still fell
back to the padded/contiguous DeepGEMM materialization path, even though the
compact FP8 decode kernels already handle real routed rows with `max_count`
chunks. The only blocker was a too-narrow route-count gate: the compact lane was
limited to B=1 (`total_routes <= 8`).

## What Worked

`DSV4_DECODE_GEMV_MAX_ROUTES` now reuses `DSV4_DECODE_CONTIG_MAX_ROUTES` (128
routes). That keeps B<=~16 decode rows on the compact FP8 lane and leaves larger
prefill shapes on tensor-core DeepGEMM. No new runtime knob; the old padded
fallback remains for larger route counts or table-build failure.

## Verification

Local gates:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
```

Result: passed. Existing warning in `crates/cli/src/eli.rs` (`Path` unused) remains
unrelated.

Pod build:

```bash
CARGO_TARGET_DIR=/host/arle-build/target-nccl-dsv4 \
CARGO_NET_OFFLINE=true CUDARC_CUDA_VERSION=12090 CUDA_HOME=/usr/local/cuda \
RUSTC_WRAPPER= CARGO_BUILD_RUSTC_WRAPPER= \
cargo build --release --no-default-features --features cli,cuda,nccl --bin arle
```

Result: passed in 51.87s on container `sglang-test`, commit `c0cc44c2`.

Correctness smoke after rebuild:

- 3 concurrent short prompts returned `hello` for all three, token ids `[33310, 1]`.
- Longer c=4 probe remained coherent for ordinary list prompts. The synthetic
  repeated-context c=4 prompt still emits token `0`; this was already attributed to
  prompt shape, not the MoE change, because the same prompt emits all-zero tokens
  in single-request control.

## Results

Phase profile, TP4, `ARLE_DSV4_DECODE_PHASE_TIME=1`, n=2 decode rows:

| metric | before | after | delta |
|---|---:|---:|---:|
| MoE phase | ~28.5ms | ~20.8-21.1ms | ~-26% |
| sw_attn phase | ~31.7-33.9ms in the old n=2 profiled run | ~20.8-22.3ms | improved in this probe |

Representative after lines:

```text
[decode-phase] n=2 sw_attn=20.8ms (prep=10.9 [proj=3.0 compidx=4.7 ...] fwd=2.2 finish=6.3) moe=20.9ms
[decode-phase] n=2 sw_attn=21.0ms (prep=11.1 [proj=3.0 compidx=5.0 ...] fwd=2.2 finish=6.4) moe=20.9ms
```

Same HTTP shape as the earlier TP4 sample (4 x 1456 prompt tokens + 32 decode
count tokens):

| metric | before | after | delta |
|---|---:|---:|---:|
| wall | 9.836s | 9.093s | -7.6% |
| output tok/s | 13.01 | 14.08 | +8.2% |
| total tok/s | 605.1 | 654.5 | +8.2% |

## Problems

- The phase logger is sync-heavy and should not be used as a throughput number.
  The HTTP table above is the comparable wall-clock sample.
- The repeated-context synthetic prompt still generates token 0. Token ids are now
  observable via `return_token_ids=true`; this prompt is not a correctness gate.
- `prefix_cache.hit_rate` can exceed 1.0 after resident reuse because hits are not
  strictly lookup-scoped; that metric needs a small semantic cleanup.

## Rule

Do not leave a proven compact decode kernel artificially B=1 when its kernel ABI
already accepts `max_count` for B>1. For decode-band route counts, avoid padded
DeepGEMM materialization unless the compact kernel itself is unsupported.
