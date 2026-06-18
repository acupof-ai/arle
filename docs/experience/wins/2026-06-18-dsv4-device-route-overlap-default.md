# DSv4 device route cleanup + B1 overlap default

## Context

Follow-up to the DSv4 decode regression investigation on .62. The code had two
bad half-states:

- `ARLE_DSV4_GPU_ROUTER` still implied a DSv4 host-route escape path even though
  the device route had already been licensed and the scratch-pool meaning had
  been removed.
- `ARLE_DSV4_COMM_OVERLAP=1` was the faster B=1 allreduce path, but production
  default still required an env flip.

## What Changed

- Deleted the DSv4 GPU-router env gate, the host D2H route fallback, the host
  route oracle helper, and the host `hash_tid2eid` table.
- Made B=1 allreduce shared-expert/route all-reduce overlap the default path;
  DeepEP keeps the non-overlap ordering.
- Left MTP explicit; it is correct-shape but still slower than no-spec on this
  prompt.

## Verification

Local:

```bash
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo test -p infer-cuda --release --no-default-features --features cuda,no-cuda spec_decode --lib
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo check -p cli --release --no-default-features --features cuda,no-cuda --lib
```

Remote .62 clean tree:

- Bundle SHA256: `b7773441a3d2d097f595962f1fcbba6363565c4a1e37528b7a559fef12de3a61`
- Clean checkout: `/data01/arle-gpu-verify-087df440`, HEAD `087df440`
- Build: `scripts/dsv4_fast_build.sh`, `release-fast`, `features=cuda,nccl`,
  `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`
- Symbol check: `ARLE_DSV4_GPU_ROUTER` and `ARLE_DSV4_COMM_OVERLAP` absent from
  `target/release-fast/arle`; `dsv4_fp8_grouped_swiglu_decode_kernel` and
  `dsv4-mtp` present.
- Remote unit: `cargo test -p infer-cuda --profile release-fast --features cuda,nccl spec_decode --lib`

## Results

All c-sweep runs used `/data01/models/DeepSeek-V4-Flash`, TP=8, H20x8,
`--num-slots 64`, output 128, profiling off. This is not the full SLO shape.

| arm | c | stagger | ok | output tok/s | note |
|---|---:|---:|---:|---:|---|
| `087df440` default no-spec | 1 | 0s | 1/1 | 39.68 | B=1 overlap now default |
| `087df440` default no-spec | 1 | 1s | 1/1 | 39.58 | B=1 overlap now default |
| `087df440` MTP D2 topk=1 | 1 | 0s | 1/1 | 35.23 | `draft_rows=2 verify_rows=3` |
| `087df440` MTP D2 topk=1 | 1 | 1s | 1/1 | 35.21 | `draft_rows=2 verify_rows=3` |
| `087df440` MTP D2 topk=2 | 1 | 0s | 1/1 | 35.34 | `draft_rows=2 verify_rows=3` |
| `087df440` MTP D2 topk=2 | 1 | 1s | 1/1 | 35.36 | `draft_rows=2 verify_rows=3` |

ShareGPT small probe, c=1, 8 prompts, 128 output tokens, warmed after a cold
first no-spec pass. This is a cheap coherent-prompt check, not a full SLO sweep.

| arm | success | wall | output tok/s | TTFT p50 | ITL p50 | E2E p50 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| `087df440` no-spec hot | 8/8 | 26.93s | 38.0 | 207.4ms | 24.7ms | 3372.5ms | warmed control |
| `087df440` MTP D2 topk=2 | 8/8 | 33.93s | 30.2 | 196.6ms | 59.5ms | 4224.5ms | `draft_rows=2 verify_rows=3` |

Immediate A/B anchors from the same .62 investigation:

| arm | output tok/s |
|---|---:|
| `2283d864` clean default no-spec | 35.65 / 35.62 |
| `2283d864` with `ARLE_DSV4_COMM_OVERLAP=1` | 38.83 / 38.69 |
| `2283d864` with `ARLE_DSV4_COMM_OVERLAP=1 ARLE_DSV4_DECODE_COMPRESSOR_BATCH=1` | 39.71 / 39.70 |

## Problems

- Historical 44 tok/s no-spec and 53 tok/s MTP were not reproduced from clean
  bundle builds on current .62 conditions. A clean `3e3e50e0` rebuild measured
  about 32.7 tok/s, so the old number is not a valid direct baseline for this
  diff.
- MTP D2 is still slower than no-spec here. Shape is now correct (`verify_rows=3`
  for D2/T2, topk=1 and topk=2), including on ShareGPT, but `seq_len>1` verify
  does not use the B=1 comm-overlap path. Keep MTP explicit until a new verify
  optimization clears ShareGPT/SLO-shape A/B.

## Rule

DSv4 device routing is the production path, not a runtime knob. B=1 allreduce
comm overlap is the default. Top-k speculation changes candidate acceptance, not
target verify row count.

## Artefacts

- `/data01/arle-gpu-verify-087df440/bench-output/head_csweep_c1_default_087df440.c_sweep.log`
- `/data01/arle-gpu-verify-087df440/bench-output/head_csweep_c1_mtp_d2_default_087df440.c_sweep.log`
- `/data01/arle-gpu-verify-087df440/bench-output/head_csweep_c1_mtp_d2_default_087df440.mtp_tail.txt`
- `/data01/arle-gpu-verify-087df440/bench-output/head_csweep_c1_mtp_d2_topk2_default_087df440.c_sweep.log`
- `/data01/arle-gpu-verify-087df440/bench-output/head_csweep_c1_mtp_d2_topk2_default_087df440.mtp_tail.txt`
- `/data01/arle-gpu-verify-087df440/bench-output/sharegpt8_default_hot_087df440.sharegpt.log`
- `/data01/arle-gpu-verify-087df440/bench-output/sharegpt8_mtp_d2_topk2_087df440.sharegpt.log`
- `/data01/arle-gpu-verify-087df440/bench-output/sharegpt8_mtp_d2_topk2_087df440.mtp_tail.txt`
