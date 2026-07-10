# DeepSeek V4 8xH20 Startup Gate And RDMA Smoke

## Goal

Validate the DeepSeek V4 multi-GPU serving path after adding the runtime
notification gate for CUDA startup warmup.

## Hypothesis

The failure was not a generic DeepSeek V4 model-load issue. Distributed
schedulers were entering startup warmup before every rank was ready, and the
DeepEP dispatch path was still unsafe after that race was removed.

## Params

- Build: `cargo build --release -p infer --features cuda,nccl --bin infer`
- Serving: 8 CUDA workers, `max_seq_len=1024`, FP8 KV cache, `num_slots=1`
- Model path: local checkpoint path intentionally not recorded
- Validation request: one non-streaming chat request, `max_tokens=2`,
  temperature `0`
- Stable MoE transport: local routed experts plus EP all-reduce
- Unsafe transport: DeepEP dispatch kept behind `ARLE_DSV4_MOE_BACKEND=deepep_unsafe`

## Env

- GPU: 8x NVIDIA H20, 97,871 MiB each
- Driver: 535.161.08
- CUDA toolkit: 12.2
- NCCL: 2.21.5+cuda12.4
- RDMA: four 400G NDR ports active and used by NCCL/IB
- GDR: `nvidia_peermem` loaded; CUDA P2P reports OK for every GPU pair;
  GDRCopy userland/device support was absent, so GDRCopy data-plane bandwidth
  was not verified

## Results

| Check | Result |
|---|---|
| Local `runtime_notify` unit tests | pass, 3/3 |
| Remote release CUDA+NCCL build | pass |
| NCCL smoke | pass over IB |
| Startup warmup gate | pass; all 8 schedulers waited before warmup release |
| DeepSeek synthetic decode warmup | disabled by model capability on multi-rank |
| 1-layer 8-GPU HTTP request | pass, response `42` |
| 43-layer 8-GPU HTTP request | pass, response `42` |

43-layer request trace:

| metric | value |
|---|---:|
| prompt tokens | 15 |
| completion tokens | 1 |
| total tokens | 16 |
| TTFT | 398.739 ms |
| total latency | 490.968 ms |
| finish reason | stop |
| request error | none |

NCCL smoke evidence: NCCL selected the IB network and completed the smoke test.

## Problems

- DeepEP variable-count prefill dispatch failed with CUDA illegal address during
  local-count receive.
- After prefill fallback, DeepEP padded decode also failed with CUDA illegal
  address during payload unpack/count receive.
- Therefore `ARLE_DSV4_MOE_BACKEND=deepep` now logs a fallback notice and uses
  the stable all-reduce path. The unsafe dispatch path requires the explicit
  `deepep_unsafe` value.
- GDRCopy data-plane validation remains deferred because the userland/device
  pieces were not installed on the validation machine.

## Learnings

- Distributed startup warmup needs a rank-wide notification gate; per-rank
  readiness is not enough when warmup can enter collectives.
- Model warmup capability must be model/config-specific. DeepSeek V4 can keep
  single-rank decode warmup, but multi-rank synthetic decode warmup is unsafe.
- DeepEP dispatch needs its own controlled repair cycle. Multi-GPU DeepSeek V4
  support is stable through NCCL all-reduce today; unsafe DeepEP should not be
  reachable by the ordinary `deepep` backend name.
