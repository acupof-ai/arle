# DSv4 FlashMLA default-on link fix, but SLO target still killed at 8K

## SLO-shape probed? -- Y (8xH20 TP=8, 8K prompt ladder step, FlashMLA prefill/decode default-on)

## Context

Goal: make FlashMLA the default DSv4 attention path, keep the SLO contract
(input 32K / output 1.5K / c=8, TTFT <= 5000 ms, TPOT <= 30 ms), and beat
SGLang by 20%+ under that contract.

The first remote default-on smoke did not reach performance measurement at all:

| Probe | Result |
| --- | --- |
| `/sgl-workspace/bench-artifacts/dsv4-analysis-20260531-short-default` | Server error response: `DeepSeek V4 FlashMLA prefill failed: DriverError(CUDA_ERROR_NOT_SUPPORTED, "operation not supported")` |
| `/sgl-workspace/bench-artifacts/dsv4-analysis-20260531-short-no-fm` | HTTP 200, output `4`, usage `17 + 1` tokens |

That isolated the immediate failure to the FlashMLA path, not tokenizer/model
loading or the non-FlashMLA serving stack.

## Root Cause

`crates/cuda-kernels/build.rs` recursively collects every CUDA source under
`csrc/`, so `csrc/attention/arle_flashmla_decode_stubs.cu` was compiled even
when `vendor/flashmla` was present and FlashMLA was enabled.

The enabled build then also added the real FlashMLA shim and phase1 objects.
The static archive could therefore contain both the fallback stub and the real
implementation for:

- `arle_flashmla_sm90_sparse_prefill_fwd`
- `arle_flashmla_sm90_sparse_decode_fwd`

Because `ar rcs` updates an existing archive, simply removing the stub from the
source list is not enough: a stale stub member can survive in
`libkernels_cuda.a`. Link order can satisfy the FFI symbol from the stub first,
which turns default-on FlashMLA into a runtime `cudaErrorNotSupported`.

## Fix

The build script now:

- always removes `arle_flashmla_decode_stubs.cu` from the recursive CUDA source
  list first;
- adds the stub back only when FlashMLA is disabled;
- removes stale `libkernels_cuda.a` before rebuilding the archive, so old object
  members cannot survive across feature flips.

Remote verification after the fix:

```text
LIB=target/release/build/cuda-kernels-0cb7f5d8be65b353/out/libkernels_cuda.a
arle_flashmla_csa_prep_cuda.o
arle_flashmla_decode_shim_cuda.o
arle_flashmla_shim_cuda.o
phase1_k512_cuda.o
phase1_k512_topklen_cuda.o
phase1_k576_cuda.o
phase1_k576_topklen_cuda.o
0000000000000270 T arle_flashmla_sm90_sparse_decode_fwd
0000000000000010 T arle_flashmla_sm90_sparse_prefill_fwd
```

No `arle_flashmla_decode_stubs_cuda.o` remained in the active archive. The
binary also contained NCCL symbols (`ncclAllGather`, `ncclAllReduce`) after the
remote `--features cuda,nccl` build.

Short default-on smoke after the fix:

| Probe | Result |
| --- | --- |
| `/sgl-workspace/bench-artifacts/dsv4-analysis-20260531-fixed-short-fm` | HTTP 200, elapsed `1199.5 ms`, output `4`, usage `17 + 1` tokens |

This proves the default-on FlashMLA path links and runs real FlashMLA shims. It
does not license a performance/default claim.

## SLO Evidence

SLO-ish server envelope:

```text
Scheduling envelope (resolved | SGLang-equiv): max_num_batched_tokens=16384 | 16384,
chunked_prefill_size=16384 | 16384, max_prefill_tokens=16384 | 16384,
mem_fraction_static=0.80 | 0.85, max_slots=8 | (n/a - SGLang has no fixed cap)
TokenKVPool budget capped from 57.517 GB to 0.132 GB for DeepSeek-V4 explicit
max_seq_len=49152 across 8 slot(s); retaining scratch headroom for long-prefill kernels
```

Ladder artifact: `/sgl-workspace/bench-artifacts/dsv4-analysis-20260531-arle-slo-ladder`.

| Prompt tokens | Completion tokens | Client elapsed | Request trace TTFT | Verdict |
| --- | ---: | ---: | ---: | --- |
| 8192 | 8 | `116377.7 ms` | `115009.982 ms` | KILL: 23.0x over 5s TTFT target |
| 16384 | n/a | contaminated by manual stop | n/a | not used |
| 32768 | n/a | contaminated by manual stop | n/a | not used |

The valid 8K row is already enough to kill the requested default/performance
claim. The commissioned SLO requires 32K input, c=8, and TTFT <= 5000 ms; a
single 8K request at 115s TTFT cannot be made default under the SLO bar.

The local SGLang checkout on the same pod did not provide a usable control row
for this run:

```text
ValueError: The checkpoint you are trying to load has model type `deepseek_v4`
but Transformers does not recognize this architecture.
```

Artifact: `/sgl-workspace/bench-artifacts/dsv4-analysis-20260531-sglang-8k`.
This is recorded only as a baseline-harness failure. It is not used as ARLE
performance evidence.

## Rule

Default-on kernel integration needs two independent gates:

1. symbol/object evidence that the binary is calling the real implementation,
   not a fallback stub or stale archive member;
2. SLO-shape wall-clock evidence before any performance/default claim.

A short prompt smoke can prove reachability. It cannot prove DSv4 SLO safety,
and it cannot support "20% faster than SGLang" when the runtime already misses
the SLO by 23x at the 8K ladder step.
