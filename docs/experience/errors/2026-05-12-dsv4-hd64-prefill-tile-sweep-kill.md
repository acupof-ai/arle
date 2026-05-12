# DSV4 HD64 prefill tile sweep kill

## Context

Goal thread: optimize Qwen3.5 and DSV4 operators, with emphasis on quantized
and KV-adjacent kernels. After the DSV4-mini HD64 decode tile win, the matching
HD64 prefill substrate still had no component benchmark. A benchmark was added
for the existing TileLang symbol:

```bash
NVCC_CCBIN=/usr/bin/g++-14 \
INFER_TILELANG_PYTHON=$PWD/.venv/bin/python \
TORCH_CUDA_ARCH_LIST=8.9 \
cargo bench -p infer --features cuda --bench ops_bench -- \
  ops_cuda/tilelang_prefill_hd64_dsv4mini --quiet
```

Bench shape:

| Param | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER |
| SM | 89 |
| q_len | 2048 |
| batch_size | 1 |
| page_size | 16 |
| num_q_heads | 16 |
| num_kv_heads | 1 |
| head_dim | 64 |
| q_dim | 1024 |
| kv_dim | 64 |
| KV dtype | BF16 |

## Root Cause

The baseline HD64 prefill TileLang parameters are:

| Param | Value |
|---|---:|
| `BLOCK_M` | 64 |
| `BLOCK_N` | 64 |
| `NUM_STAGES` | 2 |
| `NUM_THREADS` | 128 |

Three single-axis candidates were tested, restoring the baseline after each
candidate:

| Candidate | Result | Verdict |
|---|---:|---|
| baseline `BLOCK_M=64`, `BLOCK_N=64` | `170.21-170.30 us`, point `170.26 us` | baseline |
| `BLOCK_M=64`, `BLOCK_N=128` | `182.33-182.42 us`, point `182.37 us` | kill: +7.11% latency |
| `BLOCK_M=64`, `BLOCK_N=32` | `171.94-172.02 us`, point `171.98 us` | kill: +1.01% latency |
| `BLOCK_M=128`, `BLOCK_N=64` | `483.98-484.46 us`, point `484.32 us` | kill: +184.45% latency |
| `BLOCK_M=32`, `BLOCK_N=64` | AOT failure: layout conflict between `p` and `p_bf16` | kill: does not compile |

The `BLOCK_M=32` AOT failure was:

```text
Layout infer conflict between p and p_bf16 in T.Parallel loop
```

This matches the broader TileLang 0.1.9 sensitivity already seen on HD64
decode when reducing padded `BLOCK_M`.

## Fix

No runtime TileLang parameter changed. Keep HD64 prefill at
`BLOCK_M=64`, `BLOCK_N=64`.

The new `ops_cuda/tilelang_prefill_hd64_dsv4mini` bench remains committed as a
baseline harness for future DSV4/HD64 prefill work. Final restored baseline
rerun:

| metric | value |
|---|---:|
| time interval | `170.21-170.28 us` |
| time point | `170.25 us` |
| throughput interval | `25223-25233 Gelem/s` |
| throughput point | `25227 Gelem/s` |

## Rule

Do not transfer the HD64 decode `BLOCK_N=256` intuition to HD64 prefill. For
the DSV4-mini 2048-token prefill substrate on sm_89, the existing
`BLOCK_M=64`, `BLOCK_N=64` is the measured stable point among the tested tile
axes. Any future prefill attempt needs a different mechanism, such as
algorithmic layout changes or a newer TileLang version, not simple M/N tile
scaling.
