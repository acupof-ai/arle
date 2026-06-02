# DSv4 Compressor Start-Pos Pointer

## Context

The target lane is DSv4-Flash, TP8, EAGLE, CUDA graph, 256K/1500, hot GPU
cache, with a reference target around TTFT 0.44s, TPOT 4.85ms, E2E 7.7s, and
196 output tok/s.

Full decode graph replay cannot safely capture the DSv4 HCA compressor while
`pending_len` and `compressed_rows` are passed as host-by-value kernel
arguments. On replay, those captured constants would stay at the capture-step
values while the next decode token needs a new compressor offset.

## What Worked

Add a decode-oriented compressor ABI that reads `start_pos` from device memory:

- `dsv4_compressor_update_cuda` remains unchanged for existing prefill/debug
  callers;
- `dsv4_compressor_update_start_pos_ptr_cuda` derives `pending_len`,
  `compressed_base`, and `has_prev_overlap` inside the CUDA kernel from the
  current device `start_pos`;
- the batch HCA FlashMLA decode core now passes each row's
  `start_pos_gpu[row]` pointer to the compressor update;
- Rust validates that host metadata is still consistent with the same
  `start_pos` formula, so graph-readiness mistakes fail before corrupting KV.

Local checks passed at commit `01f3dd15638d3cd71aa78b17dee2ff499ee14a72`:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

Remote build and decode correctness passed on the DSv4 pod:

- remote code: `/data01/build/arle` at
  `01f3dd15638d3cd71aa78b17dee2ff499ee14a72`;
- build artifact: `/tmp/dsv4_compressor_startpos_20260603_build/build.log`;
- build time: `release-fast` finished in 7m03s, rebuilt CUDA because the
  `crates/cuda-kernels` manifest changed, then harvested fresh
  `/data01/build/arle/target/dsv4-cuda-kernels-prebuilt` artifacts;
- binary symbol check found `dsv4_compressor_update_start_pos_ptr_cuda`;
- validation artifact: `/tmp/dsv4_compressor_startpos_reach_20260603`;
- `scripts/dsv4_batched_decode_validate.py 18085` exited 0, printed
  `ANSWER_PASS`, and completed c8 with zero HTTP errors;
- normal EOS and forced 32-token decode both returned real text containing
  `406`; forced decode reported 32 completion tokens;
- operator trace proved the updated HCA batch path executed:
  `attn_hca_batch_cache_pack` 2240 calls and
  `attn_hca_batch_flashmla_decode` 2240 calls;
- local MoE stayed on the previous device-count path:
  `ffn_expert_deepgemm_device_counts` 44720 calls,
  `ffn_route_count_d2h` 0 calls, and `ffn_expert_loop` 0 calls;
- after cleanup, `nvidia-smi --query-compute-apps` reported no remaining
  compute apps.

This is a graph-safety prerequisite, not a target performance result. The
validation deliberately used operator trace and `--disable-cuda-graph`.

## Rule

CUDA graph replay cannot depend on decode-step counters captured by value.
For DSv4 HCA decode, derive compressor offsets from device-visible
`start_pos` or keep the path eager.
