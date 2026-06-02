# DSv4 Local MoE Device Counts

## Context

The target lane is DSv4-Flash, TP8, EAGLE, CUDA graph, 256K/1500, hot GPU
cache, with a reference target around TTFT 0.44s, TPOT 4.85ms, E2E 7.7s, and
196 output tok/s.

The replicated-token TP8/all-reduce lane still routed local MoE through a
host-count path: after the local expert count kernel it copied `local_counts`
to CPU, derived host offsets and total routes, then launched a per-expert host
loop. That path is not CUDA-graph compatible and is structurally unlike the
SGLang-style device-orchestrated MoE path.

## What Worked

Keep local MoE route counts on device for the DeepGEMM backend:

- reuse the existing `ARLE_DSV4_DEEPGEMM_DEVICE_COUNTS=1` gate for the local
  all-reduce MoE path when runtime scratch is present;
- run the local expert exclusive scan on GPU immediately after the count kernel;
- initialize padded route slots to `-1`, pack into fixed route capacity, and
  let `dsv4_scatter_all_route_slots_cuda` skip padded slots;
- call the existing all-expert DeepGEMM device-metadata path instead of copying
  counts to CPU and running `ffn_expert_loop`.

Local checks passed at commit `70e1e6ffc38a2ae31035738577403db0e084abab`:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

Remote build and decode correctness passed on the DSv4 pod:

- remote code: `/data01/build/arle` at
  `70e1e6ffc38a2ae31035738577403db0e084abab`;
- build artifact: `/tmp/dsv4_local_device_counts_20260603_build/build.log`;
- build time: `release-fast` finished in 17.75s and used the prebuilt CUDA fast
  path;
- validation artifact: `/tmp/dsv4_local_device_counts_reach_20260603`;
- `scripts/dsv4_batched_decode_validate.py 18085` exited 0, printed
  `ANSWER_PASS`, and completed c8 with zero HTTP errors;
- normal EOS and forced 32-token decode both returned real text containing
  `406`; forced decode reported 32 completion tokens;
- operator trace proved the new path replaced the old path:
  `ffn_expert_deepgemm_device_counts` 44720 calls,
  `ffn_route_device_pack_setup` 44720 calls,
  `ffn_route_offset_scan_gpu` 44720 calls,
  `ffn_route_count_d2h` 0 calls, and `ffn_expert_loop` 0 calls;
- after cleanup, `nvidia-smi --query-compute-apps` reported no remaining
  compute apps.

This is a graph-compatibility and orchestration fix, not a target performance
result. The validation deliberately used operator trace and
`--disable-cuda-graph`.

## Rule

For DSv4 decode, the fast path must keep route metadata device-resident. A
per-layer D2H count copy followed by a host expert loop is not a performance
baseline; it is a debug/fallback path.
