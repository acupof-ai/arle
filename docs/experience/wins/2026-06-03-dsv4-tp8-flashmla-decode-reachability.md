# DSv4 TP8 FlashMLA Decode Reachability

## Goal

Move ARLE toward the matched target workload:
DSv4-Flash, TP8, EAGLE, 256K/1500, hot GPU cache, with the SGLang reference
lane at about TTFT 0.44s, TPOT 4.85ms, E2E 7.7s, and 196 output tok/s.

## Hypothesis

ARLE's FlashMLA decode path was structurally unreachable at TP8 because the
MODEL1 decode shape gate checked `local_heads`, which is 8 for a 64-head model
under TP8. SGLang-style FlashMLA sees global `h_q=64` by all-gathering Q across
TP ranks, then slices the rank-local output slab back after the kernel.

## Params

Code tranche only:

- batch HCA FlashMLA decode computes `h_q = local_heads * tp_world`;
- TP>1 all-gathers `q_prepared`, repacks rank-major Q into
  `[B, h_global, d]`, runs FlashMLA at global `h_q`, and slices back to local
  heads;
- single-token CSA/HCA FlashMLA decode uses the same TP-aware all-gather,
  repack, and output-slice pattern;
- output projection remains the validated per-row path.

## Env

Local checks on macOS with no CUDA runtime execution:

- `cargo fmt --check`
- `git diff --check`
- `cargo check -p infer --no-default-features --features no-cuda`
- `CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,nccl,no-cuda`

## Results

Local checks passed.

Remote build and reachability validation passed on the DSv4 pod at commit
`fd05a7177a2a593749ec2a47e7ba7bcfd6953818`:

- build artifact:
  `/tmp/dsv4_tp8_flashmla_decode_20260603_build/build.log`;
- validation artifact:
  `/tmp/dsv4_tp8_flashmla_decode_reach_20260603`;
- `scripts/dsv4_batched_decode_validate.py 18085` exited 0, printed
  `ANSWER_PASS`, and completed c8 with zero HTTP errors;
- c1/c4 byte parity remained diagnostic-only false, but every c1/c4/c8 output
  contained the expected `406` answer token;
- operator trace proved the new paths executed:
  `attn_flashmla_decode` 16408 calls and
  `attn_hca_batch_flashmla_decode` 2240 calls;
- after cleanup, `nvidia-smi --query-compute-apps` reported no remaining
  compute apps.

Target workload TPOT is still pending. The reachability run deliberately set
`ARLE_DSV4_OPERATOR_TRACE=1`, `ARLE_DSV4_OPERATOR_TRACE_EVENTS=1`, and
`--disable-cuda-graph`, so its timing is not performance evidence.

## Problems

This tranche intentionally allocates TP all-gather/repack/full-output scratch in
the decode body. That is acceptable for reachability and correctness, but it is
not the final high-performance CUDA-graph-compatible implementation.

The single-token `attn_flashmla_decode` trace includes per-layer synchronizing
operator-trace overhead, so its early ~3-4 ms samples should not be compared to
SGLang TPOT. The phase's presence, not the measured trace latency, is the
evidence from this run.

## Learnings

Do not gate FlashMLA decode by the rank-local head count. The correct contract is
the head count passed to FlashMLA after any TP request-ownership transform.
