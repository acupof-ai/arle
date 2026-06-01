# DSv4 CSA select top-k-cover fast path

## SLO-shape probed?  N

This was a targeted p2047 decode probe, not the campaign SLO shape. It cannot
license the "ARLE > SGLang by 20%" claim. It only licenses keeping the
`available <= topk` CSA selector fast path if correctness stays clean.

## Goal

Remove DSv4 compressed-sparse-attention selector work when `topk` already
covers every causally available compressed block.

## Hypothesis

For DSv4 Flash config (`compress_ratio=4`, `index_topk=512`), prompts around
2047 tokens produce decode steps where `available=floor(pos/4) <= 512`. In
that region the old selector scored and sorted all blocks even though the
selected set must be the full causal range.

## Environment

- Backend: CUDA, 8x H20 pod via `~/bin/pod`
- Repo: local and remote both at `d89fc23016f82dadd61cd1df1519b913e934644b`
- Dirty runtime diff: `crates/cuda-kernels/csrc/misc/dsv4_attention.cu`
- Build: `CARGO_TARGET_DIR=/sgl-workspace/arle/target-pod cargo build --release -p infer --features cuda,nccl --bin infer`
- Model: `/data01/models/DeepSeek-V4-Flash`
- Launch: `target-pod/release/infer --num-slots 1 --max-seq-len 4096 --mem-fraction-static 0.85 --kv-cache-dtype fp8 --deepseek-distributed-layers 43`
- Env: `INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7`, `ARLE_DSV4_MOE_BACKEND=allreduce`, `ARLE_DSV4_INCREMENTAL_KV=1`, DeepGEMM roots under `/root/DeepGEMM`

## Results

No-trace warm pair, same binary:

| Probe | prompt tokens | output tokens | TTFT ms | total ms | TPOT after first ms | Output |
|---|---:|---:|---:|---:|---:|---|
| p2056/o8 warm, fast path mostly not hit | 2056 | 8 | 3795.4 | 4672.6 | 125.3 | normal text, length finish |
| p2047/o8 warm, fast path hit | 2047 | 8 | 3706.4 | 4344.0 | 91.1 | normal text, length finish |
| p2047/o32 warm | 2047 | 32 | 3711.5 | 7342.8 | 117.1 | normal text, length finish |

Trace probe (`ARLE_DSV4_TRACE_LAYER=1`, p2047/o8):

| Metric | Value |
|---|---:|
| `attn_csa_select_kernel` decode samples | 1176 |
| total traced selector ms | 847.495 |
| avg ms | 0.721 |
| p50 ms | 0.014 |
| p90 ms | 2.491 |
| samples `<=0.05 ms` | 840 |
| samples `>2.0 ms` | 336 |

Interpretation: the fast path fires for the early decode steps where top-k
covers the full causal compressed range; later tokens still take the old
score/sort path. This matches the no-trace behavior: p2047/o8 improves, but
p2047/o32 is still dominated by later non-fast-path tokens.

## Problems

- This does not move the long-output target enough. Warm p2047/o32 is still
  117.1 ms TPOT after first token. The older 60-64 ms campaign target is now
  deprecated for SGLang comparison; with the user-supplied 18 ms SGLang
  reference, a raw-target `>20%` win would require `<=14.4 ms/token`.
- The paired control is same-binary/near-shape (`p2056`, fast path not hit),
  not a rebuilt same-shape pre-patch binary. Accepting this entry as a local
  optimization is reasonable, but it is not enough for a default/SLO claim.
- The probe harness exits with code 143 after terminating the child server,
  but all response JSON, request traces, and summaries were written before
  cleanup. No server process remained afterwards.

## Learnings

Top-k-cover is a valid selector-elision boundary, but it is a narrow boundary.
For the current replicated-token fallback route, the next useful CSA work is the
`available > topk` region: scoring and partial sort still cost about 2.49 ms per
traced sample there. For the SGLang-path campaign, this is subordinate to the
path contract and the MoE/EP priority reset in
`docs/plans/2026-06-01-dsv4-sglang-path-alignment.md`.

## Artifacts

- No-trace fast-path-miss control: `/sgl-workspace/bench-artifacts/dsv4-csa-fastpath-d89fc230/no-trace-warm2/summary.json`
- No-trace p2047 warm pair: `/sgl-workspace/bench-artifacts/dsv4-csa-fastpath-d89fc230/no-trace-p2047-warm2/summary.json`
- Trace p2047/o8: `/sgl-workspace/bench-artifacts/dsv4-csa-fastpath-d89fc230/trace-p2047-o8/summary.json`
