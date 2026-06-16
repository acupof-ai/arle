# Qwen3.5 Batched Decode Full-Attention Launch Collapse

## Context

Qwen3.5/Qwen3.6 batched decode still used the old per-row full-attention loop:
for each full-attention layer and decode batch of B rows, the model launched
prep + attention + gate once per row. The batched CUDA kernel already existed
(`fused_gqa_attention_decode_batched`, grid.z = batch) but was not wired into
`full_attention_batch_rows`.

The 35B model load is currently confounded by issue #101's single-threaded slow
startup, so this tranche used `Qwen3.5-0.8B` as the fast-load surrogate. It is
the same hybrid Qwen3.5 full-attention path (`head_dim=256`, full-attn every 4
layers), so it validates the wiring and batch-launch behavior without waiting
on 35B startup.

## What Worked

- Replaced the per-row full-attention loop in `qwen35.rs` with one batched
  `fused_gqa_attention_decode_batched` launch plus one reduce and one batched
  gate per full-attention layer.
- Staged per-row `positions` and `seq_lens` from scheduler `DecodeRow.kv_seq_len`
  into device `[B]` arrays; slot `seq_len` is now an invariant check, not the
  source of truth for the batched kernel.
- Added per-full-layer K/V cache pointer tables, staged with the existing
  row-to-slot pointer table path.
- Fixed the batched kernel to match Qwen3.5/Qwen3.6 full-attn layout:
  `q_proj` is `[B, num_qheads * 2 * head_dim]` (Q + gate), q/k RMSNorm uses
  `1 + weight`, and RoPE is partial (`rotary_dim`, 64 on Qwen3.6).

## Results

Environment:

- Host: H20 pod, `CUDARC_CUDA_VERSION=12090`
- Build: `/data01/arle-qwenfp8-smoke`, `cargo build --release --features cuda --bin arle`
- Model: `/data01/models/Qwen3.5-0.8B`
- Serve: `--num-slots 8 --total-pages 128 --page-size 16 --max-prompt-tokens 1024 --max-total-tokens 1152`
- A/B: same binary; before uses `ARLE_QWEN35_BATCHED_DECODE_ATTENTION=0`

Correctness:

| Arm | c=1 BLUE-73-MANGO | c=8 BLUE-73-MANGO |
| --- | ---: | ---: |
| after batched | PASS | 8/8 PASS |
| before per-row | not rerun | 8/8 PASS |

Throughput probe:

| Arm | c | Completion tokens | Wall s | tok/s | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| before per-row | 8 | 768 | 0.9846 | 780.01 | baseline |
| after batched | 8 | 768 | 0.6234 | 1231.88 | +57.9% |

Launch shape:

| Path | Per full-attn layer attention core launches at B=8 |
| --- | ---: |
| before | 8x `nonpaged_prefill_attention_devpos_cuda` |
| after | 1x `fused_gqa_attention_decode_batched` (`grid.z = batch`) |

## Rule

For batched decode, do not leave row loops around kernels that already accept a
batch dimension and per-row device metadata. The device kernel should consume
`[B]` positions / sequence lengths and per-row cache pointer tables directly;
host row loops are the fallback arm only.

## Follow-up

Run the same c=8 BLUE-73-MANGO and throughput A/B on
`Qwen3.6-35B-A3B` after issue #101's slow startup is fixed or a preloaded 35B
server is available.
