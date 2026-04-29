# BF16-shadow mixed-batch path is architecturally infeasible

## Context

After the 2026-04-29 trace agent confirmed that 34% of c=16 wall-time
is spent in pure-prefill `StepPlan::Split` blocks because
`supports_mixed_batch` rejects FP8/INT8 KV
(`infer/src/model/qwen3/forward.rs:585`), a follow-up audit agent
proposed lifting the gate by reusing the bf16 working buffer
(`k_work`/`v_work`) as a "shadow" of the FP8 paged KV during the
mixed step.

The proposed flow:

1. `decode_prep_paged_cuda` + `prefill_attention_paged_prep_cuda`
   write fresh K/V into `k_work`/`v_work` (these kernels already do
   this for FP8 pools — they output bf16).
2. `flashinfer_tc_run_layer` reads `k_work`/`v_work` (treated as a
   bf16 alias of the layer's KV).
3. After attention, `quantize_scatter_kv_fp8_range` commits the
   bf16 scratch back into the FP8 pool, mirroring the post-prefill
   `commit_layer` path.

## Root cause of the dead end

`k_work` / `v_work` are **single-layer scratch buffers shared across
all layers within one forward pass** (`crates/cuda-kernels/src/paged_kv.rs:389-402`),
**NOT** full historical-KV mirrors:

- `decode_prep_paged_cuda` writes only the freshly RoPE'd new K/V
  row at `(page_id, last_len-1)`
  (`crates/cuda-kernels/csrc/attention/decode_prep_paged.cu:152-159`).
- `prefill_attention_paged_prep_cuda` writes only the current
  prefill chunk's `[start_pos, start_pos+seq_len)`
  (`crates/cuda-kernels/csrc/attention/prefill_attention_paged_prep.cu:81-100`).
- Historical pages of the same logical sequence are **never staged
  into `k_work`** — they live exclusively in `k_data` as FP8/INT8.

The existing FP8 pure-decode flow confirms this design:
`batch_decode.rs:1202-1240` writes one new row to `k_work`, then
**immediately** calls `quantize_paged_kv_fp8` to push it into
`k_data`, **then** runs `decode_attention_fp8` which reads the
**full historical context from `k_data`** (FP8 native attention).
`k_work` never holds more than the freshest single-step write per
layer.

If we flipped the K2 gate and pointed `flashinfer_tc_run_layer` at
`k_work_ptr`, attention would see only newly-written rows and
**zeros for every historical position** — silently corrupt logits.
Not even "!!!!!" output (which would at least be a clear NaN
signal); just bad text indistinguishable from natural sampling
drift.

## Real fix surface (ranked)

1. **Add a `decode_attention_varlen_{fp8,int8}` kernel** — varlen Q
   + causal mask + FP8/INT8 dequant in attention. Drop-in
   replacement for `flashinfer_tc_run_layer` in the mixed path
   when KV is quantized. Touches `csrc/attention/` — proper kernel
   work, multi-day. **This is the correct long-term path.**

2. **Per-layer full-context dequant into a true shadow** — needs
   `dequantize_paged_kv_fp8_cuda` (a NEW kernel; only INT8 has
   `dequantize_paged_kv_cuda` today at
   `crates/cuda-kernels/src/kv_quant.rs:101-134`) plus a per-layer
   full-context dequant call before attention and a re-quantize of
   new rows after. Cost-comparable to staying in BF16 KV
   altogether — likely a perf regression, not a fix.

3. **Reduce Split-block cost via scheduler** — make each prefill
   block shorter so decode rows wait less. Already validated
   experimentally: `--chunked-prefill-size 512` instead of the
   default 2048 gives **+38% tok/s** at c=16/4096-in (105.22 →
   145.30; ITL 86 → 71.6 ms). This is the current ship-ready fix.

## Rule

- **Don't propose buffer reuse without verifying buffer lifetime.**
  The audit agent assumed `k_work` was a full-context bf16 mirror
  because of its size; in fact it's scratch that's overwritten
  every layer within every step. Audit hypotheses about reusing
  existing kernels need to walk the **read** side of the data flow,
  not just the write side. The `decode_attention_fp8` call site at
  `batch_decode.rs:1240` (which reads `k_data`, not `k_work`) is
  the one piece of evidence that disproved the reuse hypothesis.

- **K2 gate stays in place**. Document the constraint, ship the
  Option-3 workaround now, file the Option-1 kernel as the
  long-term project. Per
  `feedback_root_cause_not_patches.md`, don't paper over with
  Option 2.

## Implementation status

- 2026-04-29: Option 3 validated (`--chunked-prefill-size 512`,
  +38% tok/s). To be codified as the FP8/INT8 KV default in the
  next commit.
- Option 1 (FP8/INT8 varlen attention kernel): filed as next
  milestone. Tracked separately in `docs/projects/`.
