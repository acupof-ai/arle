# Fused Qwen3.6-27B GDN kernels (TileLang)

Fused TileLang kernels for the Qwen3.6-27B gated-delta-rule (GDN) linear-attention
sub-layer. They collapse what the current pipeline launches as several separate
kernels into fused launches, cutting per-op kernel-launch overhead and the
intermediate HBM round-trips between stages.

These are **standalone, self-validating kernels** — they are not yet wired into
`build.rs` / `kernels.toml` / the Rust FFI or any `qwen35.rs` call site. Each file
has an embedded `_self_check()` that compiles the kernel and compares it against an
fp32 PyTorch reference. Run any file directly to validate:

```
python crates/cuda-kernels/tools/tilelang/qwen36_gdr_decode_fused.py
python crates/cuda-kernels/tools/tilelang/qwen36_prefill_wy.py
python crates/cuda-kernels/tools/tilelang/qwen36_prefill_scan_o.py
python crates/cuda-kernels/tools/tilelang/qwen36_solve_tril.py
```

All validated on H20 / sm_90 against fp32 references (out + state within bf16
tolerance; solve_tril bit-exact at realistic GDN scale).

## Kernels

**Decode (single token), value-head-tiled, grid `(num_value_heads, B)`** —
`qwen36_gdr_decode_fused.py`:
- `gdr_decode_gated_norm` — recurrent gated-delta state update + gated RMSNorm in
  one kernel (replaces two launches: `gdr_decode_batch` + `rms_norm_gated`).
- `gdr_decode_conv_gated_norm` — additionally folds the depthwise conv1d+SiLU in
  front, so the full `conv1d -> gdr -> gated RMSNorm` chain is ONE launch
  (replaces three). GQA conv-state ring writes are done by the group
  representative to avoid a shared-channel race.

**Chunkwise prefill (chunk=64)** — split at the one unavoidable state-scan barrier:
- `qwen36_prefill_wy.py` `prefill_wy` (chunk-parallel, grid `(num_chunks, B*num_value_heads)`):
  L2-norm q/k → chunk-local `cumsum(g)` → `A = beta_i·exp(g_i-g_j)·(k_i·k_j)`
  (strict-lower) → `solve_tril` → `u = A^{-1}(v·beta)`, `w = A^{-1}(k·beta·exp(gcs))`.
- `qwen36_prefill_scan_o.py` `prefill_scan_o` (chunk-serial): running state scan
  `v_new = u - w@h`, fused chunk output `(q@h + causal(q@k^T)@v_new)·scale`, gated
  decay `h = h·exp(g_last) + k^T@v_new`. Drops the `h`→HBM epilogue write (the
  fusion win). End-to-end A→B matches a full chunked-GDR reference (abs_max 3e-5).
- `qwen36_solve_tril.py` `solve_tril` — standalone `(I + StrictLower(A))^{-1}` via
  exact forward substitution (strict-lower L is nilpotent, so the inverse is exact,
  not a truncated Neumann series).

## Notes / follow-ups

- Contractions are per-thread serial loops (correctness-first, mirroring the
  recurrent decode path). Retiling the chunkwise GEMMs onto WGMMA (chunk=64 gives
  `m=64`, WGMMA-eligible) is the next perf step.
- Dims are fixed to the Qwen3.6-27B GDN config (16 key / 48 value heads,
  key=val=128, chunk=64). Wiring into the AOT `gen_tilelang_aot.py` generator
  (dynamic `T.symbolic` seq_len, `get_kernel(name)` entry, WrapperSpec) is a
  separate integration step, intentionally not included here.
- Gate uses `T.exp` consistently across decode and prefill.
