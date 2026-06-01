# Subagent Review Fixes

## Context

The batched subagent review of the last three days found correctness gaps across
HTTP tool calling, Metal cancellation, OPD teacher logits, CUDA cancellation,
Qwen3.5 offload preflight, KV parity defaults, and DSv4 FlashMLA/DeepGEMM
guards.

This entry records the local verification for those review fixes. It does not
claim a throughput win.

## What Worked

- Streaming chat tool-call extraction now accepts missing-close JSON/native XML
  payloads, strips DSML tool-call blocks, and suppresses structured deltas while
  still hiding raw tool markup for `tool_choice="none"`.
- DeepSeek-V4 chat validation now rejects forced tool selection because the
  prompt renderer only supports auto/none behavior.
- Metal mixed prefill+decode admission now treats cooperative cancellation the
  same way it treats dropped delta receivers, and the auto wired-limit path uses
  runtime HF cache/home env vars instead of compile-time `HOME`.
- OPD teacher logits now reject last-row-only logits for full KL windows, and
  CUDA raw-logits fallback fills a real `[seq_len, vocab]` buffer.
- CUDA request preemption preserves the cooperative cancel flag.
- Qwen3.5 MoE offload fails before destructive weight movement.
- KV parity defaults are back to a full 64-token audit; `KV_PARITY_PROFILE=smoke`
  is the explicit 4-token fast path.
- DSv4 FlashMLA defaults now detect stub-linked builds before dispatch, FlashMLA
  decode is MODEL1-only, DeepGEMM auto disables itself after a runtime
  compile/load failure, and the DSv4 beat-SGLang script's default `both` mode
  starts, waits, benches, and cleans up the selected server.

## Verification

```text
cargo fmt --check
  passed

bash -n scripts/dsv4_beat_sglang_bench.sh
  passed

cargo test -p chat tool_call -- --nocapture
  20 passed

cargo test -p train api_teacher --lib
  5 passed

cargo test -p train validate_logits_shape_rejects_last_row_teacher_logits_for_full_kl --lib
  1 passed

cargo test -p infer --no-default-features --features no-cuda chat_ -- --nocapture
  37 passed

cargo test -p infer --no-default-features --features metal,no-cuda --lib \
  auto_wired_limit_uses_runtime_home_for_hf_cache -- --nocapture
  1 passed

cargo test -p infer --no-default-features --features metal,no-cuda --lib \
  mixed_batch_eligibility_rejects_cooperative_cancel -- --nocapture
  1 passed

cargo test -p infer --test kv_parity_config --no-default-features --features no-cuda -- --nocapture
  4 passed

cargo clippy -p chat -- -D warnings
  passed

cargo clippy -p train -- -D warnings
  passed

cargo clippy -p infer --no-default-features --features no-cuda -- -D warnings
  passed

cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings
  passed

CUDARC_CUDA_VERSION=12080 cargo check -p infer --no-default-features --features cuda,no-cuda
  passed, with pre-existing cuda/no-cuda warnings
```

The local Mac host cannot link or run CUDA test binaries because
`/usr/local/cuda/lib64/stubs` is absent and `no-cuda` skips the CUDA kernel
archive, leaving FFI symbols unresolved at link time. CUDA runtime tests and any
DSv4 SLO statement remain pending remote H20 validation.

## Rule

Review fixes are not verified by parser/unit tests alone when they touch CUDA
runtime behavior. Local `cuda,no-cuda` typecheck is a build-surface gate; CUDA
runtime and DSv4 default claims still need a clean remote GPU worktree.
