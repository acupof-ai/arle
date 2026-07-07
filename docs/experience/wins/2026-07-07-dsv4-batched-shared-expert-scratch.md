# DSv4 batched-decode shared-expert scratch reuse

> Status: measured — 2026-07-07. Commit `de6fc4fd`.

## Change
`forward_decode_batch_stream_impl` shared-expert block: replaced per-layer
`HiddenStates::uninit(hidden, seq_len)` + `dsv4_shared_expert_forward` (6 allocs +
4 H2D inside `dsv4_shared_expert`) with `dsv4_shared_expert_forward_decode_scratch`
reusing the model-wide `kv_adapter.shared_expert_scratch` (`Dsv4SharedDecodeScratch`,
allocated when the model has a shared-expert layer; `max_m = 128`).

## Compile
BUILD_EXIT=0 (cuda,nccl,deepep), clippy-clean. Borrow: the MoE-half
`kv_adapter.shared_expert_decode_mut()` borrow does not overlap the attention-half
`layer_dsa_and_flashmla_batch_mut` borrows (sequential per layer).

## Correctness (greedy, TP=4/EP=4, DSv4-Flash-FP8, GPU 4-7)
- "capital of France, one word" → reasoning_content "…the answer is straightforward: Paris."
- "one sentence about the sea" → coherent reasoning_content.
- No garbage / NaN / empty generation.
- needle_gate.py reported all-miss `out=''`; the model outputs into
  `reasoning_content` (think), `content` empty at max_tokens=24, finish_reason=length.

## Wall-clock
Measured in the cumulative A/B with commit 2:
`errors/2026-07-07-dsv4-alloc-removal-sweep-wall-wash.md` (c1+c2 vs baseline
−0.27% mean wall, within ±0.7% run-to-run spread).
