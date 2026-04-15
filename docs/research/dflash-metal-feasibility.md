# DFlash / Block-Diffusion Speculative Decoding on Apple Silicon

_Date: 2026-04-14_

## What is DFlash?

DFlash is a speculative decoding approach using **block-diffusion** (discrete diffusion models) as the
draft mechanism. Instead of an autoregressive small model, a lightweight denoising head proposes
multiple tokens in parallel via a few denoising steps, then the target model verifies in one pass.

Key properties:
- **Parallel token generation**: draft K tokens simultaneously (not autoregressively)
- **Hidden-state extraction**: intermediate activations from target layers feed the drafter
- **Pluggable architecture**: adapter files isolate model-specific logic

## DFlash-MLX: Apple Silicon Implementation

GitHub: https://github.com/Aryagm/dflash-mlx — active project, directly relevant.

**Key implementation details:**
- Uses **hidden-state extraction**: intermediate activations from target model layers feed draft model
- Does **not require a separate KV cache** for the draft model (uses target's cached states)
- **Per-layer cache rollback**: handles Qwen3.5's hybrid attention/recurrent layers correctly
- **Batched verification**: all proposed tokens verified in one target forward pass
- Accepts "the longest matching prefix plus one bonus correction token"
- Warmup phase separated from measurement for accurate Metal GPU profiling

**Supported models:**
- Qwen3-4B BF16 (primary target + DFlash draft)
- Qwen3.5-4B MLX BF16 (functional, with hybrid attention rollback)
- Additional: Llama 3.1, Qwen3 Coder (upstream checkpoints)

## Alternative Apple Silicon Approaches

| Approach | Repo | Speedup | Notes |
|----------|------|---------|-------|
| **mlx-lm `--draft-model`** | ml-explore/mlx-examples (merged Jan 2025) | 1.5–1.8x | No batching; prompt cache conflicts |
| **mlx-community/speculative-decoding** (Swift) | mlx-community/speculative-decoding | 2–3x | Mamba drafters 86 tok/s experimental |
| **dflash-mlx** | Aryagm/dflash-mlx | TBD | Block diffusion; Qwen3/Qwen3.5; per-layer rollback |
| **Apple ReDrafter** | machinelearning.apple.com/research | 2.3x | RNN drafter; requires distillation training |
| **EAGLE-3 on MLX** | Not ported | N/A | Python/CUDA only |

**Key insight**: mlx-lm's built-in `--draft-model` is the easiest integration point
but has no batch support. For agent-infer's use case (single request, Metal backend),
it's a viable first step.

## Feasibility for agent-infer Metal Backend

### Pros
1. **Directly relevant models**: Qwen3-4B and Qwen3.5-4B are exactly what agent-infer uses
2. **No separate model checkpoint**: draft uses target's hidden states → zero extra memory
3. **KV rollback for hybrid models**: DFlash-MLX already solved the Qwen3.5 recurrent state
   rollback problem — similar to what agent-infer's `backend/metal/prefix_cache.rs` would need
4. **MLX-native**: fits within the existing `mlx-sys` / MLX bridge architecture

### Cons / Risks
1. **External C++ dependency**: integrating DFlash-MLX's C++ draft head would require
   additions to `crates/mlx-sys/` (the MLX + C++ bridge)
2. **Checkpoint availability**: DFlash draft heads are fine-tuned on the target model's
   hidden states — need the appropriate checkpoint for Qwen3-4B
3. **Agent-infer Metal is single-request**: current Metal scheduler handles one request at a time;
   speculative decoding would add draft loop complexity to `backend/metal/generate.rs`
4. **Performance measurement**: Metal GPU profiling needed post-integration to verify speedup

### Integration Sketch

```rust
// In backend/metal/generate.rs, modify generation loop:
loop {
    // 1. Draft: extract hidden states at target layer N, run DFlash denoising
    let draft_tokens = dflash_head.propose(&hidden_states, k=5);
    
    // 2. Verify: run target forward pass on [prefix + draft_tokens]
    let (target_logits, accepted) = model.forward_speculative(&draft_tokens);
    
    // 3. Accept/reject + KV rollback
    // Use verify_tokens() from infer/src/speculative.rs
    let result = verify_tokens(&proposal, &mut rng);
    
    // 4. Rollback rejected KV entries
    kv_cache.rollback(result.rejection_index);
    
    // 5. Advance by result.total_tokens()
    emit_tokens(&result.accepted, result.bonus_token);
}
```

The `verify_tokens()` function in `infer/src/speculative.rs` is already correct and
would be reused directly.

## Recommendation

**DFlash on Metal: Feasible, medium priority.**

Prerequisites:
1. Source the DFlash draft checkpoint for Qwen3-4B
2. Expose intermediate hidden states from `infer/src/model/qwen3/forward.rs`
3. Add KV rollback API to `backend/metal/kv_pool.rs`
4. Integrate DFlash C++ draft head via `mlx-sys` bridge

This is a 2–3 week project. Recommended sequencing:
1. First implement on CUDA (larger benefit, better tooling for debugging)
2. Port to Metal once CUDA implementation is validated

**vs. standard draft model**: DFlash has higher acceptance rate (no extra model needed)
but requires a fine-tuned checkpoint and more complex integration. Standard draft model
(Qwen3-0.5B → Qwen3-4B) is simpler to start with.
