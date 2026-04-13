# Speculative Decoding Feasibility — CUDA Backend

_Date: 2026-04-14_

## Current State

`infer/src/speculative.rs` already has a solid CPU-verifiable framework:

| Component | Status | Notes |
|-----------|--------|-------|
| `SpecConfig` | ✅ done | num_speculative_tokens, min_acceptance_rate |
| `TokenProposal` | ✅ done | draft_probs, target_probs, target_bonus_dist |
| `VerificationResult` | ✅ done | accepted tokens, bonus token, rejection index |
| `verify_tokens()` | ✅ done | rejection sampling, tested |
| `AcceptanceTracker` | ✅ done | rolling window, should_disable() |
| `DraftModel` trait | ✅ done | GPU stub + MockDraftModel for tests |
| `expected_speedup()` | ✅ done | Chen et al. 2023 formula |
| **GPU integration** | ❌ missing | draft model loading, KV cache, scheduler |

ROADMAP says 4.2 Speculative depends on Phase 1.x (multi-architecture models).

## Approaches Researched

### Standard Draft Model (Leviathan et al. 2023)
- Load a small model (e.g. Qwen3-0.5B) as draft, large model as target
- Draft K tokens autoregressively → target verifies in one forward pass
- **Pros**: works with any pair; **Cons**: requires loading 2 models, KV cache for both

### EAGLE / EAGLE2 / EAGLE3 (Li et al. 2024-2025)
- Draft model reuses **target model's KV cache** — draft only computes a lightweight feature head
- EAGLE2: tree-structured draft proposals (exponentially more candidates per target pass)
- EAGLE3 (2025): improved acceptance rate via training on intermediate representations
- **Key insight**: Draft model is a single transformer layer + embedding; zero separate KV overhead
- GitHub: `SafeAILab/EAGLE` — PyTorch reference; no Rust port found as of 2026-04
- Acceptance rate on Qwen3 family: ~0.7–0.8 (EAGLE2); ~0.85 (EAGLE3)
- Expected speedup with K=5, α=0.75: ~2.8×

### Medusa (Cai et al. 2024)
- Multiple decode heads on the target model (no separate model)
- **Cons**: requires fine-tuning the target; acceptance rate ~0.6

### MTP Heads (DeepSeek-V3)
- Built into model architecture; zero additional cost
- Not applicable to Qwen3/Qwen3.5

## Integration Plan for agent-infer CUDA Backend

### What Needs to be Built

**1. DraftModel GPU Implementation** (`infer/src/backend/cuda/speculative.rs`)
- `DraftEngine`: wraps a second loaded model (e.g. Qwen3-0.5B)
- Reuses CUDA scheduler infrastructure but runs single-threaded
- `draft_batch()` → calls `model.forward()` K times, collects token + probability

**2. KV Cache Interaction**
- Draft tokens get tentative KV cache entries in a separate pool
- On rejection: free the rejected draft KV entries (rollback)
- On acceptance: "commit" draft KV entries to the main pool
- Current `PagedKvPool` in `backend/cuda/paged_kv.rs` needs a "tentative" concept

**3. Scheduler Integration** (`infer/src/scheduler/cuda/`)
- `SpeculativeScheduler`: wraps draft + target schedulers
- Decode loop: draft K tokens → target verifies → advance by `num_accepted + 1`
- Need to handle CUDA Graph invalidation when K tokens are accepted (batch size changes)

**4. Probability Collection**
- Target model's `forward()` currently returns only the sampled token
- Need to return full softmax distribution (or at least per-token probabilities)
- `ModelForward::forward_with_probs()` new trait method

### Dependency Order

```
Phase 1: DraftModel loading (no scheduler needed)
Phase 2: TokenProposal with real probabilities (modify forward pass)
Phase 3: KV rollback API in PagedKvPool
Phase 4: SpeculativeScheduler integration
Phase 5: CUDA Graph adaptation (2-phase: draft pass + verify pass)
```

### Rust Ecosystem

No production-quality Rust speculative decoding implementation found (2026-04).
Candle (HuggingFace Rust) has a basic draft-model speculative decoding example
(`candle-examples/examples/falcon/`) but uses eager (non-paged) attention.

## Estimated Speedup on agent-infer Workloads

Based on `expected_speedup(k=5, alpha=0.75)` with standard draft model:

- C=1 (single request): 2.8× decode speedup, TTFT unchanged
- C=4 (batch): lower speedup (~1.5×) due to batch size overhead in draft stage
- Long contexts: higher α → higher speedup

## Recommendation

Start with **standard draft model** (Phase 1–4 above) using Qwen3-0.5B as draft.
EAGLE requires model checkpoint availability and a training pipeline; defer to Phase 4.2.2.

Key risk: KV rollback in `PagedKvPool` — needs design review before implementation.
