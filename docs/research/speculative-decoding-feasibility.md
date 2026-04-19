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

### EAGLE / EAGLE2 / EAGLE3 (Li et al. 2024–2025)
GitHub: https://github.com/SafeAILab/EAGLE — Python, integrates with vLLM/SGLang/TensorRT-LLM.

- **EAGLE-1** (ICML 2024): Draft extrapolates second-to-last-layer hidden states of target model.
  Small autoregressive head, 1–2 days training on 8x RTX 3090. 3x speedup vs greedy on 13B.
- **EAGLE-2** (EMNLP 2024): Dynamic candidate tree shaped by draft confidence scores. 4x speedup,
  1.4x over EAGLE-1.
- **EAGLE-3** (NeurIPS 2025): Multi-layer feature fusion (low+mid+high layers, not just second-to-last).
  Introduces "training-time test" to calibrate acceptance rates. **6.5x speedup** vs greedy,
  1.4x over EAGLE-2, **1.38x throughput gain in SGLang at batch=64**.
  NVIDIA ships checkpoint: `nvidia/gpt-oss-120b-Eagle3-long-context` on HuggingFace.
  arXiv: https://arxiv.org/html/2503.01840v1
- **Key insight**: Draft model is a single transformer layer + embedding; zero separate KV overhead
- GitHub: Python only; no Rust port as of 2026-04
- Acceptance rate on Qwen3 family: ~0.7–0.8 (EAGLE2); ~0.85 (EAGLE3)
- Expected speedup with K=5, α=0.75: ~2.8×

### Medusa (Cai et al. 2024)
- Multiple decode heads on the target model (no separate model)
- **Cons**: requires fine-tuning the target; acceptance rate ~0.6

### MTP Heads (DeepSeek-V3)
- Built into model architecture; zero additional cost
- Not applicable to Qwen3/Qwen3.5

## KV Cache + Paged Memory Pattern

The key challenge: paged KV allocates in fixed-size blocks. Draft tokens speculatively occupy
KV slots before acceptance. On rejection, those slots must be freed.

**vLLM/SGLang pattern** (confirmed via research):
- Paged KV applies to the **committed prefix** only
- Draft token tree is held in a **small non-paged suffix buffer**
- On acceptance: fold accepted tokens into the paged prefix
- On rejection: discard the suffix (no complex page-level rollback needed)
- Reference: "Transactional KV Caching for Speculative Decoding under Paged KV Memory" (TechRxiv 2025)

For agent-infer CUDA: `PagedKvPool` in `crates/cuda-kernels/src/paged_kv.rs`
(post `a4e12f5`) does not need per-page rollback. Instead, draft tokens operate
on a per-request contiguous buffer, and only on acceptance are they appended to
the paged pool. This is simpler than full transactional semantics.

## Integration Plan for agent-infer CUDA Backend

### What Needs to be Built

**1. DraftModel GPU Implementation** (new file, would land alongside `infer/src/speculative.rs` — current CPU-only framework location — with GPU kernels contributed to `crates/cuda-kernels/`)
- `DraftEngine`: wraps a second loaded model (e.g. Qwen3-0.5B)
- Reuses CUDA scheduler infrastructure but runs single-threaded
- `draft_batch()` → calls `model.forward()` K times, collects token + probability

**2. KV Cache Interaction**
- Draft tokens get tentative KV cache entries in a separate pool
- On rejection: free the rejected draft KV entries (rollback)
- On acceptance: "commit" draft KV entries to the main pool
- Current `PagedKvPool` in `crates/cuda-kernels/src/paged_kv.rs` needs a "tentative" concept

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

**No production Rust implementations exist** (confirmed 2026-04). `candle-vllm` has
PagedAttention + continuous batching but no speculative decoding. The technique lives entirely
in Python-land: vLLM, SGLang, TensorRT-LLM. Agent-infer would be the first production Rust
implementation. The core algorithm is straightforward; the KV cache integration (paged suffix
pattern above) is the only non-trivial part.

## Estimated Speedup on agent-infer Workloads

Based on `expected_speedup(k=5, alpha=0.75)` with standard draft model:

- C=1 (single request): 2.8× decode speedup, TTFT unchanged
- C=4 (batch): lower speedup (~1.5×) due to batch size overhead in draft stage
- Long contexts: higher α → higher speedup

## Recommendation

Start with **standard draft model** (Phase 1–4 above) using Qwen3-0.5B as draft.
EAGLE requires model checkpoint availability and a training pipeline; defer to Phase 4.2.2.

Key risk: KV rollback in `PagedKvPool` — needs design review before implementation.
