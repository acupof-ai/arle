---
title: 2026-05-10 Task #28 Medusa scaffold readiness audit (post-REFUTATION pivot)
date: 2026-05-10
type: research
status: open (informs next-session pickup)
related_docs: [`9735b47-w4a8-longctx-prompt16384-hybrid-extrapolation-refuted`, `M_medusa-required-path.md`, `M_medusa-phase1a-dataset-directive.md`, `9350767` session-tail summary §5 strategic matrix]
---

# Task #28 Medusa scaffold — readiness audit

> **Why now**: `9735b47` REFUTATION promoted Option A (Medusa) to
> dominant single-axis investment for ALL contexts (was: only short-ctx,
> long-ctx was Option B Hybrid). Hybrid Option B caps at -14.2% measured.
> Audit here surveys what's READY vs BLOCKING for Task #28 pickup.

## §1 What's READY (existing infrastructure)

### §1.1 Spec-decode runtime scaffold
- `infer/src/speculative.rs` — **721 LOC** existing:
  - `SpecConfig` (pos 47) + `Default` (pos 75)
  - `TokenProposal` (pos 91) — propose K tokens
  - `verify_tokens` (pos 201) + `verify_tokens_greedy` (pos 250) — accept/reject
  - `AcceptanceTracker` (pos 303) — α tracking
  - `MockDraftModel` (pos 392) — placeholder for draft generation
  - `expected_speedup(k, α)` (pos 436) — Leviathan formula

### §1.2 Plans + briefs READY
- `M_medusa-required-path.md` — full Phase 1-4 plan with:
  - Phase 3 formula corrected (post-`528844c`): K=4, α=0.7 → 2.62× speedup
  - License threshold 1.5×, soft-win 1.2×, kill < 1.0×
- `M_medusa-phase1a-dataset-directive.md` — dataset selection ready:
  - Recommended: `lmsys/lmsys-chat-1m` (1M chats, natural distribution)
  - Alternative: `tatsu-lab/alpaca` (52k, simpler single-turn)
  - Wire-up via `crates/train/src/hub_dataset.rs` + `arle data download`
- `63769be` Medusa Alpaca cross-link committed (Task #28 unblock point)

### §1.3 Training infrastructure
- `crates/train/src/hub_dataset.rs` — HF Hub JSONL loader exists
- `crates/autograd/` — from-scratch autograd + AdamW + lr-schedule (per CLAUDE.md workspace)
- W3/W4 baseline: established at agent shape (per `370a267`)

### §1.4 Strategic mandate
- Per `9735b47` + `114aca4`: Option A is now dominant for ALL contexts
- Per `M_medusa-required-path.md` Phase 1: 1 week training + 1 day integration + 1 day bench
- Per session-tail §5: Option A is the lowest-blocker single-axis investment

## §2 What's BLOCKING (gates before commit)

### §2.1 Architectural gaps (codex-own substrate)
1. **4 Medusa heads** (~6.5M params each, ResBlock + reused lm_head):
   - Need new `crates/cuda-kernels/csrc/spec/medusa_head.cu` or TileLang variant
   - Must integrate with Qwen3.6 (Metal canonical) AND Qwen3-4B (CUDA dev)
2. **Tree-attention** for top-T candidate verification:
   - Standard attention only verifies linear sequences
   - Tree-attention needs causal mask construction per candidate path
   - Existing `infer/src/speculative.rs` has flat `verify_tokens` only
3. **Training pipeline**:
   - SFT-style head training on frozen target
   - Data flow: HF dataset → tokenize → forward target → train heads
   - Storage: ~26M head params + optimizer state (~104 MB)

### §2.2 Empirical gaps (require measurement)
- α value for Qwen3-4B on agent W3/W4 is **PREDICTED** 0.6-0.8, not measured
- No prior Medusa training data in this repo (M_medusa Phase 1.A reported 584 tokens vs 100k+ target)
- Wall-clock training budget: ~1 week on sm_89 (single-GPU), unverified

### §2.3 Decision gates (user direction needed)
- Which target model? Qwen3-4B (CUDA, easier) vs Qwen3.6-35B-A3B (Metal canonical, harder fit on 16GB GPU during training)
- Which dataset? Alpaca (52k, ~2 days train) vs lmsys-chat-1m (subset, more representative, ~1 week train)
- Which integration target? CUDA scheduler first vs Metal scheduler first vs both parallel

## §3 What Claude can do vs codex

### §3.1 Claude (CPU-bound prep, no GPU)
- ✅ Survey existing spec-decode scaffold (DONE this audit)
- Survey vLLM/SGLang Medusa implementations for prior art:
  - vLLM `vllm/spec_decode/medusa/` (TODO)
  - SGLang `python/sglang/srt/spec_decode/` (TODO)
- Write detailed Phase 1.B substrate scaffold brief (head module + tree-attn)
- Write Phase 1.C training data pipeline brief
- Bench harness for α measurement on baseline (no Medusa training needed —
  can measure inherent self-spec α as floor)

### §3.2 Codex (GPU + training)
- Implement 4 Medusa heads kernel (~500 LOC + .cu file)
- Implement tree-attention extension to existing scheduler
- Run training pipeline (Alpaca 52k subset, ~2-3 days wall-clock)
- Run α measurement bench post-training
- License-or-kill at 1.5× tok/s threshold

### §3.3 USER decision points
- Pick dataset (Alpaca vs lmsys-chat-1m)
- Pick target model (Qwen3-4B vs Qwen3.6-35B-A3B)
- Approve ~1 week wall-clock training investment

## §4 Recommended next actions (priority order)

1. **USER decision**: Alpaca + Qwen3-4B (fastest path, 2-3 days train) vs
   lmsys-chat-1m + Qwen3.6 (best representativeness, 1 week train)
2. **Claude** (this session if user approves): vLLM `medusa/` survey →
   port-suitability analysis brief
3. **Codex** (next pickup): Phase 1.B substrate scaffold per existing plan
4. **Claude + Codex parallel**: training pipeline (codex) +
   α-measurement harness (Claude)

## §5 Cross-references
- `M_medusa-required-path.md` — full Phase 1-4 plan
- `M_medusa-phase1a-dataset-directive.md` — dataset selection
- `9735b47` REFUTATION wins entry — strategic pivot trigger
- `114aca4` — strategic matrix integrated REFUTATION
- `infer/src/speculative.rs` (721 LOC) — existing scaffold
- `crates/train/src/hub_dataset.rs` — HF Hub data loader
- `9350767` session-tail summary §5 — Option A dominance after REFUTATION
