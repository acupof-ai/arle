# DSpark Train Sidecar — Acceptance-Weighted Training for Draft Models

> Status: **Phase 1 shipped** (2026-07-20). Experience capture + acceptance-weighted
> trainer + Markov-head weight hot-swap verified end-to-end on H20
> (Qwen3.6-27B-FP8 + dspark-aeon draft): 6 training steps, loss −4.04→−3.18,
> zero errors. Phase 2 (PPO clip, util guard, checkpointing) pending.

## Decision

Train the DSpark draft model **in-production** via a sidecar training loop that
optimizes the actual acceptance rate, instead of the offline CE/L1 proxy loss
used by DeepSpec/SpecForge. The draft model already runs in the inference
engine (`crates/infer-cuda/src/qwen35/dspark.rs`); we add an asynchronous
training thread that consumes (context, draft_logits, accepted_count) tuples
emitted by the verify step and updates the draft weights without blocking
serving.

## Why in-production training over offline

| | Offline (DeepSpec) | Train sidecar (this) |
|---|---|---|
| Data | ~38 TB precomputed target cache | live traffic, zero storage |
| Objective | CE + L1 proxy for acceptance | **direct** acceptance reward |
| Distribution drift | fixed at train time | continuously adapts |
| Cold start | from scratch | fine-tune released checkpoint |
| Infra | separate training cluster | same GPU, idle cycles |

The acceptance rate is already counted at `spec_decode.rs:297`
(`mtp_accepts` / `mtp_rejects`). The reward signal is free.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Inference engine (hot path)                         │
│                                                      │
│  spec_step()                                         │
│    1. draft_chain()  → draft_logits                  │
│    2. verify()       → accepted_count (reward)       │
│    3. [hook] push (ctx, draft_logits, accepted)      │
│         → lock-free ring buffer                      │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│  Sidecar training thread (cold path)                 │
│                                                      │
│  loop:                                               │
│    batch = ring_buffer.drain(n=64)                   │
│    if batch empty: sleep(100ms); continue            │
│                                                      │
│    # Acceptance-weighted policy gradient                        │
│    log_prob = draft_logits.log_softmax().gather(tokens) │
│    advantage = accepted - baseline_ema               │
│    loss = -(log_prob * advantage).mean()             │
│    loss.backward()                                   │
│    optimizer.step()  (AdamW, lr=1e-5)                │
│                                                      │
│    # atomic weight swap                              │
│    draft_head.update_weights(new_state_dict)         │
└─────────────────────────────────────────────────────┘
```

## Reward design

```
reward = accepted_count / block_size    # normalized [0, 1]
```

The draft proposes `block_size` tokens; the target verifies all of them in one
forward. `accepted_count` is the prefix length that matched. Normalizing by
`block_size` gives a stable [0, 1] reward regardless of block size.

**Baseline**: exponential moving average of reward (α=0.01). This is the
`accept_ema` already computed at `spec_decode.rs:73` for the gating decision.

## Components

### 1. Experience capture hook (`crates/infer-cuda/src/executor/dspark_train.rs`)

After the DSpark verify+accept step in `qwen35.rs`, push a tuple to a global
`DsparkExperienceBuffer`:

```rust
pub struct DsparkExperience {
    draft_tokens: Vec<u32>,     // [block_size]
    draft_logits: Vec<f32>,     // [block_size, vocab] — D2H copy
    target_logits: Vec<f32>,    // [block_size, vocab] — D2H copy (for L1 reg)
    accepted: usize,            // reward signal
    block_size: usize,
    vocab_size: usize,
}
```

The hook runs **after** the verify, on already-computed data. It copies the
logits to host (small: block_size × vocab bf16→f32) and returns immediately.
No sync, no blocking of the hot path.

### 2. Sidecar trainer (`crates/train/src/dspark_train.rs`)

A standalone thread that:
- drains the `DsparkExperienceBuffer` in batches (`batch_size=64`)
- computes acceptance-weighted loss using the **train-crate autograd** (`CpuBackend`)
- runs AdamW step on the **Markov head only** (`w1` embedding [vocab, rank] +
  `w2` linear [rank, vocab]); the parallel backbone stays frozen
- calls back into the inference engine to swap Markov-head weights

The Markov head forward is two autograd ops: `embedding(w1, tokens)` →
`matmul(_, w2)`, added as a bias to the draft logits before log-softmax.
Vocab size is lazily inferred from the first drained experience so the
trainer is model-agnostic at construction.

### 3. Weight hot-swap (`crates/infer-api/src/loaded.rs`)

`LoadedInferenceEngine::update_dspark_markov_weights(&self, w1, w2)` runs the
weight swap on the engine thread via `run_on_engine` (invalidates the prefix
cache so stale KV from the old head is not reused). The trainer pushes updated
weights after every step; the swap is atomic w.r.t. the hot path.

## Implementation phases

### Phase 0 — Instrumentation ✅ (shipped)
- Add `ExperienceBuffer` + capture hook in `dspark_train.rs` (called from `qwen35.rs`)
- Log acceptance distribution to confirm reward signal quality
- **Exit**: see real acceptance rates per step

### Phase 1 — CPU trainer + weight swap ✅ (shipped 2026-07-20)
- Implement `DsparkTrainer` in autograd (Markov head only: `w1` embedding + `w2` linear; backbone frozen)
- Implement acceptance-weighted loss + AdamW + EMA baseline
- Wire weight hot-swap via `LoadedInferenceEngine::update_dspark_markov_weights`
- **Exit**: end-to-end verified on H20 — 6 training steps, loss −4.04→−3.18, zero errors

### Phase 2 — Production hardening (pending)
- PPO clip for stability
- Advantage normalization
- GPU utilization guard: train only when util < 70%
- Checkpointing: save draft weights every N steps
- **Exit**: no SLO regression, sustained acceptance gain

## Files touched

| File | Change |
|------|--------|
| `crates/infer-cuda/src/executor/dspark_train.rs` | `DsparkExperienceBuffer` + `capture_dspark_experience` hook (called from `qwen35.rs` hot path after verify) |
| `crates/infer-cuda/src/executor/qwen35.rs` | calls `capture_dspark_experience` after each DSpark accept step |
| `crates/train/src/dspark_train.rs` | sidecar trainer: `DsparkTrainer` (acceptance-weighted + AdamW + EMA baseline) + `spawn_dspark_train_sidecar` (RAII guard) + `InferCudaExperienceSource` adapter |
| `crates/cli/src/serve.rs` | `run_dspark_serve`: loads engine, spawns train sidecar, serves over engine's local router so weight updates land in the same running engine |
| `crates/infer-api/src/loaded.rs` | `update_dspark_markov_weights(&self, ...)` (hot-swap via `run_on_engine`) + `dspark_experience_buffer()` accessor |
| `crates/infer-api/src/serve.rs` | `bind_and_serve` made `pub` so the CLI can serve an already-built router |
| `crates/infer-api/src/lib.rs` | re-export `bind_and_serve` + `ServeShutdown` |

## Risks

| Risk | Mitigation |
|------|-----------|
| Training thread steals GPU from serving | CPU-only trainer; draft model is 5 layers, <1GB |
| Reward hacking (draft copies target) | L1 reg on draft logits vs target logits (cheap, target logits already computed in verify) |
| Weight swap race with hot path | double-buffer + atomic pointer flip; hot path checks flag once per step |
| Cold start from random weights | initialize from DeepSpec released checkpoint (`deepseek-ai/dspark_qwen3_4b_block7`) |
| Non-stationary reward | EMA baseline + PPO clip; cap update magnitude |

## Success criteria

- Phase 1: acceptance rate improves ≥5% over frozen baseline on a fixed
  gsm8k/humaneval workload, measured by `scripts/bench_throughput.py`.
- Phase 2: no p99 latency regression (>2ms) at concurrency 1, 4, 8, 16.
