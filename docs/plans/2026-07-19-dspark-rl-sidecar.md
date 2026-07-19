# DSpark RL Sidecar — Test-Time Training for Draft Models

> Status: Active

## Decision

Train the DSpark draft model **in-production** via a sidecar RL loop that
optimizes the actual acceptance rate, instead of the offline CE/L1 proxy loss
used by DeepSpec/SpecForge. The draft model already runs in the inference
engine (`crates/infer-cuda/src/qwen35/dspark.rs`); we add an asynchronous
training thread that consumes (context, draft_logits, accepted_count) tuples
emitted by the verify step and updates the draft weights without blocking
serving.

## Why RL over offline training

| | Offline (DeepSpec) | RL sidecar (this) |
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
│    # REINFORCE with advantage                        │
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

### 1. Experience capture hook (`spec_decode.rs`)

After `chain.accept_path()` returns `accepted`, push a tuple to a global
`ExperienceBuffer`:

```rust
struct Experience {
    slot_idx: usize,
    start_pos: usize,
    draft_logits: Vec<bf16>,   // [block_size, vocab] — copied out
    draft_tokens: Vec<u32>,    // [block_size]
    accepted: u32,             // reward signal
    ctx_features: Vec<bf16>,   // target hidden at target_layer_ids (for value fn)
}
```

The hook runs **after** the verify, on the already-computed data. It copies
the logits/tokens out (small: block_size × vocab is ~7 × 152k ≈ 1MB bf16)
and returns immediately. No sync, no blocking.

### 2. Sidecar trainer (`crates/train/src/dspark_rl.rs`)

A standalone thread (or `tokio` task) that:
- drains the `ExperienceBuffer` in batches
- computes REINFORCE loss using the **train-crate autograd**
  (`CpuBackend` — the draft model is tiny, 5 layers)
- runs AdamW step
- calls back into the inference engine to swap weights

The draft model forward is re-implemented in autograd ops (matmul, sdpa,
embedding, rmsnorm, rope — all already exist in `crates/autograd/src/ops.rs`).
Weights are shared via the same safetensors format the inference engine loads.

### 3. Weight hot-swap (`qwen35/dspark.rs`)

`Qwen35DsparkHead` gets an `update_weights()` method that takes a new
state dict (host-side bf16 tensors) and atomically replaces the device
pointers. Double-buffered: new weights are uploaded to a staging buffer,
then the pointer is flipped under a lock that the hot path checks once per
`spec_step`.

## Implementation phases

### Phase 0 — Instrumentation (no training yet)
- Add `ExperienceBuffer` + capture hook in `spec_decode.rs`
- Log acceptance distribution to confirm reward signal quality
- **Exit**: see real acceptance rates per step

### Phase 1 — CPU trainer + weight swap
- Implement `DSparkDraftModel` in autograd (reuse `qwen35.rs` layers)
- Implement REINFORCE loss + AdamW
- Wire weight hot-swap into `Qwen35DsparkHead`
- **Exit**: acceptance rate improves over 1000 steps on a fixed workload

### Phase 2 — Production hardening
- PPO clip for stability
- Advantage normalization
- GPU utilization guard: train only when util < 70%
- Checkpointing: save draft weights every N steps
- **Exit**: no SLO regression, sustained acceptance gain

## Files touched

| File | Change |
|------|--------|
| `crates/infer-cuda/src/executor/spec_decode.rs` | add `ExperienceBuffer` + capture hook after verify |
| `crates/infer-cuda/src/qwen35/dspark.rs` | add `update_weights()` hot-swap method |
| `crates/train/src/dspark_rl.rs` **(new)** | sidecar trainer: REINFORCE + AdamW + weight sync |
| `crates/train/src/dspark_model.rs` **(new)** | autograd impl of DSpark draft forward |
| `crates/train/src/lib.rs` | register `dspark_rl` module |

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
