# Bench — train_multi_turn batched group rollout, 2026-04-22

## Goal

- Optimization: remove serial per-episode rollout overhead from `train_multi_turn` by batching group sampling while preserving per-episode RNG streams.

## Hypothesis

- If multi-turn rollout is batched across the group, total copied bytes per step will stay roughly flat, but model forward calls and host sync points will collapse enough to cut wall time materially on Metal and measurably on CPU.

## Command

```bash
cargo run -p train --release --features metal --bin train_multi_turn -- \
  --backend metal \
  --iters 20 \
  --group-size 8 \
  --turns 2 \
  --prompt-len 16 \
  --agent-tokens 16 \
  --obs-tokens 16 \
  --d-model 128 \
  --n-layers 4 \
  --n-heads 4 \
  --d-head 32 \
  --d-ff 256 \
  --vocab 256

cargo run -p train --release --features metal --bin train_multi_turn -- \
  --backend cpu \
  --iters 20 \
  --group-size 8 \
  --turns 2 \
  --prompt-len 16 \
  --agent-tokens 16 \
  --obs-tokens 16 \
  --d-model 128 \
  --n-layers 4 \
  --n-heads 4 \
  --d-head 32 \
  --d-ff 256 \
  --vocab 256
```

## Environment

- **Backend:** Metal and CPU helper bench
- **Model:** train-side scratch Qwen3.5 dense/full-attention toy config (`hidden=128`, `layers=4`, `heads=4`, `ff=256`, `vocab=256`, LoRA rank `8`)
- **Hardware:** Apple M4 Pro, 20-core GPU, 48 GB unified memory, Metal 4
- **OS:** `Darwin 25.3.0`
- **Code under test:** `9b39968` (`perf(qwen35): batch multi-turn rollout sampling`)
- **Feature set:** `cargo build -p train --release --features metal --bin train_multi_turn`
- **Non-default flags / env vars:** `--iters 20 --group-size 8 --turns 2 --prompt-len 16 --agent-tokens 16 --obs-tokens 16 --d-model 128 --n-layers 4 --n-heads 4 --d-head 32 --d-ff 256 --vocab 256`

## Results

| backend | wall (s) | iter/s | episode/s | token/s | seq_len | group |
|---|---:|---:|---:|---:|---:|---:|
| metal, pre-change | 75.48 | 0.26 | 2.12 | 135.7 | 64 | 8 |
| metal, batched rollout | 11.76 | 1.70 | 13.60 | 870.5 | 64 | 8 |
| cpu, pre-change | 193.00 | 0.10 | 0.83 | 53.1 | 64 | 8 |
| cpu, batched rollout | 143.76 | 0.14 | 1.11 | 71.2 | 64 | 8 |

Additional measured facts:

- `metal_evals_fwd=6`, `metal_evals_bwd=223`, `metal_evals_opt=1` stayed unchanged after the optimization. The gain came before the tracked train-step forward/backward section.
- Logged training throughput is `512 tokens/step` (`group * seq_len`), but actual forward work per step at this shape is `11,648 token-positions/step` because rollout re-forwards growing prefixes.
- The optimization cut model forward calls per step from `266` to `35` (`7.6x` fewer) and explicit `to_host` sync points from `267` to `36` (`7.42x` fewer).
- Explicit copied bytes per step stayed roughly flat at `10.88 MiB`, which is why this was a launch/sync-shape fix rather than a bandwidth fix.

Toy-shape memory model (`d_model=128`, `layers=4`, `ff=256`, `vocab=256`, LoRA rank `8`):

| bucket | size |
|---|---:|
| policy params | 3.29 MiB |
| frozen ref clone | 3.29 MiB |
| optimizer trainable state (`param + grad + m + v`) | 1.12 MiB |
| persistent total | 7.70 MiB |

## Problems

- This is a local helper bench on the train binary, not a canonical `guidellm` server sweep.
- The workspace contained unrelated local modifications outside `crates/train/` while the bench was run. I committed the code-under-test first and am recording that code sha here, but the live tree was not globally clean because those unrelated paths were user-owned and intentionally left untouched.
- The remaining dominant bottleneck is still backward fragmentation on Metal (`metal_evals_bwd=223` per step).

## Learnings

- For `train_multi_turn`, batching the rollout group matters more than optimizer tweaks once device-backed AdamW is already enabled.
- This path was not bandwidth-bound at the toy shape; it was shape-bound by repeated small forwards and host syncs.
- The same rollout-shape optimization helps CPU too, which confirms the change is backend-agnostic. Metal benefits much more because the previous per-token synchronization pattern was especially costly there.
- The next optimization target should be autograd backward fragmentation, not more rollout surgery.

## Δ vs baseline

- **Baseline source:** same-session pre-change local measurement with the exact command above. There was no prior committed wins entry because historical train-speed notes were intentionally deleted earlier on 2026-04-22.

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| metal wall (s) | 75.48 | 11.76 | -84.4% |
| metal token/s | 135.7 | 870.5 | +541.4% |
| cpu wall (s) | 193.00 | 143.76 | -25.5% |
| cpu token/s | 53.1 | 71.2 | +34.1% |

## Artefacts

- Terminal output is embedded in this entry; no separate `bench-output/` directory was created because this turn measured a local train-binary helper bench rather than the HTTP `guidellm` path.

## Notes

- Commissioning context: [rust-agent-rl-single-node.md](../../plans/rust-agent-rl-single-node.md), especially the `train_multi_turn` / multi-turn RL sections.
- What changed in code: `train_multi_turn` now batches multi-turn rollout across the group via `rollout_episode_group`, and sampling now supports one RNG per row so reproducibility stays exact.
- Follow-up: attack the `tape.backward(...)` fragmentation next; this change intentionally does not change the tracked `metal_evals_bwd` count.
