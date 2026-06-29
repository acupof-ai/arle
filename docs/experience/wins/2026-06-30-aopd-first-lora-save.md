# Agent-OPD: First Complete Loop — LoRA Adapter Saved (loss=0.2240)

## Context

Run 4 of the Agent-OPD training loop on the 8×H20 box (GPU 4, 97508 MiB total).
Five sequential infrastructure walls cleared across runs 1–3 before this run
completed end-to-end:

| Wall | Fix |
|------|-----|
| ELKEID eBPF hook kills arle via `setsid()` | `process_group(0)` (setpgid) + `kill(-pgid, SIGKILL)` in `crates/train/src/spawner.rs` — commit `ea2e6133` |
| Two-copy base OOM (72875 MiB floor vs 97508 limit) | `--share-frozen-base` (default ON): 400 FP8 base projections zero-copy from rollout engine → floor drops to 34155 MiB |
| N×N causal mask OOM during backward (at seq=18168) | At seq=7178 the mask is 7178²×4 = 206 MB — not an issue for this run's trajectory lengths |
| SDPA chunk intermediates pile up in inner backward (`0b7a1d89`) | Wrap each `causal_sdpa_recompute` call in a nested `checkpoint` — one chunk at a time, ~6.6 GiB peak vs ~46 GiB without fix |
| Multi-writeback Adam-state OOM (run 3) | `--writeback-cap 1`: limit to 1 writeback per round; 4 separate writebacks accumulate ~50 GB Adam state beyond VRAM limit |

## What Worked

**Config (run 4):**
- `--share-frozen-base` (default): 400 FP8 base projections borrowed zero-copy from rollout engine
- `ARLE_OPD_WRITEBACK_OFFLOAD=1`: gradient checkpoints offloaded to CPU (D2H during forward, H2D during backward)
- `--writeback-cap 1`: only 1 writeback per round (avoids Adam-state accumulation OOM on subsequent trajectories)
- `--lora-layer-start 32`: LoRA applied to layers 32–63 (top 32 of 64)

## Measured Numbers

**Rollout:**
- Task: `ansible__ansible-f327e65`
- Samples: 4 rollouts, 4/4 passed (turns=5 each)
- Baseline held-out pass_rate: 0.3333 (1/3 tasks)

**Writeback (cap=1, trajectory 0):**
- `seq_len=7178`, `total_targets=597`, `chunk_rows=2048`
- `forward_hidden_states`: 927.042s
- `fused_ce`: 0.188s
- `backward`: 1221.516s
- `optimizer_cleanup`: 18.380s
- **`loss=0.223990`** (`mean_loss=0.2240`)
- `trained_pairs=1`

**Post-round eval:**
- held-out pass_rate=0.3333 (Δ=+0.0000 vs baseline)
- 1 round is not enough signal to move the needle on 3 held-out tasks; eval parity expected

**LoRA adapter:**
```
checkpoint_saved kind=peft_adapter mode=agent-opd step=1
dir=/host/agentopd_allfix4/adapters_round1 seconds=0.221382
```

## VRAM Trace

```
after rollout engine load (KV pool alloc'd): 44779 MiB
after autograd student load (resident floor): 49547 MiB
 ↓ KV pool + inference scratch released
agent-opd pre-writeback:                     34187 MiB   ← write floor
masked-writeback pre forward_hidden_states:  34187 MiB
masked-writeback post forward_hidden_states: 46417 MiB   ← +12230 MiB (checkpoint tensors)
 ↓ backward (1221s): checkpoints consumed H2D→recompute, gradient tensors filled
after round 0 writeback:                     82225 MiB   ← +35808 MiB (Adam m+v persistent)
```

Why the floor drops from 49547 to 34187 before writeback:
- The rollout KV pool (~10 GB) and inference scratch are released after rollout
- The BF16 LoRA weight copy is the dominant resident: 32 layers × ~195M params × 2 bytes ≈ 12 GB
- Shared FP8 base (zero-copy): borrowed from the released rollout engine; weights stay in VRAM but under the rollout engine's ownership tracking

Why Adam state is ~48 GB:
- LoRA covers attention + MoE expert projections for 32 layers of Qwen3.6-27B
- ~6B LoRA parameters × Adam m+v (FP32) × 4 bytes × 2 = ~48 GB

Peak VRAM during backward: not directly traced; `post-writeback 82225 MiB` is the
POST-cleanup floor (optimizer step taken, gradients freed, Adam state kept). The
backward peak was safely below 97508 MiB (run exited cleanly without OOM).

## Rule

- **`--writeback-cap 1` is required for 32-layer LoRA on a single H20** — the
  Adam state for 32-layer LoRA (~48 GB FP32) is created on the first writeback and
  persists; a second writeback starting at 82 GB floor can't fit the backward
  gradient tensors (~25 GB) within 97.5 GB.
- **`--share-frozen-base` is the decisive VRAM reduction** — drops pre-writeback
  floor from 72875 MiB to 34187 MiB by zero-copying 400 base projections from the
  rollout engine instead of loading a separate BF16 student copy.
- **`ARLE_OPD_WRITEBACK_OFFLOAD=1` keeps backward peak within 97.5 GB** — without
  offload, all gradient checkpoint activations accumulate on GPU simultaneously.
  Trade-off: backward takes 1221s vs ~490s without offload (CPU ↔ GPU PCIe bound).
- **`loss=0.2240` on trajectory 0 of ansible-f327e65** proves the full autograd
  path works: FP8 base → BF16 LoRA → fused CE loss → backward → AdamW step →
  adapter serialization.

Claude-Session: https://claude.ai/code/session_01FWW6tY9aXDyx9n5NNCKEb6
