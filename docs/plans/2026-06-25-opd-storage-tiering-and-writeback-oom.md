# OPD storage tiering + the 32K writeback OOM

Agent-OPD trains Qwen3.6-27B-FP8 LoRA in-process: a rollout engine generates,
the autograd student writes back passing trajectories as masked CE. At ~32K-token
trajectories the writeback OOM'd (`cuda alloc_zeros failed (add_into_device)`).
This maps every OPD-chain GPU/host/disk allocation by **access frequency** —
hot (per-token/step) → HBM, warm (write-once-read-once) → host RAM (L2), cold
(rare) → SSD (L3) — and pins the OOM to one wrong placement, now fixed.

## The tiering map (grounded in code)

| Tier | Item | Size @27B/32K | Access | Status |
|---|---|---|---|---|
| **HBM** (per-token/step hot) | FP8 base weights | 27 GB | every forward | ✅ engine+student share one copy zero-copy (`SharedFrozenBaseEntry`) |
| | active KV pages | ~0.3 GB/seq | per token | ✅ tiered (`CudaKvTierStore`) |
| | active recurrent state (linear-attn) | ~63 MB/slot | per token | HBM (idle→L2 candidate) |
| | LoRA adapters | 248 MB | every fwd/bwd | ✅ HBM |
| | optimizer m,v / grads | ~1 GB | per step | ✅ HBM (small) |
| | student embedding (FP32, tied) | 3.1 GB | every forward | ✅ HBM |
| | **current window's live activation** | **~8 GB/layer** | **in compute** | **OOM source — see below** |
| **host RAM** (L2, write-once-read-once) | grad-checkpoints (layer inputs) | ~30 GB | fwd write, bwd read | ✅ offloaded (`offload_to_host`, confirmed `cuMemFree`) |
| | evicted KV pages | — | occasional | ✅ L2 (4 GiB default) |
| | idle recurrent state | ~8 GB @full conc. | released, idle | ❌ candidate (`qwen35.rs:~519` notes future L2 spill) |
| **SSD** (L3, cold) | long-idle KV | — | rare | ✅ L3 (`--kv-ssd-path`) |
| | decode-graph images | 6–12 GB | per slot | ❌ candidate |
| | PEFT adapter saves / dataset / sandbox | — | per round/task | ✅ already on disk |

Resident HBM is only **~31 GB** (weights + student). The cold data is already
correctly tiered (host/SSD). So the OOM is **not** a tiering gap.

## Root cause of the OOM — and why "checkpoint → SSD" wasn't the fix

The OOM is the **hot, in-compute forward activation** of the writeback, which
cannot be tiered (it is being computed). Two corrections to earlier guesses:

- **Not the gradients.** `add_into_device` names the failing op, but base /
  lm_head / embedding are **frozen** — only LoRA (~500 MB) has gradients. The
  alloc fails because the GPU is already full, not because the alloc is big.
- **Not the offloaded checkpoints.** They're written-once / read-once-in-backward
  = warm → host RAM (L2) is the right tier; SSD would only slow the backward
  fetch. And they're already off-GPU (offload confirmed to `cuMemFree`).

The real consumer: **group gradient-checkpointing held all K layers' live
activations.** A checkpoint runs its group's forward with the tape disabled and
frees intermediates only at the end (`free_new_except`), so a group accumulates
`K × per_layer_activation` where per-layer ≈ `seq × (hidden + 3×intermediate) ×
4B` ≈ 8 GB at 32K. Fixed `CKPT_GROUP=4` ⇒ ~48 GB live ⇒ 31 + 48 + logits/CE ≈
OOM. (`crates/train/src/qwen35.rs` `checkpoint_sequential` caller.)

**Fix (committed `e7e5a334`):** `ckpt_group_size(seq_len, hidden, intermediate)`
bounds a group's live activation to ~8 GiB — 1 layer at 32K (peak ≈ one layer,
~41 GB total), up to 8 at short seq (keeps the host-offload/PCIe win). Mirrors
the adaptive head-chunk (`02d05d8e`). Grouping was a PCIe optimization that hurt
the binding constraint (GPU peak) at long seq; adaptive sizing honors both.

## Open candidates (not on the OPD writeback critical path)

KV-side L2/L3 already exists. The next tiering levers are ckl's KV territory:
idle-recurrent-state spill to L2 (~8 GB at full concurrency) and decode-graph
images to L3. Neither blocks the writeback.
