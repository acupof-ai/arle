# OPD writeback single-GPU memory plan — 40960 → 128K, no precision/attention/sequence change

> **Superseded (2026-07-29)** by
> [`docs/plans/2026-07-29-single-gpu-256k-training-repair.md`](../plans/2026-07-29-single-gpu-256k-training-repair.md),
> the single current capability page. The 128K figures below are pre-repair
> estimates; the measured single-GPU ladder lives in that plan's T5.
>
> Approach doc. Architectural (>5 files across the ladder), so this is written for
> sign-off before any code. Grounded in a code-level audit (file:line below) and
> the 2026 SOTA for long-context single-GPU training. Nothing here changes
> precision, attention windows, effective sequence, or the mask/loss.

## Problem, in one line

On a clean HEAD build, seq=40960 masked-CE writeback OOMs on a single H20 (97.5 GB)
in **every** config — it is a real capacity wall, not mempool, not a regression.
The wall is a **single-layer forward transient peak** that eats 58 GB → 1.6 GB in
one checkpoint group.

## Why it happens (code-confirmed)

`checkpoint_sequential` runs each group's forward **tape-disabled** inside a
closure (`crates/autograd/src/ops/checkpoint.rs:27`). Every intermediate is a pure
transient — backward will recompute, not read it — yet they are **all held live
simultaneously** and freed only once, at closure exit
(`checkpoint.rs:40`, `free_new_except`). At long seq `ckpt_group_size()=1`
(`qwen35.rs:292-303`), so the group is already one layer; the boundary cannot cut
below a layer. Result: peak = the **sum** of one layer's intermediates.

For the OOMing layer (group 3 = layer 2, a GDN linear-attention layer + dense MLP),
that sum at seq=40960:

| block | tensors | GB |
|---|---|---|
| GDN core scratch (forward-only, freeable) | q/k/v/g_cumsum/a_inv/raw_output | ~2.3 |
| dense-MLP SwiGLU | gate + up + silu + mul, each `[40960,17408]` f32 = 2.66 | 10.6 |
| all-linear LoRA tax | per projection: `low_rank` + `delta` + `add`, ~3× the projection intermediates | ~+19 |

all-linear dies in **forward** on the SwiGLU `mul [40960,17408]` (2.66 GB) with
1.6 GB free. attention-qv (fewer LoRA tensors, live 922 vs 1852) survives forward
and dies in **backward** on the full-attention gated-Q grad `add_into
[1,40960,24,512]` (1.9 GB).

## What's already at SOTA (do not rebuild)

**Cut Cross-Entropy is already present.** `fused_linear_ce_loss_indexed_device`
(`crates/autograd/src/ops/fused_linear_distill.rs:589-661`) fuses the final linear +
log_softmax + CE, chunks masked positions at `chunk_rows=32`, and frees each
`[32, vocab]` tile at line 660. The dense `[40960, 248320]` f32 = 40.68 GB logits
tensor is **never materialized**. The loss head is not the OOM site — no work needed.

## SOTA reference (2026)

- **Cut Cross-Entropy** (Apple, ICLR 2025 Oral) — fused chunked linear+CE, never
  materialize `[seq,vocab]`. *Already in ARLE.*
- **Tiled MLP** (Snowflake *Arctic Long Sequence Training* → Unsloth 2026) — tile
  hidden states along seq **before** the MLP projections; `num_shards =
  ceil(seq_len / hidden_size)` (40960/5120 = 8), written as a nested checkpoint;
  −40% single-layer VRAM; gpt-oss-20b 290K → 500K on one H100. This is the
  structural fix for the SwiGLU peak.
- **Selective activation recomputation** (Megatron, arXiv:2205.05198) — recompute
  cheap-but-heavy ops in backward instead of full-block checkpointing.
- **Enhanced checkpoint offload** (Unsloth) — offload activations *as soon as
  produced* via CUDA streams (+0.1% overhead), vs ARLE's free-only-at-closure-exit.

## Gap table — ranked by GB-saved per unit effort (seq=40960)

| # | Move | Status | Site | GB | Which peak | Effort |
|---|---|---|---|---|---|---|
| 1 | **Intra-closure as-you-go freeing** (Tiled-MLP's liveness idea, minimal form) | GAP | `qwen35.rs:1173-1177`, `lora.rs:243-248`, gated `!tape.enabled` | **~19-22** | **FORWARD (the OOM)** | LOW — `store.free()` calls, no kernel |
| 2 | GDN dead-store elimination | PARTIAL (backward already recomputes; forward still saves) | `linear_attention.rs:939-947` | 2.27 | backward | LOW — struct fields → None |
| 3 | GDN preact+qkv_conv recompute in backward | GAP | `backend_cuda.rs:4558` | 2.52 | backward | MED — 1 conv1d+silu pass |
| 4 | chunk_state N-stride coarsening | GAP | `linear_attention.rs` | ~1.5 (N=4) | fwd+bwd | MED — tape param + mini-scan |
| 5 | **Fused chunked SwiGLU** (Liger/Tiled-MLP structural) | GAP | `qwen35.rs:1175-1176`; bf16 inference kernel exists `split_qkv.cu:35`, not wired to training autograd | ~7.8 @40960, **O(chunk) seq-independent** | FORWARD | HIGH — f32 kernel + autograd op + backward + Metal |
| 7 | Cut Cross-Entropy | **ALREADY HAVE** | `fused_linear_distill.rs:589` | 0 add'l | — | N/A |

## The plan (ordered)

### Move 1 — intra-closure as-you-go freeing (ship first)

The highest-leverage, lowest-entropy change, and it hits the actual OOM (forward).

- **`crates/train/src/qwen35.rs:1173-1177` (Dense SwiGLU):** after `silu(gate)→gate2`
  free `gate`; after `mul(gate2,up)→act` free `gate2` and `up`; after
  `down_proj(act)` free `act`. Do **not** free `h` (feeds gate_proj, up_proj, and
  the residual).
- **`crates/train/src/lora.rs:243-248` (`LinearWithLora::forward`):** after
  `add(projected, delta)→out` free `projected` and the `low_rank`/`delta`
  intermediates.
- **Gate every free on `if !tape.enabled`.** This makes it a strict no-op on the
  backward-replay pass (backward runs a fresh `inner_tape` with `enabled=true`,
  `tape.rs:391/897`, and must keep intermediates for `backward_collect`) and on the
  short-seq non-checkpointed path. `store.free` (`tensor.rs:323`) just returns pages
  to the pool. No kernel, no tape/SavedContext change, no numeric change.

**Does 40960 fit after Move 1?** Yes, for the binding (forward) constraint:

```
persistent floor F ≈ 97.5 − 58        = 39.5 GB  (FP8 weights ~27 + grad_hidden 0.84 + saved-input set + optim/LoRA)
forward transient  T (today)          ≈ 56.4 GB  → peak 95.9 GB, right at the wall → OOM
Move 1: all-linear MLP+LoRA transient 29.7 → ~10.6 GB (−19); GDN fwd scratch −~3
T → ~34 GB  ⇒  peak ≈ 39.5 + 34       = 73.5 GB  ⇒  headroom ~24 GB. Fits.
```

Narrowest version (free silu only, −2.66 GB) fits but fragile (~4 GB margin) — do the
full LoRA+SwiGLU as-you-go for the 24 GB margin.

**Caveat:** Move 1 is `!tape.enabled`-gated, so it does **not** fire in backward. It
does not touch the attention-qv backward gated-Q death (1.9 GB, full-attention
layer). Whether that now fits needs a measured backward peak (see Unknowns) or Moves
2-3.

### Moves 2-4 — GDN backward high-water (do together, all in GDN core)

- **2. Dead-store elimination** — `q/k/v/g_cumsum/a_inv/raw_output` are still saved
  in `LinearAttentionCtx` (`linear_attention.rs:939-947`) but backward already
  regenerates them from `qkv_conv` (`backend_cuda.rs:4576-4593`). Set them to `None`.
  −2.27 GB, zero recompute. (A dead-store fix, not new recompute.)
- **3. preact + qkv_conv recompute** — regenerate both from the saved `qkv` via one
  conv1d+silu pass at the head of `cuda_linear_attention_backward_device_row`
  (`backend_cuda.rs:4558`); drop their forward saves. −2.52 GB backward.
- **4. chunk_state N-stride** — store every N-th 64-tok boundary; replay a
  per-super-chunk `gated_delta_rule_prefill_chunk_state` mini-scan in backward.
  N=4 → −1.5 GB, also lowers the forward closure peak.

### Move 5 — fused chunked SwiGLU (the structural move for 128K)

Move 1 is O(seq): at 128K, `gate`+`up` co-live at up_proj = 2×8.5 = 17 GB even after
freeing. The structural fix is **Tiled MLP**: loop gate/up/silu/mul along seq in
chunks (mirror the existing `fused_linear_ce_loss_indexed` chunk-tape at
`fused_linear_distill.rs:589`), making the MLP transient **O(chunk) ≈ 0.2 GB,
seq-independent**. Extend the chunk to enclose the LoRA matmuls to kill the LoRA tax.
HIGH effort (f32 `silu_mul` kernel — adapt bf16 `split_qkv.cu:35` — + autograd op +
backward + Metal), mandatory for 128K, not needed for 40960.

## Path to 128K (the ladder)

131072 = 3.2× of 40960; forward transient scales linearly, so Move 1 alone OOMs
again past ~50K. Order of attack:

1. Move 1 — necessary at every length, insufficient past ~50K.
2. Move 5 (fused chunked SwiGLU) — makes the MLP transient seq-independent. The
   structural unblock.
3. Moves 2-4 (GDN backward) — scale with seq (chunk_state at 128K ≈ 6.4 GB/layer).
4. **`offload_checkpoints` ON + host-RAM budget.** The saved-input set = hidden
   `[1,131072,5120]` f32 = 2.68 GB × 64 layers ≈ **172 GB** — must live on host (path
   exists, `checkpoint.rs:50-57` / L3 tiering). This is the real 128K floor, a
   resident set, not a transient. Verify host RAM + accept PCIe traffic.
5. Attention backward liveness for full-attention layers (the gated-Q path); at 128K
   ~6 GB. A genuine backward-side liveness change (not `!enabled`-gated).

## Unknowns — need a measured run before/with implementation

`ARLE_OPD_VRAM_TRACE` is built in (`[ckpt-group-vram]` per group `checkpoint.rs:164`;
`[autograd-op-mem]` per op `tape.rs:997`). Before trusting the arithmetic:

1. Exact 56.4 GB decomposition on the OOMing layer — does all-linear also wrap the
   GDN projections (extra 3× tax there)? Op-mem trace on that layer.
2. True as-you-go co-live peak after inserting frees — read off the trace, not the
   static tally.
3. Does Move 1 alone clear the attention-qv backward `add_into` (1.9 GB)? Backward is
   tape-enabled, Move 1 doesn't fire there — measure post-fix backward peak.
4. Does backward become the new binding constraint after Move 1? Full fwd+bwd at
   40960 with the trace on.
5. Host-RAM feasibility + PCIe cost of the 128K ~172 GB saved-input set.

## Verify

Per the bench gate: each runtime change lands a dated `docs/experience/wins/` (or
`errors/`) entry with a matched before/after at seq=40960 (and the length ladder for
128K). Correctness gate = the needle/lever gate ×3 same-config vs baseline; masked-CE
writeback must produce the same loss on a shorter runnable sequence (Move 1 is
numerically a no-op, so loss must be bit-identical).

## Bottom line

Move 1 (a few `store.free()` calls, gated `!tape.enabled`, no kernel) makes 40960 fit
with ~24 GB headroom. 128K needs the structural fused-chunked SwiGLU (Move 5) plus
host offload of the saved-input set. CCE is already done; the loss head is not the
wall. Nothing in this plan touches precision, attention, sequence, or the mask/loss.
