# agent-OPD writeback OOM closed — root cause was TWO checkpoint-forward VRAM leaks (frozen-group hidden pin + per-chunk SDPA pileup), not the per-group accumulation alone

## Context

Mainline: close the agent-OPD (train-infer-unified) loop end-to-end. Blocker was
the masked-CE writeback forward OOM at seq ~16–20K on the 8×H20 box
(Qwen3.6-27B-FP8, `--lora-layer-start 32`, `student_seq=32768`). The prior agent
([errors/2026-06-26-...accumulation-is-real-wall](../errors/2026-06-26-agent-opd-kv-pool-freed-but-forward-accumulation-is-real-wall.md))
landed the KV-pool release lever (−19.8 GB, verified) and attributed a per-group
`+23 device-tensors / +200 MiB` accumulation + a `+15 GB` final-group spike, but
did not root-cause to a fix. This entry closes the loop.

## What Worked

Decoded the survivors **per device allocation** (deduped by `Arc` storage pointer,
true element width — not the `size×4` aggregate that over-counts FP8 as f32) with a
gated `ARLE_OPD_VRAM_TRACE` probe dumping `device_survivors_by_alloc` after each
checkpoint group's `free_new_except`. The wall was **two distinct leaks**, both in
the checkpoint-forward path under `tape.offload_checkpoints`:

**Leak 1 — frozen-group hidden pin (the `+312 MiB/group` accumulation).** Measured at
seq=8000: every group's post-free device residency grew `+156 MiB` (one extra
`[1, seq, hidden]`) — group1 2 hiddens → group2 3 → group3 4, monotonic over the
frozen prefix. Cause: for a FROZEN group (`requires_grad=false`, layers 0–31 under
`--lora-layer-start 32`), `checkpoint()` skips the whole `if requires_grad` block →
no `offload_to_host`, no tape entry. The input hidden (= prior group's output) sits
in `keep` here (it's the saved input) and then in EVERY later group's `live_before`,
so no `free_new_except` ever reclaims it. **Fix:** for a frozen group with offload on,
drop the input hidden's device residency (`TensorStore::drop_device_residency` — no
host readback, the frozen hidden has no backward replay). Gated to the offload path;
default forward byte-identical.

**Leak 2 — per-chunk SDPA pileup (the binding `+43 GB single-group` wall).** Even with
Leak-1 fixed and `group_size→1`, a single `full_attention` layer's forward jumped
+43–50 GB → OOM. Cause: `head_chunked_sdpa_recompute` loops over head-chunks, but
each chunk's `causal_sdpa` allocates 4 simultaneous `[chunk, seq, seq]` buffers
(scores/scaled/masked/probs) into the store, and with `tape.enabled=false` (recompute
inside the checkpoint forward) NOTHING freed them between chunks — they piled up
across the head loop (~5 chunks × ~20 GB ≈ 100 GB at seq=16000). The `ckpt_group_size`
budget also under-counted: it modeled only the MLP `seq×(hidden+3×intermediate)` term
and ignored the ~12 GiB attention transient, returning group_size=2 where even 1
didn't fit. **Fix:** (a) free each head-chunk's transients before the next chunk when
tape is disabled (heads independent ⇒ numerically exact); (b) chunk budget accounts
for the 4 live `[chunk,seq,seq]` buffers (`8GiB/(4×seq²×4)`), so chunk→1 at long seq;
(c) `ckpt_group_size` adds the ~12 GiB attention floor → group_size→1 at seq≳8000.

## Verification (8×H20, GPU 6/7, Qwen3.6-27B-FP8, lora-layer-start 32, window 512)

Synthetic-writeback (`--synthetic-writeback-seq`), `ARLE_OPD_VRAM_TRACE=1`,
post-KV-release floor = 38987 MiB:

| metric | BEFORE (prior wall) | AFTER (both fixes) |
|---|---|---|
| seq=16000 group-2 forward peak | 97487 MiB → **OOM** | bounded ~53007 MiB |
| per-group post-free device | +156 MiB/group (unbounded) | **flat 10303–10801 MiB** |
| seq=16000 full 64-group forward | OOM at group 3 | **completes, `post forward_hidden=41999 MiB`** |
| frozen groups (fns=0) → grad groups (fns→32) | n/a | both phases bounded, recorded |
| **masked-writeback loop close (seq=40)** | n/a | **`DONE loss=5.203334`, `post-writeback used=40335 MiB`, 0 OOM** |

**Loop closes** (seq=40 synthetic-writeback, GPU 1): full forward (64 groups: 32
frozen `fns=0` + 32 grad `fns→32`) + chunked-CE + backward + AdamW step complete
end-to-end — `[synthetic-writeback] DONE loss=5.203334 elapsed=493.7s`, device
residency flat at ~10.2 GiB through the whole pass, `post-writeback used=40335 MiB`
(+1.3 GiB over the 39 GiB post-KV-release floor), zero OOM. The host-side chunked-CE
(`fused_linear_ce_loss_indexed`, scalar loop over vocab=248320×hidden=5120 per
target) is ~20 s/target single-threaded — so seq=16000's 15744-target CE is a ~hours
host compute (a SEPARATE pre-existing perf issue, NOT the OOM); the seq=16000 FORWARD
(the OOM wall) completing all 64 groups is the binding proof, and seq=40 closes the
full forward+CE+backward loop with a finite loss.

Numerical exactness preserved: `qwen35_gradient_checkpointing_lora_finite_diff_gate`
+ `head_chunked_sdpa_matches_unchunked` (now also asserts per-chunk transients freed)
+ `checkpoint_frozen_group_offload_drops_input_hidden_device_residency` /
`...default_path_keeps_device_residency` all pass; `cargo test -p autograd` (23) +
`-p train` (153) green; clippy clean.

## Rule

A "barely over" OOM premise (or even a precisely-attributed `+200 MiB/group`
accumulation) can hide a SECOND, larger leak — decode survivors **per allocation**
(Arc-deduped, true dtype width), not by aggregate bytes, and verify on the
**production seq** (the prior wall), not a smoke shape. Here the frozen-hidden pin
(`+156 MiB/group`) was real but NOT binding; the binding wall was per-chunk SDPA
transients piling up under tape-disabled recompute (4×`[chunk,seq,seq]` × N_chunks).
Both live in the same `tape.offload_checkpoints` path; both are gated so the default
short-seq forward stays byte-identical. The `free_new_except(live_before, keep)`
contract leaks any tensor that survives one group via `keep` and then lands in the
next group's `live_before` — for frozen groups with no backward, drop it explicitly.
