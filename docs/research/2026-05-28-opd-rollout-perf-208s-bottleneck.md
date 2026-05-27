# OPD perf axis: student_rollout is 67% of step at rollout=128

**Date**: 2026-05-28 (Claude /loop tick 7)
**Source**: `runs/2026-05-26-rollout128-v4-diverse1k-train-60/run.txt`

## Per-step breakdown — v4 train at rollout=128

Averaged over the first 60 steps' `phase_summary` lines:

| phase | mean (s) | % of step |
|---|---:|---:|
| **student_rollout** | **208** | **67%** |
| backward | 78 | 25% |
| student_forward (full-seq autograd) | 12.5 | 4.0% |
| teacher_forward_total | 12 | 3.8% |
| infer_sync (teacher) | 2.3 | 0.7% |
| infer_forward_token_logits (teacher) | 0.007 | <0.01% |
| kl_loss | 0.07 | 0.02% |
| optimizer_step + grad_clip + zero_grad | <0.025 | <0.01% |
| **TOTAL** | ~310 | 100% |

Compare to the 3-day-old memory snapshot at rollout=8:
student_rollout=47%, backward=38%, teacher_forward=8.4%, student_forward=5.7%.
At rollout=128 the rollout dominates because it scales linearly with
token count while backward scales sub-linearly (chunked KL with
chunk_size=16).

## What's actually happening in those 208s

`crates/train/src/opd.rs:1648` opens the rollout phase with:

```rust
// 1. Greedy rollout — tape disabled, no backward graph for sample tokens.
let phase_started = Instant::now();
tape.entries.clear();
tape.set_enabled(false);
```

then loops 144 times calling
`forward_rollout_cached_device_token(student, store, tape, ...)`
(`opd.rs:1684`). Each call is **one student forward at one token
position**, through the train-crate's manual Qwen35 implementation
(`crates/train/src/qwen35.rs:1892` `forward_rollout_cached_device_token_profiled`).

208 s / 144 tokens = **1.44 s/token** for a 0.8 B model.

`student_forward` (the *separate* full-sequence forward done after
rollout to compute KL gradients, `opd.rs:1786+`) does the same 143
tokens in 12.5 s = **87 ms/token**. Same model, same hardware. The
delta is per-call vs batched: a single `student.forward(input_ids_143)`
runs as one wide-matmul launch, while 144 individual decode calls each
incur:

- per-call train-crate autograd dispatch (even with tape disabled,
  the rollout `store`/`tape` state-tracking still runs)
- no CUDA-graph capture/replay (the train-crate manual Qwen35 path
  doesn't have the infer-side graph capture)
- per-step retain bookkeeping at `should_retain_rollout_step`
  (`opd.rs:1705`)

**16× per-token slowdown** vs the batched train-crate forward.

The teacher pays no such cost because it's routed through the infer
engine (`teacher_id=infer` in the run.txt config) — teacher's
12 s for 143 tokens through full-seq forward = 84 ms/token, same
order as student_forward (also batched).

## Hypothesis — what would close the gap

Route student rollout through the **infer engine**, the same way
teacher already runs. Infer has:

- CUDA-graph capture/replay for decode (`--cuda-graph=true` default)
- paged KV decode kernel optimized for autoregressive single-token append
- no autograd machinery on the rollout path
- BF16 0.8 B decode reaches ~5–10 ms/token in standalone benches
  (cf. `docs/experience/wins/2026-05-25-kv-tier-observability-serve-baseline.md`)

Expected per-token: **~10-30 ms** for 0.8 B BF16 decode through infer
(conservative — needs measurement). 144 tokens × 20 ms = **~3 s
student_rollout** vs 208 s today. **~70× faster** on this phase alone.

End-to-end step time at rollout=128 would drop from ~310 s to
~310 - 208 + 3 = **~105 s** (3× step throughput).

At rollout=256 (where v7 dryrun was killed at 19 min/step):
extrapolated current cost 208/144 × 256 = 370 s in student_rollout
out of ~520 s step = pretty bad. Same fix → 256 × 20 ms = 5 s
student_rollout, total step ~150 s. Unblocks rollout=256 entirely.

## Risk and license-or-kill

**Risk**: the rolled-out tokens must be **bit-identical** to what
the train-crate forward+argmax would have produced, otherwise the
subsequent student_forward (re-running the full sequence through
autograd) doesn't match what the rollout produced — KL would be
computed against a different trajectory.

Mitigations:
- Use **greedy argmax** decoding only (already the case at
  `opd.rs:1694` `device_argmax_token`). No sampling. Determinism is
  on the kernel side.
- LoRA weights must be **mirrored bit-identically** from the
  train-crate `student_params` into the infer engine's adapter slot
  before each rollout. Currently infer loads LoRA from disk via
  `INFER_LORA_PATH`; the train path would need an in-memory hand-off
  or a per-step write/load.
- Numeric drift between train-crate's manual BF16 ops and infer's
  TileLang/CUDA C kernels is the main hazard. If trajectories
  diverge mid-rollout the KL signal is wrong.

**Kill criterion**:
- Pass: paired step-time at rollout=128 drops from ~310 s to ≤ 150 s
  (≥ 2× speedup) AND rollout trajectory matches train-crate forward
  bit-identically on ≥ 95% of tokens at step 1 (sampling check).
- Kill: < 1.5× speedup OR trajectory match < 90% at step 1.
- Action on kill: keep current train-crate rollout, investigate the
  per-call autograd overhead inside the train-crate path (might be
  fixable without crossing the train↔infer boundary).

## Sequencing vs the effect-axis null verdict

The 5-seed paired analysis (this session, tick 6) says current OPD
gives **null capability effect**. Perf optimization alone doesn't
change that — but it does:

1. **Unblock more training**: faster step → can run ≥10× more
   steps in the same wall-clock. Tests the "60 steps was too few"
   hypothesis cheaply.
2. **Unblock rollout=256+**: where the per-token effect might show
   up (longer rollouts → more student-on-policy signal).
3. **Unblock multi-seed-from-the-start training**: 3 train runs at
   different seeds × ~5 h each = ~15 h at current speed, vs ~5 h
   total at the post-fix speed.

So the perf fix is on the critical path to actually testing whether
*any* OPD config gives a non-null effect at this scale. It is not
itself a capability win.

## Not in scope here

- Implementation. Crossing the train↔infer boundary for per-step
  LoRA hand-off is a substantial refactor (>5 files), needs
  approach-first per CLAUDE.md.
- Backward optimization (78 s, 25% of step). Worth a follow-up but
  the rollout dwarfs it.
- Different decode (sampling, beam) — out of OPD scope for now.
