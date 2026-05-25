# Sequence-Windowed Forward Design

Status: design only. Do not implement until ckl approves one route.

Related evidence:

- [`docs/research/2026-05-24-bf16-frozen-base-impl-path.md`](../research/2026-05-24-bf16-frozen-base-impl-path.md)
- [`docs/experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md`](../experience/errors/2026-05-24-gkd-real-corpus-tape-oom-kill.md)
- [`docs/experience/errors/2026-05-25-chunked-kl-real-corpus-512-kill.md`](../experience/errors/2026-05-25-chunked-kl-real-corpus-512-kill.md)
- [`docs/experience/errors/2026-05-25-gkd-real-corpus-256-mitigation-kill.md`](../experience/errors/2026-05-25-gkd-real-corpus-256-mitigation-kill.md)
- [`docs/research/2026-05-24-arle-opd-end-to-end-trace.md`](../research/2026-05-24-arle-opd-end-to-end-trace.md)

## Problem

T5a made KL intermediates chunkable, but T5b and T15 proved the real OOM is
earlier: teacher/student forward still materializes full `[B, S, V]` logits
before KL receives them. A real fix must make the forward producer emit and
consume sequence windows.

For Qwen3.5 0.8B/4B in this repo, `V = 248_320` from the local configs. A
single f32 logits tensor costs:

```text
bytes(S) = 1 * S * 248_320 * 4
```

| S | one logits tensor | teacher + student logits only |
|---:|---:|---:|
| 512 | 508,559,360 B = 485.0 MiB | 970.0 MiB |
| 256 | 254,279,680 B = 242.5 MiB | 485.0 MiB |
| 520 = 512 + rollout 8 | 516,505,600 B = 492.6 MiB | 985.2 MiB |
| 264 = 256 + rollout 8 | 262,164,480 B = 250.0 MiB | 500.1 MiB |

Those are lower bounds. Baseline KL also creates `softmax(teacher)`,
`log_softmax(student)`, and `weighted`; chunked KL shrinks those later
intermediates but not the full logits already resident.

## Current Inventory

| Surface | Current path | Allocation impact |
|---|---|---|
| Eval KL | `mean_prompt_kl` clears the tape, builds full prompt positions, then calls `teacher.forward_logits_device(prompt, &positions, ...)` and `student.forward(..., prompt, &positions)` before KL (`crates/train/examples/opd_step_cuda_infer_teacher_train.rs:990-1011`, exact full-forward calls at `:994-995`). | At prompt cap 512, at least two 485 MiB logits tensors per prompt before any chunked KL. T15 shows cap 256 still fails before `eval_summary step=0`. |
| Train rollout KL | OPD builds `positions = 0..rollout.len()`, calls full teacher scoring, validates `[1, rollout.len(), vocab]`, then calls full student forward before `kl_distill_loss_for_config` (`crates/train/src/opd.rs:1095-1135`, full teacher/student/KL sequence at `:1097-1124`). | At 512+8 rollout shape, teacher+student logits alone are about 985 MiB. |
| Corpus-truth SFT anchor | `corpus_truth_sft_loss` concatenates `prompt_ids + corpus_tokens`, builds positions for the whole sequence, and calls `student.forward` once (`crates/train/src/opd.rs:643-674`). | Even if KL is windowed, real GKD can still full-materialize `[1, prompt+completion, V]` student logits. T16 must cover this path too. |
| Full Qwen35 forward | `Qwen35Model::forward` calls `forward_batch(..., seq_len=position_ids.len())` (`crates/train/src/qwen35.rs:2129-2137`). Normal batch forward embeds `[batch, seq_len, hidden]`, runs every layer, final RMSNorm, then `linear_forward(hidden, lm_head)` over the full sequence (`crates/train/src/qwen35.rs:1903-1938`). | The LM head is the point where hidden-size memory becomes vocab-size memory. A windowed path should slice hidden before lm_head, not slice logits after lm_head. |
| In-process teacher | `InProcessTeacher::forward_logits_device` delegates to `Qwen35Model::forward` and ensures the full logits tensor is device-resident (`crates/train/src/teacher_infer.rs:513-528`). | The in-process teacher has the same full-logit behavior as the student. |
| Chunked KL | `kl_distill_loss_chunked` loops over seq chunks but first receives already-full logits; its core chunking is `slice(teacher_logits)` and `slice(student_logits)` (`crates/train/src/loss.rs:91-125`, exact slices at `:109-110`). | This reduces KL intermediates only. CUDA slice backward allocates a zero grad with the full input shape (`crates/autograd/src/backend_cuda.rs:5500-5534`), which explains T5b's `c1` slice failure. |

## Route Options

### Route A: Add `window_range` To Existing Forward APIs

Shape:

```rust
forward(..., input_ids, positions, window_range: Option<Range<usize>>) -> TensorId
```

Pros:

- One canonical forward surface.
- Existing callers can pass `None`.
- Easy to enforce shape: `None => [B, S, V]`, `Some(a..b) => [B, b-a, V]`.

Cons:

- High churn: every current `Qwen35Model::forward`, `CausalLm`, teacher, test,
  and helper caller must be audited.
- Easy to create half-states where some callers pass `None` and still OOM.
- API says "forward", but the causal implementation must still consume prefix
  `[0..window.end)` while only projecting hidden positions in `window`.

Estimated touch: 8-12 files.

### Route B: New `SequenceWindowedForward` Side Trait (Recommended)

Shape:

```rust
struct SequenceWindow {
    start: usize,
    end: usize,
}

trait SequenceWindowedForward {
    fn forward_logits_window(
        &self,
        store: &mut TensorStore,
        tape: &mut Tape,
        input_ids: &[u32],
        position_ids: &[u32],
        window: SequenceWindow,
    ) -> Result<TensorId>;
}

trait TeacherWindowedForward {
    fn forward_logits_window_device(
        &self,
        input_ids: &[u32],
        position_ids: &[u32],
        window: SequenceWindow,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<DeviceLogits>;
}
```

Implementation rule:

- For causal correctness, each window forward receives the full context needed
  for the window, normally tokens/positions `0..window.end`.
- It runs layers over that prefix, slices hidden to `window.start..window.end`,
  then applies `lm_head` only to the hidden window.
- Do not compute full logits and slice them. Slicing hidden has input shape
  `[B, S, H]`; slicing logits has input shape `[B, S, V]` and repeats the T5b
  failure mode.

Pros:

- Existing `forward` stays unchanged for baseline parity and non-GKD code.
- The route is explicit at callsites: eval/GKD must opt in to a window size.
- It matches T5a: keep baseline path alive, add a memory-saving sibling, prove
  parity before switching defaults.

Cons:

- Adds a second forward contract that must stay shape-compatible with
  `TeacherForward`.
- Needs new cleanup discipline because the caller must backward/cleanup per
  window.
- API-teacher fallback may still full-materialize unless separately
  implemented; the 16 GB acceptance should require the in-process teacher
  windowed path first.

Estimated touch: 5-8 files.

### Route C: Change KL To Accept Per-Window Logit Closures

Shape:

```rust
kl_distill_loss_windowed(num_positions, window_size, |range| -> (student, teacher))
```

Pros:

- KL callsite makes the producer/consumer coupling impossible to miss.
- The loss function could reuse T5a chunk weighting directly.

Cons:

- If the function returns one final `TensorId` and the caller backprops once,
  it retains every window graph and is not a memory fix.
- If the function calls `backward` internally per window, it stops being a loss
  function and starts owning optimizer/tape lifecycle.
- It does not naturally cover the `corpus-truth` SFT anchor, which is not KL.

Estimated touch: 3-6 files for a prototype, but high semantic risk. Not
recommended for mainline.

## Recommended Design

Use Route B plus an explicit windowed train/eval loop.

1. Add a default-off `--logits-window-size N` CLI flag. Omitted means current
   full-logit behavior.
2. Add `SequenceWindowedForward` for `Qwen35Model`.
3. Add `TeacherWindowedForward` for `InProcessTeacher`.
4. Add windowed eval KL: each prompt loops windows, computes teacher/student
   window logits, computes weighted KL for that window, reads the scalar, and
   cleans temporaries before the next window.
5. Add windowed OPD KL: rollout generation stays unchanged; scoring uses
   windowed teacher/student logits.
6. Add windowed corpus-truth SFT: window only the completion-scored next-token
   positions, weighting by target-token count so the CE scale matches the
   full-sequence baseline.
7. Keep the old full-logit path and run parity tests before making any default
   switch.

## Backward Graph

The hardest part is not the forward API; it is where the graph is allowed to
live.

Do not accumulate one scalar loss across all windows and call `tape.backward`
once. That retains every per-window graph until the end and defeats the memory
goal.

The windowed path should instead:

1. `optimizer.zero_grad(...)` once at the start of the OPD step.
2. For each KL/SFT window, build only that window's loss with the correct
   global weight.
3. Call `tape.backward(window_loss, store)?` immediately.
4. Clean temporaries for that window while retaining model params and their
   `.grad` tensors.
5. After all windows, run grad clip and one optimizer step.

This is compatible with current autograd. `Tape::backward` walks only the graph
relevant to the provided loss (`crates/autograd/src/tape.rs:262-358`).
Gradients reaching the same tensor are merged during backward
(`crates/autograd/src/tape.rs:401-468`), and persistent parameter grads are
accumulated in `TensorStore::accumulate_grad` (`crates/autograd/src/tensor.rs:332-407`).

Scale rules:

- KL window loss weight: `window_positions / total_kl_positions`, same scale
  as T5a's chunked KL weighting (`crates/train/src/loss.rs:91-125`).
- Corpus-truth SFT window loss weight: `target_tokens_in_window /
  total_target_tokens`.
- GKD mixing can backward KL and SFT component windows separately as long as
  the component weights include `(1 - lambda)` and `lambda` before backward.
  The logged step loss is the host sum of all weighted window scalars.

Eval has no backward. It can use the same producer loop, host-sum weighted KL
windows, and cleanup after every window.

## Acceptance Gates For Implementation

Correctness gates before GPU:

- Small CPU/CUDA parity: window size 1, 8, and full length match current
  full-logit KL loss and student gradient within the existing T5a epsilon.
- Corpus-truth SFT parity: windowed CE over completion targets matches the
  full-sequence SFT loss and gradients.
- Baseline omitted flag keeps current full-logit callsites unchanged.

License gates on 16 GB CUDA:

- PASS hardware: real-corpus GKD at `--prompt-max-tokens 512`, rollout 8,
  `--gkd-lambda 0.3`, `--sft-anchor corpus-truth`, and windowing enabled
  reaches `eval_summary step=0` plus at least 10 `train_step` lines.
- PASS perf: mean step time is not worse than 1.5x the P5 same-machine OPD
  anchor unless the doc explicitly licenses a slower real-corpus shape.
- KILL hardware: OOM remains before step 0 or before 10 train steps.
- KILL perf: mean step time exceeds 2x the P5 anchor.

Follow-on GKD value gate after hardware PASS: run to eval step 100+ and require
at least one heldout KL below step 0 before claiming the corpus-truth anchor has
useful signal.

T2 measured P5 prompt-16 OPD at 5.05 s mean step, with backward already 42.83%
of train wall-clock and student rollout 41.91%. Any sequence-windowed design
that recomputes prefixes must be measured against that backward/rollout budget;
it cannot hide a huge backward regression behind a memory PASS.

## Risk And Cost

Likely files if Route B is licensed:

- `crates/train/src/qwen35.rs`: hidden-window forward method.
- `crates/train/src/teacher_infer.rs`: windowed in-process teacher adapter.
- `crates/train/src/opd.rs`: windowed KL, windowed corpus-truth SFT, per-window
  backward/cleanup lifecycle.
- `crates/train/examples/opd_step_cuda_infer_teacher_train.rs`: CLI flag,
  eval path switch, logging of window size.
- `crates/train/src/loss.rs`: reuse or extract T5a weighting helpers; avoid
  moving optimizer/backward lifecycle here.
- `crates/train/src/trainer.rs` or a local helper: cleanup helper may need to
  retain params plus grad tensors between windows.
- Unit tests in `crates/train/src/*` and the OPD example smoke tests.
- Experience entry for PASS/KILL.

Cost estimate:

- Route B implementation: 4-7 focused working days, plus GPU acceptance.
- Route A: 1.5-3 weeks because the existing forward contract changes globally.
- Route C: 2-5 days to prototype, but likely fails SOLID because it either
  retains all graphs or hides backward lifecycle inside a loss API.

T5a reuse:

- Reuse the weighting math, parity-test pattern, and default-off rollout
  discipline from `kl_distill_loss_chunked`.
- Do not rely on T5a's `slice(student_logits)`/`slice(teacher_logits)` as the
  memory fix. The slice must move before `lm_head`, where the tensor is
  `[B, S, H]`, not after `lm_head`, where it is `[B, S, V]`.

## Open Questions Before Coding

- What window size should be first licensed? Start with 8 because it matches
  T5/T15 and keeps one logits tensor near 7.6 MiB.
- Should windowed real-corpus GKD initially support only in-process teacher?
  Recommended: yes. API teacher can be a later route because a fallback full
  teacher forward would invalidate the memory acceptance.
- Should per-window backward be a new OPD step variant or a branch inside the
  current function? Recommended: branch inside current GKD path gated by
  `logits_window_size`, keeping the public entrypoints stable until acceptance.
