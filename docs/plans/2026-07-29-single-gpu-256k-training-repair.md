# Single-GPU 256K OPD Training Repair Plan

> Status: accepted — implementation and remote validation in progress.
>
> Scope: current repository state, including relevant committed code and the
> training/autograd WIP already present in the shared worktree. This is not a
> branch-diff review.

## Execution status

- T1: wide elementwise, GDN, and slice offsets implemented; CUDA gate pending
  for the final slice tranche.
- T2.1: long-query fused forward implemented; CUDA 65,536 gate passed.
- T2.2: CP-only attention deleted; both paths use rectangular recompute.
- T2.3: 512 GiB host capture removed; CUDA boundary capture streams state only.
- T2.4: byte-budget mono fallback and context estimator implemented.
- T3.1: persistent CUDA gradient accumulation implemented; CUDA gate passed.
- T3.2: indexed CE implemented; CPU parity and exact 64K CE passed.
- T3.3: checkpoint replay chunks MLP; device-only accumulation remains pending.
- T4: programmatic checkpoint default aligned; remaining items pending.
- T5: exact `b5f078ae0` passed 64K forward/CE, then OOMed in the first
  full-attention backward replay. Last-use activation release and wide slice
  offsets are committed in `e01aa6606`; rerun pending.

## Outcome

One H20 completes one real 256K OPD update:

```text
rollout/teacher path
  -> student forward
  -> masked loss
  -> backward
  -> optimizer step
```

The run must preserve the exact sequence, attention semantics, loss, LoRA target
set, and optimizer. Passing a synthetic sub-path is diagnosis, not acceptance.

## Current truth

- The default admission fence drops updates above 23,000 tokens.
- The latest completed single-GPU update remains 49,152 tokens.
- Wide elementwise, GDN, attention, and slice paths remove the known 256K
  indexing and quadratic-attention walls; the final slice tranche is pending
  CUDA validation.
- No existing run proves a complete 64K, 128K, or 256K update.
- Exact `b5f078ae0` completed 64K forward in 749.656 seconds and CE in 2.956
  seconds. Backward OOMed after 181 seconds in layer 63 gated-q slice backward:
  3,072 MiB requested with 25 MiB free; peak was 97,483/97,508 MiB.

Therefore the capability is **49,152 verified; 256K unsupported** until the final
gate below passes.

## Constraints

1. Single GPU is the target. Context Parallel cannot be counted as progress.
2. No precision downgrade, sliding window, token truncation, loss approximation,
   or frozen-prefix approximation may license the correctness baseline.
3. No per-chunk D2H synchronization in the backward hot loop.
4. Reuse existing recompute, device storage, scatter, and collective primitives.
5. Each tranche is independently buildable, testable, measurable, and revertible.
6. Preserve unrelated worktree edits; stage and commit only explicit paths.
7. Every runtime tranche gets a dated `docs/experience/wins/` or `errors/` entry.

## Dependency order

```text
T0 truthful baseline
  |
  +--> T1 addressable 256K kernels
  |      |
  |      +--> T2 bounded attention/GDN forward and backward
  |                |
  |                +--> T3 bounded gradient/loss/MLP memory
  |                          |
  |                          +--> T4 close training orchestration
  |                                    |
  |                                    +--> T5 64K -> 128K -> 256K gate
  |
  +--> T6 remove or finish unrelated CP half-state
```

T1–T4 are the single-GPU critical path. T6 cannot delay T1–T5 unless its current
WIP overlaps the same autograd op.

## T0 — Pin a truthful baseline

### Change

- Keep the 23K production fence.
- Add an explicit experimental override for the length ladder; do not silently
  change the default.
- Make logs distinguish `filtered`, `forward OOM`, `backward OOM`, `index wall`,
  and `completed optimizer step`.
- Mark older contradictory 128K/256K research notes as superseded by one current
  capability page.

### Files

- `crates/train/src/runtime_flags.rs`
- `crates/train/src/update_strategy.rs`
- `crates/cli/src/args.rs`
- `crates/cli/src/train_cli.rs`
- `docs/research/2026-07-27-opd-writeback-128k-plan.md`
- `docs/research/2026-07-27-opd-writeback-wall-decomposition.md`
- `docs/experience/wins/2026-07-28-mlp-seq-chunked-recompute-256k.md`

### Gate

- A 256K input with the default configuration reports `filtered`, not success.
- The explicit experiment path reaches forward and records the terminal reason.
- No default flip occurs in this tranche.

## T1 — Make 256K indexable

### Root cause

Large tensors use `i32` for total element counts and flattened offsets. Dimension
sizes fit; the products do not.

### Change

1. Change GDN total lengths and flattened offsets to `usize`/`size_t` or `i64`
   end-to-end:
   - Rust launch arguments in `crates/autograd/src/backend_cuda.rs`
   - CUDA indices in
     `crates/autograd/src/backend_cuda/kernels/linear_attention.cu`
2. Do the same for binding elementwise kernels:
   - add / add-into
   - mul and mul backward
   - activation kernels used by the MLP
3. Keep small dimensions (`seq`, heads, hidden dimension, block counts) as `i32`
   where the launch ABI requires it.
4. Add boundary tests for `2^31 - 1`, `2^31`, and `2^31 + 1` total elements
   without allocating those tensors.

### Files

- `crates/autograd/src/backend_cuda.rs`
- `crates/autograd/src/backend_cuda/kernels/linear_attention.cu`
- `crates/autograd/src/backend_cuda/kernels/add_into.cu`
- `crates/autograd/src/backend_cuda/kernels/elementwise.cu`
- `crates/autograd/src/backend_cuda/kernels/mul_backward.cu`
- `crates/autograd/src/backend_cuda/kernels/activation_backward.cu`

### Gate

- Host contract tests prove the launch arguments preserve values across `2^31`.
- A CUDA boundary smoke reaches and completes each changed kernel with a logical
  total above `2^31`; use a strided/synthetic harness rather than a huge alloc.
- `ncu` or an equivalent kernel measurement confirms no regression on the
  existing 49,152 shape.

## T2 — Bound attention and GDN memory

### T2.1 Full-attention forward

Replace the composed `[q_len, kv_len]` fallback with query-tiled fused attention
for long queries. Head chunking alone is insufficient.

Reuse the existing fused attention implementation and extend its grid/loop; do
not add a second attention API.

Files:

- `crates/autograd/src/backend_cuda.rs`
- `crates/autograd/src/backend_cuda/kernels/attention.cu`
- `crates/autograd/src/ops/attention.rs`

Gate:

- CUDA forward parity at head dimension 256 for q lengths 65,535 and 65,536.
- Peak scratch is bounded by query tile size, not total sequence length.

### T2.2 Full-attention suffix backward

Route both the frozen-prefix generated suffix and CP through the owning
`causal_sdpa_recompute_with_q_start` path. Delete the CP-only wrapper.

Files:

- `crates/autograd/src/ops/attention.rs`
- `crates/autograd/src/tape.rs`
- `crates/train/src/qwen35.rs`

Gate:

- CUDA `q_start > 0` forward and gradient parity.
- Long-KV peak scales with `q_chunk * kv_len`, not `q_len * kv_len`.

### T2.3 GDN prefix boundary capture

The capture caller needs only `final_state + conv_tail`; the host reference
currently allocates output and a 512 GiB `state_history` at 256K.

Add a boundary-only device streaming mode to the existing GDN forward. It must
not allocate output, `state_history`, or backward-only intermediates.

Files:

- `crates/autograd/src/backend.rs`
- `crates/autograd/src/backend_cuda.rs`
- `crates/autograd/src/backend_cuda/kernels/linear_attention.cu`
- `crates/autograd/src/ops/linear_attention.rs`
- `crates/train/src/qwen35.rs`

Gate:

- Boundary state and conv tail match the small-shape host reference.
- 256K boundary capture has O(state + conv-window) retained memory.

### T2.4 GDN backward

For long sequences, select the existing constant-scratch mono path before the
staged path allocates full-sequence `g_in`, `M`, `B`, and `state`. Only build a
streaming staged path if mono wall-clock is later proven unacceptable.

Also update `linear_attention_ctx_bytes` to count the actual retained tensors,
not deleted saves.

Files:

- `crates/autograd/src/backend_cuda.rs`
- `crates/autograd/src/ops/linear_attention.rs`
- `crates/train/src/qwen35.rs`

Gate:

- Small-shape staged-vs-mono gradient parity.
- The long-sequence selector uses a byte budget, not a magic sequence number.
- The estimator matches the allocation ledger within 5%.

## T3 — Bound gradient, loss, and MLP memory

### T3.1 Persistent gradient accumulation

`add_into_device` currently allocates a third full-size buffer for
`old_grad + new_grad`. Persistent leaf gradients have no remaining tape readers,
so mutate the accumulator in place at the shared accumulation boundary.

Files:

- `crates/autograd/src/backend.rs`
- `crates/autograd/src/backend_cuda.rs`
- `crates/autograd/src/backend_cuda/kernels/add_into.cu`
- `crates/autograd/src/tensor.rs`
- `crates/autograd/src/tape.rs`

Gate:

- Tied/multi-touch parameter gradients match the functional baseline.
- The 57,344 failure site no longer needs a third 2.82 GB allocation.

### T3.2 Indexed CE

Replace per-chunk full-table embedding gradients with one preallocated
`d_hidden`. Each chunk scatters its selected rows into that buffer. Combine
batched target rows into one call while preserving the current mean-of-row-means
contract.

Files:

- `crates/autograd/src/ops/fused_linear_distill.rs`
- `crates/autograd/src/backend.rs`
- `crates/autograd/src/backend_cuda.rs`
- `crates/autograd/src/backend_cuda/kernels/scatter_add.cu`
- `crates/train/src/opd.rs`

Gate:

- Loss and gradients match the current host oracle.
- Empty target rows are valid participating no-ops.
- Peak contains one full `d_hidden`, not one per chunk or batch row.

### T3.3 MLP sequence chunking

Keep `d_input` and parameter accumulators on device for the entire layer. Scatter
chunk input gradients into a preallocated device tensor and accumulate parameter
gradients on device.

Define chunk size as total rows:

```text
seq_chunk = ceil(target_total_rows / batch)
```

Route both normal forward and frozen generated-segment forward through one
`forward_mlp_maybe_chunked` helper.

Files:

- `crates/autograd/src/ops/checkpoint.rs`
- `crates/autograd/src/tape.rs`
- `crates/autograd/src/tensor.rs`
- `crates/train/src/qwen35.rs`
- `crates/train/src/runtime_flags.rs`

Gate:

- B=1 and B>1 gradient parity.
- Nested frozen-generation checkpoint parity.
- No per-chunk `to_host`.
- 49,152 backward wall-clock improves over 2,460 seconds.

## T4 — Close the real training path

### T4.1 Checkpoint host residency

Add a hard byte cap and step-end trim to `CheckpointOffloadPool`. Expose the
existing host/L3 limits as supported CLI configuration. Report peak RSS and
spill bytes.

Files:

- `crates/autograd/src/tensor.rs`
- `crates/autograd/src/runtime_flags.rs`
- `crates/train/src/runtime_flags.rs`
- `crates/cli/src/args.rs`
- `crates/cli/src/train_cli.rs`

Gate:

- Pool retained bytes never exceed the configured cap.
- Step cleanup returns unused buffers.
- 128K/256K runs record host RSS and L3 spill totals.

### T4.2 Frozen-prefix semantics

Keep frozen-prefix KV disabled in the correctness baseline. Rename/document it as
approximate until prefix carry backward is implemented. Do not use its memory
result to license exact 256K training.

Files:

- `crates/cli/src/args.rs`
- `crates/train/src/opd.rs`
- `crates/train/src/qwen35.rs`
- `crates/train/tests/test_frozen_prompt_kv_writeback.rs`

### T4.3 BF16-resident experts

Replace the process-global setter with a per-load option:

- all all-linear rollout students: `true`
- attention-only students: `false`
- every teacher: `false`

Cover `opd`, `rubric-opd`, and `agent-opd`.

Files:

- `crates/infer-api/src/loaded.rs`
- `crates/infer-cuda/src/runtime_flags.rs`
- `crates/infer-cuda/src/qwen35.rs`
- `crates/cli/src/train_cli.rs`

Gate:

- 50 optimizer steps with expert LoRA re-merge.
- Teacher memory remains unchanged.
- No global flag leaks between sequential model loads.

### T4.4 GKD teacher logits

Make teacher logits truly windowed or top-k streamed. A full
`[1, 256K, vocab]` tensor is forbidden.

Files:

- `crates/train/src/opd.rs`
- `crates/train/src/teacher_infer.rs`
- `crates/autograd/src/ops/fused_linear_distill.rs`

Gate:

- Short-shape loss/gradient parity.
- Teacher-logit peak is O(window * vocab), not O(sequence * vocab).

### T4.5 Default convergence

Remove duplicate runtime defaults so CLI and programmatic callers get the same
gradient-checkpoint behavior. Raise `max_update_seq` only after T5 passes.

Files:

- `crates/train/src/runtime_flags.rs`
- `crates/train/src/qwen35_loader.rs`
- `crates/cli/src/args.rs`

## T5 — DevOps-owned validation ladder

The DevOps agent executes and archives commands; the implementation owner reads
the raw logs and signs the verdict.

### Local gates on every tranche

```bash
cargo fmt --check
cargo test -p autograd --release
cargo test -p train --release
cargo clippy -p autograd -p train --release -- -D warnings
CUDARC_CUDA_VERSION=12080 \
  cargo check -p train --release \
  --no-default-features --features cuda,no-cuda
```

Kernel tranches additionally require:

```bash
CUDA_HOME=/usr/local/cuda cargo test -p autograd --release --features cuda
```

Run the smallest named CUDA tests first; do not use a full workspace CUDA build
as the debugging loop.

### Remote H20 ladder

Use one clean H20, one run at a time, release binary, fixed model, fixed LoRA
target set, fixed seed, and fresh process after any OOM:

1. 49,152 baseline re-anchor.
2. 57,344 root-cause gate.
3. 65,536.
4. 131,072.
5. 262,144.

For each point archive:

- git SHA and binary SHA256
- exact command and environment
- GPU model, driver, CUDA version
- model path and checksum
- loss and optimizer-step completion
- peak device allocated/reserved/driver-used bytes
- peak host RSS and L3 spill bytes
- forward, backward, optimizer, and total wall-clock
- terminal status and full stderr

Stop at the first failure, diagnose it, and do not average through OOMs.

### Correctness

Before the 256K run:

- Short-shape loss and gradient parity for every changed op.
- `scripts/needle_gate.py` and `scripts/lever_gate.sh` ×3 against the baseline
  envelope for runtime-visible changes.
- No NaN/Inf in loss, gradients, optimizer state, or updated LoRA weights.

### Acceptance

256K is accepted only when all are true:

1. A real trajectory reaches one completed optimizer step.
2. Exact training semantics are used; approximate frozen-prefix mode is off.
3. Device peak fits one H20 without retry or hidden truncation.
4. Host RSS stays below the declared machine budget.
5. No `i32` boundary, host fallback, or per-chunk D2H sync appears.
6. Correctness gates pass.
7. A dated experience entry contains the raw artifact paths and verdict.
8. `CHANGELOG.md` records the capability phase exit and default decision.

Passing only `--synthetic-writeback-seq` is not acceptance.

## T6 — Context Parallel disposition

Context Parallel is multi-GPU and is not required for the single-GPU goal. The
current code is not wired end-to-end and contains collective/loss-normalization
gaps.

Default disposition for this project:

- preserve current unowned WIP;
- remove CP from all single-GPU 256K claims and documentation;
- after T5, either complete it as a separate multi-GPU milestone with real
  multi-rank NCCL tests, or delete the unused half-state.

Do not add adapters or parallel CP/non-CP attention implementations. Both paths
must converge on the shared rectangular recompute op.

## Commit plan

Each commit is self-contained and names only its files:

1. `fix(cuda): use wide offsets for long training tensors`
2. `fix(train): stream long-context attention boundaries`
3. `fix(autograd): bound long-context attention backward`
4. `fix(autograd): accumulate persistent gradients in place`
5. `fix(train): keep chunked MLP backward on device`
6. `fix(train): bound indexed CE gradient storage`
7. `fix(train): cap checkpoint host residency`
8. `fix(cuda): scope BF16 expert residency per model load`
9. `docs(train): license single-GPU 256K OPD`

Do not combine unrelated tranches merely to reduce commit count.

## Explicit non-goals

- No Context Parallel as a substitute for single-GPU acceptance.
- No new general tensor compiler or allocator abstraction.
- No new attention API when an existing op can be extended.
- No precision change hidden inside a memory fix.
- No default flip based on extrapolation.
- No claim from CPU tests, typecheck, or synthetic forward alone.
