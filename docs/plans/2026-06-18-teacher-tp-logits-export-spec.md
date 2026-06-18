# Teacher TP Logits Export Spec

Date: 2026-06-18
Scope: read-only research to implementation spec. No code was changed while
writing this document. Do not run H20/tmux2/GPU for this spec.

## Goal

Unblock `InferTeacher` scoring with the real DSv4-Flash teacher at TP=8. The
student path already asks the teacher for dense `[seq_len, vocab]` logits through
`LoadedInferenceEngine::forward_token_logits`; the missing piece is making that
surface execute as a TP-collective control op instead of a rank-0-only call.

Minimum implementation sentence:

Broadcast one raw-logits control envelope to every worker rank, have all TP ranks
run the same private-slot DSv4 forward, and return rank0's full-vocab logits
through the existing `RawLogits` contract.

## Current Blockers

- `CudaExecutorModel::forward_token_logits` delegates only Qwen35 and bails on
  DSv4 (`crates/infer-cuda/src/executor.rs:373-392`).
- Qwen35 raw logits call `ensure_not_collective("forward_token_logits")`, so the
  existing OPD surface is explicitly single-rank only
  (`crates/infer-cuda/src/executor.rs:2655-2696`).
- The guard explains the failure mode: rank-0 control-seam calls would desync the
  NCCL collective sequence and diverge resident weights
  (`crates/infer-cuda/src/executor.rs:2390-2401`).
- Offload/reload currently have the same guard on Qwen35 and DSv4 enum arms bail
  outright (`crates/infer-cuda/src/executor.rs:406-439`,
  `crates/infer-cuda/src/executor.rs:2360-2387`).
- `InferTeacher` needs the existing dense `RawLogits` shape. It calls
  `engine.forward_token_logits`, validates `seq_len` and `vocab`, then imports
  the bf16 device buffer as f32 with shape `[1, seq_len, vocab]`
  (`crates/train/src/teacher_infer.rs:717-791`). The windowed path just slices
  the dense tensor after this full forward (`crates/train/src/teacher_infer.rs:794-832`).

## TP=8 Ownership

Current ARLE runtime state:

- Every DSv4 rank loads a full `embed.weight` and full `head.weight` through
  `load_dsv4_global_matrix` (`crates/infer-cuda/src/dsv4.rs:1064-1068`).
- `load_dsv4_global_matrix` loads a whole BF16/F32 or block-scaled matrix and has
  no TP shard parameter (`crates/infer-cuda/src/loader.rs:3001-3014`).
- Attention is loaded with the TP config and MoE uses an EP split
  (`crates/infer-cuda/src/dsv4.rs:1041-1048`,
  `crates/infer-cuda/src/dsv4.rs:1070-1081`).
- The static DeepSeek spec still marks `head.weight` and `embed.weight` as
  vocab-parallel (`crates/deepseek-spec/src/v4.rs:185-188`,
  `crates/deepseek-spec/src/v4.rs:1225-1228`).

Conclusion: under the code in this tree, TP=8 ranks hold sharded attention/MoE
work, but each rank holds a full DSv4 vocab head at runtime. This is a current
runtime fact, not a claim about the checkpoint's ideal sharding contract.

Implication for logits export:

- V1 does not need logits all-gather. After every rank participates in the DSv4
  forward collectives, rank0 can project with its local full `lm_head` and return
  full `[seq_len, vocab]` logits.
- If ARLE later changes DSv4 to actually load `head.weight` as vocab-parallel,
  the export path must add a logits gather step; see "Future Vocab-Sharded Head".

## Why All Ranks Must Run

DSv4 hidden states are not rank0-local. The forward path contains real TP/EP
collectives:

- TP runtime can be NCCL-backed and `is_collective()` returns true for NCCL
  (`crates/infer-cuda/src/tp.rs:55-80`).
- Attention output is all-reduced in both batched and decode paths
  (`crates/infer-cuda/src/dsv4.rs:2861-2868`,
  `crates/infer-cuda/src/dsv4.rs:3742-3748`).
- DeepEP low-latency MoE runs collective dispatch/combine even when a rank owns
  zero tokens, then gathers owned columns by all-reduce
  (`crates/infer-cuda/src/dsv4.rs:3810-3888`).
- Non-DeepEP routed MoE performs an explicit all-reduce
  (`crates/infer-cuda/src/dsv4.rs:3948-3956`).

Therefore a rank0-only call to `forward_token_logits` is invalid. It can hang
NCCL, or worse, mutate only one rank's transient state while the workers remain
at a different collective point.

## Existing Logits Building Blocks

DSv4 already has the important inner pieces:

- Private KV adapter/slot construction exists via `new_kv_adapter` and
  `new_slot_state` (`crates/infer-cuda/src/dsv4.rs:1153-1201`).
- `forward_tokens_verify` runs a contiguous token prefix, writes a slot like a
  normal forward, and returns logits for every row
  (`crates/infer-cuda/src/dsv4.rs:1573-1606`).
- `verify_logits_from_stream` folds each row through head HC, RMS norm, and a
  batched `lm_head` projection (`crates/infer-cuda/src/dsv4.rs:1609-1648`).
- `lm_head_project_batch` is already `[m, hidden] -> [m, vocab]`
  (`crates/infer-cuda/src/dsv4.rs:4690-4712`).
- Existing MTP top-k treats logits as row-major `[m, vocab]`, masking by
  `offset = row * vocab + id` (`crates/infer-cuda/src/dsv4.rs:4754-4806`).

The Qwen35 precedent is also the right outer shape: validate non-empty matching
tokens/positions, require contiguous positions, allocate a private transient slot,
then return a `(DeviceVec, [seq_len, vocab])`
(`crates/infer-cuda/src/executor.rs:2655-2696`). Do not copy its
`ensure_not_collective` guard for DSv4.

## Upstream Practice

vLLM uses a vocab-parallel LM head model: local logits are computed first, then
TP logits are gathered before processors see the full vocabulary. It chooses
gather-to-rank0 or all-gather based on platform
([vLLM logits_processor.py at d682968](https://github.com/vllm-project/vllm/blob/d682968aa9fcd7e7a78218b548c52fc198a87a6c/vllm/model_executor/layers/logits_processor.py#L75-L103)).
For top-1 it avoids full-vocab communication by gathering only `(value, index)`
pairs and reducing them
([vLLM logits_processor.py at d682968](https://github.com/vllm-project/vllm/blob/d682968aa9fcd7e7a78218b548c52fc198a87a6c/vllm/model_executor/layers/logits_processor.py#L106-L156)).

SGLang also computes local LM-head logits and all-gathers across TP when enabled
([SGLang logits_processor.py at 97e3b89](https://github.com/sgl-project/sglang/blob/97e3b8998dc0f331423438091067ec0201d35e54/python/sglang/srt/layers/logits_processor.py#L830-L856)).
Its logprob path either materializes full logits/logprobs or chunks the
logprob work to bound peak memory
([SGLang logits_processor.py at 97e3b89](https://github.com/sgl-project/sglang/blob/97e3b8998dc0f331423438091067ec0201d35e54/python/sglang/srt/layers/logits_processor.py#L383-L413),
[SGLang logits_processor.py at 97e3b89](https://github.com/sgl-project/sglang/blob/97e3b8998dc0f331423438091067ec0201d35e54/python/sglang/srt/layers/logits_processor.py#L670-L827)).
Top-k logprobs are taken after log-softmax on the gathered/full logprob tensor
([SGLang logprob.py at 97e3b89](https://github.com/sgl-project/sglang/blob/97e3b8998dc0f331423438091067ec0201d35e54/python/sglang/srt/layers/utils/logprob.py#L29-L61),
[SGLang logprob.py at 97e3b89](https://github.com/sgl-project/sglang/blob/97e3b8998dc0f331423438091067ec0201d35e54/python/sglang/srt/layers/utils/logprob.py#L148-L219)).

Takeaway for ARLE: do not invent a separate sampling path. Export either full
rank0 logits after a collective forward, or for a future vocab-sharded head,
gather logits/top-k with explicit TP collectives.

## V1 Spec

### 1. DSv4 raw logits method

Add a DSv4-specific executor/model path that mirrors Qwen35's input validation
without the single-rank guard:

- Validate `input_ids` non-empty.
- Validate `input_ids.len() == positions.len()`.
- Validate `positions[i] == positions[0] + i`.
- Allocate a private DSv4 KV adapter and slot sized for `input_ids.len()` using
  `Dsv4Model::new_kv_adapter` and `Dsv4Model::new_slot_state`
  (`crates/infer-cuda/src/dsv4.rs:1153-1201`).
- Call `Dsv4Model::forward_tokens_verify` with that transient slot and adapter
  (`crates/infer-cuda/src/dsv4.rs:1573-1606`).
- Return the logits buffer as `DeviceVec` with shape `[seq_len, lm_head.rows]`.
  The row-major layout is already assumed by `mtp_topk_device`
  (`crates/infer-cuda/src/dsv4.rs:4754-4806`).

Do not reuse serving slots. OPD scoring is a one-shot full prefix forward, so the
transient state should be allocated, advanced once, and dropped exactly like the
Qwen35 raw-logits surface describes (`crates/infer-cuda/src/executor.rs:2655-2696`).

### 2. Enum dispatch

Change the DSv4 branch in `CudaExecutorModel::forward_token_logits` from bail to
delegate to the new DSv4 executor/model method
(`crates/infer-cuda/src/executor.rs:373-392`).

Keep dense Qwen3 bailing unless it becomes an OPD target.

### 3. Multiproc collective control

The existing `ServeHandle::run_on_executor` runs a closure only on rank0's engine
thread (`crates/infer-server/src/lib.rs:469-510`). The engine loop drains those
control closures before admissions/steps (`crates/infer-server/src/execution.rs:193-197`,
`crates/infer-server/src/execution.rs:308-319`). Worker ranks only understand
`TickAdmissions` today (`crates/infer-server/src/multiproc_relay.rs:156-164`,
`crates/cli/src/serve_multiproc.rs:325-366`).

Add a worker-executed control envelope, not an HTTP request:

```text
RelayEnvelope::ControlForwardTokenLogits {
    seq: u64,
    input_ids: Vec<u32>,
    positions: Vec<u32>,
}
```

Worker behavior:

- Extend `run_lockstep_driver` to match this envelope.
- Validate a monotonic control sequence, or fold control envelopes into the same
  ordered lockstep sequence as `TickAdmissions`.
- Call a new `CudaWorkerEngine::forward_token_logits_discard(&mut self, ...)`
  that runs `self.0.executor_mut().forward_token_logits(...)` and drops the
  result.
- Do not return worker logits to rank0 in V1.

Rank0 behavior:

- In the rank0 control closure, broadcast `ControlForwardTokenLogits` to all
  workers through the existing relay coordinator before running the local
  executor forward.
- Then run local `executor.forward_token_logits(...)`.
- Return local rank0 logits through `RawLogits`.

The relay already has a process-global tick broadcaster installed by the CLI
coordinator (`crates/infer-server/src/multiproc_relay.rs:67-90`,
`crates/cli/src/serve_multiproc.rs:203-218`) and a coordinator broadcast method
(`crates/infer-server/src/multiproc_relay.rs:378-387`). Reuse that plumbing:
add a control broadcaster beside the tick broadcaster rather than adding a second
transport.

Ordering rule: the control envelope and rank0 local forward must be paired at the
same engine-loop boundary, before normal admissions/steps. A worker that misses
one control envelope can deadlock the next DSv4 collective, so broadcast failure
must abort rank0 before it enters the local forward.

### 4. Public API contract

Keep `RawLogits` unchanged for V1:

- It already carries one row-major `[seq_len, vocab]` `DeviceVec` and one
  `DeviceContext` (`crates/infer-api/src/types.rs:18-63`).
- This is valid because rank0 holds a full runtime `lm_head`.
- `InferTeacher` needs no shape/API change for dense KL
  (`crates/train/src/teacher_infer.rs:717-791`).

If future code returns top-k only, add a separate teacher method and data type.
Do not overload `RawLogits`; dense KL/windowed code currently assumes full vocab.

## Future Vocab-Sharded Head

If DSv4 runtime is changed to obey the static vocab-parallel head spec, each rank
will produce local logits `[seq_len, vocab_per_rank]`. Then V1's "rank0 already
has full logits" assumption is false.

Full dense logits gather:

- Use `TpRuntime::all_gather_bf16_raw`, which requires every rank to call with the
  same `sendcount` and a receive buffer sized `sendcount * world_size`
  (`crates/infer-cuda/src/tp.rs:486-522`).
- The one-shot path has the same rank-major all-gather contract
  (`crates/infer-cuda/src/tp.rs:1187-1212`).
- Gather layout should be rank-major `[tp, seq_len, vocab_per_rank]`.
- Add a small reorder kernel or D2D scatter to produce row-major
  `[seq_len, vocab]` on rank0. Workers may allocate the receive buffer and discard
  it, because the current primitive is all-gather, not gather-to-root.

Top-k/logprob export:

- For exact top-k over vocab-sharded logits, compute local top-k per row on every
  rank, add `vocab_start = rank * vocab_per_rank`, all-gather `(value, global_id)`
  pairs, and merge top-k on rank0.
- For top-1 this is exactly the vLLM communication pattern: O(seq * 2 * tp)
  values instead of O(seq * vocab).
- For top-k, use `[seq_len, k, 2]` pairs. Packing ids as f32 is safe only while
  vocab ids are exactly representable; a generic raw all-gather over `(bf16/f32,
  u32)` is cleaner if this path becomes production.

Do not implement vocab-sharded head and dense-logits export in the same patch
unless memory forces it. The current DSv4 blocker can be removed without changing
head ownership.

## Collective Offload/Reload

Current state:

- `InferTeacher` may call `offload_engine_weights` after teacher scoring and
  `reload_engine_weights` before the next teacher forward
  (`crates/train/src/teacher_infer.rs:839-858`).
- The serve surface routes those calls through rank0-only `run_on_executor`
  (`crates/infer-server/src/lib.rs:513-526`).
- Qwen35 offload/reload snapshots and restores resident weights only after full
  device synchronization (`crates/infer-cuda/src/qwen35.rs:1803-1823`,
  `crates/infer-cuda/src/qwen35.rs:1951-1962`).

DSv4 TP rule:

- Do not offload or reload only rank0. That diverges resident weights across the
  TP group.
- Add `ControlOffloadWeights { seq }` and `ControlReloadWeights { seq }` using
  the same worker-control relay pattern as logits.
- Each rank offloads/reloads its local resident weights and releases local
  workspaces. Rank0 returns its local freed bytes. If the caller needs global
  freed bytes, add a separate sum later; the existing API returns one `usize`.
- Offload/reload must run at the same engine-loop boundary as other controls,
  never while a request step is in flight.

Implementation warning: DSv4 has no existing `model.offload_engine_weights`
equivalent. Before adding one, enumerate every resident `DeviceMatrix`,
block-scaled cache, MTP matrix, norm vector, DeepEP/DeepGEMM cache, workspace, and
slot-owned buffer. Weights can move; slot KV/recurrent state should remain
resident unless the caller explicitly wants to destroy in-flight serve state.
This is a separate implementation unit from logits export.

V1 recommendation: land collective logits first. Keep DSv4 offload/reload
disabled unless the OPD run actually needs teacher/student VRAM time-share. If it
is needed, implement offload/reload as a second collective-control patch.

## Line-Level Change Map

1. `crates/infer-cuda/src/dsv4.rs`
   - Add a DSv4 full-logits helper near `forward_tokens_verify`
     (`crates/infer-cuda/src/dsv4.rs:1573-1606`) or near the head projection
     helper (`crates/infer-cuda/src/dsv4.rs:1609-1648`).
   - Reuse `new_kv_adapter`/`new_slot_state`
     (`crates/infer-cuda/src/dsv4.rs:1153-1201`).
   - Return row-major logits `[seq_len, lm_head.rows]`.

2. `crates/infer-cuda/src/executor.rs`
   - Replace the DSv4 bail in `CudaExecutorModel::forward_token_logits`
     (`crates/infer-cuda/src/executor.rs:388-392`).
   - Add DSv4 executor method with Qwen35-style validation
     (`crates/infer-cuda/src/executor.rs:2655-2696`) but no
     `ensure_not_collective`.
   - Leave DSv4 offload/reload bails until the separate collective offload patch
     (`crates/infer-cuda/src/executor.rs:418-439`).

3. `crates/infer-server/src/multiproc_relay.rs`
   - Add worker control envelope variants next to `TickAdmissions`
     (`crates/infer-server/src/multiproc_relay.rs:156-164`).
   - Add a process-global control broadcaster beside the tick broadcaster
     (`crates/infer-server/src/multiproc_relay.rs:67-90`).
   - Reuse `RelayCoordinator::broadcast`
     (`crates/infer-server/src/multiproc_relay.rs:378-387`).

4. `crates/cli/src/serve_multiproc.rs`
   - Install the control broadcaster next to the tick broadcaster
     (`crates/cli/src/serve_multiproc.rs:203-218`).
   - Extend `run_lockstep_driver` to execute `ControlForwardTokenLogits`
     (`crates/cli/src/serve_multiproc.rs:325-366`).
   - Add a worker engine method to run raw logits and discard output
     (`crates/infer-api/src/loaded.rs:1316-1357`).

5. `crates/infer-api/src/serve_engine.rs`
   - In `ServeInferenceEngine::forward_token_logits`, broadcast the worker control
     envelope before rank0's local executor forward
     (`crates/infer-api/src/serve_engine.rs:184-199`).
   - Keep returned `RawLogits` unchanged.

6. `crates/train/src/teacher_infer.rs`
   - No dense path change required. It should keep consuming `RawLogits`
     (`crates/train/src/teacher_infer.rs:717-791`).
   - Add a future top-k trait method only if the loss path stops requiring dense
     full vocab logits.

## Verification Plan

No verification was run for this read-only spec. Future implementation gates:

1. CPU/typecheck gate: `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`.
2. Single-rank CUDA smoke: DSv4 `forward_token_logits` returns shape
   `[seq_len, vocab]` and matches `forward_tokens_verify` argmax for a tiny input.
3. TP=8 reachability gate: all ranks log entry/exit for the same
   `ControlForwardTokenLogits seq`, no NCCL hang, rank0 returns logits.
4. Correctness gate: for a fixed token prefix, rank0 raw-logits top-1 matches the
   DSv4 normal greedy next-token path at the final position.
5. OPD gate: `InferTeacher` dense KL path imports `[1, seq_len, vocab]` from
   DSv4 TP=8 without changing train-side shape assumptions.

Do not claim a capability improvement from this work. It only unblocks the real
teacher scoring surface; quality A/B still needs the OPD capability matrix.
