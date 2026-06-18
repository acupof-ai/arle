# OPD Route B External Trainer Integration

Date: 2026-06-18
Scope: design only. No code, H20, GPU, tmux2, or `infer-cuda` source changes.

## Goal

Route B keeps student training in verl and uses ARLE only as the real teacher
scoring service:

```text
verl rollout/FSDP student
  -> student prompt + response token ids
  -> ARLE teacher top-k logprob service
  -> verl teacher_ids + teacher_logprobs tensors
  -> verl FSDP forward_kl_topk loss
```

The integration should not port a student trainer into ARLE. The only ARLE
contract needed by verl is batched top-k next-token logprobs over the student
on-policy sequence.

## Source Snapshot

- ARLE TP logits export spec:
  `docs/plans/2026-06-18-teacher-tp-logits-export-spec.md`.
- verl research snapshot: `/tmp/verl` at
  `14574ecf52e310055e4d6e9f116bcb14d343d7e0`.

## Current Facts

### ARLE teacher export

- The DSv4 TP logits spec says the current TP=8 runtime loads a full DSv4
  `lm_head` on every rank, so V1 can run all TP ranks collectively and return
  rank0 full-vocab logits without a logits all-gather
  (`docs/plans/2026-06-18-teacher-tp-logits-export-spec.md:39-64`).
- The same spec requires every rank to participate in the DSv4 forward because
  attention and MoE execute TP/EP collectives
  (`docs/plans/2026-06-18-teacher-tp-logits-export-spec.md:66-84`).
- V1 keeps the dense `RawLogits` shape for ARLE internal OPD, but the spec also
  records the future vocab-sharded case where top-k/logits need explicit TP
  collectives (`docs/plans/2026-06-18-teacher-tp-logits-export-spec.md:108-130`,
  `docs/plans/2026-06-18-teacher-tp-logits-export-spec.md:212-220`).
- ARLE train already has a sparse top-k distill loss surface:
  `fused_linear_distill_loss_sparse(hidden, lm_head, teacher_topk_log_probs,
  teacher_topk_indices, ...)` is forward-KL only
  (`crates/train/src/loss.rs:362-392`), and its test checks sparse top-k against
  dense KL when missing teacher mass tends to zero
  (`crates/train/src/loss.rs:1536-1595`).

### verl OPD path

- verl's async OPD recipe is exactly the desired system shape: student rollout,
  teacher returns top-k logprobs plus token ids, then sparse token-level KL
  (`/tmp/verl/docs/advance/async-on-policy-distill.md:13-23`).
- Its documented batch inputs are top-k teacher logprobs, top-k teacher indices,
  and the attention mask; loss injection happens at the final logits stage
  (`/tmp/verl/docs/advance/async-on-policy-distill.md:52-63`).
- The Qwen3.5 35B-A3B to 4B FSDP example already matches the target model scale:
  default student `Qwen/Qwen3.5-4B`, teacher `Qwen/Qwen3.5-35B-A3B`, teacher
  world size 8, and top-k 64
  (`/tmp/verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh:8-18`).
- That same launcher wires FSDP student training and vLLM rollout
  (`/tmp/verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh:61-103`)
  plus distillation config and teacher TP/EP
  (`/tmp/verl/examples/on_policy_distillation_trainer/run_qwen3_5_4b_fsdp.sh:118-134`).
- In verl, the top-k loss consumes only `data["teacher_logprobs"]` and
  `data["teacher_ids"]`; it dispatches to FSDP or Megatron by actor strategy
  (`/tmp/verl/verl/trainer/distillation/losses.py:123-156`).
- The FSDP loss expects teacher tensors shaped `[bsz, seqlen, topk]` and can use
  a chunked path to avoid materializing `[B, T, V]` log-softmax
  (`/tmp/verl/verl/trainer/distillation/fsdp/losses.py:26-63`,
  `/tmp/verl/verl/trainer/distillation/fsdp/losses.py:75-130`).
- verl warns that `forward_kl_topk` is most effective as supervised
  distillation with `use_policy_gradient=False`
  (`/tmp/verl/verl/workers/config/distillation.py:112-118`).

## Design Decision

Use verl's native async OPD/FSDP training path unchanged at the loss boundary.
Add only an external ARLE teacher client/server adapter that produces verl's
existing teacher tensors.

Concretely:

1. ARLE exposes an external teacher scoring endpoint that accepts token ids for
   the full student prompt plus response and returns top-k token ids/logprobs.
2. verl gets a minimal `external_arle` teacher backend that conforms to the
   existing `LLMServerClient.generate` contract used by
   `AsyncTeacherLLMServerManager`.
3. The adapter returns `extra_fields["prompt_ids"]` and
   `extra_fields["prompt_logprobs"]`, so verl's existing manager converts them
   into `teacher_ids` and `teacher_logprobs` without touching the FSDP loss.
4. Student rollout, actor update, checkpointing, FSDP, and loss computation stay
   in verl.

## Wire Contract

### Request

The ARLE service should accept batched requests. JSON is enough for bring-up;
switch to Arrow/msgpack only if measured host overhead matters.

```json
{
  "schema": "arle-opd-teacher-topk-v1",
  "requests": [
    {
      "request_id": "sample-0001",
      "input_ids": [151644, 77091, 198, 1234],
      "prompt_len": 2,
      "response_len": 2,
      "top_k": 64,
      "temperature": 1.0,
      "alignment": "verl_prompt_logprobs_v1"
    }
  ]
}
```

Rules:

- `input_ids` is exactly `prompt_ids + response_ids` from verl's agent loop.
  verl already calls the teacher on that concatenated sequence
  (`/tmp/verl/verl/experimental/agent_loop/agent_loop.py:913-936`).
- `top_k` must equal `distillation.distillation_loss.topk`.
- `temperature` is fixed to `1.0` for the first integration. verl's current
  vLLM teacher helper rejects non-1.0 prompt-logprob temperature
  (`/tmp/verl/verl/experimental/teacher_loop/teacher_manager.py:30-43`).
- The response must mimic verl's existing prompt-logprob alignment, not invent a
  new shift convention. verl's vLLM adapter skips the no-context first token and
  appends one dummy row so the returned row count equals the input sequence
  length (`/tmp/verl/verl/workers/rollout/vllm_rollout/utils.py:434-467`).

### Response

```json
{
  "schema": "arle-opd-teacher-topk-v1",
  "results": [
    {
      "request_id": "sample-0001",
      "prompt_ids": [[50256, 318], [77091, 25], [198, 13], [0, 0]],
      "prompt_logprobs": [[-0.4, -1.1], [-0.2, -1.8], [-0.7, -0.9], [0.0, 0.0]],
      "top_k": 2,
      "alignment": "verl_prompt_logprobs_v1"
    }
  ]
}
```

Rules:

- `prompt_ids`: int32-compatible `[seq_len, top_k]`.
- `prompt_logprobs`: fp32-compatible `[seq_len, top_k]`.
- Row count must equal `len(input_ids)`, because
  `AsyncTeacherLLMServerManager` asserts both tensors match the input length
  (`/tmp/verl/verl/experimental/teacher_loop/teacher_manager.py:102-128`).
- The dummy row uses token id `0` and logprob `0.0`, matching verl's vLLM
  extraction convention (`/tmp/verl/verl/workers/rollout/vllm_rollout/utils.py:462-467`).
- The ARLE HTTP client can immediately convert these arrays into a lightweight
  object whose `extra_fields` keys are `prompt_ids` and `prompt_logprobs`.

## ARLE Teacher Service Plan

### V1: full-head top-k on rank0

1. Use the TP logits export work to run DSv4 raw logits as a collective control
   op on every TP rank.
2. Because the current runtime keeps a full vocab head on rank0, compute
   `log_softmax` and top-k on rank0 for each scored row.
3. Return only top-k ids/logprobs to verl. Do not send dense `[seq, vocab]`
   logits over the external API.

This is not the final memory-optimal path, but it is the shortest correct path
once TP logits export lands.

### V2: vocab-sharded top-k

If DSv4 later loads `lm_head` as a real vocab-parallel shard:

1. Each rank computes local logits for its vocab shard.
2. Each rank computes local top-k `(logit, global_token_id)` pairs per row.
3. All-gather the fixed-size local top-k pairs to rank0.
4. Rank0 selects global top-k, applies global logsumexp normalization, and
   returns top-k logprobs.

This follows the upstream pattern captured in the TP logits spec: vLLM/SGLang
gather full logits for general processing, and vLLM also has a pair-gather path
for small-k cases
(`docs/plans/2026-06-18-teacher-tp-logits-export-spec.md:108-130`).

## verl Integration Points

### Keep unchanged

- FSDP student loss: it already consumes `teacher_logprobs` and `teacher_ids`
  (`/tmp/verl/verl/trainer/distillation/losses.py:123-156`).
- FSDP top-k KL implementation: it already gathers student probabilities at
  teacher ids and supports chunked top-k for memory control
  (`/tmp/verl/verl/trainer/distillation/fsdp/losses.py:75-130`).
- Agent loop tensor plumbing: teacher tensors are already padded, concatenated,
  and emitted into the `DataProto`
  (`/tmp/verl/verl/experimental/agent_loop/agent_loop.py:720-770`,
  `/tmp/verl/verl/experimental/agent_loop/agent_loop.py:957-959`).
- Actor update path: the trainer marks `distillation_use_topk` and calls
  `update_actor` with the batch
  (`/tmp/verl/verl/trainer/ppo/ray_trainer.py:1296-1328`).

### Minimal verl-side patch

Current verl assumes distillation teachers are Ray-managed vLLM/SGLang servers:

- Trainer creates `MultiTeacherModelManager`, obtains a `TeacherModel` resource
  pool, and passes `teacher_model_manager.get_client()` to `AgentLoopManager`
  (`/tmp/verl/verl/trainer/ppo/ray_trainer.py:913-953`).
- `DistillationTeacherModelConfig` only accepts `inference.name` of `vllm` or
  `sglang` for top-k validation
  (`/tmp/verl/verl/workers/config/distillation.py:188-218`).
- Single-teacher config derives `num_replicas` from the internal teacher
  resource pool (`/tmp/verl/verl/workers/config/distillation.py:289-307`).
- `TeacherModelManager` launches rollout replicas and returns an
  `LLMServerClient` over its load balancer
  (`/tmp/verl/verl/experimental/teacher_loop/teacher_model.py:62-101`,
  `/tmp/verl/verl/experimental/teacher_loop/teacher_model.py:196-204`).

Patch only this teacher-manager boundary:

1. Add `distillation.teacher_models.<name>.inference.name=external_arle`.
2. Add one endpoint field, for example
   `distillation.teacher_models.<name>.inference.engine_kwargs.external_arle.url`.
3. In config validation, allow `external_arle` when `use_topk=True`; it has no
   vLLM `max_logprobs` boot cap.
4. In trainer initialization, if all configured teachers are external, skip
   `Role.TeacherModel` resource-pool allocation and construct an
   `ExternalArleTeacherClient` dict directly.
5. Keep `AsyncTeacherLLMServerManager` unchanged by returning an object with the
   same `extra_fields["prompt_ids"]` and `extra_fields["prompt_logprobs"]`
   contract as `LLMServerClient.generate`.

This is a small verl adapter, not a trainer rewrite.

## Suggested verl Config Shape

```bash
distillation.enabled=True
distillation.n_gpus_per_node=0
distillation.nnodes=0
distillation.teacher_models.teacher_model.key=default
distillation.teacher_models.teacher_model.model_path=arle-external
distillation.teacher_models.teacher_model.inference.name=external_arle
distillation.teacher_models.teacher_model.inference.engine_kwargs.external_arle.url=http://ARLE_HOST:PORT/v1/opd/teacher/topk-logprobs
distillation.distillation_loss.loss_mode=forward_kl_topk
distillation.distillation_loss.topk=64
distillation.distillation_loss.use_task_rewards=False
distillation.distillation_loss.use_policy_gradient=False
distillation.distillation_loss.use_chunked_topk=True
```

Rationale:

- `forward_kl_topk` is the native sparse distributional objective.
- `use_policy_gradient=False` follows verl's own warning for top-k KL.
- `use_task_rewards=False` isolates teacher distillation first; rewards can be
  added after teacher contract correctness is proven.
- `use_chunked_topk=True` is conservative for long-context 4B training because
  it avoids a full `[B, T, V]` log-softmax buffer.

## Acceptance Gates

### Offline contract gate

No GPU required.

1. Fake ARLE endpoint returns deterministic `prompt_ids` and `prompt_logprobs`
   for two sequences with different prompt/response lengths.
2. `ExternalArleTeacherClient.generate` returns the same `extra_fields` shape
   expected by `AsyncTeacherLLMServerManager`.
3. `_pad_teacher_outputs` pads to `[1, prompt_width + response_width, topk]`
   without shape errors.
4. `left_right_2_no_padding` converts teacher tensors to nested tensors; FSDP
   `compute_forward_kl_topk` returns finite losses on fake student logits.

### Alignment gate

No GPU required if using synthetic logits.

1. Build a toy sequence with known logits.
2. Convert ARLE service output into verl `prompt_ids`/`prompt_logprobs`.
3. Verify the row convention matches verl's vLLM extraction convention:
   first no-context token skipped, one dummy row appended, final row count equals
   `len(input_ids)`.
4. Verify only response positions contribute to loss through the existing
   response mask.

### Remote teacher gate

Run only after TP logits export exists and H20 is available.

1. Start ARLE DSv4-Flash TP=8 teacher.
2. Call the top-k endpoint on a fixed token sequence twice; require identical
   shapes, finite logprobs, sorted top-k, and `sum(exp(logprobs)) <= 1.0`.
3. Compare one short sequence against dense `RawLogits` top-k on rank0 to prove
   the external top-k endpoint is an adapter over the real teacher logits.
4. Run one verl dry step with external ARLE teacher and fake/small data; no
   student quality claim yet.

## Open Risks

- Tokenizer identity must be exact. The verl student tokenizer and ARLE teacher
  tokenizer must map the same prompt/response text to the same ids, or the
  teacher scores the wrong sequence.
- The top-k row shift is easy to get wrong. The adapter must mimic verl's
  existing prompt-logprob convention rather than an intuitive next-token table.
- A Ray-external teacher needs failure semantics: timeout, retry budget, and a
  clear "fail the step" behavior. Silent zero-filled teacher rows would corrupt
  training.
- Full-head V1 computes dense logits on rank0 before top-k. This is acceptable
  for first correctness, but long-context throughput should move to V2
  vocab-sharded/top-k collectives if dense projection or host transfer becomes
  the bottleneck.

## Implementation Order

1. Land ARLE TP collective logits export per the TP logits spec.
2. Add ARLE top-k service adapter over that export; keep dense logits internal.
3. Add the tiny verl `external_arle` teacher client/config bypass.
4. Run offline contract/alignment gates.
5. Run one remote teacher correctness gate.
6. Only then start OPD capability training with verl FSDP student + ARLE teacher.
