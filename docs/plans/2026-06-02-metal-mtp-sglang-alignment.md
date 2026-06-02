# Metal MTP SGLang alignment plan

Date: 2026-06-02
Status: active control plan
Scope: local Metal Qwen3.6 MTP speculative decode

## 0. Decision

Metal MTP stays explicit opt-in. It is not a default, not an auto-loaded path,
and not a performance win until token-level greedy parity and multi-prompt
wall-clock evidence both pass.

Current measured state:

- `--mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit` is wired.
- Draft depth 2 is the only reasonable experiment for this checkpoint because
  the draft config declares `block_size = 3`.
- Draft depth 3 and 4 are killed as production candidates for this checkpoint:
  they exceed the declared block shape and lowered acceptance enough to erase
  the verifier-call reduction.
- The 2026-06-02 long-output sweep showed MTP2 can win code prompts, regress
  essay and KV Q&A prompts, and does not yet produce byte-identical greedy
  output against the baseline.

Controlling evidence:

- `docs/experience/errors/2026-06-02-metal-mtp-depth-sweep-default-kill.md`
- `infer/src/backend/metal/runtime.rs:3124-3158`
- `infer/src/backend/metal/request_state.rs:5457-5593`
- `infer/src/backend/metal/mtp.rs:553-663`

## 1. What SGLang does

This section records the source-survey findings that must not be lost when
ARLE changes the Metal path.

### 1.1 Speculative decode is a worker path

SGLang initializes a draft/spec worker and makes it the scheduler's
`model_worker`; the target worker is passed into that wrapper. It does not run
speculative decode as an ad-hoc side branch outside the scheduler.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/managers/scheduler.py:756-807`

ARLE implication: an MTP implementation that stays as scalar per-request
fallback cannot be considered SGLang-aligned. The scheduler must see MTP rows
as a batchable decode mode, with one owner for draft, verify, accept, and
requeue semantics.

### 1.2 Frozen-KV MTP has a strict KV contract

SGLang's Frozen-KV MTP worker documents the key rule: the assistant reads
target KV only, reuses EAGLE verify input/output, and owns the recurrent draft
loop because there is no assistant-side KV extension.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker.py:14-18`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker.py:121-131`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_utils.py:35-90`

ARLE implication: keep the current "draft owns no persistent KV" rule. Do not
add draft KV unless a separate design proves it is a different algorithm.

### 1.3 Frozen-KV positions use the last committed target slot

SGLang sets the draft RoPE phase from `seq_lens - 1`, and repeats it for top-k
frontiers when needed.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_utils.py:93-151`

ARLE implication: the current Metal rule "RoPE uses `target_cache_len - 1` for
the frozen-KV draft step" is correct and must remain a parity invariant.

### 1.4 Target verify preallocates the whole draft block, then accepts

SGLang's EAGLE verify path assigns draft token IDs to the target batch,
allocates KV slots for the whole verify block, writes request-to-token mapping,
runs target verify, computes accept indices, frees rejected slots, moves
accepted cache slots when needed, and advances `seq_lens` by accepted count.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/eagle_info.py:123-170`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/eagle_info.py:242-348`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/eagle_info.py:448-617`

ARLE implication: packed MTP cannot only return a scalar next token. It needs
row-wise accepted counts, accepted token spans, rejected-tail rollback, and
seed-hidden selection for the accepted row.

### 1.5 Tree verification and top-k are first-class in SGLang

SGLang builds tree metadata before verify and supports top-k frontiers in the
same EAGLE contract. The DeepSeek docs use top-k 1 by default, but the code
keeps the tree structures.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker.py:575-653`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker.py:655-716`
- `/Users/bytedance/code/sglang/docs/basic_usage/deepseek_v3.md:175-198`

ARLE implication: keep ARLE's next local step linear and greedy first. Do not
start top-k/tree work before greedy parity and packed verify are licensed.

### 1.6 Draft loop graphing is an optimization, not the first missing piece

SGLang captures the recurrent Frozen-KV draft loop with fixed CUDA graph
buffers for request indices, positions, sequence lengths, top-k state, and
hidden states.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py:54-149`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py:253-323`
- `/Users/bytedance/code/sglang/python/sglang/srt/speculative/frozen_kv_mtp_cuda_graph_runner.py:329-414`

ARLE implication: do not chase MLX draft-loop graphing first. The local MTP
profile shows draft time is small relative to verify/block overhead; packed
target verify and parity gates come first.

### 1.7 SGLang exposes accept metrics

SGLang reports average speculative accept length in server info and records
accept/correct-draft counters at request level.

Evidence:

- `/Users/bytedance/code/sglang/python/sglang/srt/managers/scheduler.py:3432-3439`
- `/Users/bytedance/code/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py:669-688`
- `/Users/bytedance/code/sglang/python/sglang/srt/managers/tokenizer_manager.py:2167-2187`

ARLE implication: every Metal MTP benchmark must report acceptance next to
TTFT, TPOT, total latency, and throughput. A latency table without acceptance
is not usable evidence.

### 1.8 SGLang is still fixing correctness edges

Local `/Users/bytedance/code/sglang` is behind `origin/main` by 45 commits as
of this note. The fetched origin range includes EAGLE KV-canary work and a
chunked-prefill next-token chain fix. That does not change the Frozen-KV MTP
contract above, but it is a warning: speculative decode correctness needs
canaries, not text-snippet inspection.

ARLE implication: a token-level parity harness is a prerequisite, not optional
polish.

## 2. ARLE Metal MTP current state

What is already real:

- split draft model validation and loading;
- one-layer Qwen3.6 MTP drafter;
- frozen target KV read by the draft layer;
- target verify through the C++ Qwen3.5/Qwen3.6 compiled model;
- GDR tape rollback on partial accept;
- seed hidden refreshed from the accepted verifier row;
- acceptance metrics logged at request cleanup;
- server-level MTP counters in Prometheus, `/v1/stats?format=json`, and
  `metal_bench` artifacts.

Evidence:

- `infer/src/backend/metal/mtp.rs:151-230`
- `infer/src/backend/metal/mtp.rs:553-663`
- `infer/src/backend/metal/request_state.rs:5457-5593`

Current gaps:

- MTP rows are explicitly scalar-only in the scheduler runtime.
- MTP rows bypass the standard packed/double-buffer decode fast path.
- Greedy baseline and MTP2 are deterministic independently, but not
  byte-identical to each other on the checked long-output prompt.
- Acceptance is prompt-sensitive and too low for depth 3 or 4.
- Per-request tokenizer-manager-style MTP metadata is not yet exposed.

Evidence:

- `infer/src/backend/metal/runtime.rs:3124-3158`
- `infer/src/backend/metal/request_state.rs:5507-5565`
- `docs/experience/errors/2026-06-02-metal-mtp-depth-sweep-default-kill.md`

## 3. Non-negotiable invariants

These are the invariants future MTP work must preserve.

1. Draft owns no persistent KV cache.
2. Draft attention reads committed target full-attention KV only.
3. Frozen-KV draft RoPE phase is the last committed target slot.
4. Target verify is the only path allowed to commit target KV/GDR.
5. Accepted input count is `matched_prefix + 1`; the verifier next token is
   still emitted after the accepted draft prefix.
6. GDR rollback is required on partial accept. Length truncation alone is not
   enough for recurrent state.
7. The next MTP seed hidden is the target verifier final hidden at
   `accepted_inputs - 1`, after final RMSNorm and before `lm_head`.
8. Raw target-step TPOT and effective speculative output-token TPOT must be
   reported separately.
9. Any benchmark must include acceptance length/rate and per-case latency
   deltas. Average-only tables are not sufficient.
10. A default flip requires exact greedy token parity under `temperature=0`,
    not merely coherent-looking text.

## 4. Execution order

### P0 - Keep the opt-in boundary

Status: done.

Keep `--mtp-draft-model` explicit. Do not auto-enable MTP from target model
metadata and do not treat the split MTP checkpoint as a target replacement.

License gate: already killed as default by the 2026-06-02 depth sweep.

### P1 - Token-level greedy parity harness

Add a local Metal harness that runs the same prompt through baseline target
decode and MTP verify under `temperature=0`, then records:

- baseline target token IDs;
- MTP emitted token IDs;
- first divergence position;
- draft block tokens;
- matched prefix length;
- accepted input count;
- verifier next token;
- final seed row index;
- target cache length before and after verify.

PASS: baseline and MTP2 token IDs match for the full generated window across
essay, code, debugging, QA, and ops prompts.

KILL or block: first divergence cannot be explained by sampling settings,
prompt template drift, or known benchmark artifact.

### P2 - Observability parity

Status: partial landed on 2026-06-02. Server metrics and `metal_bench`
now expose MTP block/acceptance/scalar-fallback counters. Tokenizer-manager
style per-request surfaced metadata is still deferred behind the parity
harness.

Expose Metal MTP counters in server stats and bench output:

- blocks;
- block size;
- accepted input sum;
- correct draft sum;
- verify count;
- suffix accept rate;
- average accept length;
- scalar fallback count for MTP rows.

PASS: every benchmark artifact can explain a speedup or regression from both
latency and acceptance.

### P3 - Packed target verify for MTP rows

Prototype a row-batched MTP verify path before touching draft-loop graphing.
Reuse the existing Metal C++ verify primitives where possible:

- `CppQwen35Model::verify_block_batched_sampled`;
- `qwen35_rollback_to_accepted_varlen`;
- the DFlash packed verifier shape in `execute_qwen35_dflash_packed_batch`.

Required design points:

- collect only rows with compatible block size and sampling mode;
- build a `[B, block]` draft-token matrix;
- pass row-wise cache positions and RoPE offsets;
- return row-wise accepted counts and next tokens;
- replay or rollback GDR per row;
- update each request's token buffer and seed hidden;
- preserve scheduler finish/requeue semantics.

PASS: same-binary same-prompt A/B, c=2/4, `max_tokens >= 192`, MTP2 improves
total latency by at least 10 percent on the multi-prompt average, with no
per-case regression above 10 percent and exact greedy parity intact.

KILL: packed verify cannot beat scalar MTP and standard decode after acceptance
and token parity are both controlled.

### P4 - Scheduler-level speculative row ownership

If P3 passes, move MTP row handling out of per-request scalar fallback and into
one scheduler-owned speculative executor. This is the SGLang alignment step:
the executor owns draft, verify, accept, rollback, metrics, and requeue.

Do not create parallel old/new paths. If this lands, scalar MTP should become
a fallback only for unsupported row shapes, with explicit fallback metrics.

### P5 - Draft-loop graph or fixed-buffer optimization

Only after P3/P4 pass, evaluate MLX-side fixed-buffer or compile/cache reuse
for the MTP draft loop.

PASS: measured draft+verify wall-clock drops enough to improve end-to-end
latency, not just a narrow draft-window timing.

KILL: if verify/block overhead remains dominant or MLX compile reuse is not
available, stop and return to standard Metal decode optimization.

## 5. Benchmark protocol

Use the canonical target model:

```text
mlx-community/Qwen3.6-35B-A3B-4bit
```

Use the split draft model only as drafter:

```text
--mtp-draft-model mlx-community/Qwen3.6-35B-A3B-MTP-4bit
```

Minimum local matrix:

| Workload | Prompts | max_tokens | temperature | Required metrics |
| --- | --- | ---: | ---: | --- |
| short sanity | 1 | 32 | 0 | TTFT, TPOT, total, acceptance |
| long mixed | essay/code/debug/QA/ops | 192 | 0 | per-case deltas, acceptance |
| quality window | essay/code/QA `/no_think` | 512 | 0 | parity hashes, first divergence |
| concurrency | same prompt, c=2/4 | 192 | 0 | scalar fallback count, packed win |

Report raw target-step TPOT separately from effective output-token TPOT. If
the path is speculative, acceptance rate is part of the metric definition.

## 6. What not to do

- Do not auto-enable MTP based on model metadata.
- Do not spend time on draft depth 3 or 4 for the current split checkpoint.
- Do not use text snippets as correctness evidence.
- Do not compare speculative effective TPOT to raw baseline TPOT without
  labelling both.
- Do not chase draft-loop graphing before packed target verify is licensed.
- Do not claim SGLang alignment while MTP rows still route through scalar
  `execute_decode_single`.

## 7. SOLID gaps

Known gaps that remain deliberately deferred:

- SGLang origin/main should be re-read after syncing the local checkout before
  copying any newer canary details.
- The first ARLE code change after this plan must be the parity harness, not a
  hot-path optimization.
- Exact line-level callgraph for `tokenizer_manager` per-request metadata was
  surveyed but not imported into ARLE yet; it belongs in P2.

This is acceptable because this document is a control plan, not a performance
claim.
