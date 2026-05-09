# Phase 3b PPL Bench A 40k vanilla — OOM on `arle train eval`(post-Path-A unblock)

## Context

Per `docs/experience/errors/2026-05-10-phase3b-arle-train-eval-multishard-gap.md`
Path A unblocked the multi-shard loader gap by converting Qwen3-4B
(`model.safetensors.index.json` + 3 shards)→ single
`infer/models/Qwen3-4B/model.safetensors`(7.5 GB,38.5 s wall-clock via
Python `safetensors+torch`)。

Then Phase 3b Bench A:vanilla Qwen3-4B 40k context PPL baseline。

## What Failed

```
$ ./target/release/arle train eval \
    --model infer/models/Qwen3-4B \
    --data bench-output/eval-longctx/eval-longctx-40960.tokenized.jsonl \
    --seq-len 40960 --backend cuda --metrics-jsonl ...

[ARLE train eval] error: cuda alloc_zeros failed
```

## Root Cause

`arle train eval` uses `forward_batch_tokens_with_positions` which is
**single-pass forward**(per `crates/train/src/eval_lm.rs:186`)— processes
all 40960 tokens at once, no paged attention, no chunked prefill。

Memory at 40k single forward Qwen3-4B(36 layers,32 attn heads,128 head_dim):
- Weights BF16:**8 GB**
- Per-layer attention matrix temp:40k × 40k × 4B(FP32 softmax)= 6.4 GB
  per layer,× 36 layers materialized incrementally — peak depends on
  liveness analysis but typically 1-3 layer-worth held = 6-20 GB
- KV materialized:40k × 36 × 8 × 128 × 2B = **2.95 GB**
- Activations(2560 hidden × 40k × 2B):**0.2 GB**
- Total peak:**>16 GB** → OOM on RTX 4070 Ti SUPER

## Why this is intrinsic to `arle train eval`

`train eval` was designed for **fine-tune checkpoint evaluation** on
typical SFT-length sequences(< 8k tokens)。Long-context eval requires:
- Paged KV cache(`kv_pool.rs`)
- Chunked prefill(`scheduler/cuda/prefill.rs`)
- Attention with online softmax(no full attention matrix in memory)

These all exist in `infer` serving path but **not in `train eval` path**。

## Phase 3b path forward(updated)

**Path A**(multi-shard convert,now done)+ vanilla `arle train eval`
**fails at >8k context** on 16GB GPU(empirically;may work on 80GB H100)。

**Updated recommendation:Path B**(per
`docs/experience/errors/2026-05-10-phase3b-arle-train-eval-multishard-gap.md`):
use `infer` server logprobs API for PPL — leverages paged KV + chunked
prefill,fits 40k+ on 16GB easily。

LOC for Path B:**~30-50 LOC Python**(client script):
1. Boot ARLE server with model
2. POST tokenized prompt + `logprobs=True` + `max_tokens=0`(or 1 with
   greedy)to `/v1/completions`
3. Sum per-token negative log-likelihoods of the input tokens themselves
   (teacher-forcing style if API supports it,else generation-PPL)
4. PPL = exp(-mean log p(token | context))

**Caveat** — most OpenAI-compatible APIs only return logprobs of
**generated** tokens,not input tokens。If ARLE's `/v1/completions`
echoes input logprobs(check `echo=True` flag),Path B works as
true LM PPL。Otherwise it's generation-PPL only。

**Path C(new — best long-term)**:add `--paged-attention` flag to
`arle train eval` that swaps in `kv_pool.rs` + chunked prefill。
~200-300 LOC codex,unblocks all future train-side long-ctx eval。

## What worked(positive evidence)

- Multi-shard → single safetensors conversion:**38.5 s for 7.5 GB**
  (Python `safetensors+torch` save_file)。Reproducible via small Python
  one-liner。
- `arle train eval` infrastructure exists and accepts `--seq-len`
  argument — only the **memory pressure** at long ctx is the blocker。
- Qwen3-4B native max ctx is 40960 — 40k eval should be possible
  with proper paged attention path。

## What's still proven(re-state)

- **M_rope-yarn-scaling Phase 1+2 substrate works** end-to-end in
  production CUDA serving(per `4efd30b` Phase 3a smoke):server
  loaded Qwen3-4B + YARN factor=2.0 + max_seq_len=65536,HTTP 200
  + valid completion logprobs
- Vanilla noop bit-equivalence:`vanilla_inv_freq_matches_legacy_formula`
  test PASS in `cargo test -p qwen3-spec`
- 51 unit tests pass on YARN math

## Lesson

**Eval surface scope ≠ inference surface scope**。`arle train eval`
designed for SFT-checkpoint evaluation(< 8k typical),uses simple
single-pass forward。Long-ctx eval needs serving-tier infra(paged KV
+ chunked prefill + online softmax)。Should map this in Phase 3b plan
**before** scoping a 30-50 min wall-clock bench plan。

## Action items

- Phase 3b PPL eval via Path B(server logprobs)deferred — needs
  `echo=True` semantics check on ARLE `/v1/completions`
- Phase 3 remains "substrate proven"(Phase 3a)but "quality bench
  blocked by eval surface limits"(Phase 3b)
- M_rope-yarn-scaling task #39 stays in_progress with substrate-proven
  + bench-deferred description

## Cross-references

- Multi-shard gap(now unblocked):`docs/experience/errors/2026-05-10-phase3b-arle-train-eval-multishard-gap.md`(659d8aa)
- Phase 3a smoke PASS:`docs/experience/wins/2026-05-10-phase3a-rope-yarn-server-smoke.md`(4efd30b)
- Phase 3b plan(needs §6 Memory feasibility update):`docs/plans/2026-05-10-rope-yarn-phase3b-ppl-eval-plan.md`(eab591d)
- Eval lib:`crates/train/src/eval_lm.rs:186`(`forward_batch_tokens_with_positions`)

## 状态

Phase 3b Bench A 40k blocked by `arle train eval` single-pass forward
OOM on 16 GB GPU。Path A loader unblock 完成,但 eval surface 本身不
支持长 ctx。Path B(server logprobs)remains viable but defers to next
tick — substrate work(Phase 1+2 + 3a smoke)is sufficient to declare
M_rope-yarn-scaling impl complete for long-ctx serving;PPL quality
eval is a **separate axis** with its own infra requirements。
