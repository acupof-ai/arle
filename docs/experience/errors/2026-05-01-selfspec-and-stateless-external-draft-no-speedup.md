# SelfSpec and stateless external draft do not produce CUDA spec speedup

## Context

After P2.3/P2.4, `spec_draft_k=5` was exposed as a single-token canary rather
than a true speculative decoder. The next requested direction was to stop
SelfSpec multi-token work and prioritize an external Qwen3 small draft model.

`Qwen/Qwen3-0.5B` is not a public Hugging Face repo. The nearest official
text-generation draft candidate available today is `Qwen/Qwen3-0.6B`, downloaded
locally to `infer/models/Qwen3-0.6B`.

## Root Cause

There are two separate no-speedup traps:

1. SelfSpec without MagicDec-style sparse KV is just the target Qwen3-4B running
   K draft forwards plus target verification. That is K+1 target forwards and
   cannot beat normal decode.
2. The current CUDA `DraftEngine` external path is stateless. Each
   `draft_batch()` creates a fresh draft state and prefills the full prefix
   before decoding K draft tokens. For longctx-32k, that redoes a 32k prefill
   every decode step and erases any benefit from a 0.6B draft model.

Target verification also needs rejected-suffix rollback: the live CUDA target
state is backed by `PagedKVPool`, which currently has append/free-slot
primitives but no per-slot truncate/release path for speculative suffixes.

## Fix

Do not enable either fake path:

- keep SelfSpec K>1 rejected until MagicDec sparse-KV exists
- reject `DraftMode::External` while the scheduler lacks persistent draft KV
  state and target paged-KV rollback

The real external-draft implementation must land in this order:

1. Add per-request draft-state lifecycle for the external Qwen3 draft model.
2. Prefill draft state once at admission and advance/rollback it with accepted
   target tokens.
3. Add target paged-KV provisional append plus truncate/release on rejection.
4. Add a model-side K-position verifier output for Qwen3 target logits.
5. Only then allow `--spec-draft-model external:<path>` with `spec_draft_k>1`.

## Rule

External draft support is not just a CLI enum. It is only real when both draft
KV and target verifier KV have correct lifecycle/rollback semantics.
