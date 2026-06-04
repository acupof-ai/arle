# `infer/` deletion gate — train's CUDA OPD-teacher surface

**Status:** the SOLE remaining blocker to deleting legacy `infer/`. `agent` and `cli`
are off direct `infer` (`d11bf3f8`, `63d968c5`); the only path left is
`cli → train → infer` and `train → infer`. `train` needs `infer-api` to expose the
CUDA OPD-teacher surface its `teacher_infer.rs` + `infer_student.rs` consume.

**Scope (from a 2026-06-04 read-only scoping pass): ~800-1200 LOC new, ~3-4 weeks,
GPU-verified.** Not a thin adapter — it ports three legacy CUDA features the rewrite
stack does not yet have. Documented here so it can be executed deliberately (or
deferred with `infer` retained as a *train-only* OPD-teacher dependency until then).

## The 4 gaps (consumer = train, CUDA-gated)

1. **Raw logits** — `forward_token_logits(input_ids,positions) -> RawLogits` +
   `RawLogits{ logits: DeviceVec, shape:[seq,vocab], device }` (`seq_len/vocab_size/
   to_host_f32/with_logits_device_ptr`). HAVE: the executor already computes a logits
   `DeviceVec` (`infer-cuda/src/executor.rs`, `sample_decode_logits`). BUILD: a
   forward variant that returns FULL `[seq_len, vocab]` (Qwen3 dense currently keeps
   only the last row for sampling) + the infer-api type/method. **Biggest risk: per
   executor (Qwen3 / Qwen3.5 / DSv4) logits-shape — verify full-seq support before
   committing; if only last-row, OPD-teacher scope shrinks or needs a full-seq
   backport.** infer-api type+method = SMALL/local; full-seq forward = MEDIUM, GPU-verify.

2. **Weight offload/reload** — `offload_engine_weights()->usize` + `reload_engine_weights()`.
   HAVE: legacy `infer/src/server_engine/loaded.rs:206`. BUILD: not in the rewrite —
   D2H evict all weight matrices (track bytes) + H2D restore, per executor variant.
   LARGE (~150-200 LOC + per-variant), GPU-verify.

3. **Per-step LoRA re-merge** — `remerge_student_lora(StudentLoraUpdate)` + the
   `StudentLora{Update,Layer,Matrices}` types. HAVE: legacy has the types + a
   *load-time* merge (`infer/src/model/qwen35/lora.rs`); the rewrite has only
   name mentions. BUILD: a pristine base-weight cache + an in-place
   `W = base + (alpha/rank)·BᵀAᵀ` re-merge on the resident Qwen3.5 q/v projections.
   LARGE (~150-200 LOC), GPU-verify (numeric correctness + OPD convergence).

4. **Import swap** — `train/{Cargo.toml, teacher_infer.rs, infer_student.rs}`:
   `infer::server_engine::*` → `infer_api::*`; drop `dep:infer`. SMALL, local — but
   gated on 1-3 landing (else a forbidden half-state: train compiles, OPD broken).

## Order / verification

A (types+stubs, local) → B (full-seq logits, GPU) → C (offload/reload, GPU) →
D (LoRA re-merge, GPU) → E (import swap + delete `infer/`). Steps B-D each need pod
verification. Deletion checklist: all three surfaces working + GPU-verified, train
integration green, `grep infer:: crates/train` empty, then remove `infer/` from the
workspace + the root dev-dep.

## Recommendation

Two honest options:
- **(a) Build it** — commit ~3-4 weeks to the OPD surface (GPU-verified). Then
  `infer/` deletes and the rewrite is the sole stack.
- **(b) Pragmatic boundary** — accept the rewrite as the serving truth (4.5/5 goal
  axes done) and retain `infer/` *solely* as train's OPD-teacher backend behind the
  narrow CUDA-OPD interface above, deleting it when (a) is done. `infer` is then a
  train-only, clearly-scoped dependency — not on any serving path.

Either way, `agent` + `cli` are already off `infer`; this surface is the only thing
keeping the crate alive.
