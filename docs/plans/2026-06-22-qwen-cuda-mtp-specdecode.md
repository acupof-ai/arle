# Qwen3.6 NextN-MTP Speculative Decode — CUDA Port Plan

**Status:** scoped 2026-06-22 (parallel 4-reference Workflow); implementation in progress.
Increment 1 (tensor names) + increment 2 (`Qwen35MtpHead` struct + gated FP8 load)
**LANDED** (`0a57ce35`, `e1897d50`).
**Lever:** ~2-3x decode on Qwen3.6-27B-FP8, FP8-preserving. MTP head present in the
27B-FP8 checkpoint (mtp.fc + mtp.layers.0.*; mtp_num_hidden_layers=1).

## Measured facts (checkpoint + config, not inferred) — 2026-06-22

- **MTP tensors present & named correctly.** `model.safetensors.index.json` has
  **22** `mtp.*` tensors = the **15 logical** names `mtp_tensor_names()` generates
  + **7** `_scale_inv` companions. q/k/v/o + gate/up/down are **FP8** (scale_inv
  present); `mtp.fc.weight` has **no** scale_inv → **BF16**. `load_matrix_quant_aware`
  auto-selects FP8 vs BF16 per tensor → increment 2's load finds every tensor.
  Risk #1 reduced to *fc-concat-order + partial-rotary fidelity* (name existence
  is settled).
- **The MTP head layer is FULL attention** (`mtp.layers.0.self_attn.{q,k,v,o}_proj`)
  — the draft head never touches trunk linear-attn state.
- **The TRUNK is HYBRID: `layer_types` = 48 `linear_attention` + 16 `full_attention`.**
  ⇒ the verify pass (step 3) advances 48 layers' gated-delta recurrent state +
  conv ring. **Risk #2 (linear-attn rollback) is REAL — it does NOT evaporate.**
  This is the dominant difficulty of the port (see §3 + risk #2, rewritten below).

---

I have all the evidence needed. The DSv4 executor's `mtp_draft_tokens` already flows from CLI through `CudaExecutor::new` (executor.rs:199) into the DSv4 constructor — but the Qwen35 constructor (executor.rs:2518) ignores it. I have confirmed the seam shape (`submit`→`StepOutput.tokens: Vec<u32>`, line 848 emits one token per decode row but can emit multiple), the verify primitive (`forward_token_logits_full` already returns full per-row logits), the trunk-hidden gap (Qwen35 exposes `last_hidden` only inside `lm_head_logits`, no public `forward_tokens_with_hidden`), and the missing spec config in `qwen35-spec`. Here is the concrete plan.

---

# Qwen3.6 NextN-MTP Speculative Decode — CUDA Port: Implementation Plan

## Reference architecture (grounded, not inferred)
- The seam is `submit(plan)`→`StepOutput { tokens: Vec<u32> }` (`infer-seam/src/lib.rs:97`; emit site `executor.rs:848`). **Spec decode is 100% executor-internal** — a decode row commits N tokens instead of 1; the engine/scheduler never learns it was speculative. No seam change.
- DSv4's `executor/spec_decode.rs` is the exact orchestration template: `spec_step` (B=1) and `spec_step_batched` (B>1), with `DraftChain`/`accept_path`/`verify_schedule` as pure host logic (already unit-tested without a GPU).
- **Two architectural deltas vs DSv4** that make this NOT a copy-paste: (a) DSv4 MTP is a hyper-connection wide-stream head reading `e_proj+h_proj`; Qwen3.6 NextN is the `qwen3_5_mtp` head: `fc(concat[pre_fc_norm_embedding(embed(tok)), pre_fc_norm_hidden(h)])` → one **gated full-attention partial-rotary** transformer layer → its own hidden → shared `norm`+`lm_head` (canonical shape at `infer-metal/src/dflash.rs:239-246`). (b) Qwen runs **single-GPU dense/MoE on a contiguous full-attn KV cache**, not FlashMLA sparse rings — so the **full-attn** KV rollback is `truncate_slot`-style length reset. **BUT** the trunk is hybrid (48/64 layers linear-attn), so the **linear-attn** recurrent + conv state DOES need snapshot/restore on partial accept — like DSv4's ring, not simpler. The rollback is hybrid: length-reset for 16 full layers + snapshot/restore for 48 linear layers (see §3 buffer enumeration).

---

## 1. MTP head: struct + load (`crates/infer-cuda/src/qwen35.rs`)

- **New struct `Qwen35MtpHead`** (mirror `dsv4.rs:321` `Dsv4MtpLayer`), fields:
  - `pre_fc_norm_embedding: DeviceVec`, `pre_fc_norm_hidden: DeviceVec` (the two RMSNorms over embed/hidden before `fc`)
  - `fc: DeviceMatrix` (`[hidden, 2*hidden]` concat projection)
  - `layer: Qwen35Layer` (reuse the existing struct at `qwen35.rs:874` — a gated full-attention layer + dense/MoE FFN; partial-rotary handled by the existing attn config)
  - `norm: DeviceVec` (head's final RMSNorm before the shared lm_head)
  - Note: lm_head + embed_tokens are **shared with the base** (`self.output_projection()` / `self.embed_tokens`) — do NOT reload.
- **Add field to `Qwen35Model`** (`qwen35.rs:889`): `pub(crate) mtp: Option<Qwen35MtpHead>` and `spec_decode_on: bool`.
- **Load in `Qwen35Model::from_safetensors`** right after the layer-load loop (~`qwen35.rs:1733`, before the `Ok(Self{...})` at 1819): `if spec_decode_on { mtp = Some(load_mtp_head(...)) }`. Every projection via `loader.load_matrix_quant_aware` (line 898) — `fc` + the head layer's q/k/v/o/gate/up/down; norms via `load_matrix`/`load_dsv4_vec`-equivalent. Tensor names come from step-1.5 below.
- **Thread `spec_decode_on` into the constructor signature**: `from_safetensors(path, max_seq_len, mtp_draft_tokens: Option<usize>)`.

### 1.5 Spec config + tensor names (`crates/qwen35-spec/src/lib.rs`) — **this is net-new, qwen35-spec has zero MTP today**
- Add `pub num_nextn_predict_layers: usize` to `Qwen35Config` (parse from HF `config.json`, default 0). Mirror `deepseek-spec/src/v4.rs:69`.
- Add `fn mtp_tensor_names(&self) -> Qwen35MtpTensorNames` returning the `model.layers.{N}.*` names for the NextN block + `pre_fc_norm_embedding.weight`, `pre_fc_norm_hidden.weight`, `fc.weight`, head `norm` — match the `Qwen3.6-27B-MTP` checkpoint layout. Mirror `v4.rs:240` + `v4.rs:1133` `DeepSeekV4MtpTensorNames`. **Use the `mlx-community/Qwen3.6-27B-MTP-4bit` tensor names the Metal loader already resolves (`dflash.rs:298` `qwen3_5_mtp`) as the source of truth.**

---

## 2. MTP draft forward — `Qwen35Model::mtp_forward_level` (`qwen35.rs`, new method near `forward_token_logits_full` 2506)

Op sequence per draft level (reuse existing primitives, all already imported at `qwen35.rs:56`):
1. `embedding_batch(&self.embed_tokens, [chain_token], &mut emb)` — the candidate token's embedding.
2. `rms_norm_offset(emb, &mtp.pre_fc_norm_embedding) → emb_n`; `rms_norm_offset(h_prev, &mtp.pre_fc_norm_hidden) → h_n` (`rms_norm_offset` at 2527).
3. Concat `[emb_n ; h_n]` into one `[2*hidden, 1]` buffer (D2D memcpy into a scratch, like `dsv4.rs:4970` h_prev gather), then `gemm_batch(&mtp.fc, concat) → h'` (the `fc` projection).
4. One transformer layer over `h'`: reuse the **exact layer body** from `forward_hidden_staged` (`qwen35.rs:2161-2330`) — input RMSNorm → gated full attn (writes the MTP head's OWN per-slot KV, position = `start_pos+level`) → residual → post-attn norm → MoE/dense FFN → residual.
5. `rms_norm_offset(h_layer, &mtp.norm) → normed`; `gemv(self.output_projection(), normed) → logits`; argmax (+ top-k via existing sampler) → next `chain_token`. Return `(candidates, h_layer)` so the chain feeds the head's own hidden into level+1 (autoregressive, per `dflash.rs:244`).

`draft_chain(...)` host loop is a **direct reuse** of `spec_decode.rs:582-633` — only `mtp_forward_level` differs by backend.

---

## 3. Verify + accept + KV rollback (mirror `spec_decode.rs:242-363`)

- **Verify pass**: ONE `Qwen35Model::forward_token_logits_full(slot, ws, &chain.tokens(), start_pos)` — **this primitive already exists** (`qwen35.rs:2506`, returns `[seq_len, vocab]` device logits, every row, no sampling). It runs the **base** model layer stack over the draft chain at positions `start_pos..start_pos+depth`, writing the base KV cache for those rows. Then host-argmax each row.
- **Accept**: reuse `DraftChain::accept_path` (`spec_decode.rs:190`) verbatim — pure, already unit-tested. Returns accepted prefix + bonus.
- **KV rollback** (HYBRID — measured: 48 linear + 16 full trunk layers, so NOT pure length-reset): the verify wrote `depth+1` rows; accepted = `k`. **§0.1 buffer enumeration** (the buffers `Qwen35SlotState` mutates per verify token, `qwen35.rs:352`):
  - (i) **`k_caches`/`v_caches`** (16 full-attn layers, `max_seq_len*kv_dim` bf16) → **self-heal**: position-indexed; reset `slot.seq_len` to `start_pos + k + 1` and the stale rows `[start_pos+k, start_pos+depth]` are overwritten by the next real token at that position. Precondition: next write targets the same positions (true after the seq_len rewind). No snapshot.
  - (ii) **`gdr_states`** (48 linear layers, `Vh*Kd*Vd` **f32** gated-delta recurrent) → **MUST snapshot/restore**: advanced **in-place, content-based, no position index** (`gated_delta_rule_decode_cuda`/`_prefill_recurrent_cuda`, `qwen35.rs:4051/4067`) → cannot self-heal by rewinding a cursor.
  - (iii) **`conv_states`** (48 linear layers, `qkv_dim*(kernel-1)` bf16 ring) → **MUST snapshot/restore**: the ring window (`kernel-1`, tiny) is narrower than a multi-token draft burst → speculative shifts alias live slots → no self-heal (exactly the DSv4 EAGLE `sw_window` ring-slot lesson).
  - (iv) **`seq_len`** (host) → rewind to `start_pos + k + 1`. Trivial.
  - (v) **`spec_slot.pending`/`.hidden`** → bonus token + accepted row's base hidden.
- **MVP de-risk — start at `depth=1`** (single draft token): accept ∈ {0,1}, so the worst case is undoing **one** speculative advance → a **single pre-verify snapshot** of the 48 layers' (`gdr_states` f32 + `conv_states` bf16) into reusable scratch (sized once), restored only on reject. No per-token ladder, no re-run. Prove accept-rate + tok/s at depth=1 first, THEN extend.
- **depth>1 (the surfaced design decision — needs a parent call + GPU iteration):** verify runs `depth+1` tokens through the linear layers in ONE prefill-recurrent launch (advances state by `depth+1` atomically), so per-token rollback can't read an intermediate state. Two options: **(A) snapshot-once + re-run the `k` accepted** through the linear in_proj→conv→gdr path (`k` tiny; cost ≈ `k/(depth+1)` of the linear verify); **(B) per-token snapshot ladder** — break verify into `depth+1` single-token recurrent steps, snapshotting after each (no re-run, `depth+1`× scratch, loses the batched-prefill speedup). Pick after measuring the depth=1 accept-rate — if accept-rate is high, (A)'s re-run is cheap; if low, depth>1 may not be worth it at all.
- `spec.hidden = base verify hidden of the accepted row` (seed for next step's level-0 draft), exactly `spec_decode.rs:348`.

---

## 4. Orchestration: new file + executor wiring

- **New file `crates/infer-cuda/src/executor/qwen_spec_decode.rs`** (sibling of `executor/spec_decode.rs`). Contains: `DraftChain` (or `pub(crate)` re-export of the DSv4 one — it's model-agnostic host logic, prefer extracting to a shared `executor/spec_chain.rs`), `Qwen35CudaExecutor::spec_step` and `::spec_step_batched`, `draft_chain`, `spec_depth`/`spec_topk`/`spec_requested`. **Copy the DSv4 control flow, swap the model calls**: `capture_spec_rings`→(drop, use truncate), `mtp_forward_level`→Qwen version, `forward_tokens_verify_scheduled`→`forward_token_logits_full`, `commit_accepted_fold`→`truncate slot.seq_len`.
- **`executor.rs:2476` `Qwen35CudaExecutor`**: add fields `spec_slots: Vec<SpecSlotState>`, `spec_draft_tokens: Option<usize>`, `spec_draft_topk: Option<usize>` (mirror `Dsv4CudaExecutor` 1064-1071).
- **`from_qwen35_safetensors` (executor.rs:2518)**: add `mtp_draft_tokens: Option<usize>` param, pass to `Qwen35Model::from_safetensors`, init `spec_slots`.
- **`CudaExecutor::new` (executor.rs:186)**: pass `mtp_draft_tokens` into the Qwen35 arm (today only DSv4 gets it — line 203 vs 186).
- **`submit_decode_row` (executor.rs:2957)**: at entry, `if self.spec_requested() { return self.spec_step(row.slot, row.kv_seq_len, position) }` returning `Vec<u32>`; bubble the multi-token result up through `submit` (executor.rs:2843) into `StepOutput.tokens` (the emit site already takes a Vec). Disable the decode-graph lane when spec is on (graph captures single-token).

---

## 5. Gating (already plumbed — minimal additions)
- CLI `--spec-type mtp` / `--mtp-draft-tokens` / `--mtp-draft-topk` **already exist** (`cli/src/args.rs:502,521`; lowered at `serve.rs:248-250` into `engine_config.mtp_draft_tokens`). DSv4-only today; just route the same config to the Qwen35 executor (step 4). **Default off**: `spec_type` defaults `None` (`serve.rs:277`); `mtp_draft_tokens=None` → `spec_decode_on=false` → no MTP load, byte-identical baseline.
- Keep an `ARLE_QWEN35_SPEC_DECODE` env fallback (mirror `dsv4_spec_decode_enabled`) for bring-up only.

---

## 6. Implementation order (smallest-first, each independently commit-able + tested)

1. **qwen35-spec MTP config/names** (step 1.5). Test: unit-parse a `Qwen3.6-27B-MTP` `config.json`, assert `num_nextn_predict_layers==1` + tensor names. No GPU. **Commit.**
2. **Extract `DraftChain`/`accept_path`/`verify_schedule` to shared `executor/spec_chain.rs`**, re-export into DSv4 (no behavior change). Test: the existing DSv4 unit tests still pass. **Commit.**
3. ✅ **DONE (`e1897d50`)** — **`Qwen35MtpHead` struct + gated FP8 load** (step 1). Single-GPU only (TP errors loudly, deferred); `mtp_draft_tokens=None` path byte-identical. Tensor names verified against the live checkpoint index (22 tensors, FP8/BF16 split correct). Mac-typecheck + clippy clean. (Device-load smoke deferred: load path is reachable only via the param, which the executor passes `None` until step 7 wires the flag — first real device load happens at step 4's `mtp_forward_level` test.)
4. **`mtp_forward_level`** (step 2). Test: single draft step produces a finite top-1 token; shapes match. **First real device load of the head** (gate it behind the env fallback or a temp `Some(1)` to reach the load path). **Commit.**
5. **`forward_token_logits_full` verify reuse + `spec_step` (B=1, depth=1) + hybrid rollback** (steps 3-4 B=1, depth=1). The depth=1 MVP rollback is a **single pre-verify snapshot/restore** of the 48 linear layers' `gdr_states`+`conv_states` (§3) — no ladder, no re-run. **Commit.** depth>1 (option A/B, §3) is a separate increment after the depth=1 accept-rate is measured.
6. **`spec_step_batched` (B>1)** + executor `submit` multi-token wiring. **Commit.**
7. **CLI routing to Qwen35 executor** (step 4 wiring) + bench entry. **Commit.**

### Correctness gate (the §0 SOLID bar)
**Spec decode MUST be output-equivalent to greedy no-spec** — same prompt, `--spec-type mtp --mtp-draft-topk 1` greedy vs baseline greedy no-spec, on the **same binary/shell** (per `feedback_correct_inference_not_baseline_identity`). Because MTP verify uses the base model's own argmax to accept, greedy-vs-greedy should be **token-exact** for dense layers (no MoE non-determinism in dense Qwen) — assert exact match on ≥2 prompts. For the MoE 35B/27B target, fall back to the needle-gate + same-config-twice floor + self-consistency (token-exact is confounded by MoE atomic-scatter order). Bench entry: `scripts/bench_guidellm.sh` Δ% vs the Qwen3.6 CUDA no-spec baseline; report accept-rate + tok/s. Spec only ships if tok/s clears no-spec.

---

## 3 hardest / riskiest steps
1. **The NextN head shape + tensor-name fidelity (step 1/1.5)** — Qwen3.6 NextN ≠ DSv4 MTP (it's `concat[pre_fc_norm_emb, pre_fc_norm_hidden]→fc→gated-partial-rotary layer`, not the hyper-connection `e_proj+h_proj`). Getting `fc` input concat order, the two pre-fc norms, and partial-rotary wrong = silent garbage drafts (0% accept, not a crash). **Mitigation: lift the exact layout from the Metal `qwen3_5_mtp` loader (`dflash.rs:298`) which already runs this head correctly; A/B the draft's level-0 top-1 against the base model's next-token on a warm prompt — they should frequently agree.**
2. **Hybrid KV rollback — the dominant risk, MEASURED REAL (step 5).** The trunk is **48 linear-attn + 16 full-attn** layers (config `layer_types`), so verify advances 48 layers' `gdr_states` (f32 recurrent) + `conv_states` (bf16 ring) **in-place, content-based, no position** — they do NOT self-heal under a length truncation (per `reference_disabled_event_tracking_premature_buffer_free` + the DSv4 EAGLE-rollback anchor). The full-attn K/V caches DO self-heal (position-indexed). **This risk does NOT evaporate.** **Mitigation (de-risked): start at `depth=1`** → worst-case undo is ONE speculative advance → a single pre-verify snapshot/restore of the 48 layers' (gdr+conv) into reusable scratch (§3 buffer enumeration). Prove depth=1 accept-rate + tok/s, then decide depth>1 (option A re-run / option B ladder) — **surface to ckl before committing depth>1** (the prefill-recurrent kernel advances all `depth+1` atomically, so the intermediate-state design is a real fork in the road).
3. **Trunk-hidden plumbing for level-0 draft seed (step 2/4)** — Qwen35 currently exposes the trunk hidden only *inside* `lm_head_logits` (`qwen35.rs:2371`, `last_hidden`); there is no public `forward_tokens_with_hidden` like DSv4 has (`executor.rs:1897`). The decode step that produces the pending token must also return its post-final-norm-pre-lm_head hidden as the MTP `h_prev`. **Mitigation: add a thin `forward_tokens_with_hidden` returning `(token, last_hidden_clone)` — small, mirrors the DSv4 method exactly.**

## Files to touch
- `crates/qwen35-spec/src/lib.rs` — `Qwen35Config{+num_nextn_predict_layers}`, new `mtp_tensor_names()` + `Qwen35MtpTensorNames`.
- `crates/infer-cuda/src/qwen35.rs` — new `Qwen35MtpHead`; `Qwen35Model{+mtp,+spec_decode_on}`; `from_safetensors` MTP load + signature; new `mtp_forward_level`; new `forward_tokens_with_hidden`; reuse `forward_token_logits_full` (verify).
- `crates/infer-cuda/src/executor/spec_chain.rs` — **new**, extracted `DraftChain`/`accept_path`/`verify_schedule` (shared host logic).
- `crates/infer-cuda/src/executor/qwen_spec_decode.rs` — **new**, `Qwen35CudaExecutor::{spec_step, spec_step_batched, draft_chain, spec_depth, spec_topk, spec_requested}`.
- `crates/infer-cuda/src/executor.rs` — `Qwen35CudaExecutor{+spec_slots,+spec_draft_tokens,+spec_draft_topk}`; `from_qwen35_safetensors(+mtp_draft_tokens)`; `CudaExecutor::new` Qwen arm passes config; `submit_decode_row`/`submit` multi-token spec path; disable graph when spec on.
- `crates/infer-cuda/src/executor/spec_decode.rs` — re-point to shared `spec_chain` (no behavior change).
- `crates/cli/src/serve.rs` — relax the `--spec-type only CUDA` gate to also accept Qwen3.6 model (already CUDA-gated; just confirm it doesn't DSv4-special-case).
- `docs/experience/wins/2026-06-22-bench-guidellm-qwen36-mtp-spec.md` — bench entry (`pending-remote` stub if not runnable locally; CUDA-on-Mac).

No `infer-seam` / `infer-core` change — spec decode stays inside the executor.
