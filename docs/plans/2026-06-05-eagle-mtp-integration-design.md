# ARLE EAGLE/MTP Speculative-Decode Integration Design (DSv4-Flash, adopt-best-first)

Lever #3 of `docs/plans/2026-06-05-dsv4-endgame-architecture-adopt-best-first.md`. **ADOPT** the in-checkpoint `mtp.0.*` draft head (no training) + SGLang's draft→tree→verify→extend loop structure. **WRITE** only ARLE-scheduler glue. Target: ~16 ms kernel base → ~8 ms (×1.93).

## 0. Key structural finding (governs everything below)

DSv4-Flash MTP is **not vanilla EAGLE**. The draft head consumes the **wide hyper-connection stream** (`hidden_size × hc_mult`), not a plain hidden vector. SGLang `deepseek_v4_nextn.py:148-155` views the carried hidden as `[n_tokens·hc_mult, d]`, runs `hnorm→h_proj` on it, `enorm→e_proj` on the new token embedding, sums them, then runs **one full `DeepseekV4DecoderLayer`** (MLA attn + HC + MoE incl. shared experts) + `hc_head` fold + `shared_head.norm` + `lm_head`. ARLE already produces exactly this wide stream: `dsv4.rs:617` "the residual is the `hidden_size * hc_mult`-wide hyper-connection STREAM"; the base forward folds it via `head_hc` at `dsv4.rs:~822` (`head_hc` fold → `norm` → `lm_head` → `sample_cuda_token`). **The draft head taps the stream one step *before* that fold.** This means: (a) the draft head reuses ~all DSv4 layer kernels already in `infer-cuda`; (b) ARLE must expose the wide-stream row, which the base forward currently discards after folding.

## 1. Loading `mtp.0` (weight layout + where)

**Tensor-name infra already exists and is correct** — `crates/deepseek-spec/src/v4.rs:889-950` `DeepSeekV4MtpTensorNames` covers `mtp.0.{enorm,hnorm,e_proj,h_proj,attn_norm,ffn_norm,norm, hc_attn/hc_ffn/hc_head_*, attn.{wq_a,wq_b,wkv,wo_a,wo_b,q_norm,kv_norm}, ffn.{shared_experts.*,experts.*,gate}}` with TP shard mapping (`v4.rs:928` `shard_for`). `config.mtp_tensor_names(0)` (`v4.rs:181`) / `config.mtp(0)` (`v4.rs:602`). Config fields present: `num_nextn_predict_layers` (`v4.rs:56`), `hc_mult` (`v4.rs:53`).

**Loaders already exist** — every primitive the MTP layer needs is in `crates/infer-cuda/src/loader.rs`: `load_dsv4_attention` (:1078), `load_dsv4_moe_layer` (:915), `load_dsv4_hyper_connection` (:1046), `load_dsv4_global_matrix` (:1061), `load_dsv4_vec` (:689). The MTP layer is structurally identical to a `Dsv4Layer` plus the `enorm/hnorm/e_proj/h_proj` front-end and `hc_head/norm` head.

**WRITE (new):** a `Dsv4MtpHead` struct in `dsv4.rs` (sibling of `Dsv4Layer`, `dsv4.rs:578`) holding: `enorm,hnorm: DeviceVec`; `e_proj,h_proj: DeviceMatrix`; one `Dsv4Layer` (compress_ratio=0, i.e. dense-attn MTP per SGLang `deepseek_v4_nextn.py:48,106` `COMPRESS_RATIO_NEXTN_LAYER=0`); `hc_head: Dsv4HyperConnection`; `norm: DeviceVec`. The `lm_head` is **shared** with the base model (`deepseek_v4_nextn.py:235` prefix `model.shared_head.head` resolves to the base lm_head row layout — reuse `Dsv4Model.lm_head`, do not re-load). Build it in `from_dsv4_fp8_safetensors` right after the layer loop (`dsv4.rs:588`), gated on `config.num_nextn_predict_layers > 0`:
```
let names = config.mtp(0);              // v4.rs:602
let mtp_layer_plan = compress_ratio=0;  // dense MTP attn
let mtp = (config.num_nextn_predict_layers > 0).then(|| Dsv4MtpHead {
    enorm: loader.load_dsv4_vec(&ctx, &names.enorm)?,  hnorm: …hnorm,
    e_proj: loader.load_dsv4_global_matrix(&ctx, &names.e_proj)?,  h_proj: …h_proj,
    layer: Dsv4Layer { hc_attn/hc_ffn/attn_norm/ffn_norm/attention/moe via the SAME loaders as :578-587 },
    hc_head: loader.load_dsv4_hyper_connection(&ctx, &names.hc_head)?,
    norm: loader.load_dsv4_vec(&ctx, &names.norm)?,
})?;
```
Store as `Option<Dsv4MtpHead>` on `Dsv4Model` (`dsv4.rs:593-607`). Sharding: `names.shard_for(config, name, tp_size)` (`v4.rs:928`) already returns correct per-rank shards (e_proj/h_proj Replicated, attn/ffn match the base layer), so the TP/EP split (`dsv4.rs:548-553`) applies unchanged.

## 2. Draft forward (one MTP layer fwd, reuse resident KV)

SGLang reference: `deepseek_v4_nextn.py:133-203` (the single MTP forward) + `eagle_worker.py:839-925` (`draft_forward`, the multi-step loop). For DSv4 with `num_nextn_predict_layers=1`, the natural config is **EAGLE topk=1, num_steps=K** (a linear chain, not a tree) — SGLang's `select_top_k_tokens` degenerates to the chain when topk=1, which keeps the verify path a single linear-prefix accept (simplest correct first cut; tree topk>1 is a later optimization).

**WRITE:** `Dsv4MtpHead::draft_forward(&self, slot, wide_stream_row, last_token, position) -> (draft_token, new_wide_stream_row)` in `dsv4.rs`. Per SGLang `deepseek_v4_nextn.py:140-201`:
1. `embed = embedding(last_token)` — reuse `ops::embedding_batch` / `embed_tokens` (`dsv4.rs:682`).
2. `e = e_proj(enorm(embed))`; `h = h_proj(hnorm(wide_stream_row.view(hc_mult, d)))` — `rms_norm_vec` (`dsv4.rs:~828`) + the existing GEMM helper. `hidden = e[:,None,:] + h` → wide `[hc_mult, d]` (`deepseek_v4_nextn.py:152-155`).
3. Run the MTP `Dsv4Layer` forward — **identical kernel sequence to a base decode layer** (`gen_mhc(hc_attn)→hc_pre→attn_norm→mla_attention→hc_post`, then `hc_ffn→ffn_norm→dsv4_moe_forward`, per `dsv4.rs:619-621`), against the **MTP layer's own KV slot** (see §2-KV below).
4. `hc_head` fold → `shared_head.norm` → **shared** `lm_head` → `argmax` (`ops::argmax`, `ops.rs:293`) for the greedy draft token. Also return the **pre-`hc_head` wide stream** as the carry for step i+1 (`deepseek_v4_nextn.py:196` `pre_hc_head = hidden_states.flatten(1)` is the carry; `eagle_worker.py:916` `hidden_states = logits_output.hidden_states`).
5. Loop K times (`eagle_worker.py:869-919`), `position += 1` each step (`eagle_worker.py:919`), collecting `draft_tokens[0..K]`.

**Resident-KV reuse — the load-bearing constraint.** The MTP layer has its **own** MLA KV (its `wkv`/`kv_norm` differ from base layers), so it needs a **second small MLA KV arena per slot** sized to the MTP layer only (1 layer vs `num_hidden_layers`). The base model's resident KV (`Dsv4SlotState.kv_arena`, `dsv4.rs:554`) is *read context* — the MTP attention attends over the same logical sequence but writes to its own 1-layer cache. **WRITE:** add `mtp_kv: Option<Dsv4MlaKvArena>` (1-layer, `Dsv4MlaKvArena::from_config` with `num_layers=1`, `dsv4.rs:79`) to `Dsv4SlotState` (`dsv4.rs:221`). During draft step i, the MTP attn appends one row at `position` to `mtp_kv` and attends over `[0..position]` of `mtp_kv`. **Note (event-tracking keepalive):** the draft loop produces async intermediates the same way the base forward does — reuse the `Dsv4ForwardKeepalive` pattern (`dsv4.rs:663`) per `reference_disabled_event_tracking_premature_buffer_free.md`, or the chained draft buffers get freed mid-kernel.

**MTP-KV prefill (draft-extend).** SGLang `eagle_worker.py:1144` `forward_draft_extend_after_decode` is the step that, after a verify commits N accepted tokens, runs **one MTP-layer extend** over those N tokens to fill the MTP KV up to the new committed length and produce the seed hidden for the next draft. ARLE mirrors this: after the verify commit (§3), run `Dsv4MtpHead` over the accepted tokens (start_pos = committed_len_before, len = num_accepted) to advance `mtp_kv`, capturing the last wide-stream row as the next draft seed (`eagle_worker.py:1251-1257` `next_draft_input = EagleDraftInput{bonus_tokens, hidden_states, …}`).

## 3. Verify step (base forward on draft chain, accept-longest-prefix == greedy)

SGLang reference: `eagle_worker.py:931-1017` (`verify`) → `eagle_info.py:242-504` (`EagleVerifyInput.verify`) → `eagle_utils.py:241` (`verify_tree_greedy_func`).

**The greedy accept core (`eagle_info.py:335-348`):** when `is_all_greedy`, `target_predict = argmax(target_logits, dim=-1)` over all `draft_token_num` positions, then `verify_tree_greedy` walks the chain accepting `draft[j]` iff `draft[j] == target_predict[parent(j)]`, stopping at the first mismatch. The accepted output is `target_predict[last_accepted]` appended (the "bonus token" — the target's own next-token at the accept frontier is always correct and always taken). **This is provably == non-spec greedy:** the target argmax at each verified position is exactly the token non-spec decode would emit; accept-longest-prefix appends the identical token sequence the base model would have produced one-at-a-time.

**WRITE (ARLE, linear-chain topk=1 — no tree kernel needed):** `Dsv4Model::verify_draft_chain(&self, slot, committed_token, &draft_tokens[0..K]) -> Vec<u32>`:
1. Run **one base-model forward** over the `K+1` row batch `[committed_token, draft[0], …, draft[K-1]]` with a causal mask, producing `K+1` logit rows — reuse `forward_tokens` extended to multi-row decode (the prefill path `dsv4.rs:680-683` already does batched embed; the seq forward already handles `seq_len>1`). The KV writes land at `committed_len … committed_len+K`.
2. `target[j] = argmax(logits[j])` (`ops::argmax`, `ops.rs:293`) for j in 0..=K — the **CPU-side linear accept** replaces `verify_tree_greedy`: `accept = [target[0]]; for j in 0..K { if draft[j]==target[j] { accept.push(target[j+1]) } else { break } }` (mirrors `eagle_info.py:457-481`). `target[0]` is always accepted (the bonus token).
3. **KV rollback of rejected rows.** Accepted length = `len(accept)`; the slot's true `seq_len` advances by exactly `len(accept)` (committed = `committed_len + accept.len()`), and rows `[committed_len+accept.len() … committed_len+K]` of the base KV must be **truncated/evicted** (SGLang `eagle_info.py:483-484,501-512` `kv_committed_len += num_accept; evict_mask`). ARLE's `Dsv4SlotState.seq_len` (`dsv4.rs:496`) is the single source — set it to the committed length; the MLA arena is append-only contiguous, so truncation = resetting the logical length (no per-row free needed). **Precondition for parity:** the base forward must write KV for the draft rows it speculated, then logically drop the rejected tail — verify the arena supports length rollback (it does: contiguous append, `seq_len` is the only cursor).

## 4. Slotting into `Engine<E,K>` / scheduler step

**The plan contract is already scaffolded** — `crates/infer-plan/src/lib.rs:24-26` has `ForwardMode::{TargetVerify, DraftExtend}`, `:62-65` `SpecPlan{draft_rows}`, `:84` `ForwardPlan.spec: Option<SpecPlan>`. The engine step (`crates/infer-core/src/lib.rs:400-433`) and seam (`infer-seam/src/lib.rs:37-77` `BackendExecutor::{submit,poll}`) are the integration surface.

**Design: spec is executor-internal, the engine plan stays per-request.** The continuous-batching scheduler (`lib.rs:417-431` admit→build_plan→allocate→submit) and overlap (`lib.rs:404-415` poll-then-build) **do not change**. One decode `DecodeRow` (`infer-plan/src/lib.rs:32`) maps to **one draft→verify→commit cycle inside `Dsv4CudaExecutor::submit`**, returning **a `Vec<SlotToken>` (1..=K+1 tokens) for that slot** instead of one. The draft/verify loop is a backend-internal expansion of a single decode row — the host scheduler is oblivious, which is the cleanest seam (matches `infer-seam` "device tensors never cross this seam").

**WRITE:**
- `infer-plan`: extend `StepOutput.tokens` semantics to allow **multiple `SlotToken` per slot per step** (already a `Vec`, `infer-plan/src/lib.rs:138`); `SlotToken` (`:122`) needs no change.
- `infer-core::apply_output` (`lib.rs:529-585`): the decode-row loop (`:572-585`) currently does `tokens_by_slot.remove(&slot)` (one token/slot). **Change to drain all tokens for the slot in commit order**, pushing each to `generated_tokens` and checking stop/length per token (`finish_reason_for`, `lib.rs:557`) — first finishing token truncates the rest (mirrors `eagle_info.py:477-481`). This is the **only `infer-core` change** and it's additive.
- `Dsv4CudaExecutor::submit` (`executor.rs:598-664`): on a decode row, if `self.model.mtp.is_some()` and spec enabled, call `draft_chain` (§2) then `verify_draft_chain` (§3), emit the accepted `Vec<SlotToken>`; else the existing single-token path (`executor.rs:646-653`). Drop the `rows==1` ensure's "single decode token" assumption only for the spec arm.
- Wiring flag: `--speculative-algo MTP --speculative-num-steps K` in `crates/cli/src/args.rs` (spec already referenced there per grep), threaded to the executor ctor (`executor.rs:107` `from_dsv4_fp8_safetensors`). Default OFF until the parity gate (§6) passes, per the plan's "no default flip without matched A/B."

**Interaction with continuous batching:** because the spec cycle is single-row-internal and DSv4's executor is **already single-row only** (`executor.rs:603-608` `rows==1`), there is zero batching conflict — spec multiplies one row's tokens-per-step. Multi-row batched spec is a follow-up gated on DSv4 batched decode landing (orthogonal to this lever).

## 5. Reuse from Medusa scaffold vs write new

| Component | Verdict |
|---|---|
| `docs/plans/M_medusa-*`, `docs/research/2026-05-08-medusa-*` | **Reuse the *concepts only*** — verified-spec == greedy invariant (`M_medusa-required-path.md:127,150`), acceptance-rate counter in `/v1/stats` (`:129`), the verify-loop mental model. The α-ceiling analysis (`:180-186`) is why MTP-trained-head beats classical self-spec — it *motivates* MTP. |
| `infer/src/speculative.rs` (721 LOC, the Medusa target) | **GONE** (verified: `LEGACY_GONE`) — the pre-rewrite tree dissolved in the `infer-core`/`infer-cuda` split. Do **not** resurrect it; write against the new seam. |
| Medusa head *training* (`M_medusa-required-path.md:96-112`, ~1 week) | **Skip entirely** — the `mtp.0` head ships pre-trained in the checkpoint. This is the whole adopt-best-first win: zero training cost. |
| `infer-plan` `ForwardMode::{TargetVerify,DraftExtend}` + `SpecPlan` (`infer-plan/src/lib.rs:24-65`) | **Reuse** — already scaffolded, exactly the right shape. |
| `deepseek-spec` MTP tensor-names + shard (`v4.rs:889-950`) | **Reuse as-is** — complete and tested (`v4.rs:1173-1174`). |
| `infer-cuda` loaders + `Dsv4Layer` + HC + MLA + MoE kernels | **Reuse as-is** — MTP layer is a `Dsv4Layer` (compress_ratio=0) + 4 small tensors. |
| Tree-attention / `build_tree_kernel_efficient` / `verify_tree_greedy` CUDA kernel | **Write NOT NEEDED for v1** (topk=1 linear chain → CPU accept). Tree (topk>1) is a perf follow-up requiring the SGLang `eagle_utils.py:127,241` kernels. |
| Draft loop + verify-accept + KV rollback glue | **WRITE NEW** (§2-§4) — the genuine gap, ~300 LOC in `dsv4.rs` + `executor.rs` + the `apply_output` drain. |

## 6. Acceptance / parity gate (accepted == non-spec greedy)

**The hard invariant: spec-on output tokens MUST be bit-identical to spec-off greedy** (`M_medusa-required-path.md:150` "verified spec — bit-exact greedy"). This is *structurally* guaranteed by §3 (target argmax + accept-longest-prefix), but must be **measured**, not asserted:

1. **Token-equivalence test (`crates/infer-cuda/examples/dsv4_parity.rs` extension, the existing parity harness `executor.rs`/`examples/dsv4_parity.rs`):** run the same prompt twice — spec-off greedy vs spec-on (K=3) — and `assert_eq!(tokens_spec_off, tokens_spec_on)` for ≥3 prompts incl. the needle-retrieval prompt (`project_dsv4_compressed_attention_longctx_bug.md`, the validated long-ctx probe). Decode the actual tokens (`feedback_validate_comparison_inputs_before_bug.md`, `errors/2026-05-26-fp8-kv-catastrophic-was-test-artifact.md`) — do **not** trust an aggregate accept-rate metric.
2. **Acceptance-rate counter** in the step output (mean accepted-tokens/step) surfaced via `/v1/stats` — **diagnostic only**, not a correctness signal. α near 0 means spec is useless (kill), α high means good speedup; neither affects correctness.
3. **KV-rollback parity:** after a verify with rejections, the slot's resident KV at `[0..committed_len]` must be byte-identical to what spec-off decode produced — add a hash/compare probe in the parity example for the first rejection case (this is where a rollback bug hides: rejected draft rows leaking into committed KV → silent divergence on the *next* token, not the current one).
4. **License-or-kill (plan §"Verify-locally gates"):** wall-clock A/B at the B=1 SLO shape, **same-binary same-shell two-flag** (`wins/2026-05-27-dsv4-native-deepep-perf-ab.md`), `strings target/release/arle | grep <mtp_symbol>` confirms the pod built the change (`errors/2026-05-28-...precond-fail.md`). Gate: tokens identical (hard) **and** ≥1.5× decode tok/s (soft, `M_medusa-required-path.md:29`). Tokens-differ → **KILL the integration, it's a bug**, not a tuning miss.

**Parity precondition checklist (must hold or accept ≠ greedy):** (a) MTP attn writes to a *separate* `mtp_kv`, never the base KV (§2); (b) `target[0]` (bonus) always accepted (§3.2); (c) base forward over the K+1 chain uses a causal mask so row j sees only `[0..j]` (else target_predict is wrong); (d) rejected-tail KV length rollback is exact (§3.3); (e) the draft seed for step 0 of the next cycle is the verify's last-accepted wide-stream row, re-derived via the draft-extend over accepted tokens (§2-extend), not a stale carry.

---

## Cited anchors

**ARLE:** `crates/infer-core/src/lib.rs:400-433` (step), `:529-585` (apply_output decode drain) · `crates/infer-seam/src/lib.rs:37-77` (BackendExecutor seam) · `crates/infer-plan/src/lib.rs:24-26,62-65,84,122-138` (ForwardMode/SpecPlan/StepOutput) · `crates/infer-cuda/src/executor.rs:598-664` (Dsv4 submit), `:870` (sample_cuda_token) · `crates/infer-cuda/src/dsv4.rs:221-294` (Dsv4SlotState incl. last_hidden/last_normed), `:540-607` (model ctor + layer loop), `:617-621` (wide-stream residual), `:623-683` (forward_tokens), `:~820-840` (head_hc fold→norm→lm_head→sample), `:663` (keepalive) · `crates/infer-cuda/src/loader.rs:689,915,1046,1061,1078` (DSv4 loaders) · `crates/infer-cuda/src/hc.rs:125,378-419` (HC gen/fold) · `crates/infer-cuda/src/ops.rs:293` (argmax) · `crates/deepseek-spec/src/v4.rs:53,56,181,602,889-950` (config + MTP names + shard).

**SGLang (`/tmp/sglang-full/python/sglang/srt/`):** `models/deepseek_v4_nextn.py:48,118-203` (MTP forward, hc_head, wide-hidden view), `:235,276-280` (shared lm_head, load_weights is_nextn) · `models/deepseek_v4.py:1814-1887` (mtp.0 weight-name remap + nextn load) · `speculative/eagle_worker.py:452-559` (forward_batch_generation draft/verify/extend), `:748-925` (draft + draft_forward loop), `:931-1017` (verify), `:1144-1257` (forward_draft_extend_after_decode) · `speculative/eagle_info.py:242-348` (verify, greedy argmax accept), `:457-512` (accept commit + KV evict) · `speculative/eagle_utils.py:241-281` (verify_tree_greedy_func).
