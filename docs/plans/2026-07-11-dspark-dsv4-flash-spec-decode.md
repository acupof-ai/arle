# Plan — DSpark for DeepSeek-V4-Flash (the DSv4 throughput lever)

> Status: Active (P1 arch re-derived v3 — 3-stage; T1/T2 landed on wrong arch, refitting) — 2026-07-12 · Driver: DSv4 native
> NextN-MTP nets only ~1.03× (accept-limited); DeepSeek's official DSpark reports
> **60–85% per-user speedup over MTP-1 on V4-Flash**. This is the DSv4
> high-concurrency-throughput lever, not the Qwen3.6 track (that was the substrate
> proof — [P1 license](../experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md),
> already shipped, untouched).
>
> **2026-07-12 course-correction.** The first P1-min impl was reverted: it assumed
> DSpark = "reuse the native-MTP `Dsv4Layer` AS-IS + swap fusion tensors", which is
> the WRONG architecture. Reading `DeepSpec/modeling/dspark/qwen3/modeling.py`
> (verbatim source) shows DSpark's draft is **EAGLE-style dual-stream**, not a
> single-stream MTP layer. Correct architecture below (§P1 corrected). P0 stands
> (requant done, artifact on pod). P1 deferred until the dual-stream forward is
> written from the real source. B (Qwen3.6 DSpark) is proven unaffected — the
> revert touched only DSv4-gated code, no `qwen35/*` file.

## 2026-07-13 — ROOT CAUSE CONFIRMED (SGLang DFlash reference) + reimplementation spec

TP=4 runs correct (lockstep fixed, `d4c25f1c2`) but draft **accept = 0%**. Two
disproven cheap fixes (both no-ops on the drafts): output inverse-RoPE
(`f36675d85`, reverted `88360c888`) and absolute-position-shift alone (pure
relative RoPE → a constant base shift doesn't change scores). **Dominant bug =
context STRUCTURE**, confirmed verbatim against the operational SGLang DFlash
reference (`/sgl-workspace/sglang/.../speculative/dflash_worker.py` + `models/dflash.py`):

- **Reference context = a WINDOW of ALL committed tokens.** Each committed token →
  concat its `K=len(target_layer_ids)=3` post-layer taps → `fc(K*hidden→hidden)` +
  `hidden_norm` → ONE KV entry, K-RoPE'd at that token's ABSOLUTE position, appended
  to a persistent draft KV that accumulates the whole sequence. The block attends
  ALL of it non-causally.
- **Our impl attends only the LAST token's `hc_mult` context rows.** On block 1 of
  "The capital of France is → Paris" the draft never sees "France/capital" → can't
  predict Paris → garbage. This is why drafts are unrelated to the target.
- Confirmed NON-bugs: wide-tap capture (`main_proj [hidden, 3*hidden]` folds wide
  taps per-HC-row exactly like the proven MTP `h_proj`; Explore-verified), tied-head
  (reuses `self.lm_head`), non-causal kernel (already), output de-RoPE (none in ref).

**Reimplementation (per-committed-token context window):**
1. **Accumulate context per committed token**, not per-block-from-last-token. Each
   newly committed token (anchor forward + each verify-accepted token) contributes
   its 3 taps → `main_proj` → `main_norm` → context entries appended to the
   persistent per-stage `latent_kv` at that token's slot.
2. **RoPE at ABSOLUTE token positions** (cache offset stays draft-local — separate
   the two, per DFlash: draft-cache length is for slot alloc ONLY, never RoPE).
   Block Q/K at `target_seq_len + [0..block)`; each ctx token's entries at its
   absolute position.
3. **HC-mapping choice (the one DSv4-specific unknown, no ref):** each committed
   token yields `hc_mult` context entries (from `main_proj`'s `[hidden, hc_mult]`);
   place all `hc_mult` at the token's single absolute position. Validate empirically
   (accept > 0 confirms; if still 0, try `hc_mult` consecutive positions or fold to 1).
4. Reuse the MTP tap/stream plumbing (proven on DSv4-HC); only the draft HEAD
   (3-stage + markov + confidence) differs.

Cost: substantial (slot state + per-token tap accumulation + context append +
absolute RoPE + executor wiring). Multi-cycle. Geometry 95% specced; the HC-mapping
is the one empirical unknown.

**UPDATE 2026-07-13 — GEOMETRY SOLVED (`b350b0f90`, accept 0 → 0.143).** The
absolute-RoPE-decoupled-from-cache-offset fix landed and validated on pod TP=4:
draft `[11111,84941]` vs target `[11111,1,978]` → accepted=1; `accept_rate 0.143`
(was 0.0). HC-mapping choice (all `hc_mult` at the token's single abs position)
confirmed by nonzero accept. The relative context→block offset was the bug (buffer
stride, not the true 1) — see
[wins](../experience/wins/2026-07-13-dspark-dsv4-geometry-solved-absolute-rope.md).
**Remaining lever = per-committed-token context WINDOW** (prompt-prefix seed +
within-step accepted-draft taps) to lift accept 0.143 → reference 60-85%. That's a
trunk forward-path change (multi-row tap capture at prefill + verify, mirror the
qwen35 track's `dspark_append_ctx`), NOT geometry. HC-mapping empirical unknown
resolved.

## Verdict first

Adopt **DeepSeek's official `deepseek-ai/DeepSeek-V4-Flash-DSpark`** draft module
as a new spec-decode source for our DSv4 executor. The draft backbone is one V4
MLA+MoE+HC layer (`mtp.0`) + a 3-tap fusion + Markov + confidence heads. **But it
is NOT a drop-in of native MTP** — the draft attention is EAGLE-style dual-stream
(q from a noise residual, k/v from `cat[context, noise]`, own KV cache), which
must be written new (see §P1 corrected). Reused as-is: the verify/rollback
substrate (DSv4 MTP path), the Markov/confidence procedure (`qwen35/dspark.rs`),
the `mtp.0.{ffn,hc_*}` load path.

## What DSpark-for-V4 is (verified sources)

- Official checkpoints `deepseek-ai/DeepSeek-V4-Flash-DSpark` (+ `-Pro-`) — reuse
  frozen V4 weights + a draft module. Paper arXiv:2607.05147 (DeepSeek+PKU);
  training/eval in [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT).
- **Config** (`DeepSeek-V4-Flash-DSpark/config.json`): `dspark_block_size 5`,
  `dspark_target_layer_ids [40,41,42]` (top-3 of 43 layers),
  `dspark_markov_rank 256`, `dspark_noise_token_id 128799`, `tie_word_embeddings
  false` (reuses V4 embed + output head, frozen).
- **Draft tensors** (`mtp.{0,1,2}.*`, 3 stacked DSv4 blocks in shards
  46–48-of-48; see §P1 corrected v3 for the per-stage split): each stage =
  `attn.*` (MLA) + `ffn.*` (256-expert MoE + shared + gate) + `attn_norm`/
  `ffn_norm` + `hc_attn`/`hc_ffn`. Stage extras: `main_proj`/`main_norm` @mtp.0;
  `hc_head`+`norm`+`markov_head.markov_w1/w2`+`confidence_head.proj` @mtp.2.
  Output head tied to `embed.weight` (no `lm_head`).
- **Procedure** (DeepSpec `eval/dspark/draft_ops.py` + `modeling/dspark/
  markov_head.py`, read verbatim): `_forward_backbone(target_hidden, noise_emb=
  embed(noise_ids), is_causal=False)` → crop draft-KV to `start` (noise rows
  self-heal) → `base_logits = output_head(block_hidden)` → Markov semi-AR:
  per position `logits_i += markov_w2(markov_w1[prev_token])`, sample L→R →
  confidence head: `sigmoid(conf) < threshold` truncates to the confident prefix
  (dynamic draft length) → verify block. Lossless (greedy re-check / rejection
  sampling at temp>0).

## Existing substrate (reuse verbatim)

| Piece | Where | Status |
|---|---|---|
| V4 MLA+MoE+HC layer forward (mtp.0), EP=8 expert shard | `dsv4.rs`, native MTP load/forward | shipped |
| MTP verify + rollback (`truncate_decode_len`), demote/restore preserving MTP stream | `executor/dsv4.rs` | shipped |
| Adaptive spec gate (accept EMA, skip streak) | `executor/dsv4.rs` `mtp_should_speculate` | shipped |
| MTP tensor-name + Shard contract | `deepseek-spec/src/v4.rs` `DeepSeekV4MtpTensorNames` | shipped (native flavor) |
| Block draft + Markov + confidence procedure (Rust) | `qwen35/dspark.rs` | shipped (Qwen-coupled → share/port) |
| CLI `--spec-type`, spec stats `/v1/stats` | args + server | shipped |

## Phases (license-or-kill each)

### P0 — Weights + contract — DONE 2026-07-11 (path chosen: C, requant to fp8)
1. Draft shard 46 (3.36 GB, complete `mtp.0` backbone: attn + main_proj + 256
   experts + shared + router + hc + norms; markov/confidence in 47/48, still
   downloading through a mirror outage) via `hf-mirror.com/resolve/`.
2. **Frozen-body-identity = FALSE** (measured): DSpark is fp4-requantized — its
   MoE experts are **MXFP4** (`I8 [2048,2048]`, group-32, E8M0 scale), our base
   is fp8 (`F8_E4M3 [2048,4096]`). Divergent dtype AND shape → no drop-in. ckl
   chose **option C**: requant the draft's experts → fp8, load as an all-fp8
   `mtp.0` over our fp8 base (accept the fp4-trained / fp8-served hidden shift —
   speculative-safe, measured by accept-rate).
3. **Requant done + validated** (`scripts/requant_dspark_mxfp4_to_fp8.py`, manual
   raw-byte safetensors I/O — the numpy framework can't materialize E8M0):
   Frobenius **0.0000** (fp4 magnitudes {0,.5,1,1.5,2,3,4,6} ⊂ fp8 e4m3 + pow-2
   scales ⇒ EXACT upcast, zero requant loss). Output `mtp0-fp8.safetensors`
   (6.63 GB, all-fp8), experts F8_E4M3 [2048,4096] + F8_E8M0 [16,32] block-128 —
   format-identical to base experts ⇒ loads on the existing `load_dsv4_moe_layer`
   fp8 path. **C's only residual error is the hidden-state shift, not requant.**
   (Our fp4 path is NVFP4 group-16, not MXFP4 — native fp4 load was out, which is
   why requant-to-fp8 is the zero-new-kernel route.)

### P1 — DSpark draft backbone on DSv4 (`--spec-type dspark`, single flavor)
Run `mtp.0` as a non-causal block-5 forward with the 3-tap `main_proj` fusion
(vs native MTP's 1-tap). Reuse the DSv4 layer forward; the only new draft state
is the noise-token block + the tapped `[40,41,42]` hidden. Verify/rollback
unchanged. Gate: needle x3 + same-config-twice (correct-inference), draft runs
without EP/TP desync. Kill: draft forward can't share the trunk's EP expert split.

### P2 — Markov + confidence heads on DSv4
Port the Markov (`markov_w2·markov_w1[prev]`, rank 256) + confidence
(`sigmoid(conf)<threshold` → confident-prefix length) heads — the Rust already
exists in `qwen35/dspark.rs`; refactor the procedure into a shared,
target-independent module so DSv4 + Qwen3.6 share it (delete the duplication).

### P3 — Confidence-scheduled verify length (THE throughput lever)
DeepSpec's hardware-aware prefix scheduler sets draft/verify length per request
from profiled throughput curves — more tokens when GPUs idle, fewer under load.
This is what turns a per-user latency win into a concurrency-throughput win.
Wire the confidence head's dynamic length to a load signal (batch occupancy /
scheduler pressure). Gate: c-sweep TTFT·TPOT·throughput vs native MTP.

### P4 — Pod A/B + license (TP=8, agent shape)
no-spec vs native-MTP vs DSpark, agent-shape c-sweep {1,4,8,16}. Correct-
inference gate + Δ% per metric. Target: DeepSeek's 1.6–1.85× per-user; license
the default flip only if it clears TTFT·TPOT·throughput on ≥2 binding shapes.

## P1 corrected architecture (2026-07-12 v3 — from the FULL checkpoint index + DeepSpec source)

**The draft is a 3-STAGE STACKED pipeline, not a single `mtp.0`.** The prior v2
("one dual-stream `mtp.0` block") was still wrong — it read only `mtp.0.*`. The
full `dspark_model.safetensors.index.json` (72,317 keys) + DeepSpec source
(`ds_common.py`, `ds_markov_head.py`, `ds_eval_draft_ops.py`) are ground truth.

**Draft = mtp.0 → mtp.1 → mtp.2, each a full DSv4 transformer block** (MLA attn +
256-expert MoE + attn_norm/ffn_norm + `hc_attn`/`hc_ffn`). Per-stage extras:

| Stage | Role | Stage-only tensors |
|---|---|---|
| **mtp.0** | entry: 3-tap fusion | `main_proj` `F8_E4M3 [4096,12288]`+scale (fp8-block) · `main_norm` bf16 |
| **mtp.1** | middle | none (bare block) |
| **mtp.2** | exit: sampling heads | `hc_head` · `norm` · `markov_head.markov_w1/w2` · `confidence_head.proj` |

Procedure (`forward_dspark_draft_block` + `_forward_backbone`, verbatim):
```
context = main_norm(main_proj(concat(out40, out41, out42)))   # extract_context_feature: hidden[layer_id+1], ids [40,41,42] → 3×4096=12288; fused ONCE at mtp.0 entry
noise   = embed(noise_ids)                                     # create_noise_embed: per block pos0 = anchor (last accepted) token, pos1..4 = mask 128799; embed = base embed.weight (tied, no lm_head)
block_hidden = stack[mtp.0, mtp.1, mtp.2](context, noise, is_causal=False, own draft KV cache cropped to `start` per block)
base_logits  = compute_logits(block_hidden)                   # base embed.weight^T (tied head), after mtp.2 norm(hc_head(·))
sample_draft_tokens: semi-AR L→R over block_size=5, step logits += markov_w2(markov_w1[prev_token])   # VanillaMarkov: low-rank [vocab→256→vocab], HIDDEN-INDEPENDENT (mtp.2 has NO gate_proj)
confidence   = sigmoid(confidence_head.proj(hidden, prev_tok)) ; truncate block at first pos < threshold   # dynamic draft length
```

Verified verdicts (full index + DeepSpec source, 2026-07-12):
- **3 stacked draft blocks** (not 1) ⇒ draft cost ≈ 3/43 layer-equiv per block,
  amortized over accept≈3.5 → **~2–7% of the verify forward**. Re-prices the plan
  doc's "1 layer" by 3× but is NOT a blocker; **accept-rate stays load-bearing.**
- **Own draft KV cache** — each stage has full `attn.*` (MLA wq_a/b, wkv, wo_a/b,
  q_norm, kv_norm, attn_sink); `past_key_values_draft.crop(start)` per block. NOT
  the frozen-target-KV shortcut (that's only the target-side VERIFY, which DOES
  reuse frozen KV per the 2026-06-06 amortization wall).
- **markov is VanillaMarkov, hidden-independent** — mtp.2 ships only `markov_w1`
  (`nn.Embedding[vocab,256]`) + `markov_w2` (`nn.Linear[256,vocab]`), no
  `gate_proj` ⇒ the cheap `w2(w1[prev])` transition, no hidden mixing.
- **confidence is a LEARNED head already in the ckpt** (`mtp.2.confidence_head.proj`)
  ⇒ P3's dynamic-length signal is a checkpoint tensor, not something to invent.
- **No `lm_head`** — output head tied to `embed.weight`; reuse base embed for logits.
- **main_proj is fp8-block** (`F8_E4M3`+`F8_E8M0` scale), NOT bf16 → load via the
  fp8 path, not `load_dsv4_global_matrix`.

Reusable AS-IS: `forward_tokens_verify` (`dsv4.rs:2453`) + `capture_spec_rings`/
`restore_spec_ring_tail` (`dsv4.rs:1072/1098`) for the frozen-KV target verify;
the per-stage block load path (`load_dsv4_attention`/`load_dsv4_moe_layer`/
`load_dsv4_hyper_connection`). **Written NEW**: the 3-stage backbone forward
(3× dual-stream MLA with per-block own draft KV) + main_proj fp8 fuse + markov
semi-AR + confidence truncation. This is the real P1+P2 cost.

**Landed (correct 3-stage)**: T1/T2 tensor-name + `Dsv4DsparkDraft` load scaffolding
(`3b1921a7a` → refit `59138561b`); P0 requant extended to all 3 stages
(`mtp{0,1,2}-fp8.safetensors`, 19.92 GB, Frobenius 0, `bd7bcc4e2` — codex found
mtp0-only was incomplete); **T3 3-tap capture** (`ac9152e3d`): captures the wide
HC stream at `dspark_target_layer_ids` in the eager forward, gated `is_dspark()`,
default byte-identical.

**T3 design note — DSpark decode runs EAGER (no CUDA graph)**: T3 adds
`!is_dspark()` to the decode-graph gate so the tap fires at c=1 decode (mirrors
native MTP's `last_hidden_out` eager-force). Correct for P1 (spec-decode is a
different control flow than the single-token decode graph anyway). **P3 caveat**:
when the confidence scheduler drops draft-len→0 under load, those non-speculating
steps still pay the eager tax with no spec gain — P3 must re-enable the graph path
when not speculating. **T4 must rebase on HEAD** — a concurrent session is churning
`cuda-kernels/csrc` (`9fc53e7e4`/`a07a48d90`).

**T4 attention design RESOLVED** (MLA internals map, 2026-07-12): the DSpark-DSv4
draft attn is **full MLA geometry** (`wq_b [32768,1024]`=64h×512, `wkv [512,4096]`
single compressed latent, `wo_a/b` O-LoRA) — K==V==latent, NOT the small symmetric
MHA the Qwen3.6 DSpark track uses. So the dual-stream is `latent =
cat[wkv(context), wkv(noise)]` and MLA attention runs over that small explicit
latent. Consequences: ① every existing MLA backend (FlashMLA decode/prefill,
scalar `dsv4_swa`/`dsv4_hybrid`) is **pool/ring-bound** — none takes explicit
cat K/V; ② Qwen's `nonpaged_prefill_attention` (the shipped Qwen3.6 DSpark draft
kernel) is **256-cap + symmetric** — can't hold MLA's 512 asymmetric latent →
**one new small dense MLA-latent attention kernel is the only new CUDA** (draft KV
is tiny: `context_len + block_size` rows, no paging/compression). **Structure
mirrors the shipped `qwen35/dspark.rs`** (draft `SlotState` w/ small `k_ctx`/
`v_ctx` per stage, `DsparkScratch`, block-draft over explicit KV, per-row RoPE via
`dsv4_prepare_qk_fused_batch_start_pos`), swapping only that kernel; wq_a/wkv via
`dsv4_linear`, wo via `mla_oproj`, MoE+HC via the `mtp_forward_level` pattern all
reuse. T4 = "adapt the working Qwen dspark to MLA latent geometry", not design from
scratch. Kernel is pod-gated (no nvcc on Mac).

**#2 KERNEL ITEM — 9216-key ceiling (codex kernel review P1), coordinate with T4.4.**
The kernel uses `__shared__ float logits[9216]` and rejects `kv_len > 9216`. The
draft `latent_kv.ctx_end` GROWS per block; unless T4.4 bounds it (rebase/crop —
DeepSpec does `past_key_values_draft.crop(start)`), a long prompt (needle gate uses
long context) pushes `kv_len > 9216` → deterministic fail. Fix is COUPLED with
T4.4's draft-KV window management: either T4.4 keeps the draft context bounded
(rebase/crop → `kv_len` stays small, ceiling safe + declared) OR the kernel goes
dynamic-shared-mem / tiled softmax. Decide once T4.4 lands + shows its cache model.
Handle in the pod kernel-iteration session (where nvcc can compile-test).

**#1 post-attention inverse-RoPE — IMPLEMENTED (f36675d85) then DISPROVEN as the
dominant bug (A/B, 2026-07-13).** The base DSv4 MLA attention (`dsv4_swa.cu:86-101`)
de-RoPEs the output's `rope_dim` slice at the query position before `mla_oproj`; the
draft kernel omitted it, so the hypothesis was the draft's DeepSpec-trained
`wo_a/wo_b` expect de-RoPE'd values. Added it (mirror of swa, `abs_pos =
block_start + token`). **Pod TP=4 A/B verdict: the draft ids are byte-identical
pre/post fix** (`[dspark-dbg]` dump: `anchor=603 drafts=[68745]` unchanged;
`8760→[9515,85158]` unchanged) and accept stayed 0%. The output de-RoPE is a no-op
on these drafts → the dominant garbage is UPSTREAM of the attention output. Kept the
fix (correct per swa, harmless) but it is NOT sufficient. **Remaining suspects
(ranked):** ① context-fusion tap layout (POD-VERIFY, `dspark.rs` tap concat
`513-522` reads the wide HC stream as `[row r @ r*hidden]` — unverified); ② draft
attention Q/K RoPE positions / context-K RoPE consistency; ③ MoE routing. Eliminated
so far: tied-head tensor (draft reuses `self.lm_head`, proven by correct MAIN output)
and tap-capture width (`capture_mtp_stream_hidden`, proven MTP path, `stream_dim =
hidden*hc_mult` matches). Next: DeepSpec DSv4 reference (recon) or a runtime norm
sweep (tap/context/block_hidden L2) to localize the broken stage.

**Flag #1 (V-output width) RESOLVED** (pod build + `dsv4_oproj_group_shape`
`attention.rs:6648`): the draft attention V-output = **`local_heads × head_dim`
(512), NOT nope 448**. wo_a `[8192,4096]` → `groups = local_width(32768)/wo_a_cols
(4096) = 8`; `local_attn.hidden_dim = groups × cols_per_group = 32768 = local_width
= 64h×512`. The value spans the full latent (512/head) and feeds `mla_oproj`
identically to the main model. The T4.1 stub FFI (`out = local_heads × 448`) must
change to `head_dim` (512) — resize `scratch.attn_heads` + span the kernel
weighted-sum over the full 512 — done together with the kernel impl on the pod.

**Backbone RESOLVED** (DeepSpec `dspark/qwen3/modeling.py` `_forward_backbone`,
verbatim): `hidden = noise_embed`; `context = main_norm(main_proj(concat taps))`
computed ONCE; `for stage in [mtp.0, mtp.1, mtp.2]: hidden = stage(hidden,
context)` — **context injected as k/v at EVERY stage** (`k = cat[k_proj(context),
k_proj(hidden)]`, `v` likewise, `q = q_proj(hidden)`); exit `norm(hc_head(hidden))`
(DSv4 adds hc_head vs qwen3's bare norm); `logits = hidden @ embed.weight^T`
(tied, no lm_head). Draft attention is SMALL (q_len=block_size=5, kv_len =
context_len + draft_kv) — a dense dual-stream MLA, NOT the paged FlashMLA path.
Architecture is now 100% unambiguous → T4 unblocked.

Safety patch of the first (wrong-arch) impl:
`scratchpad/dsv4_dspark_p1min_wrongarch.patch` — reference only; do not re-apply.

## Risks (named, not priced)

- **EP=8 draft MoE**: mtp.0's 256-expert MoE must shard across the same EP split
  as the trunk; the draft forward runs one extra MoE layer per block step —
  P1 measures whether the draft MoE cost eats the acceptance gain at TP=8.
- **main_proj 3-tap vs native 1-tap**: new fusion loader; the tapped layers
  [40,41,42] must be exposed from the trunk forward (we tap for native MTP at 1
  layer today — extend to 3).
- **Confidence scheduler is the novel bit**: P3 is where the throughput win
  lives and where the least prior art in-tree exists; treat as the load-bearing
  phase.
- Frozen-body-identity (P0.2) gates the cheap 10 GB download vs the full 167 GB.

## Non-goals
- Training our own V4 DSpark heads (the official checkpoint has them — no P3-train
  like the Qwen3.6 track needed).
- Eagle3 flavor (DSpark dominates it by 26–31% accept-length).
