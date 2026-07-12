# Plan — DSpark for DeepSeek-V4-Flash (the DSv4 throughput lever)

> Status: Active (P1 reverted, re-scoped) — 2026-07-12 · Driver: DSv4 native
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
- **Draft tensors** (`mtp.0.*`, 4705 tensors in 3 dedicated shards
  46–48-of-48, cleanly separated from the 45 body shards): `attn.*` (MLA:
  wq_a/b, wkv, wo_a/b, kv_norm, q_norm, attn_sink), `ffn.*` (full 256-expert
  MoE + shared_experts + gate), `hc_*` (hyper-connections), `main_proj`/
  `main_norm` (the 3-tap fusion — DSpark flavor, vs native MTP's
  enorm/e_proj/h_proj 1-tap), `markov_head.markov_w1/w2`, `confidence_head.proj`,
  `norm`.
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

## P1 corrected architecture (2026-07-12 — supersedes the reverted reuse-map)

**The draft is NOT a native-MTP `Dsv4Layer` reuse.** Source of truth: DeepSpec
`modeling/dspark/qwen3/modeling.py` (procedure is target-independent). The draft
layer is **EAGLE-style dual-stream** on the DSv4 MLA:

```
context = main_norm(main_proj(concat(tap40, tap41, tap42)))   # [seq, hidden]; taps = layer OUTPUTS at 40,41,42
hidden  = noise_embedding = embed(noise_ids)                   # residual stream IS noise; block pos0 = anchor tok, pos1.. = mask 128799
# per draft layer (MLA), q from noise only; k/v from BOTH streams, concatenated along seq:
q = q_proj(input_layernorm(hidden))
k = cat([kv_proj(context), kv_proj(hidden)], dim=seq)
v = cat([v_proj(context), v_proj(hidden)], dim=seq)
... MLA attn + MoE + HC ...
logits = base_lm_head(self.norm(hidden))                       # reuse BASE norm + head (no bare mtp.0.norm in ckpt)
```

Verified verdicts (checkpoint index + shard-46 header on pod, DeepSpec source):
- **Noise stream is load-bearing** — not optional even at block_size=1 (pos0 =
  last accepted token embedding). The reverted impl dropped it → would collapse accept.
- **Draft has its OWN attention block** (`mtp.0.attn.*` = full MLA wq_a/b, wkv,
  wo_a/b, q_norm, kv_norm, attn_sink) ⇒ **own draft KV cache** (separate
  `Dsv4LayerAttentionState`, seeded over context, cropped on reject). NOT the
  native frozen-target-KV shortcut.
- **Final norm = base `self.norm`** — checkpoint has attn_norm/ffn_norm/main_norm
  but **no bare `mtp.0.norm`**; draft reuses base norm + head. (codex flagged a
  missing draft norm — false, verified twice.)
- **Quant: no requant beyond experts.** attn + main_proj are already fp8
  `F8_E4M3` 128×128-block (all dims ÷128) = base format; only the 256 routed
  experts are MXFP4 → requant to fp8 (done; `mtp0-fp8.safetensors` 6.63 GB on pod).

Reusable AS-IS (the scaffolding the revert also discarded — regenerate when P1
resumes): `deepseek-spec/v4.rs` DSpark config fields + `DeepSeekV4MtpTensorNames`
dspark flavor; the `mtp.0.{ffn,hc_*}` load path; `forward_tokens_verify`
(`dsv4.rs:2453`) + `capture_spec_rings`/`restore_spec_ring_tail`
(`dsv4.rs:1072/1098`) for verify/rollback; the Markov/confidence procedure in
`qwen35/dspark.rs`. **What must be written NEW**: the dual-stream MLA attention
forward (q from noise, k/v = cat[context, noise]) + the draft KV cache — this is
the real P1 cost, ~a few hundred lines, not a fusion-tensor swap.

Safety patch of the reverted (wrong-arch) impl:
`scratchpad/dsv4_dspark_p1min_wrongarch.patch` (883 lines) — reference only; the
fusion is wrong, do not re-apply.

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
