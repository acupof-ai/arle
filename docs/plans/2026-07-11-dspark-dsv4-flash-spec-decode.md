# Plan — DSpark for DeepSeek-V4-Flash (the DSv4 throughput lever)

> Status: Active — 2026-07-11 · Driver: DSv4 native NextN-MTP nets only ~1.03×
> (accept-limited); DeepSeek's official DSpark reports **60–85% per-user speedup
> over MTP-1 on V4-Flash** at matched throughput. This is the DSv4
> high-concurrency-throughput lever, not the Qwen3.6 track (that was the
> substrate proof — [P1 license](../experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md)).

## Verdict first

Adopt **DeepSeek's official `deepseek-ai/DeepSeek-V4-Flash-DSpark`** draft module
as a new spec-decode source for our DSv4 executor. It is an EXTENSION of our
existing native MTP, not a from-scratch drafter: the draft backbone is one V4
MLA+MoE+HC layer (`mtp.0`, which we already load + forward for native MTP), plus
a 3-tap target-hidden fusion, a Markov head, and a confidence head. The
procedure (non-causal block draft → Markov semi-AR sampling → confidence-
scheduled dynamic length → verify) is target-independent and already implemented
in Rust for Qwen3.6 (`qwen35/dspark.rs`); the verify/rollback substrate is
already the DSv4 MTP path.

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

## P1 implementation spec (file:line reuse map — copy-ready)

Confirmed against the DSv4 native-MTP forward (`dsv4.rs`):
- `Dsv4MtpLayer` (`dsv4.rs:370`): `layer: Dsv4Layer` (MLA+MoE+HC) reused AS-IS for
  the draft backbone. DSpark flavor swaps the fusion tensors: drop
  `enorm/hnorm/e_proj/h_proj`, add `main_proj: DeviceMatrix` (`[hidden, 3*hidden]`)
  + `main_norm: DeviceVec` + `markov: Option<..>` + `confidence: Option<..>` +
  `target_layer_ids: [40,41,42]`. Load `mtp.0.{attn,ffn,hc_*}` identically.
- `mtp_forward_level` (`dsv4.rs:5909`) is the template. Native fusion (5947–6004):
  `stream = e_proj(enorm(emb(tok))) + h_proj(hnorm(h_prev))` via
  `dsv4_mtp_add_eproj_hproj_cuda`. DSpark fusion: `stream = main_norm(main_proj(
  concat(tap40,tap41,tap42)))` combined with `noise_embed = embed(noise_ids)` —
  one new fuse kernel (analogous to the add kernel), over the block-5 positions.
- Draft driving: native runs m separate draft rows at one position; DSpark runs
  ONE non-causal block of `block_size=5` positions (mask = block sees itself +
  trunk ctx). Reuse the `Dsv4Layer` forward with a non-causal block mask.
- Markov + confidence: port the procedure from `qwen35/dspark.rs`
  (`markov_w2·markov_w1[prev]` L→R, `sigmoid(conf)<thr` → confident-prefix len)
  into a shared target-independent module — both backends call it.
- Verify/rollback: reuse `forward_tokens_verify` (`dsv4.rs:2453`) + `capture_spec_rings`/
  `restore_spec_ring_tail` (`dsv4.rs:1072/1098`) — they already verify a
  multi-token chain + roll back on partial accept; a 5-block is a chain of 5.
- Tap exposure: native taps 1 trunk layer (`mtp_frozen_target_layer_idx`,
  `dsv4.rs:6337`); DSpark needs the trunk forward to expose hidden at [40,41,42]
  — extend the single-tap capture to 3.

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
