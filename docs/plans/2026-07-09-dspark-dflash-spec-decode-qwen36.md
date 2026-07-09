# Plan — DSpark/DFlash draft for Qwen3.6 spec decode (OPD rollout lever)

> Status: Active — 2026-07-09 · Driver: OPD rollout is decode-bound
> (decode 80.4% of rollout wall, B=1 ~11 tok/s @45K after the sidecar
> prefix-reuse fix). Native NextN-MTP nets only **1.03×** at depth 2
> ([2026-06-23](../experience/wins/2026-06-23-mtp-replay-elimination-h20-net-win.md))
> — accept-length-limited, not verify-limited anymore.

## Verdict first

Adopt the **DFlash block drafter + DSpark Markov head** as an alternative draft
source for the existing Qwen3.6 spec-decode path. The hard substrate — verify
forward, gdr/conv snapshot, linear-only partial-accept replay (bit-equal),
full-attn cursor rewind, on-device argmax — is already built and licensed
correct. What changes is ONLY the drafter: 1-layer NextN chain (depth 2) →
7-token block draft, which is where our 1.03× is capped.

## What DSpark is (verified sources)

- **Paper**: DSpark — Confidence-Scheduled Speculative Decoding with
  Semi-Autoregressive Generation (DeepSeek + PKU,
  [arXiv:2607.05147](https://arxiv.org/abs/2607.05147)).
- **Code**: [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
  (MIT) — training + eval for three drafters: DSpark, DFlash, Eagle3. Released
  draft checkpoints: Qwen3-4B/8B/14B + Gemma4-12B (`*_block7`).
- **Mechanism**: DFlash drafts a whole K-token block in ONE parallel forward
  from mask inputs (position k can't see sampled k−1 → "suffix decay" caps
  acceptance). DSpark adds a low-rank **Markov head** — a per-position logit
  bias conditioned on the previous token — so the block samples left-to-right
  (semi-AR) at negligible cost, plus **confidence-scheduled** dynamic draft
  length. Verification unchanged → lossless (greedy re-check, or rejection
  sampling at temp>0).
- **Qwen3.6-27B prior art**:
  [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
  draft weights exist for our exact target family;
  [hikarioyama/dspark-aeon-27b](https://github.com/hikarioyama/dspark-aeon-27b)
  measured DSpark-vs-DFlash on a 27B hybrid via ABBA A/B: aggregate **+10.9%**,
  **tool-call +14.1%**, accept-rate +0.078 — biggest wins on exactly our
  workload (agentic tool-call, single-stream, temp>0). Caveats they report:
  Markov head helps at temp>0, can hurt at greedy; win compresses at high
  concurrency. B=1 temp>0 is precisely the OPD rollout regime.

## Existing substrate (what we reuse verbatim)

| Piece | Where | Status |
|---|---|---|
| Verify forward + per-row logits, on-device argmax | `qwen35.rs` spec_step | shipped |
| Recurrent rollback: `gdr_snap`/`conv_snap` + `Qwen35LinearCapture` linear-only replay | `qwen35.rs:1057-1110` | shipped, bit-equal ([06-23](../experience/wins/2026-06-23-mtp-replay-elimination-h20-net-win.md)) |
| Full-attn cursor rewind on partial accept | `qwen35.rs:901` | shipped |
| Adaptive gate (accept EMA, skip streak) | `executor.rs:1823-1831` | shipped (DSv4 lane) |
| CLI: `--spec-type`, `--mtp-draft-model`, `--mtp-draft-tokens/topk` | `cli/src/args.rs:685-738` | shipped |
| Draft-corpus source: rollout dumps (`--dump-messages-dir` + cc-convert) | scripts | shipped |

## Phases (license-or-kill each)

### P0 — Contract probe (no engine code)
1. Fetch `z-lab/Qwen3.6-27B-DFlash`; diff its config/tensor shapes against
   `Qwen3.6-27B-FP8` (vocab, hidden, rope). Mismatch with our checkpoint ⇒
   the weight is unusable as-is → P3 (train own) becomes the entry cost.
2. Read DeepSpec `deepspec/modeling` DFlash + DSpark head forward to spec the
   exact draft computation (block mask input, target-hidden conditioning,
   Markov rank-256 bias). Output: a one-page tensor-level draft-forward spec.
   Gate: shapes match + forward spec fits our loader. Kill: architecture
   requires target internals we don't expose (then Eagle3-from-DeepSpec is the
   fallback drafter, same substrate).

### P1 — DSpark superset behind `--spec-type dspark` (heads optional)
Scope per ckl 2026-07-09: the deliverable is the DSpark SCHEME — Markov head +
confidence-scheduled dynamic draft length — not DFlash. One module, both heads
optional (present iff their tensors exist in the checkpoint), `layer_types`
config-driven, so the z-lab DFlash checkpoint loads as a backbone-only DSpark
for end-to-end path validation until our own trained checkpoint adds the heads.
1. `qwen35-spec`: DSpark tensor-name contract beside `mtp_tensor_names`
   (backbone + optional `markov_w1/w2` + confidence head).
2. `infer-cuda`: block draft forward (K from config); Markov left-to-right
   block sampling (`logits_i += markov_w2·markov_w1[prev]`); confidence
   truncation seam (`confident_prefix_len`, = K when head absent); verify/
   rollback untouched (`Qwen35LinearCapture` is depth-parameterized).
3. A/B on H20, OPD rollout shape (20–45K ctx, tool-call heavy, B=1):
   no-spec vs MTP-d2 vs DSpark-backbone-only. Gates: needle x3 +
   same-config-twice (correct-inference, NOT byte-vs-baseline), tok/s Δ.
   Kill: ≤1.15× vs no-spec.

### P2 — temp>0 rejection-sampling verify
Current verify is argmax-only; OPD think-rollouts sample. Verify draws from
the exact reported draft distribution (incl. Markov bias) — required for the
rollout lane regardless of drafter. Gate: rollout-lane A/B inside a real OPD
round (tok/s + pass-rate unchanged).

### P2.5 — Prefix-restore partial-ctx drafting (OPD-decisive)
Prefix-cache-hit requests silently degrade to plain decode: after restore the
draft ctx has a gap, so `executor.rs` never re-seeds `pending`
(`df.ctx_len == row.start_pos` fails, then `ctx_len == total_tokens` fails).
At OPD's ~91% hit rate DSpark is near-inert in the rollout lane until fixed.
Draft = 4× sliding(2048) + 1× full_attention (measured, DFlash config), so a
suffix-only ctx is exact for the sliding layers once the tail ≥ window and
approximate only for the full layer. Cheapest-first:

1. **Partial-ctx (this phase)** — `Qwen35DsparkSlotState.ctx_base` absolute
   offset; draft attn `lo = max(ctx_base, …)`, buffer index = abs − base
   (RoPE already keyed to absolute positions); prefill gate resets the ctx at
   `start_pos` on gap instead of bailing; `pending` requires
   `ctx_end == total_tokens`, not coverage from 0. No flag — activates only
   where today's path degrades to plain. Gate: per-chain accept counter split
   by `ctx_base>0` vs `==0`; accept collapsing toward ~1/16 (verify overhead
   eats the win) → KILL, go to 2.
2. **Fallback: sidecar the draft ctx K/V** (exact, ~61 KB/token, roughly
   doubles sidecar) — only if 1 kills.

### P3 — Train our DSpark heads (DeepSpec, warm-start z-lab backbone)
No public DSpark checkpoint exists for Qwen3.6-27B; the Markov + confidence
heads must be trained. DeepSpec `Qwen3DSparkTrainer`, backbone warm-started
from z-lab DFlash (shape-compatible), corpus = rollout dumps (on-policy
tool-call — the aeon recipe, +14.1% tool-call vs generic). Cache math:
61.4 KB/token (6×5120 bf16) → 50–200M tokens = 3–12 TB, fits pod NVMe;
full-perfectblend-from-scratch (~76 TB) is storage-infeasible. Small patches:
`text_config` nesting, mask_token_id 248070, max_length. Refresh per N OPD
rounds to track LoRA drift.

## Risks (named, not priced)

- z-lab weights may target a different Qwen3.6-27B base revision → P0 decides.
- Draft forward cost: DFlash backbone > our 1-layer MTP head; B=1 is
  latency-bound so draft cost eats acceptance gains — that is what the P1 A/B
  measures, no pre-estimate.
- K=7 verify rows widen the verify GEMV past its depth-2 tile-matching
  ([06-22](../experience/wins/2026-06-22-tile-matched-amortizing-verify-gemv.md));
  re-measure, don't assume.
- 45K-context draft conditioning: confirm in P0 how DFlash consumes target
  hidden at long ctx (it does not re-attend the trunk context).
