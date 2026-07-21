# Plan — DSpark/DFlash draft for Qwen3.6 spec decode (OPD rollout lever)

> Status: Active — P0+P1 LICENSED, P2.5 + memory-clamp RESOLVED 2026-07-11 ·
> Driver: OPD rollout is decode-bound (decode 80.4% of rollout wall, B=1 ~11
> tok/s @45K). Native NextN-MTP capped at 1.03×; **DSpark/DFlash backbone nets
> 2.4–3.8×** vs no-spec, greedy
> ([P1 license](../experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md)).
> **P2.5 was already implemented** (`ctx_base`/`rebase` partial-ctx drafting) +
> verified holding under prefix hit (accept 0.18–0.22, 100% partial-ctx chains).
> **Draft-KV memory clamp fixed** (`1ee72d809`): full layer caps at
> `min(max_seq_len, max_total_tokens)`, 544→64 MB/slot, slots 32→256, long-ctx
> unblocked ([win](../experience/wins/2026-07-11-dspark-draft-kv-cap-per-request-ceiling.md)).
> Remaining before OPD default: P2 temp>0 rejection-sampling verify + P3 train
> the DSpark heads (Markov/confidence) for the +14.1% tool-call lift.

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
| Recurrent rollback: `gdr_snap`/`conv_snap` + `Qwen35LinearCapture` linear-only replay | `qwen35.rs:1057-1110` | shipped, bit-equal (06-23) |
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

## Next wall — small-M FP8 dense GEMM (2026-07-10)

> Verdict 2026-07-10: MIN_M 16→2 LICENSED (+5–9% dspark greedy, matched A/B);
> memoize + M=1 GEMV variants KILLED by measurement; achievable read BW is
> 3.5 TB/s (not 4.0), so the honest M=1 residual is ~1.6×.
> [wins](../experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md)
>
> **Marlin W8A16 next-lever: NO-GO (2026-07-10 research).** DeepGEMM is already
> a Hopper wgmma tensor-core FP8 GEMM and is already benched at M=1:
> 1.4–1.87 TB/s, tied with / below the 1.78 TB/s GEMV in 2/3 shapes (that's why
> MIN_M=2 keeps M=1 on the GEMV). The M=1 wall is the shared x-load + fp8→f32
> decode tail (2.8–3.0 TB/s hard ceiling), NOT the MAC path — CUDA-core→tensor-
> core cannot move it. In-tree Marlin is all W4/INT4 + sm_89 (ERR_ARCH on
> Hopper); a W8A16 port would be net-new and compile to DeepGEMM's already-measured
> class. GEMM is ~66% of M=1 (15/23 ms); even a hypothetical 3.0 TB/s kernel caps
> at ~17 ms/tok (−26%) and nothing on this stack reaches 3.0 at M=1. The only
> lever that moves M=1 is raising M — spec decode (done) and concurrency (#17).

Evidence: dense_ffn 26 ms/step profiled at M=16 vs ~3.2 ms weight-read floor
(~8×); plain decode 23 ms/tok vs ~7 ms roofline (27 GB FP8 / 4 TB/s H20).
Wins here speed BOTH plain decode (M=1 GEMV lane) and DSpark verify (M=17
DeepGEMM lane).

**Survey (adopt-official-first).** SGLang on Hopper routes FP8 block-scaled
dense GEMM to DeepGEMM `gemm_fp8_fp8_bf16_nt` for ALL M (decode M=1 included;
persistent scheduler + TMA keeps it BW-bound), Triton `w8a8_block_fp8_matmul`
as fallback. vLLM uses CUTLASS SM90 blockwise with swap-AB at M≤64, DeepGEMM
optional. We already carry the DeepGEMM native bridge
(`deepgemm_native.cu::launch_sm90_dense_nt`) — the highest-leverage adoption
is extending that lane down the M axis, not a new kernel.

**Decomposition (file:line).**
1. Per-call host overhead in the DeepGEMM bridge: every
   `dsv4_deepgemm_fp8_gemm_nt_cuda` call re-runs `get_best_config` (layout
   search), `generate_kernel_code` (multi-KB string build), `hex_digest`
   (hash of the full source), and 2× `std::filesystem::exists`
   (`deepgemm_native.cu:1334-1346,1573-1576`). At M=16 verify that is
   ~200 calls/step — hypothesis for most of the 26−4 ms gap. Fix: memoize
   `(kind, m, n, k)` → `{config, runtime}` in a host map; TMA descriptors
   stay per-call (they embed device pointers).
2. Routing floor `QWEN_FP8_DEEPGEMM_DENSE_MIN_M = 16`
   (`quant_linear.rs:25`): M∈1,16) runs the scalar warp-per-row GEMV
   (`quantized_gemv.cu::fp8_f32_block_gemv_batch_kernel`), measured ~30% of
   HBM BW at B=1 e2e. After (1), A/B GEMV vs DeepGEMM per
   M∈{1,2,4,8,16,17,32} per shape (FFN 17408×5120 / 5120×17408, attn-sized
   5120×5120) and set the constant to the measured crossover.

**Expected ceiling.** Per-op: weight-read floor + ~10 µs/call (pack_quantize
one K-pass + launch). dense_ffn @M=16 → ~4–5 ms/step (floor 4.3 ms:
64×3×89 MB / 4 TB/s). Plain decode GEMM share → near-floor; e2e gain bounded
by attention/gdr share — measured, not promised. If DeepGEMM at M∈{1..8}
loses to the GEMV (occupancy at 78 persistent CTAs), keep the GEMV where it
wins and report the measured residual gap honestly.

License gates: micro-bench table old/new/floor per M; e2e vs 07-10 anchors
(42.6–43.6 plain / 104–108 dspark greedy / 64–106 sampled); needle 738291 ×3
exact both lanes + same-config-twice self-consistency.

## Risks (named, not priced)

- z-lab weights may target a different Qwen3.6-27B base revision → P0 decides.
- Draft forward cost: DFlash backbone > our 1-layer MTP head; B=1 is
  latency-bound so draft cost eats acceptance gains — that is what the P1 A/B
  measures, no pre-estimate.
- K=7 verify rows widen the verify GEMV past its depth-2 tile-matching
  ([06-22);
  re-measure, don't assume.
- 45K-context draft conditioning: confirm in P0 how DFlash consumes target
  hidden at long ctx (it does not re-attend the trunk context).
