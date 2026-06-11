# DSv4 MTP spec decode — DSA compressed-KV rollback gap fixed (self-heal, SGLang-aligned); non-batched now passes the needle gate

## Context

MTP (depth-1 self-speculative) decode is already serve-wired in `executor.rs`
(`forward_decode_tokens`: `mtp_forward` draft → `forward_tokens_verify` →
accept `[draft,bonus]` / reject `truncate+restore_spec_rollback`), opt-in via
`ARLE_DSV4_SPEC_DECODE=1`. It was validated **byte-identical +61% on 2026-06-08**.
Re-validating on the current tree (`1e0f05e1`, post efficiency-tranche +
official-DSA rewrite) for productionization, it was **BROKEN**: every domain
(问答/创作/代码/客服) decoded **degenerate garbage** (loops, `或或或`, `\ \ \`),
while plain decode (spec off) was fully coherent — so a rewrite since 6-08
regressed the spec path. (ckl's hint: "KV + 位置编码 + 历史 CSA 得算出来存下来"
+ "看 sglang"; the perf table measured before this was garbage-speed, void.)

## Two bugs found (root-caused)

1. **DSA compressed-KV rollback gap (this fix).** `dsv4_dsa_official_enabled()`
   defaults ON; CompressedSparse layers write the official DSA key cache every
   decode (`csa_select_official`, `official.packed_rows = indexer_rows_after`,
   attention.rs:5438). But `capture/restore_rollback_snapshot` cover
   sw/fp8/compressor/indexer **only — NOT `dsa_official`**, and
   `truncate_decode_len`→`advance_decode_len` roll back `compressor`/`indexer`
   `compressed.seq_len` but leave `packed_rows` at its speculative value. A
   rejected draft that crossed a compression boundary leaves a stale draft key
   in `dsa_key_cache` and a desynced counter → corrupt top-k selection →
   accumulating degeneration. Classic partial-fix gap: the post-6-08 official-DSA
   rewrite added `dsa_official` without extending the rollback's mutated-buffer
   enumeration (rollback had only recently been patched for compressor/sw/fp8).

2. **Batched 2-token verify (separate, still open).** With
   `ARLE_DSV4_MTP_BATCHED_VERIFY=1` (default) the `[pending,draft]` batched
   verify garbles even **accepted** tokens (问答 86% accept still fully garbage);
   `=0` (per-token verify) fixes 问答 → a forward-compute bug in the 2-token
   batched path (the "known col1 bug" now hitting col0; likely the compressed
   positions for 2 tokens aren't boundary-floored). Tracked separately.

## What worked — SGLang frozen-KV discipline

Dispatched a study of SGLang's DeepSeek-V3.2 DSA + nextn/MTP. Verdict: SGLang
ships **zero** compressed-cache snapshot/restore. The compressed write target is
a **deterministic function of committed `seq_len`** (`write_pos =
((seq_len-1)//ratio)*ratio`, boundary-floored, written only at `seq_len %
ratio == 0`); on reject it rolls back **only `seq_len` + frees dense KV**, and
the compressed cache **self-heals** next step (re-pack overwrites). The
compressor ring is sized larger in spec mode (16/256 vs 8/128) so a rejected
draft never aliases a live committed slot. → **Fix is (B) self-heal, not (A)
snapshot/restore.**

Applied (attention.rs `truncate_decode_len`):
```rust
self.advance_decode_len(mode, ratio, total_len);
if let Some(dsa) = &mut self.dsa_official {
    let compressed_rows = total_len / ratio.max(1);
    dsa.packed_rows = dsa.packed_rows.min(compressed_rows); // self-heal, no snapshot
}
```
Clamp `packed_rows` DOWN to the rolled-back compressed-row count so the next
real decode re-packs (overwrites) the stale draft key from the restored indexer
KV. (Why not freeze the verify writes: that loses accept-amortization + the
within-batch dependency + depth-K; the DRAFT is already frozen — `mtp_forward`
is SW-only. SGLang mutates-then-self-heals, doesn't freeze the verify.)

## Validation (gate, not eyeballed samples)

- **garbage → coherent**: non-batched 客服 went from `\ \ \` to a coherent
  customer-service reply.
- **Needle correctness gate PASS** (`dsv4_lever_gate.sh`, lengths 115/446/2000 ×2,
  spec-fix nobatch vs baseline): **zero misses, zero garbage, within the baseline
  envelope** at every length (115: spec exact=2 vs baseline exact=1/partial=1; the
  `738731/738741` partials appear in baseline too = MoE non-determinism floor).
- **Discipline caught**: an eyeballed single 客服 sample looked like a "residual
  spec bug" (looping); the rigorous gate showed it was the **MoE-non-determinism
  confound** the project warns about (`feedback_correct_inference_not_baseline_identity`).

## State / Rule

- Non-batched MTP spec decode is **CORRECT** with this fix (needle gate pass).
  **Perf is NOT yet a win**: non-batched verify = 2 forwards/step → ~wash vs
  plain decode; the +61% requires the **batched** verify, which is bug #2 (open).
  So MTP stays opt-in until #2 is fixed; default flip un-licensed.
- **Rule**: any new device-state buffer added to a forward path (here
  `dsa_official`) MUST be enumerated into the spec/rollback mutated-buffer list
  in the SAME change (`feedback_design_to_implementation_detail`). And gate spec
  correctness on the **needle ladder + same-config-twice**, never an eyeballed
  greedy sample (MoE non-determinism).

## UPDATE — Bug 1 FIXED: per-token verify attention (the +61% path restored)

Root cause (exact): `forward_tokens_stream_impl` gates the **device-side
start_pos** AND the FlashMLA/fused-wqkv decode paths on `seq_len == 1`
(dsv4.rs ~1806, attention.rs 3630/4008). The batched 2-token verify
(`seq_len == 2`) therefore got `start_pos_device = None` and fell to the
**host-start_pos prefill-style compressed/DSA path**, which is incorrect for a
2-token chunk at a fully-populated mid-sequence position — garbling even
**accepted** col0 tokens (问答 86% accept fully garbage). Non-batched
(`ARLE_DSV4_MTP_BATCHED_VERIFY=0`) dodged it because each token went through
the correct seq_len==1 decode path.

Fix (dsv4.rs, `per_token_attn` flag on `forward_tokens_stream_impl`, true only
for the batched verify): run **attention per token on the seq_len==1 decode
path** (own device `start_pos` buffer — the fn-level binding already borrows
`slot.start_pos_device`), in order so token r+1 attends to r's just-written KV;
point-wise/MoE stay batched for the weight-read amortization. Mirrors the
proven `forward_decode_batch` Step-A loop. (Build also needed a forward-decl of
`get_or_build_runtime` — a latent use-before-def in the pod tree that ckl
fixed in parallel as `d7a45894`; nvcc-order bug the no-nvcc Mac lane can't see.)

Validation (8×H20, both fixes, **default batched** `ARLE_DSV4_SPEC_DECODE=1`):
- **Garbage → coherent.** The catastrophic `或或或` / `response response` is gone.
- **Perf restored** (4-domain, vs ~44 baseline): 问答 **63.3 (+39%)**, 客服
  **63.5 (+47%)** (accept 86%/93%), 代码 46.9 (+12%), 创作 47.3 (+2%) — accept-rate-gated.
- **Needle ladder ×2 (115/446/2000): zero miss, zero garbage** — clears the gate's
  KILL criteria. Caveat: 446/2000 showed `partial=2` (e.g. `738321` — right region,
  wrong last digits) where this baseline run got `exact=2`. Partials are in the
  baseline non-determinism floor too; n=2 can't separate a real 1-digit recall
  skew from MoE noise (`errors/2026-05-28-mmlu-cross-base-was-noise.md` —
  <5pp/small-n needs multi-seed). **Follow-up: multi-seed needle to confirm the
  per-token verify is recall-neutral at long context.**
- The 客服 "loop" that looked like a residual bug was the **degenerate-prompt /
  MoE-non-determinism confound** — spec-OFF 客服 ×3 ALSO loops (`;);)`,
  `…解决方案。…解决方案。`). `退化(循环) prompt 不是有效测例`.

**State**: MTP recovered from totally-broken → **correct + fast** (both batched &
non-batched pass the needle gate's miss/garbage criteria; batched gives the
speedup). Still opt-in. Default flip awaits: (a) the multi-seed long-context
recall check, (b) productionizing the env gate to a `--spec-type mtp` CLI flag (#16).

## UPDATE — #16 CLI flag landed (`9c979dd1`), pod-e2e pending-remote

`--spec-type mtp` + `--mtp-draft-tokens N` now drive CUDA DSv4 MTP (consuming
ckl's `ServeSpecOptions` serve-spec infra from `c1655675`). Plumb:
`serve_http` lowers `--spec-type mtp` → `EngineLoadConfig.mtp_draft_tokens` →
`Dsv4CudaExecutor.spec_draft_tokens` → gate `is_some() || ARLE_DSV4_SPEC_DECODE`
(env kept as fallback). `EngineLoadConfig` serializes into
`ARLE_WORKER_ENGINE_CONFIG` so the flag reaches all TP/EP ranks. Mac
`cuda,no-cuda` typecheck clean (`-p infer-api`, `-p cli`).

**pending-remote**: pod e2e — serve `--spec-type mtp --mtp-draft-tokens 1` (no
`ARLE_DSV4_SPEC_DECODE`), confirm the `[dsv4-mtp]` accept/reject markers appear
(flag engages MTP) + the depth-1 A/B holds. Needs the pod synced to recent main
(currently at `1e0f05e1` + the git-applied MTP patches; #16 builds on ckl's
`c1655675` infra which post-dates that tree). Default flip still gated on the
multi-seed recall (#20) + depth decision.

## UPDATE — #20 multi-seed long-ctx recall: MTP is RECALL-NEUTRAL (2026-06-11)

The depth-1 validation's caveat (one n=2 run at 446/2000 showed MTP `partial=2`
vs a baseline `exact=2`) is **resolved as non-determinism floor, not regression**.
Ran the needle gate (`scripts/dsv4_needle_gate.py`, runs=5 same-config repeats,
greedy temp=0, secret `738291`) against the **baseline** serve (no MTP, pod
`19383a43`, 8×H20 TP=8/EP=8, port 18188):

| len | baseline (no MTP) runs=5 | det? |
|---|---|---|
| 446  | **exact=3 partial=2 miss=0** | NONDET |
| 1000 | exact=5 partial=0 miss=0 | NONDET |
| 2000 | **exact=3 partial=2 miss=0** | NONDET |

The baseline ITSELF partials at 446 and 2000 (outputs `7381` / `738738` — it
recalls `738` then the MoE atomic-scatter order flips digit 4+ run-to-run). The
MTP `partial=2` was one draw from this **identical** distribution at the **same
two lengths**. MTP is recall-neutral by construction (batched verify commits
exactly the target's per-position argmax, so its token stream ≡ baseline greedy
except through the same non-determinism the baseline already has) and the
empirical baseline floor seals it. Vindicates the project gate: correct-inference
= needle + same-config-twice floor, NOT byte-identity (MoE non-determinism).

**Optional final seal (deferred, not blocking):** matched MTP-on runs=5 at the
same lengths to confirm the exact/partial split lands within the baseline floor.
Requires evicting the live baseline serve to free the 8 GPUs for an MTP serve —
a ckl-owned call, not done unilaterally.

## Default-on decision (#62 exit): KEEP OPT-IN, not default-on yet

Depth-1 MTP is a **validated opt-in win**, not yet a default flip:
- ✓ correctness: needle gate clean ×2, recall-neutral (this update)
- ✓ perf: +39% 问答 / +47% 客服 B=1 decode (accept 86/93%)
- ✗ default-on license INCOMPLETE: the project default-flip rule needs ≥2
  binding production shapes cleared on TTFT *and* ITL *and* output-throughput
  with wall-clock framing; MTP is benched on decode-tok/s for 2 chat shapes, no
  prefill/TTFT-regression check. Plus the current build's single-row-forward
  limitation (`DSv4 CUDA prefill/mixed forward is single-row only`) interacts
  with batched serving — the batched lane must clear before a global default.
- Depth-K: KILLED (draft-quality wall, accept 1/4 @ K=4 — see
  `errors/2026-06-11-dsv4-mtp-depth-k-draft-quality-wall.md`).

Verdict: ship `--spec-type mtp --mtp-draft-tokens 1` as the documented opt-in;
default-on revisited after (a) the batched/single-row lane lands and (b) a
multi-shape wall-clock TTFT+ITL+throughput sweep.
