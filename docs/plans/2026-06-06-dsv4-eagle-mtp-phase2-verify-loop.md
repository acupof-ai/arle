# DSv4 EAGLE/MTP Phase 2 — greedy verify loop + KV rollback (the 1.9× decode lever)

## Superseded by later evidence

**The verify loop here LANDED CORRECT (A1, `25a92e8a`) but the "1.9× decode lever"
goal did NOT hold, and the per-token verify is −32%.** The verify-loop + rollback
state machine in this doc was implemented and made correct (full mutated-buffer
rollback: [`../experience/wins/2026-06-06-dsv4-eagle-rollback-fix-correct.md`](../experience/wins/2026-06-06-dsv4-eagle-rollback-fix-correct.md)),
but spec decode is parked at the **draft-quality wall** — 39% accept vs SGLang's 68%,
so the amortization math never pays off:
[`../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md`](../experience/errors/2026-06-06-dsv4-mtp-perf-acceptance-workload-blockers.md).
The s_q=K amortization detail is in
[`2026-06-06-dsv4-a2-sqk-verify-detail.md`](2026-06-06-dsv4-a2-sqk-verify-detail.md)
(also superseded), and the corrected approach is the frozen-KV redesign
[`2026-06-06-dsv4-frozen-kv-mtp-redesign.md`](2026-06-06-dsv4-frozen-kv-mtp-redesign.md).
6ms-via-spec is re-anchored on the
[H20 reference baseline](2026-06-06-dsv4-h20-reference-baseline.md). Kept for history
(the depth-1 state machine is correct and shipped, default-off `ARLE_DSV4_SPEC_DECODE`).

---

**Date:** 2026-06-06. **Goal:** turn the loaded MTP draft head (Phase 1,
`2e0cde16`) into a working speculative-decode loop: 1 base forward → up to 2
committed tokens at acceptance α, ~(1+α)× decode throughput (target 1.5–1.9×,
26.6 ms → ~14 ms/token effective). This is the single biggest remaining decode
lever — all raw-kernel levers stacked stay above 20 ms; only spec-decode reaches
6 ms-class.

## Why Phase 2 is small now (the blocker that isn't)

The feared blocker — "DSv4 decode is single-token FlashMLA; verifying a draft
needs a new K-token-at-`start_pos` attention kernel" — **does not exist**.
`forward_tokens(slot, tokens, start_pos, …)` (`dsv4.rs:676`) already takes a
**multi-token** slice at arbitrary `start_pos` under the contiguous-append
invariant `slot.seq_len == start_pos` (`dsv4.rs:721`); `seq_len = tokens.len()`,
and `seq_len > 1` skips the single-token decode-graph/scratch branches
(`dsv4.rs:742/748`) into the multi-token prefill-style forward (causal among the
K + against the cached prefix). `truncate_slot` (`lib.rs:258`, tested) already
rolls KV back and resets `slot.seq_len`. So Phase 2 = **one contained forward
variant + a state machine**, no new kernel.

## The one forward change — all-position greedy logits

`forward_tokens_impl` folds + lm_heads **only the last row** (`dsv4.rs:1048-1091`,
`head_hidden_from_stream(&stream, seq_len-1, …)` → norm → `lm_head_project` →
`sample_cuda_token`). Add `forward_tokens_verify(slot, tokens, start_pos)
-> (Vec<u32> argmax_per_pos, Vec<DeviceVec> hidden_per_pos)`: identical body, but
at the head stage **loop `row in 0..seq_len`** — `head_hidden_from_stream(&stream,
row, …)` → `rms_norm_vec` → `lm_head_project` → **argmax** (greedy; verify is
greedy, no sampling) — collecting K tokens and K row-hiddens. `stream` (the wide
HC stream) is alive via keepalive at that point. K is tiny (2 for depth-1), so a
per-row loop is fine; a batched lm_head over K rows is a later optimization.

## The greedy depth-1 verify state machine (executor decode loop)

DSv4 ships `num_nextn_predict_layers=1` → depth-1 → each verify forward processes
**2 tokens** `[pending, draft]`. Steady state, with `(p, pending, h)` =
(committed seq_len, token to place at position p, hidden at position p−1 for the
MTP draft):

1. `d = mtp_forward(h, pending, pos)`  — draft for position p+1.
2. `(argmax, hiddens) = forward_tokens_verify(slot, [pending, d], start_pos=p)`
   — appends KV at p, p+1; `slot.seq_len = p+2`. `b = argmax[0]` (base prediction
   at pending's position p → for p+1), `c = argmax[1]` (bonus, base prediction at
   p+1 → for p+2).
3. **Accept (`b == d`):** emit `pending`, `d`. Both KV valid (correct inputs).
   `pending = c`, `h = hiddens[1]`, `p += 2`. → **2 tokens / 1 verify forward.**
4. **Reject (`b != d`):** emit `pending` only. Position p+1's KV was written with
   the wrong input `d` → `truncate_slot(slot, p+1)` (keep pending's KV at p, drop
   p+1). `pending = b`, `h = hiddens[0]`, `p += 1`. → **1 token / 1 forward.**
5. EOS / max-new check after **each** emitted token (accept emits 2 — check both).

Prime once with the normal prefill (`forward_tokens_with_hidden` over the prompt)
→ initial `(pending, h)`. The non-spec path is unchanged; spec is gated
`ARLE_DSV4_SPEC_DECODE=1` (the flag already loads the MTP head).

## Correctness gate — greedy identity (deterministic, Codex self-checks)

Greedy verify accepts a draft **iff** it equals the base argmax, so greedy
spec-decode must emit a **byte-identical token sequence** to greedy non-spec.
Gate: `dsv4_parity` (or the multigpu parity harness) with `ARLE_DSV4_SPEC_DECODE=1`
vs `=0`, **same prompt, greedy** → assert identical `clean_tokens`. Any divergence
= a verify/rollback bug (wrong position, KV not truncated, hidden picked from the
wrong row). This is the hard gate; it must pass before any perf claim.

## Perf license (Claude owns)

Acceptance rate α (count accept vs reject over the decode) + wall-clock tok/s
**same-binary env-A/B** (`ARLE_DSV4_SPEC_DECODE` 1 vs 0) on the 64-tok decode,
TP=8/EP=8. Expected ~(1+α)× — report α and the measured speedup; license the
default-flip only if the wall-clock win holds (not the theoretical α). The
comm-overlap (`1b0222e7`) compounds here: the 2-token verify forward makes the
overlapped shared expert 2× larger, so re-run the comm-overlap A/B **under spec**
once Phase 2 lands.

## Files

- `crates/infer-cuda/src/dsv4.rs` — `forward_tokens_verify` (new, mirrors
  `forward_tokens_impl` head stage); reuse `mtp_forward`.
- `crates/infer-cuda/src/executor.rs` (or wherever the DSv4 decode loop drives
  `forward_tokens`) — the verify state machine, gated on `dsv4_spec_decode_enabled()`.
- `crates/infer-cuda/src/lib.rs` — `truncate_slot` already present; wire the
  reject path to it.
- Correctness: extend `dsv4_parity` to take `ARLE_DSV4_SPEC_DECODE` and assert
  greedy identity vs the baseline run.

## Division of labor

Deterministic state machine + the all-position forward variant + the greedy
identity gate → **Codex** (its lane: deterministic infra + correctness,
`feedback_codex_deterministic_not_perf`). Acceptance-rate + wall-clock A/B +
default-flip license → **Claude**.
