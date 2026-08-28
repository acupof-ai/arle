# The code-diff study pays out on day one: fence-free MoE decode, one math fix, one refuted suspicion

## Context / Goal

llama.cpp merged qwen4exp support ~36 hours before this entry (PR #27742,
commit ca3d5a3e1). Instead of continuing to optimize from first principles,
the operator's directive was to READ their implementation and lift what wins
("你可以直接抄成 sota 吗而不是自己苦苦探索"). A read-only agent diffed their
graph (`src/models/qwen4exp.cpp` + support files, fetched via
raw.githubusercontent) against ARLE's `crates/infer-vulkan` qwen4_exp lane,
producing a ranked lift list, a divergence table (D1–D7), and non-findings —
every claim with file:line receipts on both sides.

## The study's verdict in one line

The two implementations agree on essentially all of the math (HC order, PLE
gate/hash/EOS semantics, GDN, interleaved [q|gate], MoE renorm — all verified
receipt-by-receipt). The differences are structural: they have QSA and
chunked delta-net; we have MTP-in-flight, recurrent rollback, working Vulkan
decode, PLE prefetch I/O, and the oracle/parity infrastructure.

## Lift #1, landed same day: fence-free MoE decode (55.5 → 47.2 ms/token)

The 48 per-layer decode fences existed for ONE host duty: read the router's
ids so `scale0_for_route` could gather a slot-ordered `weight_scale_2` list
for the NVFP4 expert GEMV's SCALE0 fusion. The GEMV already bound the
device-side ids buffer — only the scale lookup forced the round-trip.

The fix inverts the fusion's indexing: `data_fuse0[expert_i0]` (slot) becomes
`data_fuse0[expert_id]` (the id the shader already loaded), matching BIAS0's
existing convention in the same shader. The binding becomes a resident
48×3×512 f32 `weight_scale_2` table (288 KB in the scratch arena), seeded per
layer on first touch — first-touch is what makes the host write race-free
without any fence. Decode's flush + logits/ids/wts read-backs survive only
under `collect_taps`; grouped prefill's three per-block scale-list uploads
per (layer, chunk) die too, and the `esc` arena region retires.

**Receipts** (same sitting, Performance mode): `profile_forward_token`
47.23 ms steady-state (was 55.5), submits/token 50 → 2 (PLE ring + logits),
2529 dispatches recorded at 1.32 µs each, subset parity harness clean,
truncated prefill=decode 0.000e0, `device_nvfp4_gemv`'s out-of-order-ids
gate green on the flipped semantics. The wall is now ~85% GPU execution
(40 of 47 ms) — the fence tax is fully paid off.

## The bug the gate caught (and why the gate exists)

First run of the fence-free decode: prefill=decode diverged at rel ~1e0 from
layer 1 on, decode's states collapsing toward zero. The deleted per-layer
flush had been doing SECOND, undocumented duty as the top-k → expert-GEMV
synchronization — without a submission boundary the GEMVs raced the router's
ids/wts writes. One explicit `barrier()` where the flush used to sit fixed
it. A change that deletes a synchronization point must re-state the ordering
it silently provided; the truncated bit-exact gate caught the omission in
its first 14 seconds.

## D1: the one confirmed numeric deviation was OURS

GDN q/k l2-norm epsilon: the shared shader used 1.0e-12 where the reference
uses 1e-6 (~1e-8 relative, documented as a known deviation). Both upstream
graphs pass `f_norm_rms_eps = 1e-6` to `ggml_l2_norm` (qwen35.cpp:432,
qwen4exp.cpp:872) — verified for BOTH lanes before touching the shared
constant. Fixed; kernel-test host reference moved with it.

## D3: the suspicion against THEM, refuted by reading one more file

The study flagged llama.cpp's QSA tail-block selection as plausibly
reference-divergent (they top-k 2051 CELLS where the reference appends the
tail unconditionally). Fetching `llama-memory-hybrid-idx.cpp` settled it:
`set_input_qsa` gives the query's own tail block bias **+1e9** (always wins
top-k → unconditional inclusion, arithmetic exact at 512 blocks + ≤3 tail
cells = 2051) and incomplete non-tail blocks **-INFINITY** (never selected,
so their wrongly-÷r pooled means never matter). llama.cpp is
reference-exact. The S6 QSA port spec is now complete and verified — and our
port should still COMPACT (gather the ≤2051 selected rows, flash over them)
rather than copy their full-cache mask surgery, which saves no FLOPs.

## Non-findings: the platform-lead receipts (then the baseline corrected one)

For qwen4exp specifically, llama.cpp has: no MTP (the MTP-on-hybrid
allowlist excludes QWEN4EXP), no recurrent-state rollback
(`llm_arch_supports_rs_rollback` excludes it — a prerequisite gap for any
verify-reject speculation on this arch), and no analog of the bit-exact
prefill=decode gate.

The study's third claimed lead — "their Vulkan path dies at QSA's
`ggml_top_k`" — did NOT survive the same-box measurement, and per the house
rule that gets said loudly: llama-bench (master ca3d5a3e1, built from source
with GGML_VULKAN, UD-Q4_K_XL 103.68 GiB, same sitting, Performance mode)
RAN on the 8060S. pp128 **187.1 tok/s**, tg32 **21.96 tok/s** — against
ARLE's 56 tok/s prefill and 21.2 tok/s decode (47.23 ms) the same day. So:
decode is a TIE on their claimed-weak backend (3.5% apart), prefill is a
3.3× gap that ranks exactly where the study's lift list pointed (chunked
delta-net — port in flight the same day). The reported topk assert did not
reproduce at any probed length: pp4096 ran at **236.3 tok/s** — no crash
with QSA active, and faster per token than pp128. Two consequences said
plainly: their prefill lead grows with context, and they serve ctx > 2048
today while ARLE's QSA stub caps at 2048 — the S6 port is capability
parity, not just speed.

## Rules Reaffirmed

- Reading a competitor's landed implementation is the highest-leverage form
  of "measure, don't reason" — three findings (fence shape, eps, D3) for one
  agent-day, each with receipts.
- A deleted fence is a deleted BARRIER: enumerate what a submission boundary
  synchronized before removing it.
- Suspicions about the other side's math earn a verdict only after the file
  that implements it is read — D3 died in 40 lines of `set_input_qsa`.

## Correction: the power-mode label on this day's numbers

Every measurement in this entry and its siblings from 2026-08-28 was
labelled "Performance mode" in the commit messages. That label was
UNVERIFIED: `powercfg -getactivescheme` read **Balanced
(381b4222-f694-41f0-9685-ff5bb260df2e)**, not the Performance fingerprint
(27fa6203), when the device-MoE-planning agent checked it at the end of the
day. Treat the absolute tok/s and ms/token figures here as
Balanced-mode numbers, not comparable to prior sittings' Performance-mode
absolutes.

What survives untouched: every ratio and A/B in this day's work, because
each was measured in ONE sitting on ONE machine — including the ARLE vs
llama.cpp comparison (ARLE 43.5 ms/token decode, llama.cpp tg32 21.96 tok/s
= 45.5 ms/token), which was run back to back on the same box under the same
scheme. The house rule is exactly why this is recoverable: record the power
mode with every number, and trust ratios from one sitting over absolutes.
