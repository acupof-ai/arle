# 2026-04-28 → 2026-04-29 perf + correctness bug roundup

Captured during the SGLang-parity push. Each row is a bug, root cause,
status, and follow-up. Issues without a fix this cycle are
hand-off-ready: the file paths and reproduction steps are concrete
enough that codex / a subagent can pick them up.

---

## ✅ Fixed (committed, on main)

### F1 — KV pool budget formula misaligned with SGLang
`83e67ff2`. Three stacked bugs in
`infer/src/scheduler/cuda/core/construction.rs`: pre-deducted
workspace, used `total` instead of `pre_model_free` for headroom, and
the `cuMemGetInfo` snapshot silently no-op'd when no `DeviceContext`
existed (Codex P2). Fix: use `pre_model_free × (1 -
mem_fraction_static)` for headroom, drop workspace pre-deduction,
ensure `DeviceContext::new()` before the snapshot.
Effect: pool 84,096 → 114,208 tokens at fraction=0.94 (+35%).

### F2 — TileLang prefill HD128 short-qlen NaN
`47bad713`. `T.reduce_max(scores, m_new, dim=1, clear=False)` left
`m_new` uninitialized; TileLang codegen emitted `m_new[i] = max(m_new[i],
m_new_clear[i])` reading stack garbage on first iteration. NaN
propagated through `exp2(scores - m_new)` → `p_bf16 @ v_tile` → final
acc_o → uniform-NaN logits → argmax = token 0 = "!". Every short
prompt (chat / e2e) emitted `"!!!!!"` for the entire generation.
Fix: `clear=True`. The kernel was bench-validated only at qlen=4096;
the partial-tile path was never exercised before e2e validation.
Errors entry:
`docs/experience/errors/2026-04-28-tilelang-prefill-short-qlen-nan.md`.

### F3 — Duplicate weight storage (qkv_proj + gate_up_proj)
`c4109b29`. Each Qwen3 layer kept BOTH individual q/k/v + a
`concat_rows`-built merged copy, and same for MLP gate/up + gate_up.
4.72 GB of duplicate VRAM at model load. The merged form was used by
two batched-decode call sites under the bf16 branch only; the
quantized branch already used 3 separate q/k/v GEMMs. Unify all paths
onto the existing-tested separate-GEMM flow and drop the merged
copies. Effect: post_model_load free 10.31 → 15.04 GB; KV pool
114k → 175k tokens. Wins entry:
`docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-weight-dedup.md`.

### F4 — `max_prefill_tokens` default 2048 vs SGLang 16384
`8f6965c3`. Default was bound to `chunked_prefill_size` (2048 on L4),
so each prefill step admitted exactly one 2048-token chunk regardless
of how many requests were queued. For c=16/4096-in: 32 sequential
chunked steps × ~300 ms = ~10 s of TTFT. SGLang's `PrefillAdder`
(`schedule_policy.py:603`) packs up to `max_prefill_tokens` (default
16384) per step. Fix: default to `max_num_batched_tokens`. Also adds
a "Scheduling envelope (resolved | SGLang-equiv)" log line at server
boot so future param drift is visible at a glance.

---

## ⚠️ Known but not fully fixed (this cycle)

### K1 — BF16 merged-QKV fast path lost in batched decode (Codex P2)
After F3 the bf16 batched decode does 3 separate q/k/v GEMM launches
per layer per decode step instead of 1 merged GEMM + split kernel. At
c=16/4096-in the larger pool from F3 drowns out the per-step launch
cost (admission throttling drops more than launch cost rises).
**At c=1 (decode-bound)** the codex-flagged cost is likely the
dominant signal — needs a separate bench. Fix path: store only the
individual q/k/v weights, build `qkv_proj` as a per-batch scratch view
on first call (no extra eager VRAM). File:
`infer/src/model/qwen3/batch_decode.rs:1515-1534`.

### K2 — Mixed (decode+prefill) batches don't fire on FP8/INT8 KV
`infer/src/model/qwen3/forward.rs:585` (`supports_mixed_batch`) gates
mixed batching on `KVFormat::BF16` only. We default to FP8E4M3, so
the scheduler always falls to the legacy `Split` (separate decode +
prefill launches) instead of fused FlashInfer varlen. This is the
NEXT TTFT lever after F4 — once mixed batches fire, decode rows can
piggyback on prefill steps. Fix path: implement a fused-dequant
varlen kernel that reads fp8/int8 KV; OR enable the existing path to
work with fp8/int8 by dequantizing the relevant KV tiles inside
FlashInfer's plan.

### K3 — `mem_fraction_static` default 0.88 vs SGLang 0.85
Close but not exactly aligned. Lower would give more workspace
headroom (helps bf16 KV at default config). Higher gives more pool.
Probably keep at 0.88 unless we see a real OOM cliff. File:
`infer/src/scheduler/types.rs:108`.

### K4 — Bench `--fast` preset variance
30s × c=16 produces ~10 completed requests, so single-run tok/s
swings ±50. n=3 medians stabilize attribution but a canonical sweep
(60s × multiple concurrencies) is the real protocol. The wins-entry
gate should require sweep-profile data, not `--fast`.

### K5 — e2e test asserts byte-exact match against HF baseline
`infer/tests/e2e.rs:162` panics on greedy text mismatch. Our
FlashInfer/TileLang attention drifts from HF reference after ~5-15
tokens (numerical, not algorithmic). Test "FAIL" is constant, even
when output is correct. The reasonable assertion is
"first-N-token-prefix match" or "no token 0 / NaN propagation",
not strict equality. Fix path:
`infer/tests/e2e.rs:148-167`.

### K6 — guidellm `--fast` reports "successful" on empty outputs
When the server OOMs mid-bench, completed requests get 256 tokens
worth of zero-logits → empty text → guidellm logs "successful_count
= N" but every sample is empty. The bench wrapper should reject
runs where `streaming_iterations.successful.mean == 0` AND
`request_totals.errored == 0`. File:
`scripts/bench_guidellm.sh:headline-table-extraction`.

### K7 — Slot leak after bench client disconnect
When the bench HTTP timeout (30s) fires, the client closes the
connection but the server keeps the slot allocated until decode
naturally completes (256 tokens × 80 ms = 20 s — MAY race the
timeout). Subsequent requests stall waiting for the orphaned slots
to finish. Fix path: hook `axum`'s `Connection::Close` (or HTTP
client cancellation) to call `scheduler.cancel(req_id)`. File:
`infer/src/http_server/handlers.rs::stream_completion`.

### K8 — Service-stats trace doesn't capture per-phase TTFT breakdown
We get `step breakdown: plan=prefill admission=Xus decode=0us
prefill=Yus` per scheduler step, but no aggregated "first-token wait
distribution by phase" — admission queue / first-prefill-chunk /
last-prefill-chunk / first-decode. That's the data agent needed to
prove the F4 root cause. Fix path: export a histogram in
`infer/src/scheduler/cuda/runtime/scheduler_loop.rs` of TTFT split
into (admission_us, total_prefill_us, first_decode_us).

### K9 — INT8 KV decode ITL ~30% slower than FP8 at same shape
At c=16/4096-in: fp8 ITL 75-78 ms, int8 ITL 97-104 ms (post-F4
data, n=3 each). With both at ~157k pool, the per-token cost is
visibly higher on int8. Suspect: int8 dequant in FlashInfer's
attention plan. Validate with `ncu` on a layer step.

### K10 — Wins-entry promotion gate didn't include text-correctness
`docs/experience/wins/2026-04-28-bench-guidellm-cuda-l4-tilelang-prefill-causal-bound.md`
promoted TileLang to default without an e2e correctness gate; F2 fell
through. The errors entry codifies this as a rule but the bench
wrapper should mechanically enforce it: add a 4-token-prompt smoke
test before publishing tok/s numbers.

---

## Open task list (carried forward)

- **#5 [pending]** Iterate on TileLang decode + prefill perf — actual
  perf tuning, separate from the correctness work above.
- **K1** lazy `qkv_proj` build OR document the trade-off as
  intentional after a c=1 bench shows the cost.
- **K2** is the highest-impact perf lever remaining — fused-dequant
  varlen for FP8/INT8 KV.
- **K5, K6, K10** are bench-protocol hardening, not perf — but they
  unblock confident future iteration.
- **K7** is a server-side robustness item — surfaces every time a
  bench is repeated against a live server.
