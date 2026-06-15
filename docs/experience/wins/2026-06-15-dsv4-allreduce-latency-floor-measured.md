# DSv4 collectives are latency-floor + negligible wall-clock — measured (the "15%" was a framing trap)

## Context
ckl questioned the earlier "attention collectives are ~15% / Q-allgather 10.4%" analysis:
NVLink is fast and decode passes only a few tokens, so why would all-reduce/EP comm cost
that much? Measured it directly (`comm_bench`, pod 8×H20 NVLink, TP=8 NCCL) instead of
trusting the inference.

## Measurement — all-reduce p50 µs at the decode shapes
| shape [N,7168] bf16 | bytes | NCCL | nccl_sym | car_1stage |
|---|---|---|---|---|
| [1,7168]  | 14 KB  | 17.3 | 7.6 | 5.0 |
| [4,7168]  | 57 KB  | 22.3 | 8.0 | 5.0 |
| [8,7168]  | 115 KB | 21.7 | 7.8 | 6.4 |
| [16,7168] | 229 KB | 17.1 | 7.5 | 9.5 |
| [32,7168] | 459 KB | 17.5 | 7.5 | 14.8 |
Q-allgather: nccl ~11-14 µs.

## Findings
1. **LATENCY-FLOOR-bound, not bandwidth-bound.** NCCL all-reduce is ~17-22 µs FLAT from
   14 KB to 459 KB (32× the data, same time). At decode sizes the payload is trivial; the
   cost is the fixed NCCL ring/launch/sync latency. ckl's instinct is correct — NVLink
   bandwidth is irrelevant here.
2. **The cost is the COUNT, not the data.** DSv4 = 60 layers × 2 all-reduces (attn+moe) =
   **120 all-reduces/forward × ~17 µs ≈ 2.0 ms** (+ Q-allgather ~0.8 ms) ≈ **2.8 ms/forward**,
   FIXED regardless of batch size (latency-floor).
3. **As a wall-clock fraction it's small and SHRINKS with concurrency.** Fixed ~2.8 ms /
   step: ~9% of a B=1 step (~34 ms) but **~2.7% of a high-concurrency batched step**
   (~75-130 ms) — matches the prior DP-attn doc's "2.7% at B=16".
4. **The "15%" was a framing trap (§0).** It was "% of kernel time" / an NVTX-window
   share, NOT wall-clock (NVTX "X% of window" ≠ wall-clock X%).
5. **Decisive cross-confirmation:** comm_bench shows a 3.5×-faster all-reduce exists
   (`car_1stage` 5 µs, `nccl_sym` 7.5 µs vs NCCL 17 µs), yet the 2026-06-10 custom-AR A/B
   measured it **wall-NEUTRAL** on single-node H20. A 3.5×-faster collective giving zero
   wall change ⇒ collectives are a negligible wall-clock fraction. The wall is the DSv4
   **MoE compute** (verify-compute-bound), which batched MTP +77% addressed by amortizing.

## Implication
- DP-attention (removes the attention collectives) is confirmed a ~2.7% decode lever
  (its value is prefill/scaling, not decode) — consistent with deferring it.
- A faster all-reduce arm (nccl_sym / car) is NOT a throughput lever (wall-neutral); at
  most a small B=1 latency tweak (collective is ~9% at B=1) — but the 2026-06-10 A/B
  already found it neutral, so not worth wiring.

## Rule
- **"X% of kernel time" or "X% of an NVTX window" ≠ X% of wall-clock.** Always
  cross-check a collective/overhead % against a direct latency microbench + a "make-it-
  faster, did-wall-move?" A/B. A 3.5×-faster collective with zero wall change is the
  cleanest proof the collective doesn't bound the wall.
- **Small-tensor collectives are latency-floor-bound** (~17 µs NCCL on H20 NVLink,
  flat 14KB→459KB) — the cost is the per-op count × the floor, not bandwidth. Reducing
  collective COUNT (fewer layers' all-reduces) beats reducing collective SIZE.
