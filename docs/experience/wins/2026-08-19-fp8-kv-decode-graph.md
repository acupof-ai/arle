# The decode graph under FP8 KV — CUDA, 2026-08-19

> Status: Shipped, both arms measured

## Context

`--kv-cache-dtype fp8` silently turned off the Qwen3.5/3.6 whole-step decode
graph. Nothing logged the downgrade: a serve started with FP8 KV looked identical
to one without, and every NVFP4 measurement this campaign ran without the graph.

The gate is `paged_kv_bf16()` (`executor/qwen35.rs:1019`, true only when
`full_attn_kv.format == KVFormat::BF16`), read by `try_graph_decode_paged`
(`:2106`). It also gates the batched DSpark draft (`:2343`), which this change
leaves alone.

## Why the gate was wider than the constraint

Quantised KV needs three device buffers a BF16 decode does not, and only one of
them is a problem:

| buffer | size at B=1 | grows with context | read by decode |
|---|---:|---|---|
| `new_token_rows` | 1 i32 | no | yes (`qwen35_attention.rs:608`) |
| `quant_decode_meta` | 3 i32 (2b+1) | no | yes (`attention.rs:591`) |
| `prefix_token_rows` | `start_pos` i32 | **yes** | **no** — quant PREFILL only (`attention.rs:407`) |

The one buffer that would break a capture is the one decode never touches. The
other two are exactly the "fixed size, contents change per step" shape
`PageMeta::refresh_decode` already implements for every BF16 field.

The kernel side was already capture-safe and had to be checked, not assumed:
`decode_attention_quantized.cu` launches `dim3 grid(num_splits, total_q_heads)`
and `choose_decode_num_splits(batch, heads, head_dim, total_q_heads,
workspace_bytes)` takes no KV length — so the grid is fixed at B=1, the true
length is read on device from `kv_indptr`, and the workspace is pool-owned
(`pool.quantized_attn_workspace()`), not a per-call allocation.

## Change

- `PageMeta::persistent_decode` takes the pool format and allocates the two small
  quant buffers when it is not BF16. `prefix_token_rows` stays `None`.
- `refresh_decode` drops its BF16-only assert for a shape-consistency check and
  writes both buffers in place, reusing `pool.build_quantized_decode_indptr(&[slot])`
  so there is one source of truth for the packed meta.
- The capture gate becomes explicit about which lane each format captures: BF16
  the FA3 lane (its ceiling pinned by `seqlen_k_capture`), FP8/INT8 the split-KV
  lane, everything else eager.

Failure stays safe: `run_or_capture` already downgrades to the eager forward and
disarms on any capture error.

## Results

1xH20, synthetic prompt, 30 s/point, `--kv-cache-dtype fp8`, no spec, same
binary for both arms, engine probed with a real completion before each grid.

| c | NVFP4 before | NVFP4 after | FP8 before | FP8 after |
|---:|---:|---:|---:|---:|
| 1 | 66.4 | **74.2** (+11.7%) | 56.8 | **61.6** (+8.5%) |
| 2 | 102.6 | 103.3 | — | 100.2 |
| 4 | 163.5 | 165.3 | — | 196.6 |
| 8 | 215.3 | 216.8 | — | 360.6 |
| 16 | 236.1 | 235.9 | — | 636.7 |

Aggregate tok/s. `GRAPH_CAPTURES` 20 (NVFP4) and 18 (FP8), zero server errors on
both.

c>=2 does not move at all — 102.6 -> 103.3, 163.5 -> 165.3, 236.1 -> 235.9 — which
is the mechanism confirming itself: the graph is gated to `plan.decode_rows.len()
== 1` (`qwen35.rs:2283`), so a batched step never captures or replays.

## The gain is symmetric and does not move the NVFP4-vs-FP8 ratio

Both checkpoints are `qwen3_5`, both were served with FP8 KV, both had lost the
graph. Measuring NVFP4 on the fixed binary against FP8 on the old one reads
+30.7% at c=1; the matched arms read **+20.5%**. The difference is entirely this
change accruing to both sides.

Worth stating because the tempting version of this entry claims a 30% headline.
`docs/baselines.md` rule 3 already lists the serve flags as part of the
fingerprint; this is what enforcing it costs and buys.

## Rule

A capability gate written as a format check can be far wider than the constraint
it stands for. Before accepting one, enumerate what the other format actually
needs and check each item against the real requirement — here two of the three
extra buffers were fixed-size and the third was never read on the gated path.
