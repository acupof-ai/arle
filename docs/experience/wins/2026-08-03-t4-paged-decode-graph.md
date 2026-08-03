# T4 whole-step decode graph under paged KV: −7.9% decode ITL — CUDA, 2026-08-03

> Status: **Shipped** (#196 T4). c=1 W8A16 decode ITL p50 **23.21 → 21.37 ms**;
> cumulative vs pre-#196 baseline **26.88 → 21.37 (−20.5%)**. 17 captures /
> 4100+ replays measured in the serve log (not an ARMED line — API-event
> counted per the 2026-08-01 lesson). Same 32k c=1 protocol.

## What shipped

The whole-step decode-graph lane, previously unreachable under the paged-KV
serving default, now captures the paged decode step:

- **`PageMeta::persistent_decode` / `refresh_decode`** (`loader.rs`): per-slot
  fixed-capacity page-table device buffers whose ADDRESSES never change; each
  step rewrites contents (tiny H2Ds) outside the graph. Also kills the eager
  path's 8 per-step `upload_i32` allocations for graphed steps.
- **FA3 `seqlen_k` pinned to capacity** (`PageMeta.seqlen_k_capture`): FA3's
  device-side `prepare_varlen_num_blocks` re-derives real scheduling from
  `seqused_k` each replay — the dynamic-shape mechanism FA3 ships for graphs.
  `num_splits` and accum scratch are capture-constant.
- **TileLang fallback refuses capture** (hard `ensure!`): that kernel bakes
  `num_pages` as a host arg and would replay stale — any capture reaching it
  errors and permanently downgrades the lane to eager (never a silent bad
  graph).
- **`try_graph_decode_paged`** (`executor/qwen35.rs`): reuses the proven
  contiguous-lane machinery (per-slot `CudaGraphState`, bake fingerprint of
  staged-input/logits pointers + workspace epoch, warm→capture→replay,
  any-failure→eager). Sampling stays outside the graph. Gates: armed flag ∧
  BF16 pool ∧ FA3 hd256 lane ∧ single-row ∧ within max_seq_len.

## Numbers

| arm | ITL p50 | ITL p99 |
|---|---:|---:|
| T2 (eager paged) | 23.21 | 23.84 |
| **T4 (graphed)** | **21.37** | **22.64** |
| SGLang, same kernel + same weights | 17.07 | 18.67 |

Correctness: greedy reasoning-channel output byte-identical to the eager T2
binary across a 120-token trajectory (~8 page-boundary crossings); 16/16
bench requests complete; MMLU-100 parity run recorded in #196.

## Learnings

- The ledger predicted 3.6 ms for launch idle; the graph bought 1.84 ms. The
  residue is the per-step host tail that replay does not remove: 8 small
  refresh H2Ds, sampling D2H+sync, scheduler/HTTP. Next targets: coalesce the
  refresh into one staged H2D, then nsys the graphed step to re-split the
  remaining ~4.3 ms vs SGLang.
- Dynamic-shape kernels are the graph's failure class. The safe pattern:
  device-read metadata at fixed addresses (FA3), refuse capture where a host
  arg bakes shape (TileLang), and count capture/replay API events — an ARMED
  log proves nothing ([[reference_qwen35_decode_graph_unreachable_under_paged_kv]]).
