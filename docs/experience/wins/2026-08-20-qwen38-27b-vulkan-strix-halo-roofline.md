# Qwen3.8-27B on Vulkan / Strix Halo: 9.4 tok/s is 59% of a hard 15.9 tok/s ceiling

## Context / Goal
First end-to-end run of Qwen3.8-27B-Q4_K_M through the Vulkan backend on the
AMD Strix Halo box, and the question that followed it: can per-layer weight
prefetch into a fast cache buy anything, and is 60 tok/s reachable.

## Hypothesis
Going in, the working guess was that decode sat far below memory roofline
(~21% of peak) and that a prefetch/residency scheme had headroom. Both parts
were wrong, and only measurement showed it.

## Params
- Backend: Vulkan
- Model: `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-Q4_K_M.gguf` (15.93 GiB)
- Tokenizer: `Qwen/Qwen3.8-27B` (the GGUF repo ships none)
- CLI: `arle serve --backend vulkan --model-path <gguf> --port 8022`
- Profiler: `ARLE_GPU_TIMESTAMPS=1` for the per-op split, off for throughput

## Env
- Host: Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, RDNA 3.5), 63.6 GB unified
  LPDDR5X, Windows 11, AMD proprietary driver 26.7.1 (LLPC), Vulkan 1.4.349
- Date: 2026-08-20
- Build: `cargo build --release -p arle --features vulkan`

## Results

### Weight bytes — the roofline denominator
`scripts/gguf_weight_bytes.py <gguf> 64`:

| Bucket | Bytes |
| --- | ---: |
| device-resident | 16.091 GB |
| host `token_embd` (gathered per token on host) | 0.715 GB |
| skipped MTP block `blk.64` | 0.290 GB |

| Per-layer | Bytes |
| --- | ---: |
| min / mean / max | 209.4 / 235.1 / 257.9 MB |
| three layers | 773.7 MB |

### Throughput (`scripts/bench_throughput.py`, no profiler)

| Concurrency | decode tok/s | output tok/s |
| ---: | ---: | ---: |
| 1 | 9.4 | 8.4 |
| 4 | 9.4 | 8.4 |

Concurrency buys nothing: `crates/infer-vulkan/src/executor.rs:200`
`ensure!(row_count == 1)` forces one row, and `:175` decodes token-serially.

### Per-op GPU time (`ARLE_GPU_TIMESTAMPS=1`, 100 positions, steady state)

| Op | ms/token | share | dispatches |
| --- | ---: | ---: | ---: |
| gemv | 78.5 | 92.4% | 497 |
| linear | 3.8 | 4.5% | 96 |
| norm | 1.1 | 1.3% | 2881 |
| quant / add / swiglu / flash / rope / sigmoid / kvpack | 0.9 | 1.1% | 1809 |
| **total GPU** | **85.0** | | **5363** |

Wall clock was 114.9 ms/position in that run, so ~30 ms/token is host-side —
plausibly the cost of recording 5363 dispatches per token.

### Roofline
LPDDR5X-8000 × 256-bit = 256 GB/s theoretical.

| Basis | Effective GB/s | % of peak |
| --- | ---: | ---: |
| GPU-busy only (85.0 ms) | 189 | 74% |
| Wall clock (107 ms, profiler off) | 151 | 59% |

Hard ceiling at one token per weight sweep: 256 / 16.09 = **15.9 tok/s**.

## Verification
| Check | Result |
| --- | --- |
| `cargo fmt --check` | PASS |
| `cargo clippy -p infer-vulkan -p kv-native-sys --all-targets -- -D warnings` | PASS |
| `cargo clippy -p infer-server -p arle -p train --all-targets -- -D warnings` | PASS |
| `cargo test -p infer-vulkan -p infer-server -p kv-native-sys` | PASS, 36 passed |
| three sequential requests, no crash | PASS (was exit 75 before the slot/epoch fix) |
| output quality, Chinese + arithmetic | PASS, fluent and correct |
| `eval_harness prefix_reuse token_reuse` | **FAIL 0/2** — see below |

## Problems
**The Vulkan lane has no prefix/KV reuse at all**, and the eval gates say so
rather than merely running slowly: `token_reuse` reports `on_hit=0 off_hit=0
delta=0`, and the server logs `prefix-lookup: prompt=57 raw_blocks=0
licensed_blocks=0`. `crates/infer-vulkan/src/kv_pool.rs` has no lookup or
licensing path — only `attach_pages` — and `forward.rs:808` bails unless
`start_pos == state.seq_len`, which is the uncached full-prefix contract stated
in its own doc comment. This predates the commits measured here (`kv_pool.rs`
was last touched by an unrelated refactor) and is a missing feature, not a
regression. `prefix_reuse` additionally timed out, because it builds 2000-token
docs and prefill runs at the decode rate.

Practical cost: every request re-prefills from scratch. A 2000-token multi-turn
conversation pays 2000 × ~107 ms ≈ 3.5 min of TTFT on each turn.

- Prefill is token-serial, so TTFT scales at the decode rate (~107 ms/token).
  **Both prefill claims above were superseded on 2026-08-20.** Prefix/KV reuse
  landed in
  [vulkan-resident-sequence-prefix-reuse](2026-08-20-vulkan-resident-sequence-prefix-reuse.md)
  (turn-2 156.5 s → 3.2 s), and prefill is no longer token-serial: a batched
  chunk GEMM plus a KHR cooperative-matrix kernel took it 3.45 → 37.0 tok/s, so
  TTFT now runs ~9× ahead of the decode rate — see
  [vulkan-coopmat-prefill-warptile](2026-08-20-vulkan-coopmat-prefill-warptile.md).
- `KV_CACHE_MAX_SEQ` is hardcoded 8192 in `forward.rs:162`; the model declares
  262144.
- The same dropped-`epoch` bug likely sits in `model_qwen36.rs:262`,
  `model_qwen3.rs:99`, `model_dsv4.rs:274`, `model_gemma4.rs:249`.

## Learnings
**There is no fast tier to prefetch into on this part.** Every `memoryType` in
heap 1 is `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` — weights live in the
same LPDDR5X the CPU uses, with no PCIe hop and no discrete VRAM. And the
on-chip capacity is not close: Vulkan reports 32 KB
`maxComputeSharedMemorySize`, and gfx1151's whole hierarchy tops out around
32 MB of MALL. One layer is 235 MB. Three layers are 774 MB — 24× the entire
on-chip budget. The scheme was never blocked on code.

**gemv is already at 74% of DRAM peak**, so hardware prefetch and wave-level
latency hiding have taken the headroom a software scheme would have chased.

**Therefore bytes-per-token, not bytes-per-second, is the only lever left.**
16.09 GB is read whether the sweep produces one token or eight, so 60 tok/s
single-stream would demand 965 GB/s — 3.8× the machine. Reaching it means
amortizing one weight sweep across many tokens (batching, then MTP), or a
model whose weights fit in 256/60 = 4.3 GB, i.e. an 8B-class checkpoint.

**Heavier quants are strictly slower here**, in direct proportion to bytes:
Q5_K_M 18.47 GiB → ~8.1 tok/s, Q6_K 21.31 → ~7.0, Q8_0 27.05 → ~5.5.

## Rule
On a unified-memory APU, do not design a weight-residency or prefetch scheme
before comparing per-layer bytes against the on-chip cache budget and checking
the achieved fraction of DRAM peak. If gemv already runs above ~70% of peak,
the remaining win is in reading the weights fewer times — batching and
speculative decode — not in moving them somewhere faster.
