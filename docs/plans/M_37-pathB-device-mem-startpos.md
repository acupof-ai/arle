---
title: M_37 Path B — device-memory `start_pos` for prefill graph reuse
date: 2026-05-10
type: plan
status: ready-for-codex-pickup
audience: codex
prereq: #24 (35fc3cf landed) + #37 Path A KILL (e462c53)
---

# Path B device-memory `start_pos` — close prefill graph reuse gap

> Per `docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md`,
> codex's #24 `35fc3cf` landed multi-key 8-d graph cache(`tokens, batch,
> pages, prefix_rows, ...`),但 scout bench showed **Δ -0.07% TTFT + capture
> key churn**(re-capture every 5s for same `(tokens, batch, pages,
> prefix_rows)` tuple)→ Path A direction KILLED。Path B
> (device-memory `start_pos`)re-licensed P0。

## 1. Root cause(re-statement)

Per #24 codex implementation: graph cache key is 8 fields **including per-
request varying fields**(start_positions, seq_lens, page_count)。Same model
shape but different positional metadata → unique key per request → cache
miss 100%。

**SGLang `PiecewiseCudaGraphRunner` 的 fix**:metadata 全 device-tensor +
replay-time refresh(`start_pos`, `seq_lens` 不进 key,作 device pointer
传入)→ single graph reused 100% across all positions。

## 2. Path B implementation scope

### 2.1 Move `start_pos` from launch scalar to device tensor

**Current**(post `35fc3cf`,`infer/src/model/qwen3/prefill.rs`):
```rust
// Capture key includes start_positions in tuple
let capture_key = (tokens, batch, pages, prefix_rows, batch_size,
                   seq_lens.clone(), start_positions.clone(), page_count);
```

**Target**:
```rust
// Capture key drops per-request varying fields
let capture_key = (tokens, batch, pages, batch_size, page_count);
// start_pos / seq_lens become device tensor refreshed per replay
```

**LOC estimate**:
- New `PrefillGraphMetadata` struct holding `start_pos: DeviceVec<u32>` + `seq_lens: DeviceVec<u32>`(~30-50 LOC)
- Allocator hoist into `PrefillGraphResources`(~20-30 LOC)
- Replay refresh hook before `cuda_graph_launch`:`metadata.copy_from_host(start_positions)?`(~20-40 LOC)
- Capture key tuple narrowing(~10 LOC)
- Prep kernel(if any reads scalar `start_pos`)to read from device tensor(~30-80 LOC,depends on which kernel)

### 2.2 Verify prep kernel reads device pointer

**Locate kernels that read `start_pos`**:
- `crates/cuda-kernels/csrc/attention/`: prep prefill metadata kernels
- `crates/cuda-kernels/tools/tilelang/`: TileLang attention with start_pos param

If existing kernel takes `start_pos` as launch scalar(`int start_pos`):**must
change to `int* start_pos`**(device pointer)+ caller passes `metadata.start_pos.device_ptr()`.

If TileLang DSL kernel:may need DSL-level change(check `tools/tilelang/batch_prefill_paged_hd128.py`)。

### 2.3 Validation gates

1. **Correctness**:
   - `cargo test --release -p infer --features cuda --test e2e PASS`
   - `cargo test --release greedy_consistency -- --nocapture`(尤其 device-mem
     read 数值 vs scalar 等价 check)
2. **Functional smoke**:
   - `INFER_PREFILL_GRAPH=1` server boot + 200 OK + valid completion
   - Server log:capture key count **远小于** request count(reuse evidence)
3. **Throughput license**(`docs/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`):
   - Bench A(off baseline)vs B(on)matched-control 4k/c=4 n=3
   - **License**:TTFT p50 Δ ≥ +10% σ < 5%
   - **Strong proceed**:Δ ≥ +25%
   - **KILL**:Δ < +5% OR cache hit rate < 50%(per request 至少 1 capture
     reused across requests)
4. **Anti-pattern check**:
   - `cudaGraphLaunch` count ≫ `cudaGraphInstantiate` count
   - `prefill graph capture key` count < 30(for 60s c=4 4k/256 sustained)

### 2.4 Total LOC + risk

| 子 task | LOC | 风险 |
|---------|----:|------|
| `PrefillGraphMetadata` struct + lifecycle | 50-80 | 低 |
| Capture key tuple narrow | 10 | 低 |
| Replay refresh hook | 20-40 | 低 |
| Prep kernel device pointer change | 30-80 | 中(数值精度 verify)|
| Tests + greedy_consistency | 30-50 | 低 |
| **总** | **140-260** | 中 |

## 3. Phased execution(codex pickup recommended)

### Phase 1 — Implementation(2-3 days codex)

- 实施 §2.1 + §2.2
- Build + cargo check + clippy
- Run validation §2.3 step 1 + 2(correctness + functional smoke)

### Phase 2 — Throughput bench(1 day,Claude run)

- ./scripts/post_p24_commit_pipeline.sh full(or manual A/B)
- Fill `wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`
- License decision per §2.3 step 3

### Phase 3(可选)— Cache key further narrow

If Path B passes license but cache hit < 80%,iterate:remove more fields
from key(e.g. `pages` if computable from shape)。

## 4. Cross-references

- Codex #24 implementation:`docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`(35fc3cf)
- #37 Path A KILL evidence:`docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md`(e462c53)
- Path A vs B design:`docs/research/2026-05-09-37-multikey-vs-device-startpos-design.md`(9a477c7)
- Pre-built bench template:`docs/experience/wins/TEMPLATE-2026-05-10-bench-37-w4hybrid-prefill-graph-throughput.md`(1168381)
- SGLang upstream pattern reference:`/tmp/sglang-chunked-src/python/sglang/srt/model_executor/cuda_graph_runner.py`(if needed,fetched 2026-05-09)

## 5. ROI

- **Wall-clock**:Phase 1 (codex 2-3d) + Phase 2 (Claude 1d) ≈ 3-4 days
- **Predicted gap close**:if Path B works,close 30-50% of +76.6% SGLang gap
  → ARLE TTFT 4k/c=4 reduce from 1639 ms → 1100-1300 ms range
- **Risk**:medium(prep kernel device-pointer change requires numerical
  verification via greedy_consistency)

## 6. 状态

Path B device-memory `start_pos` 完整 implementation plan ready for codex
pickup。LOC 140-260,3-4 days wall-clock,direct close 4k/c=4 SGLang
+76.6% gap target。Pre-built validation infrastructure ready
(post_p24_commit_pipeline.sh,validate_p24_phase0v3.sh,bench template)。
