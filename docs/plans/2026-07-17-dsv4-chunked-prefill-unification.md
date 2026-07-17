# DSv4 Chunked-Prefill Unification: sizing, restore-alignment, and compressor-boundary snapshots

> Status: Active — Phase 1 in progress (2026-07-17)


**File:** `docs/plans/2026-07-17-dsv4-chunked-prefill-unification.md`
**Status:** proposed · **Scope:** `infer-api` / `infer-core` / `infer-seam` / `infer-cuda` · **Sensitivity:** #146 (needle-gated), #154 prefix-restore, f59dd79af compressor carry

## 0. Problem

Every DSv4 prefill tick is exactly 128 tokens regardless of config, because two stacked mechanisms fight each other:

1. **API-layer override** — `crates/infer-api/src/loaded.rs:1944-1946` unconditionally sets `scheduler.chunked_prefill_size = 4096` for `CudaModelKind::Dsv4` (comment at :1937-1943 cites the `DSV4_PREFILL_QUERY_CHUNK` scratch bound). The CLI flag (`crates/cli/src/args.rs:688`, applied at `crates/cli/src/serve.rs:478-480`) is accepted and copied into the config (`loaded.rs:502`) and then trampled — cosmetic.
2. **Planner one-unit cap** — `crates/infer-core/src/planner.rs:82-91`: `restore_alignment = executor.prefill_restore_boundary_alignment()` = `sliding_window` = **128** (`crates/infer-cuda/src/executor/dsv4.rs:1225-1227`), and when `restore_alignment > 1` the chunk is capped to one `lcm(page_size=16, 128) = 128` unit, because a forward snapshots boundary state only at its own end (`executor/dsv4.rs:900`: `boundary = page_end == end_pos && page_end.is_multiple_of(align)`).

Measured cost: one 128-token chunk = one full 43-layer forward ≈ **137 ms** (latency-bound) → cold prefill ≈ **930 tok/s**, c1 TTFT **3.0 s** on a 2.8k uncached prompt. `pack_quantize_bf16_to_fp8` + `swiglu_quantize_w13` are **18.6%** of GPU kernel time at c16 (1.06M instances) — instance count is proportional to forward count, i.e., to `prompt/128`.

## 1. Current machinery map (verified anchors)

| Concern | Location |
|---|---|
| Config field + default 64 | `crates/infer-api/src/loaded.rs:41`, `:185`; engine-side field `crates/infer-core/src/lib.rs:60`, default 2048 at `:111`, accessor `prefill_chunk_size()` `:96` |
| Model-kind overrides | `loaded.rs:1934-1936` (Qwen `.max(2048)` floor), `:1944-1946` (Dsv4 hard `= 4096`) |
| Planner chunk sizing + one-unit cap | `crates/infer-core/src/planner.rs:49-100` (cap at :84-86, end-align at :87-91); `max_tokens_per_step` cap :55-60, populated from executor at `infer-core/src/lib.rs:424` |
| Capability trait | `crates/infer-seam/src/lib.rs:295-301` (`prefill_restore_boundary_alignment`, default 1); dispatch `crates/infer-cuda/src/lib.rs:681-693`, `crates/infer-cuda/src/executor.rs:591-596`; DSv4 impl `executor/dsv4.rs:1225-1227` |
| Boundary snapshot publish | `executor/dsv4.rs:860-934` (`publish_completed_prefix_pages`; boundary predicate :900), prefill call site :2139-2144, decode call site :1832; async fence drain `poll_prefix_captures` :802-858 |
| Capture / restore | `crates/infer-cuda/src/dsv4.rs:1443-1483` (`capture_prefix_page`, per-layer D2H), restore + 128-align ensure `executor/dsv4.rs:1239-1272` (:1264-1267), commit walk `reusable_prefix_blocks_for_prompt` :1183-1221 (commits only at `meta.boundary` pages); entry layout `crates/infer-cuda/src/attention/prefix_state.rs:44-118` (note :13-18: true sub-forward boundary granularity “needs a kernel-side per-block overlap output buffer”) |
| Scratch bound 4096 | `crates/infer-cuda/src/attention/kv_layout.rs:3` (`DSV4_PREFILL_QUERY_CHUNK`), `:35` (`DSV4_INDEXER_STAGING_RING_ROWS = 2×4096`) |
| 4096-sized buffers (all model-wide, one instance, **not** per-slot) | `Dsv4PrefillDeepGemmLinearScratch` `attention/flashmla.rs:1558-1620` (`input_fp8` M×K, `input_scales`, `qkv_raw`, `oproj_group_in`; asserts `attention.rs:1240-1246`, `:1421-1428`, FP8 staging note `:5658`); DSA `raw_indices` `attention/dsa.rs:176-191` (assert `attention.rs:9461-9468`); indexer staging ring `kv_layout.rs:49`, `:181`, `dsa.rs:438` (assert `attention.rs:7583-7588`) |
| deepep_ll per-forward cap | `crates/infer-cuda/src/dsv4.rs:1887-1904` (`max_tokens_per_step`), dispatch `executor.rs:333-337` |
| FP32 probe scratch (shared, `max_width×max_seq_len`) | `kv_layout.rs:204-234` — **no chunk constraint** |
| SW ring update kernel | `attention.rs:2114-2159` → `crates/cuda-kernels/csrc/attention/dsv4_swa.cu:180-211` (`slot = (start_pos+token) % sliding_window`, unordered writes) — call sites `attention.rs:2701` (prefill), `:282`, `:3112` |
| Qwen3.5 mid-chunk snapshot precedent | `crates/infer-cuda/src/executor/qwen35.rs:2582-2659` (`prefill_row_snapshotted`: executor-internal forward splitting at stride cuts) |
| Repair path | `planner.rs:124-165` (`fit_plan_to_kv_pages`, sheds whole prefill rows) |
| ITL-kill precedent | `docs/experience/errors/2026-05-25-axis3-chunked-prefill-size-kill.md` (2048→4096/8192 killed on ITL) |
| Gates | `scripts/eval_harness/needle_ladder.py`, `scripts/eval_harness/prefix_reuse.py`, `scripts/needle_gate.py`, `scripts/dsv4_variable_shape_dsa_gate.py`, `scripts/dsv4_concurrent_probe.py` |

## 2. Design analysis

### Cost model (basis for all arithmetic)
t(chunk) ≈ F + μ·m, with measured t(128) = 137 ms and “latency-bound” ⇒ F ≈ 115-125 ms, μ ≈ 0.10-0.17 ms/tok at low M (μ shrinks at high M as GEMMs saturate). Planning estimates, to be replaced by Phase-2 measurement:

| chunk | t(chunk) est. | tok/s | c1 TTFT, 2.8k prompt |
|---|---|---|---|
| 128 (today) | 137 ms (measured) | 930 (measured) | 3.0 s (measured, 22 forwards) |
| 512 | ~190 ms | ~2700 | ~1.2 s (6 forwards) |
| 1024 | ~250-280 ms | ~3900 | ~0.8 s (3 forwards) |
| 2048 | ~380-420 ms | ~5000-5400 | ~0.65 s (2 forwards) |
| 4096 | ~650 ms | ~6300 | ~0.65 s (2.8k < 4096 anyway) |

### A. Restore points with big chunks — recommendation: **(b) grain = chunk end**, telemetry-guarded

- **(a) mid-forward snapshots at every internal 128 boundary: rejected.** The boundary sections (compressor `prev_overlap_*`/`pending_*` registers + full bf16 SW ring, `prefix_state.rs:62-84`) never *materialize* at internal positions: each layer processes the whole chunk in single attention/pack/compressor calls, and the ring at internal boundary `b` is overwritten by rows `b..b+128` of the same forward. Capturing them requires either (i) kernel-side per-block overlap output buffers (`prefix_state.rs:18` says exactly this) — a new kernel feature squarely in #146-sensitive territory — or (ii) splitting the forward at each boundary, which is identical in cost to 128-token chunks (no win). Even granting (i), D2H volume ≈ 43 layers × (ring 128×head_dim×2 B + overlap registers) ≈ 5-7 MB per boundary × 15 internal boundaries in a 2048 chunk ≈ ~100 MB per chunk — prohibitive.
- **(b) grain = chunk size: recommended.** Content pages are still captured for *every* completed 64-token page inside the chunk (`executor/dsv4.rs:882-924`); only the `boundary` bit coarsens. `reusable_prefix_blocks_for_prompt` (`executor/dsv4.rs:1196-1221`) already *continues* past non-boundary pages and commits at the last boundary page — no restore-path change needed. Loss arithmetic vs 128-grain, partial-prefix hit, expected extra recompute = (g−128)/2 tokens: g=1024 → 448 tok ≈ **~110 ms** at ~4k tok/s; g=2048 → 960 tok ≈ **~190 ms**. Versus per-cold-prompt saving of **~2.2 s**, coarse grain wins >10×. Crucially the dominant multi-turn agent case is served by finish write-through (`--dsv4-decode-reuse`, `capture_finish_frontier` `executor/dsv4.rs:945-1037`), which restores to the *exact finish position* independent of prefill grain.
- **(c) hybrid via executor-internal segmentation (Qwen3.5 `prefill_row_snapshotted` pattern, qwen35.rs:2582): rejected as default.** Forward-splitting at grain 512 costs ≈ the same GPU time as planner-chunking at 512 (per-forward floor F dominates); it only saves engine-tick overhead. Keep the pattern documented as the follow-up if Phase-3 telemetry (partial-hit waste counter, below) shows real loss.

### B. What actually bounds max chunk
Everything is already sized for 4096; nothing is per-slot:
1. `Dsv4PrefillDeepGemmLinearScratch` — `max_m = DSV4_PREFILL_QUERY_CHUNK` (`flashmla.rs:1571-1572`), asserts at `attention.rs:1240`, `:1421`. One shared instance (`kv_layout.rs:325`, built at `:774-780`). `input_fp8` alone = 4096×hidden_size bytes (~29 MB at 7168).
2. DSA `raw_indices` topk output — chunk-sized (`dsa.rs:183`), assert `attention.rs:9461`. Query GEMM scratch is tiled by `DSV4_DSA_PREFILL_QUERY_TILE` independently (`dsa.rs:176`).
3. Indexer staging ring — `DSV4_INDEXER_STAGING_RING_ROWS = 8192` rows (`kv_layout.rs:35`); assert `attention.rs:7583` requires one forward's delta ≤ ring depth ⇒ chunk ≤ 8192, non-binding at ≤4096.
4. `deepep_ll` — `max_tokens_per_step` (`dsv4.rs:1887`) already caps the *plan* via `planner.rs:55-60`; the executor capability (below) must take `min` with it.
5. FP32 probe scratch — `max_width × max_seq_len` (`kv_layout.rs:210-222`): no constraint.

⇒ **Max chunk = 4096 with zero reallocation.** Raising beyond 4096 means bumping `DSV4_PREFILL_QUERY_CHUNK` (~2× the shared scratch, +60-100 MB) — out of scope; 4096 stays the hard ceiling.

The **128-assuming** code paths (the real Phase-2 work):
- Planner one-unit cap (`planner.rs:84-86`) — removed, replaced by capability query.
- **bf16 SW ring update kernel race**: `dsv4_swa.cu:187-193` — with `num_tokens > sliding_window`, up to `chunk/128` threads write the same ring slot with **no ordering**; nondeterministic winner ⇒ wrong SWA keys for the next chunk. Today chunks = 128 = window so each slot has exactly one writer; at 2048 this is a live race (plausibly a contributor to the historical >2048 #146 needle failures alongside the block-map divergence fixed by `Dsv4BlockMap`, `kv_layout.rs:344-373`). Fix in Phase 2 (host-side tail slice).
- Restore ensure `matched_len % 128 == 0` (`executor/dsv4.rs:1264-1267`) — stays valid: planner keeps aligning chunk *ends* to `lcm(16,128)=128`, so boundary pages remain 128-aligned.
- Publish boundary predicate (`executor/dsv4.rs:900`) — unchanged; a non-aligned *final* chunk simply doesn't mark a boundary (planner.rs:87-91 already lets the final sub-unit tail through unaligned).

### C. Config-path unification
- `EngineLoadConfig.chunked_prefill_size: usize` → **`Option<usize>`** (`loaded.rs:41`, `#[serde(default)]`; default `None` at `:185`). CLI already distinguishes explicit (`serve.rs:478-480` only assigns on `Some`) — plumb that through instead of losing it.
- `loaded.rs:1925-1946` becomes **default-not-override**: `None` → per-kind default (Qwen dense/3.5: 2048; DSv4: **1024**, see below; Metal: 64). `Some(v)` → respected, clamped into executor-safe bounds `[128, 4096]` rounded **down** to a 128 multiple, with a single `warn!` when clamped. This also fixes the Qwen `.max(2048)` floor silently trampling an explicit smaller value (`loaded.rs:1934-1936`).
- New seam capability (next to `prefill_restore_boundary_alignment`, `infer-seam/src/lib.rs:299`): `fn max_prefill_chunk(&self) -> usize { usize::MAX }` — “largest single prefill forward this executor accepts; also the restore-snapshot grain when > restore alignment”. Planner (`planner.rs:60`, :84-86) uses `chunk_cap = prefill_chunk_size().min(cap).min(executor.max_prefill_chunk())` and **drops the one-unit cap**; end-alignment to `lcm(page, alignment)` stays.
- DSv4 returns `min(DSV4_PREFILL_QUERY_CHUNK, deepep max_tokens_per_step)`; Phase 1 returns `sliding_window` (=128) so plans are byte-identical until the flag flips.
- **Default choice (data):** t(2048) ≈ 400-500 ms decode stall per interleave vs ITL p50 71 ms and the 2026-05-25 axis3 ITL kill; t(1024) ≈ ~270 ms stall but 8× fewer stalls than today (total prefill occupancy 3.0 s → 0.8 s, so ITL *p50* likely improves while p99 rises to ~stall+71 ms). **Default 1024**; `--chunked-prefill-size` honored up to 4096; drop to 512 if the Phase-3 c16 ITL-p99 gate fails.

### D. Interactions
- **Interleaving fairness**: unchanged mechanism — decode rows ride every plan (`planner.rs:36-47`), prefill budget is `prefill_step_budget().min(cap − decode_rows)` (:56-59). Bigger chunk = longer per-tick stall, fewer stalls; gated in Phase 3.
- **Repair path**: `fit_plan_to_kv_pages` (`planner.rs:145-153`) sheds whole prefill rows; a 2048-token chunk demands 128 engine pages at once → higher shed probability under warm-cache pressure, retried next tick. Risk of persistent starvation only if capacity never reaches chunk demand — add a shed counter + (optional follow-up) truncate-instead-of-shed to the largest aligned chunk that fits.
- **MTP commit-fold**: decode-lane only (small m; `attention.rs:282` fold path, boundary-crossing publish skip at `executor/dsv4.rs:894-899`) — unaffected by prefill chunk size.
- **TP lockstep**: chunk size derives from config + rank-invariant capability — no rank-local data, plans stay identical across ranks.
- **DSpark**: per-chunk prompt seeding (`executor/dsv4.rs:2125-2134`) accumulates across chunks; verify the draft context append handles 2048-row chunks (the Qwen dspark analog warns at `qwen35/dspark.rs:704`).

## 3. Phases

### Phase 1 — config unification + capability query (no behavior change)
1. `infer-seam/src/lib.rs` (~:301): add `max_prefill_chunk()` default `usize::MAX`, doc the grain semantics; keep `prefill_restore_boundary_alignment` as **end-alignment only** (reword doc at :295-298).
2. Wire dispatch: `infer-cuda/src/lib.rs:681` block (add sibling method), `infer-cuda/src/executor.rs:591` (Dsv4 arm → `d.max_prefill_chunk()`, Qwen arms `usize::MAX`), `executor/dsv4.rs:1225` (Phase 1 body: `self.model.config.sliding_window.max(1)`).
3. `infer-core/src/planner.rs:60`, :82-91: `chunk_cap` takes `min(executor.max_prefill_chunk())`; delete the `restore_alignment > 1 ⇒ chunk.min(alignment_unit)` branch (:84-86); keep :87-91 end alignment. Update the mock at `infer-core/src/lib.rs:2150-2165` and planner tests (`lib.rs:2889`, `:5273-5308` etc.).
4. `infer-api/src/loaded.rs:41/:185/:502/:1925-1946`: `Option<usize>` + default-not-override + clamp-with-warn as in §C. Keep DSv4 resolved default = 4096 *config-side* in Phase 1 (planner caps to 128 anyway) so nothing moves.
5. **Acceptance gate:** planner unit tests proving byte-identical `ForwardPlan`s for DSv4 (alignment 128, capability 128) pre/post; `cargo test -p infer-core -p infer-api -p cli`; one smoke serve confirming the log now reflects the *effective* chunk. **Rollback:** revert — no persisted state, no kernel changes.

### Phase 2 — big chunks behind a flag
1. SW ring race fix: at `attention.rs:2114-2159` slice the tail host-side — if `k_prepared.seq_len > sliding_window`, pass only the last `window` rows (`start_pos += seq_len − window`); audit call sites `attention.rs:2701`, `:3112`, `:282` (m small, no-op) and the `start_pos_device` decode variants (:4574, :4671, :6304 — m small). Alternative: make `dsv4_swa.cu:180-194` last-writer-deterministic; host slice is simpler and fewer bytes.
2. Add flag (env, matching `runtime_flags.rs:159` pattern): `ARLE_DSV4_PREFILL_CHUNK` or a `cuda` runtime-flag field via `serve.rs:487`; `executor/dsv4.rs:1225` `max_prefill_chunk()` returns `min(DSV4_PREFILL_QUERY_CHUNK, deepep cap (dsv4.rs:1887), flag)` — flag absent = 128 (unchanged).
3. Verify assert inventory at 2048/4096: `attention.rs:1240`, `:1421`, `:7583`, `:9461`; dspark chunk append; compressor batched update + FP32 probe (carry lands at chunk end — `fp32_carry_stale` semantics from f59dd79af unchanged).
4. Measure t(chunk) curve at 512/1024/2048/4096 (single-stream, `scripts/dsv4_concurrent_probe.py` c1 lane) to replace §2 estimates.
5. **Acceptance gates (all with flag ON at 2048):** needle ladder (`scripts/eval_harness/needle_ladder.py`, lengths through 4000, 3 runs, 100%); prefix-reuse L2 (`prefix_reuse.py` WARM/REUSE/CTRL) **plus a restore-hit A/B**: warm a 2.6k-token doc, re-query with a prompt diverging at token ~1.4k — assert reuse commits to the last boundary page and output matches CTRL; `dsv4_variable_shape_dsa_gate.py`; MTP + `--dsv4-decode-reuse` smoke. **Rollback:** unset flag (default path byte-identical incl. the ring tail-slice, which is a no-op at seq_len ≤ window).

### Phase 3 — default flip
1. `loaded.rs` DSv4 resolved default → 1024 (one constant), flag/CLI honored to 4096.
2. **Acceptance gate — c-sweep per the bench spec:** c1/c4/c16 (and c32 champion row per CHANGELOG 2d38b05da) measuring TTFT p50/p99, ITL p50/p99, total tok/s, shed counter. Pass = TTFT c1 ≤ 1.0 s on the 2.8k prompt, cold prefill ≥ 3500 tok/s, c16 ITL p99 regression ≤ +250 ms and p50 non-regressing (the 2026-05-25 axis3 kill criteria applied); snapshot into `benchmarks/snapshots/`.
3. **Rollback:** revert the one-line default; explicit-flag users unaffected.

## 4. Risk register

| Risk | Exposure | Mitigation |
|---|---|---|
| #146-class numeric drift at chunk >128 (compressor carry, block maps) | High-severity, history at >2048 | `Dsv4BlockMap` already structural (`kv_layout.rs:344`); needle ladder mandatory per phase; f59dd79af carry semantics untouched (carry still advances at forward end) |
| SW ring write race at chunk > window | Silent SWA corruption | Phase-2 step 1 fix + a parity test: 2048-token prefill vs 16×128 reference, ring bytes compared |
| Restore correctness (alignment ensure `executor/dsv4.rs:1264`) | Hard abort on unaligned match | Planner end-alignment retained; boundary pages only at 128 multiples; prefix_reuse gate |
| Repair-path shed amplification (128 pages/chunk) | Prefill starvation under warm cache | Shed counter; optional truncate-to-fit follow-up (`planner.rs:145-153`) |
| ITL p99 regression at c16 | Precedented kill (2026-05-25) | Default 1024 not 2048/4096; c-sweep gate owns the flip; 512 fallback |
| deepep LL buffer < chunk | Silent chunk shrink | capability takes `min` with `max_tokens_per_step`; log the effective capability at boot |
| Coarser restore grain loses partial-prefix hits | ~110 ms avg per partial hit at 1024 | Telemetry counter (wasted-recompute tokens per restore); Qwen35 `prefill_row_snapshotted` segmentation pattern reserved as follow-up |

## 5. Expected wins (to be re-measured at Phase-2/3 gates)

- **TTFT c1** (2.8k uncached): 3.0 s → **~0.8 s** at default 1024 (~0.65 s at 2048 via flag).
- **Cold prefill**: 930 → **~3900 tok/s** (1024), ~5000+ (2048).
- **Quant-pack kernel share** at c16: 18.6% / 1.06M instances → **≤8%** (~8× fewer prefill forwards ⇒ ~8× fewer pack/swiglu-quant launch instances; bytes constant, launch overhead was the cost).
- **Config honesty**: `--chunked-prefill-size` becomes real for DSv4 (clamped, warned), planner cap derived from an honest executor capability instead of two disagreeing hard-codes.

### Critical Files for Implementation
- /Users/bytedance/code/agent-infer/crates/infer-core/src/planner.rs
- /Users/bytedance/code/agent-infer/crates/infer-api/src/loaded.rs
- /Users/bytedance/code/agent-infer/crates/infer-cuda/src/executor/dsv4.rs
- /Users/bytedance/code/agent-infer/crates/infer-seam/src/lib.rs
- /Users/bytedance/code/agent-infer/crates/infer-cuda/src/attention.rs (plus `crates/cuda-kernels/csrc/attention/dsv4_swa.cu` for the ring-race fix)