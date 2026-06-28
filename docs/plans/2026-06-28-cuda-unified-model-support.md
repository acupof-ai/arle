# CUDA Unified Model Support — close the batching gaps across the staged models

> Plan, 2026-06-28. Scope: CUDA backend only. Staged models: Qwen3-4B (dense),
> Qwen3-30B-A3B (vanilla MoE), Qwen3.5-122B-A10B (MoE), Qwen3.6-27B-FP8,
> DSv4-Flash-FP8.

## 1. Goal and the "not one-forward" framing

The seam (`infer-seam` `BackendExecutor` / `KvPool` / `KvBatchDescriptor`) and
`infer-core` `Engine<E,K>` are **already model-agnostic** — they schedule, page,
and batch without knowing which model runs below. The split lives entirely below
the seam, in the CUDA executor, which has **three irreducibly different forward
arms**:

| Arm | Models today | Attention kernel family |
|-----|--------------|-------------------------|
| R6-clean dense | Qwen3-4B dense | HD128 paged-attention (TileLang `paged_attn_v1`) |
| qwen35 | Qwen3.5-122B, Qwen3.6-27B-FP8 | HD256 GQA + gated-attn + shared-expert MoE |
| dsv4 | DSv4-Flash-FP8 | MLA + DSA (FlashMLA) |

These three cannot collapse into one forward: the attention math differs (HD128
vanilla MHA/GQA vs HD256 gated GQA vs MLA latent-compressed). **"Unify" here does
NOT mean one forward function.** It means: close the capability gaps so every
staged model **serves BATCHED** (c>1 throughput scales), converging the Qwen
family onto the already-batched **qwen35 arm**, and leaving **DSv4 as its own
already-batched arm**. The seam/Engine need zero changes; all work is
below-seam capability lifting.

Target end-state: a single CLI launch per model, all serving paged + batched,
no per-model special-casing above the seam.

## 2. Capability matrix (current state)

| Model | Arm | Serves? | Batches (c>1)? | Dtype | TP | Gap |
|-------|-----|---------|----------------|-------|-----|-----|
| Qwen3-4B dense | R6-clean | yes | **no** (bsz=1 hard-gated) | BF16 | 1 | Gap 1 |
| Qwen3-30B-A3B vanilla MoE | (none — config rejected) | **no** | n/a | BF16 | n | Gap 4 |
| Qwen3.5-122B-A10B | qwen35 | **no** (TP divisibility) | yes (paged per-row*) | BF16 | TP4 fails | Gap 3 |
| Qwen3.6-27B-FP8 | qwen35 | yes | yes (paged per-row*) | FP8 | 1 | Gap 2 (verify) |
| DSv4-Flash-FP8 | dsv4 | yes | yes (batched) | FP8 | TP8/EP8 | — (reference) |

\* "paged per-row" = c>1 is accepted and produces correct tokens, but decode
currently routes through a sequential per-row paged path (throughput measured
flat). Whether the qwen35 *true-batched* path beats per-row under the paged
default is the one load-bearing unknown — see §5.

## 3. The four gaps

Each gap is anchored to file:line, with a fix sketch, its dependency, per-gap
effort, and an honest VERIFIED (anchor read in source) / UNVERIFIED tag.

### Gap 1 — R6-clean dense decode is hard-gated to bsz=1 (VERIFIED)

The batched HD128 paged-decode kernels **already exist** (BF16 q16/q32/q40/q64 +
FP8 q32); the executor refuses to use them.

- `crates/cuda-kernels/kernels.toml:447-500` — 4 BF16 batched decode kernels
  registered (`batch_decode_paged_hd128_q{16,32,40,64}_kv8`), all
  `abi = "paged_attn_v1"`, `py_module = tools/tilelang/batch_decode_paged_hd128.py`.
- `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py:88-95` —
  `Grid=(1, num_q_heads, batch_size)`; `batch_size` is the third (bz) block dim;
  kernel indexes `KV_indptr[bz..bz+1]` per batch.
- `crates/cuda-kernels/kernels.toml:646-658` — FP8 batched
  `batch_decode_paged_hd128_fp8_q32_kv8`, same grid.
- `target/release/build/cuda-kernels-*/out/ffi_tilelang_generated.rs` — exports
  `tilelang_batch_decode_paged_hd128_q*_kv8_run_cuda`,
  `tilelang_batch_decode_paged_hd128_fp8_q32_kv8_run_cuda`,
  `resolve_paged_attn_v1(...)`, `resolve_paged_attn_fp8_v1(...)`.
- `crates/infer-cuda/src/attention.rs:5100-5104` — **hardcodes `(bsz, total_q,
  max_q) = (1, 1, 1)` for decode** (verified: the `if decode { (1,1,1) }`
  literal is present; comment at :5084-5085 says the literals "reproduce the old
  per-arm literals exactly").
- `crates/infer-cuda/src/executor.rs:1038-1043` — **`ensure!(rows == 1, "R6
  clean CUDA forward is single-row only")`** at the ForwardPlan level (verified
  in source).

Fix sketch:
1. Lift the `rows == 1` ensure at `executor.rs:1038`; build a multi-row launch
   that fills `KV_indptr` / `qo_indptr` per row and passes `bsz = rows.len()`.
2. Replace the `(1,1,1)` decode literal at `attention.rs:5100` with
   `(bsz, total_q=bsz, max_q=1)` for batched decode.
3. For the FP8 dense path, remove the BF16-only gate (reported at
   `loader.rs:2576` — **UNVERIFIED**, confirm during impl) and wire
   `resolve_paged_attn_fp8_v1` alongside the existing `resolve_paged_attn_v1`.

Dependency: **Gap 1 is the only gap that depends on the §5 unknown** — the
per-row-vs-batched throughput question. If the qwen35 A/B (§5) shows true-batched
loses to per-row under paging, the same per-row-default may be the right call for
R6-clean too, and Gap 1 reduces to "remove the artificial rows==1 ceiling so
multi-row plans are *accepted*" without forcing the batched kernel as default.

Effort: medium. The kernels and FFI exist; the work is the multi-row launch
plumbing + indptr construction + the FP8 resolve wiring. No new kernel.

### Gap 2 — Qwen3.6-27B-FP8 batched-decode is verified-paged-per-row only (VERIFIED)

Qwen3.6-27B-FP8 serves today on the qwen35 arm, but the *true-batched* decode
kernel is unreachable under the paged default.

- `crates/infer-cuda/src/executor.rs:4448` — master gate:
  `if !qwen35_batched_decode_enabled() || self.recall_active() ||
  self.full_attn_paged()` falls back to **sequential per-row** (verified: the
  fallback loop calling `submit_decode_row` is at :4450-4460). `full_attn_paged()`
  (`executor.rs:3590-3591`) returns true whenever `full_attn_kv.is_some()` —
  always, since the shared-paged migration.
- `crates/infer-cuda/src/executor.rs:3246` — **always** calls
  `build_full_attn_kv_pool()` (PagedKVPool), so the contiguous per-slot
  `k_caches`/`v_caches` the batched kernel needs are never populated.
- `crates/infer-cuda/src/qwen35.rs:488-499` — documents `k_caches`/`v_caches`
  are "EMPTY by default (full-attn KV is paged); populated only by the legacy
  contiguous lane".
- `crates/infer-cuda/src/qwen35.rs:6283` — true-batched kernel entry
  (`fused_gqa_attention_decode_batched`, lines 6315-6362) is reached only when
  `qwen35_batched_decode_attention_enabled() && head_dim == 256`; else per-row
  loop at :6376-6441 (three single-row kernels per slot).
- Env gates (VERIFIED): `ARLE_QWEN35_BATCHED_DECODE` (default on, "0" disables)
  at `executor.rs:2870-2875`; `ARLE_QWEN35_BATCHED_DECODE_ATTENTION` (default on,
  "0" disables) at `qwen35.rs:72-79`.

Fix sketch: this gap is **resolved by deciding §5**, not by new code in the
common case. Two outcomes:
- If §5 shows true-batched wins: add the contiguous-KV lane the batched kernel
  needs — either (a) a flag to skip `PagedKVPool` allocation at
  `executor.rs:3246`, or (b) populate `k_caches`/`v_caches` in
  `Qwen35SlotState::acquire` (`qwen35.rs ~700`). Neither exists today.
- If §5 shows per-row is fine: Gap 2 is **closed as-is** — Qwen3.6-27B-FP8
  already serves batched-correct (paged per-row), throughput-flat is acceptable,
  document the verdict and remove the dead true-batched lane per no-half-states.

Dependency: gated entirely on §5. Independent of Gaps 1/3/4 (no shared code).

Effort: small if §5 says per-row (doc + dead-lane cleanup); medium if §5 says
batched (new contiguous-KV lane).

### Gap 3 — Qwen3.5-122B GQA replication for TP where num_kv_heads < world_size (VERIFIED feasible, source-survey sketch)

Qwen3.5-122B has 2 KV heads; TP4 fails the `num_kv_heads % world_size == 0`
divisibility check. Fix is uniform KV-head replication.

- `crates/infer-topo/src/sharding.rs:186-203` — `head_shard`; divisibility
  assumed at ~:201.
- `crates/infer-cuda/src/qwen35.rs:2127-2138` — validation ensures
  `linear_num_key_heads`/`linear_num_value_heads` divisibility (the failing
  check).
- `crates/infer-cuda/src/qwen35.rs:1676-1679` — `local_full_attn_kv_dim`
  dimension calcs.
- `crates/infer-cuda/src/loader.rs:2215-2235` — `load_qkv_head_sharded_quant_aware`
  per-head shard bounds.
- `crates/infer-cuda/src/attention.rs:6285-6333` — GQA-ratio decode batch.

Fix sketch: `head_replicate()` — when `num_kv_heads < world_size`, compute
`replicas_per_head = ceil(world_size / num_kv_heads)`. For 2 KV heads at TP4:
ranks 0-1 both own KV head 0 (`local_kv_heads=2`), ranks 2-3 both own head 1.
Changes: (1) `sharding.rs:201` return `(num_q_heads/world_size, num_kv_heads)`
under replication, not `num_kv_heads/world_size`; (2) `qwen35.rs:2127-2138`
allow the replication path past divisibility; (3) dims at :1676-1679 unchanged
(replicated ranks compute identical dims); (4) `loader.rs:2215-2235` replicated
ranks load the **full** per-head K/V slice (not a sub-slice); (5)
`attention.rs:6285-6333` `gqa_ratio = num_q_heads / local_kv_heads`, replicated
ranks read **identical read-only** K/V cache pointers.

Risk: cache coherence — `k_cache_table`/`v_cache_table` MUST be byte-identical
across replicated ranks (read-only, so safe if loaded identically). GQA-ratio
invariant: kernel needs an integral ratio; replication preserves it only if
uniform across ranks. Mitigation: unit tests `head_replicate(16,2,4)->(4,2)` and
`head_replicate(64,8,8)->(8,1)`; loader validation that replicated ranks compute
the same `local_kv_heads`; debug-assert in the decode batch that all ranks in a
replicate group pass identical cache pointers.

Dependency: **independent** of Gaps 1/2/4 (topo + loader only; no attention-arm
restructure). Needs a TP-capable box (H20) for the multi-rank parity gate.

Effort: medium. Pure sharding/loader change reusing existing per-head load path;
the kernel and forward are untouched.

### Gap 4 — Vanilla Qwen3-MoE config adapter (VERIFIED feasible, source-survey sketch)

Qwen3-30B-A3B is `model_type=qwen3_moe` (vanilla HF); the loader rejects it.

- `crates/infer-api/src/loaded.rs:1668` — rejection site; classification at
  :203-212.
- `crates/infer-cuda/src/qwen35.rs:2100-2180` — loader `construct`.
- `crates/qwen35-spec/src/lib.rs:570-625` — `Qwen35Config` schema;
  :707-728 tensor names; :708 `model_prefix()`.

Schema differences (vanilla `qwen3_moe` vs ARLE `qwen3_5`):
1. **Tensor prefix**: vanilla `model.{layer}`; ARLE `model.language_model.{layer}`
   (`qwen35-spec:707-728`).
2. **Gated full-attn**: vanilla plain `q_proj [H*hd, hidden]`; ARLE gated
   `q_proj [H*hd*2, hidden]` (`qwen35-spec:623-624` `full_attn_gated`).
3. **Shared expert**: vanilla none (pure routed MoE); ARLE has
   `shared_expert_intermediate_size` (`qwen35-spec:610`) + `shared_expert_gate`,
   `shared_expert.{up,down}_proj` (`qwen35-spec:144-148`).

Fix sketch: define a separate `Qwen3MoeConfig` matching the vanilla schema
(`num_experts`/`num_experts_per_tok`/`hidden_size` at config root, not nested);
implement `Into<Qwen35Config>` with explicit field mappings: prefix tensor names
with `model.language_model`, `full_attn_gated=false`,
`shared_expert_intermediate_size=0`, synthesize `layer_types` (vanilla uniform).
In `qwen35.rs:2100-2180`, peek `model_type` in the raw JSON before deserialize;
choose parser accordingly; validate GQA head counts and reject unsupported
features.

Risk: serde mismatch on extra/missing fields. Mitigation: separate struct + the
`Into` adapter; JSON peek to route the parser; unit test parsing a real vanilla
`qwen3_moe` config and asserting adapted prefix/`gated=false`/`shared=0`.

Dependency: **independent** of Gaps 1/2/3 (config + loader only). Note: once
loaded, the vanilla MoE runs on the qwen35 arm with `gated=false` and no shared
expert — both already conditional in that arm, so no forward change.

Effort: small-medium. One config struct + adapter + JSON-peek routing; no kernel
or forward change.

## 4. Dependency DAG and recommended landing order

```
            §5 pod A/B (qwen35 c>1 non-paged) ── load-bearing unknown
                        │
                        ▼
   ┌──────────── Gap 1 (R6-clean batched decode) ── depends on §5 verdict
   │
   │   Gap 2 (Qwen3.6 batched-or-confirm) ── also gated on §5, but no shared code
   │
   │   Gap 3 (122B GQA replication) ──┐
   │                                  ├── independent, parallelizable
   │   Gap 4 (vanilla-MoE adapter) ───┘
```

Gaps **2, 3, 4 are independent of the kernel unknown** and of each other. Gap 1
(and the "batched vs per-row" half of Gap 2) is the only work blocked on §5.

Recommended order:
1. **§5 pod A/B first** — it unblocks Gap 1 and decides Gap 2's shape. Cheap,
   one serve config + a fixed-c sweep.
2. **Gap 4 then Gap 3 in parallel** (both independent, no §5 dependency).
   Gap 4 unblocks a 5th model serving at all; Gap 3 unblocks the 122B on TP.
   Either order; both can land while §5 runs.
3. **Gap 1** after §5 returns. If §5 says per-row is fine, Gap 1 shrinks to
   removing the `rows==1` ceiling (accept multi-row, keep per-row default).
4. **Gap 2** last — it's the cleanup/decision that closes the matrix; resolves to
   doc + dead-lane removal (per-row verdict) or new contiguous-KV lane (batched
   verdict).

## 5. The one load-bearing unknown — qwen35 c>1 when non-paged

Everything else is VERIFIED in source. The single thing that requires a measured
pod A/B is: **does the qwen35 true-batched decode path beat the paged per-row
path at c>1?** The current code makes the true-batched lane unreachable under the
paged default (Gap 2), and c>1 throughput was measured flat on the paged per-row
path — but that flatness has **not** been attributed to per-row sequencing vs a
deeper bottleneck. We must not assume; we measure.

The complication (VERIFIED): there is **no launch flag today** that puts
Qwen3.6-27B-FP8 into the non-paged + batched-decode-ON config —
`executor.rs:3246` always allocates the `PagedKVPool`, and the batched lane is
gated off by `full_attn_paged()` at `executor.rs:4448`. So the A/B requires a
minimal enabling change before it can run:

- **Control (A):** paged default, per-row decode — the current shipping path.
  Launch: `arle serve --backend cuda --model-path <Qwen3.6-27B-FP8>` (paged
  on, `ARLE_QWEN35_BATCHED_DECODE=1` default but skipped by the paged gate).
- **Treatment (B):** non-paged contiguous KV + true-batched decode. Requires
  the enabling change from Gap 2 fix-sketch (a) — a flag to skip `PagedKVPool`
  at `executor.rs:3246` so `full_attn_paged()` returns false and the gate at
  `executor.rs:4448` admits the batched lane — then
  `ARLE_QWEN35_BATCHED_DECODE=1 ARLE_QWEN35_BATCHED_DECODE_ATTENTION=1`.

Measurement: `scripts/bench_guidellm.sh qwen36-batched-ab --concurrencies
1,4,16,64 --max-seconds 120` against each arm, same binary, same shell, same
prompts, two configs side-by-side (per the matched-A/B rule). Decision metric:
output throughput (tok/s) and ITL p50/p99 at c=16/64. If B does not beat A at
c>1, **per-row is the verdict** — Gap 1 keeps per-row default, Gap 2 closes with
dead-lane removal. If B wins, the contiguous-KV lane is licensed.

This must run on the H20 pod (CUDA). Until it returns, Gap 1's default-kernel
choice and Gap 2's shape are explicitly **deferred — accepting the uncertainty**.

## 6. Correctness gates per gap

Per `docs/bench-and-trace-spec.md §7.1` (correctness gate before any perf
reporting) and the KV-precision parity gate (`scripts/needle_gate.py` +
same-config-twice envelope, NOT byte-identity — MoE non-determinism). Every gap
clears its gate before a wins entry and before any default flip.

| Gap | Correctness gate |
|-----|------------------|
| 1 (R6-clean batched) | Needle ladder on Qwen3-4B at c=1 and c>1 (4/8/16); **c>1 parity** — batched-decode tokens vs the validated bsz=1 per-row baseline within the same-config-twice non-determinism envelope. FP8 dense path: needle exact vs the BF16 envelope. Smoke: 4-tok prompt, first 5 chars not all identical (`spec §7.1`). |
| 2 (Qwen3.6 batched/confirm) | If batched lane enabled: needle ladder + same-config-twice on the true-batched arm vs the paged per-row arm; c>1 parity within envelope. If closing as per-row: confirm the existing paged per-row passes the needle ladder at c=1/4/16 (it must already, since it's the shipping path). |
| 3 (122B GQA replication) | Multi-rank parity: needle ladder under TP4 with replication vs a known-good non-replicated reference (e.g. TP2 where 2 KV heads divide evenly), same-config-twice on each. Unit tests `head_replicate(16,2,4)`, `head_replicate(64,8,8)`. Debug-assert identical cache pointers across replicate group. |
| 4 (vanilla-MoE adapter) | Needle ladder on Qwen3-30B-A3B once it loads; same-config-twice envelope. Unit test parsing a real vanilla `qwen3_moe` config asserting adapted prefix / `gated=false` / `shared=0`. Smoke: 4-tok prompt non-degenerate. |

Cross-cutting: no default flip lands on a single shape — per the distilled lesson
"backend/quant/decoding default flips need multi-shape verification". Any batched
default (Gap 1 or 2) needs ≥2 binding production shapes clearing TTFT *and* ITL
*and* output throughput before the flip, not just c-reachability.
