# Metal Optimization — Complete Logical Reasoning + Analysis (2026-05-07 evening)

This entry is the synthesis the next several /loop ticks (or any
fresh session) should read FIRST before touching code. It folds
together the morning's SOTA gap analysis, the unification
recalibration, the M_d.1 namespace closure, the M_e.1 paged-KV plan
(with errata), the apples-to-apples c-sweep, and the c=1 isolation
decomposition into a single coherent picture.

Every numerical claim below is cited to a wins entry committed
earlier the same day.

## 1. Where ARLE-Metal stands today (empirical)

All measurements: M4 Pro (20-core), `models/Qwen3.5-0.8B-MLX-4bit`,
guidellm 0.6.0, 30 s `--fast`-style cells, sequential server runs.

### 1.1 Apples-to-apples by workload

| Cell | ARLE ITL p50 | mlx-lm ITL p50 | ARLE TTFT p50 | mlx-lm TTFT p50 | ARLE out tok/s | mlx-lm out tok/s |
|---|---:|---:|---:|---:|---:|---:|
| c=1 short (128/2048) | 3.95 ms | 3.18 ms | **37.4 ms** | 165.1 ms | 245.8 | 308.2 |
| c=1 long (4096/256) | 4.37 ms | 3.38 ms | **920 ms** | 1048 ms | 129.5 | 136.9 |
| c=4 long (4096/256) | 19.34 ms | 7.17 ms | **1.20 s** | 4.51 s | 147.5 | 196.1 |
| c=8 long | 39.77 ms | — | 7.61 s | — | 144.1 | — |
| c=16 long | 82.49 ms | 18.97 ms | 5.18 s | 12.69 s | 78.2 | **467.9** |

Sources: `2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md`,
`2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md`,
`2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md`.

### 1.2 Algebraic decomposition

The c=4 long-context ITL gap of 2.70× decomposes structurally:

```
ARLE     c=4 / c=1 = 19.34 / 4.37 = 4.43×
mlx-lm   c=4 / c=1 =  7.17 / 3.38 = 2.12×

Per-token kernel gap        = 4.37 / 3.38 = 1.29×
ARLE-specific batching gap  = 4.43 / 2.12 = 2.09×

2.70×  =  1.29×  ×  2.09×
```

The per-token kernel gap is small. The batching gap is the dominant
structural problem.

### 1.3 What ARLE wins on

- **Prefill / TTFT.** 1.14× (c=1 long) to 4.4× (c=1 short) faster
  than mlx-lm. The chunked prefill + decode-priority scheduler
  interleave (locked by `decode_priority_holds_under_c4_mixed_traffic`
  in `infer/src/backend/metal/scheduler.rs`) delivers regardless of
  concurrency.
- **ITL p95 stability at sweet spot.** ARLE c=4 ITL p95 = 19.74 ms;
  mlx-lm c=16 ITL p95 = 33.86 ms. ARLE has tighter tails when run
  inside its hot-path envelope.

## 2. Why the batching multiplier is 2.09× worse

`infer/src/backend/metal/request_state.rs::Qwen35PackedDecodeBatch`
(line 773-784) carries:

```rust
struct Qwen35PackedDecodeBatch<'a> {
    batch_cache_len: i32,             // shared column cursor
    left_padding: Vec<i32>,           // per-row pad
    packed_kv_flat: Vec<MlxArray>,    // ONE shared cache, per layer
    packed_gdr_flat: Vec<MlxArray>,
    ...
}
```

Every row's valid KV data sits in `[left_padding[i], batch_cache_len)`.
At c=N with prompt-length variance δ, every row pays for the LONGEST
in-flight prompt's left padding. mlx-lm's decode does not left-pad —
each request maintains its own KV slice and packing happens only at
the SDPA call with no pre-padding.

This kernel architecture choice is the root of the 2.09× ARLE-specific
batching multiplier. Plumbing knobs (e.g. `--max-running-requests` flag
shipped in commit `bbc484c`) cannot fix it; only a kernel rewrite can.

## 3. Substrate audit — what's already in tree

| Component | Path | State |
|---|---|---|
| Token-level KV pool | `infer/src/backend/metal/kv_pool.rs` | **Fully built**: alloc/share/free/write/gather APIs |
| Per-driver use under `--kv-pool` | `request_state.rs:3270-3371` | **Wired but per-driver**: each `Qwen3StepDriver` has its own pool, all writes use `METAL_REQUEST_STATE_ID` singleton key |
| Cross-request shared pool | — | **Not yet** — `MetalKVPool` API supports it (request_id parameter) but no caller threads real IDs |
| Qwen3.5 packed decode → pool | — | **Not yet** — `Qwen35PackedDecodeBatch` bypasses the pool entirely, uses left-pad concat |
| RadixCache namespace (M_d.1) | `prefix_cache.rs` + `tokenizer.rs` | **Closed**: tokenizer SHA-256 + 32-byte namespace + load_snapshot bypass guard + isolation test (commits `fc68450`–`0e1bc3d`) |
| ELI Layer-1 | `infer/src/http_server/openai_v1.rs` + smoke `curl` | **Verified end-to-end**: HTTP 200 in 70 ms with `tool_choice` + `response_format` |

Audit-error history: the original M_e.1 plan claimed "MetalKVPool is
fully built but unused on the hot path"; that grep missed
`request_state.rs`. Errata in §7 of the plan + a feedback memory
(`feedback_substrate_audit_grep_full_tree.md`) capture the lesson.

## 4. Gap → fix mapping

The three structural axes of the gap:

| Axis | Magnitude | Owner track | Effort | Status |
|---|---|---|---|---|
| **A. Batching/padding** | 2.09× | M_e.1 paged-KV | L (multi-commit) | Plan ready, errata applied, acceptance numbers tightened (217f1f8) |
| **B. Per-token kernel** | 1.29× | M_e.0 profile pass | M | Demoted from blocker; scope bench-only first |
| **C. Long-context KV scaling** | 1.05× (4096 vs 128) | Composes with A | — | Will partially close as paged-KV lands; full attention-compute optimization is post-A |

ELI integration is orthogonal to all three:

| Layer | Status |
|---|---|
| Layer 1 — nexil OpenAI adapter at ARLE `/v1` | **Live**, verified end-to-end |
| Layer 2 — native `ArleAdapter` in eli/nexil | **Designed** in `2026-05-07-eli-arle-native-provider-design.md`, impl is sibling-repo work needing user authorization |
| Layer 3 — embedded `arle-embed` | Deferred until Layer 1+2 battle-tested |

## 5. Sequencing — by leverage, ordered

The atomic-commit sequence below is the actual work plan. Each step
has a well-defined acceptance number; landing them in order keeps the
tree coherent and lets each commit revert independently.

### Phase 1 — preparation (no behavior change)

**P1.1 — Promote MetalKVPool from per-driver to runtime-shared.**
Move ownership from `Qwen3StepDriver.kv_pool` to the scheduler runtime;
each driver borrows the shared pool. Every request gets a unique
`request_id` (replace `METAL_REQUEST_STATE_ID` singleton).
- Effort: M
- Files: `metal/runtime.rs`, `metal/request_state.rs`
- Acceptance: Qwen3 `--kv-pool` path under c=4 retains today's ITL
  ± 5%. Tests still pass.

**P1.2 — Slot lifecycle wired to scheduler admit/finish.**
On admit: `pool.alloc_tokens(req_id, prompt_len)`. On finish:
`pool.free_request(req_id)`. Reads still come from the legacy concat
cache; pool slots are tracked but their data is not yet consumed by
attention.
- Effort: S
- Files: `metal/runtime.rs`, `metal/request_state.rs`
- Acceptance: prefill alloc overhead < 1 ms per request.

### Phase 2 — Qwen3.5 dual-write under opt-in flag

**P2.1 — Qwen3.5 packed decode dual-write.** Behind a new
`--metal-qwen35-pool` flag (or reuse `--kv-pool` with model dispatch),
make `Qwen35PackedDecodeBatch` write per-step K/V to BOTH the legacy
left-pad concat cache AND the shared pool via `pool.write_kv_slots`.
Attention still reads from concat (legacy correctness). Add a
property test: `pool.gather_kv(layer, req_id)` matches the concat
slice for the same request, byte-for-byte.
- Effort: M
- Files: `metal/qwen35.rs`, `metal/request_state.rs`, plus a
  `#[cfg(test)]` parity test
- Acceptance: dual-write does NOT regress c=4 ITL beyond 1 ms.

### Phase 3 — the unlock

**P3.1 — Kernel cutover under flag.** SDPA input K/V comes from
`pool.gather_kv_rows(layer, request_ids)` instead of the concat cache.
Each row's gathered tensor has exactly `current_seq_len` keys — no
left-pad. Attention mask becomes per-request scalars again, RoPE
offsets simplify.
- Effort: L
- Files: `metal/qwen35.rs`, `metal/runtime.rs`, possibly
  `crates/mlx-sys/src/mlx_qwen35_model.cpp`
- **Acceptance** (tightened 217f1f8 per c=1 isolation):
  - c=4 ITL p50 ≤ 9.3 ms (currently 19.34 ms)
  - c=16 ITL p50 ≤ 12 ms (currently 82.49 ms)
  - c=16 ITL p95 ≤ 15 ms
  - c=16 output tok/s ≥ 350 (currently 78)
  - c=1 long ITL p50 ≤ 1.05× of pre-commit 4.37 ms

### Phase 4 — promotion

**P4.1 — Flip default `max_running_requests` from 4 to 16.** After
P3.1 lands and a c-sweep confirms monotonic scaling.
- Effort: S
- Files: `metal/scheduler.rs`
- Acceptance: c-sweep at default config shows ITL p50 monotonically
  bounded as c grows; c=16 within ±25% of mlx-lm's 467 tok/s.

**P4.2 — Retire the concat path.** Once P4.1 has shipped one bench
window without rollback, drop the legacy concat cache code path
entirely. Pure deletion-style refactor per
`feedback_no_half_states.md`.
- Effort: S
- Files: `metal/request_state.rs`, `metal/qwen35.rs`, `metal/runtime.rs`

### Phase 5 — per-token kernel polish (the residual 1.29×)

**P5.1 — c=1 profile pass.** Use `mlx instruments` / metal capture +
the existing `metric.set_memory_bytes` trace at `runtime.rs:2879` to
identify the dominant per-token cost on ARLE single-stream. Hypotheses
to test:
- Extra `eval()`/`item()` boundaries on the Rust hot path
- Per-driver step path missing `mx.compile`-style fusion
- Fixed C++-bridge per-call cost
- Effort: M (profile + targeted fix)
- Acceptance: c=1 long ITL p50 ≤ 3.50 ms (was 4.37 ms; closes the
  1.29× gap to ≤ 1.04× of mlx-lm).

### Phase 6 — ELI Layer 2 (orthogonal, requires authorization)

**P6.1 — Native `ArleAdapter` in eli/crates/nexil.** Per the design
doc `2026-05-07-eli-arle-native-provider-design.md`. Six files in the
sibling repo. Two-commit sequence (overlay first, custom_headers
plumbing second). Sibling-repo write requires user authorization.

## 6. Quantitative definition of "world #1 on Metal"

After Phase 4 lands:

- **c=4 long-context output tok/s** ≥ mlx-lm c=4 of 196 (≈ parity)
- **c=16 long-context output tok/s** ≥ 0.75× mlx-lm c=16 of 467
  (= 350; ARLE acceptable up to 25% behind, given matched-c is rare
  in production agent traffic)
- **TTFT at every workload** ≤ mlx-lm TTFT (already true; preserve)
- **ITL p95 stability** ≤ mlx-lm ITL p95 at the same c (already true
  at c=4; should hold at c=16 post-paged-KV)
- **Long-context (W6, 32k prompt)** added to M6 Metal snapshot per
  `docs/plans/M6-metal-world-rank-snapshot.md`; output tok/s within
  ±15% of best Apple-Silicon baseline

After Phase 5 lands:

- **c=1 long-context ITL p50** ≤ 3.5 ms (parity with mlx-lm 3.38 ms
  ±5%) — closes the residual per-token gap

## 7. Risks + unknowns

1. **P3.1 is L effort and high-risk.** The kernel cutover touches
   the C++ bridge and the model step path. A single-tick attempt is
   not safe. Plan to land in 2–3 sessions with strong dual-write
   property-test backing.
2. **C++ bridge per-call cost is unverified.** The hypothesis is in
   §5 P5.1; until measured, we cannot rule out that per-token cost
   includes >1 ms fixed overhead that no kernel rewrite reduces.
3. **MLX 0.31.1 fast::rope quirk** (`feedback_mlx_rope_axis.md`,
   `feedback_mlx_rope_layout.md`) requires `[B, H, S, D]` ordering
   AND array-form RoPE offsets; switching to per-request gathered
   tensors must carefully preserve this. Tripwire tests in
   `metal::mlx::tests` should be re-run after P3.1.
4. **CUDA path stays untouched.** All Metal commits in this plan
   keep `infer/src/scheduler/cuda/*` and `crates/cuda-kernels/*`
   bit-identical. Verified by `cargo check -p infer
   --no-default-features --features cuda,no-cuda` + clippy at every
   commit, per CLAUDE.md.
5. **bench host availability.** Each phase needs a real Metal bench;
   they happen on this M4 Pro host. If the host changes, re-snapshot
   c=1 baseline first to recalibrate the 4.37 ms ITL anchor.

## 8. What "start optimizing" means now, concretely

Next bench-track action: implement Phase 1 (P1.1 then P1.2) as two
atomic commits + bench regression check. P1.1 is M effort and slightly
above the 12-min cron tick safety margin; expect to land it across
two ticks (first tick: read substrate + draft change + cargo check;
second tick: tests pass + commit + bench regression + push).

Next ELI-track action: confirm whether sibling-repo work in
`/path/to/code/eli/` is authorized for Layer 2 implementation.
If yes, P6.1 is independent of all Metal work and can run in
parallel.

Next CUDA-track action: nothing (out of scope per user directive).

## 9. References

Bench evidence (chronological, all 2026-05-07):
- `2026-05-07-bench-guidellm-metal-c-sweep-m4pro.md` (c-sweep, surfaces
  the c=16 collapse)
- `2026-05-07-bench-guidellm-metal-c4-apples-vs-mlxlm.md` (matched
  c=4, surfaces "two distinct gaps")
- `2026-05-07-bench-guidellm-metal-c1-isolation-decomposition.md`
  (c=1 isolation, surfaces 1.29× × 2.09× decomposition)

Plans:
- `docs/plans/M_e1-metal-paged-kv-hot-path.md` (the load-bearing
  optimization plan)
- `docs/plans/M6-metal-world-rank-snapshot.md` (canonical bench
  protocol)

Project context:
- `docs/projects/2026-05-07-metal-world-first-gap-analysis.md` (morning
  SOTA report — superseded for ARLE-specific numbers by the bench
  evidence above; still authoritative for upstream landscape)
- `docs/projects/2026-05-07-metal-world-first-recalibration-vs-unification.md`
  (frames Metal work as backend-unification ripple)
- `docs/projects/2026-05-07-eli-arle-native-provider-design.md` (Layer
  2 design, sibling-repo work)
- `docs/plans/backend-unification.md` §M-series (master roadmap;
  M1–M5 landed today; M6 in flight)

Memory:
- `feedback_metal_unification_frame.md`
- `feedback_substrate_audit_grep_full_tree.md`
- `feedback_no_speculative_interface_shaping.md`
- `feedback_no_half_states.md`
