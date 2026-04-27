# CUDA TileLang single-prefill HD128 / HD256 (contiguous-KV) — closed without implementation, 2026-04-27

> Tranche 5 stub — single-prefill (contiguous-KV) FlashInfer paths
> deliberately stay on FlashInfer. **Option A** per
> `docs/plans/tilelang-integration.md` §"Tranche ledger" Tranche 5 row:
> doc-only tranche, no code change, no bench to run. This entry exists
> so the next iteration of "全部接入" finds the decision recorded and
> does not re-investigate.

## Goal

- Optimization (per `docs/bench-and-trace-spec.md` §goal taxonomy): close
  the "全部接入 TileLang" series by deciding what to do with the two
  remaining FlashInfer attention symbols — `flashinfer_single_prefill`
  (HD128 contiguous-KV) and `flashinfer_single_prefill_hd256` (HD256
  contiguous-KV). **Decision: Option A — no implementation work.** The
  contiguous-KV single-prefill paths stay on FlashInfer; TileLang is not
  wired in here.

## Hypothesis

- Contiguous-KV single-prefill is offline-only on this codebase. T0
  verification confirmed `prefill_uses_paged_pool() = true` for both
  Qwen3 and Qwen3.5; the paged pool is always active during server
  warmup, so the OpenAI hot path never reaches these symbols. Their
  remaining callers are:
  - `infer/src/speculative/cuda.rs:124` — speculative draft model in
    offline `batch_serving` (not OpenAI server).
  - `infer/tests/bench_prefill.rs` — offline bench harness.
  - `infer/examples/regenerate_test_data.rs` — JSON baseline regenerator.
  - `infer/src/bin/bench_serving.rs` — offline serving bench.
- TileLang's AOT cubins are paged-only: `page_size = 16` with
  indirection through `kv_indices` + `kv_indptr` + `kv_last_page_len`.
  Contiguous KV is a flat `[seq_len, num_kv_heads, head_dim]` tensor
  with no paged pool to index. Closing the gap would require either a
  new contiguous-KV TileLang kernel (work for an offline-only callsite)
  or a "1-page virtual paged pool" wrapper at the call site that pays
  per-call indirection cost on a path no production caller hits.
- Hypothesis: the gain from a TileLang single-prefill kernel (if any)
  does not justify either option. The L4 evidence behind the user's
  "全部接入" instruction covers paged-prefill HD128, paged-prefill
  HD256, paged-decode HD128, paged-decode HD256, and the TC-decode
  alias — none of it covers contiguous-KV single-prefill, so there is
  no measured ceiling to chase. Combined with "不着急删除", the right
  move is to leave FlashInfer in place.

## Decision

- **Tranche 5 is closed without implementation.** The single-prefill
  FlashInfer paths remain canonical for their (non-production) callers.
  The ledger row in `docs/plans/tilelang-integration.md` §"Tranche
  ledger" is the durable record; this file is the bench-side companion
  required by `CLAUDE.md` §Benchmarks (every in-scope diff produces a
  wins entry — the "diff" here is the doc update, but the rule applies
  uniformly so a future reader does not re-open the question).

## Command

```bash
# No bench to run. Local verification only:
cargo check -p infer --no-default-features --features cuda,no-cuda
cargo check -p infer --no-default-features --features cuda,no-cuda,tilelang-attn
```

## Environment

- **Backend:** cuda (default-build only; `tilelang-attn` arm unchanged
  for this tranche).
- **Models:** N/A — no callers of single-prefill on the OpenAI hot path.
  Offline callers (speculative draft, tests/bench/examples) continue
  to use FlashInfer unchanged.
- **Hardware:** N/A — no remote bench scheduled. If the decision is
  ever revisited (e.g. speculative decoding moves on-server, making
  contiguous-KV single-prefill production-reachable), reopen this
  tranche with a fresh plan and bench.
- **Commit:** Tranche 5 doc-only commit (this file + plan ledger row).
- **Feature set:** unchanged from before — default `cuda` build keeps
  FlashInfer; `cuda,tilelang-attn` build keeps the Tranches 0/2/3/4
  TileLang cutovers and otherwise keeps FlashInfer for single-prefill.

## Canonical params (DO NOT CHANGE PER-RUN)

- N/A — no bench. Listed here only to keep the template skeleton
  consistent with sibling pending-remote stubs.

## Results

- Status: **closed without implementation** (Option A).
- Local verification completed (macOS workspace), exit 0 in both arms:
  - `cargo check -p infer --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cuda,no-cuda,tilelang-attn`
  Both commands print only the standard
  `cuda-kernels: no-cuda feature active: skipping CUDA/Triton kernel
  compilation` warning and `Finished dev profile`. No code change in
  this tranche, so the cargo check exit status is unchanged from the
  pre-tranche baseline (and from the post-Tranche-4 baseline).
- No remote H100 / L4 sweep scheduled. If the decision is ever
  revisited, the new plan must justify why a non-production callsite
  warrants either a new contiguous-KV TileLang kernel or a 1-page
  virtualization wrapper.

## Problems / surprises

- None. The tranche ran exactly as scoped: read T0 verification, read
  the FlashInfer single-prefill call sites, confirm none are on the
  OpenAI hot path, decide Option A, write the ledger row + this stub.

## Learnings

- "全部接入" as a goal is conditioned on the L4 evidence; the L4
  evidence is for paged-prefill HD128 / paged-prefill HD256 /
  paged-decode HD128 (TC) / paged-decode HD256, not for contiguous-KV
  single-prefill. When extending the series, ask "is the path on the
  OpenAI hot path?" before assuming a TileLang swap is even meaningful.
- The contiguous-KV vs paged-KV layout split is a real architectural
  boundary: TileLang's AOT cubins are paged-only by construction
  (`page_size = 16` is a compile-time invariant), and the right move is
  to keep the contiguous-KV path on FlashInfer rather than force-fit a
  virtualization wrapper.
- Recording a "decided not to implement" outcome in the wins/ ledger
  with explicit reasoning is a feature, not a process tax — it
  prevents a future reader from re-running the same investigation and
  reaching a different conclusion silently.

## Cross-links

- Plan: `docs/plans/tilelang-integration.md` §"Tranche ledger" Tranche
  5 row.
- Sibling tranches (all 2026-04-27):
  - Tranche 4 (TC decode HD128 alias):
    `docs/experience/wins/2026-04-27-bench-guidellm-cuda-tilelang-tc-decode-hd128-pending-remote.md`
  - Tranche 2 (paged-prefill HD256 swap):
    `docs/experience/wins/2026-04-27-bench-guidellm-cuda-tilelang-prefill-hd256-pending-remote.md`
  - Tranche 3 (paged-decode HD256 swap):
    `docs/experience/wins/2026-04-27-bench-guidellm-cuda-tilelang-decode-hd256-pending-remote.md`
- Phase 0 (paged-prefill HD128 cutover): `docs/plans/tilelang-integration.md` §§1–8.
