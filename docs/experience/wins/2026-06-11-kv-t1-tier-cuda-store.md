# KV T1 Tier — CUDA Host Store, Default-On (#82 D) — pending-remote

## Goal

Wire the dense-Qwen3 CUDA arm to the T1 tier seam
([host tranche](2026-06-11-kv-t1-tier-host-tranche.md)): demoted prefix
pages land in a host store and promote back on the next prefix hit.
**Default-on** (ckl 2026-06-11): tier capacity defaults to 4 GiB of host
RAM; T2 SSD stays opt-in.

## Hypothesis

`TokenKVPool::copy_pages_to_host` / `copy_pages_from_host` (monolith-era
plumbing that survived in `cuda-kernels`, full page image incl.
scales/norms, synchronous) are sufficient demote/promote primitives; a
capacity-capped `BTreeMap<u64, Vec<u8>>` host store keyed by engine tier
keys completes the executor side without touching the forward path.

## Params

- `CudaKvTierStore` in `infer-cuda/src/executor.rs`;
  `DEFAULT_KV_TIER_BUDGET_BYTES = 4 GiB` → capacity =
  budget / `storage_bytes_per_page()` (BF16 Qwen3-dense ≈ 2.4 MB/page →
  ≈1700 pages ≈ 27K tokens of demoted prefix).
- Hooks on the Qwen arm only; Qwen3.5-hybrid / DSv4 arms report capacity 0
  (their KV is not page-addressable — prefix cache disabled; DSv4 support
  is a hard requirement per ckl, routed through #85's sidecar snapshot
  protocol).
- v1 stores pageable `Vec<u8>` (what `clone_dtoh` returns); pinned-arena
  upgrade (kv-native-sys host arena + side stream) is the perf follow-up,
  licensed by the pod A/B below.
- CLI opt-out (`--kv-t1-budget-bytes 0`) wiring pending — the serve-config
  files are mid-flight in a concurrent session; tracked in #82.

## Env

- Local Apple Silicon: `CUDARC_CUDA_VERSION=12060 cargo check -p infer-api
  --release --no-default-features --features cuda,no-cuda --lib` clean;
  infer-cuda host tests 53 passed; engine semantics covered by the host
  tranche's mock tests.

## Results

**pending-remote** — CUDA execution is not possible on this machine. Pod /
single-GPU lane owes:

1. Correctness: needle ladder ×3 same-config + same-config-twice envelope
   on a workload that forces demote→promote (small `--total-pages`,
   repeated long shared prefix), per the #82 exit criteria.
2. Perf license-or-kill: matched same-binary A/B (tier on/off) on
   multi-turn long-prompt re-attach; metric = TTFT of re-attached turns vs
   full re-prefill. H2D promote of a 2.4 MB page is ~0.1–0.25 ms pageable;
   re-prefill of 16 tokens is also sub-ms on dense Qwen3 — the win case is
   LONG shared prefixes (many pages per hit), so the A/B must use the
   multi-turn agent shape, not a smoke shape.

## Problems

Default-on ships ahead of the pod gate by explicit owner decision
(2026-06-11). Risk surface: dense-Qwen3 CUDA serving only (DSv4/hybrid
report capacity 0); the demote/promote primitives are the same
`copy_pages_*` functions the monolith used. If the pod gate fails, flip
`DEFAULT_KV_TIER_BUDGET_BYTES` to 0 — one-line revert.

## Learnings

Adopt-official-first applies to our own tree too: the page host-copy
plumbing already existed in `cuda-kernels` with the right sync semantics —
the whole CUDA tranche is a store struct + dispatch, no new kernels.
