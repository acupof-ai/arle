# Unified resource budget — VRAM/RAM/SSD convergence (post-#60 systematic fix)

## Context

#60 fixed one phantom allocation (per-slot×per-layer MoE decode scratch, one
live copy). The systematic follow-up audit
([plan](../../plans/unified-resource-budget.md)) found no second phantom — every
other per-slot alloc is a legitimate stateful cache — but exposed the real
systematic gap: **budget logic was fragmented and partly absent**.

| Surface | Before | Risk |
|---------|--------|------|
| DSv4 CUDA | bottom-up clamp `free×0.9 − shared`, NCCL min-reduce (dsv4.rs) | ok |
| Qwen3.5/3.6 CUDA | **NO clamp** — requested num_slots admitted as-is | **OOM at large max_seq_len (the #60 failure class)** |
| Metal | top-down `plan_resource_budget` (its own planner) | ok, but not shared |
| Host RAM (T1) / SSD (T2) | hardcoded 4 GiB / 20 GiB constants | not machine-derived |

## What Worked

**One neutral policy kernel in `infer-seam`** (the only crate both backends
depend on) — pure arithmetic, no I/O, no backend types, 10 unit tests:
`SlotBudget::{from_free,from_limit,fits_fixed,affordable}`,
`clamp_to_affordable`, `split_host_tiers`. Backends keep what is genuinely
backend-specific (memory probing, NCCL min-reduce, byte sizing, macOS paging
guards).

- **Phase B** (`15ec44cd`): the kernel + tests. Behavior-neutral (no consumers
  wired), no bench delta possible.
- **Phase C1** (`a23336eb`): DSv4 `kv_budget_num_slots` routed through the
  kernel. **Byte-identical** — `from_free` reproduces `floor(free×0.9)−Σfixed`
  exactly (the two saturating_subs fold into one, proven by
  `resource::tests::saturating_sub_fold_is_identical`), `affordable()` ==
  `checked_div(per_slot)`, clamp == `requested>affordable`. Budget log unchanged.
- **Phase C2** (this entry): **Qwen3.5/3.6 CUDA gains the clamp it lacked** —
  `Qwen35Model::kv_budget_num_slots` computes per-slot bytes (K+V contiguous
  caches + gated-delta recurrent state + conv rings, mirroring `new_slot_state`
  exactly) and clamps through the same kernel + NCCL min-reduce. Wired in the
  executor constructor before slot allocation, post-weights `mem_get_info`.
  - **Strictly safer, no regression in the fitting regime:** when
    `requested ≤ affordable` the clamp is a no-op (byte-identical to before).
    Behavior changes ONLY when the request would previously have OOM'd — old =
    crash at slot alloc, new = clamp + serve. Dense Qwen3 is out of scope: it
    uses a `PagedKVPool` sized by `total_pages` (global pool), not
    num_slots×max_seq_len, so it has no slot-multiplied OOM.

## Evidence

- `infer-seam` unit tests: **10/10 pass** (kernel arithmetic, DSv4 fold
  byte-identity, tier caps/floors/fallback).
- Mac CUDA-Rust typecheck (`cuda,no-cuda`) + `clippy -D warnings`: **clean** for
  C1 and C2.
- DSv4 C1 byte-identity: the #60 pod 8-slot run already exercises this exact
  path (`shared MoE decode 114MB` came through `kv_budget_num_slots`); the
  refactor preserves the log line and arithmetic.

**GPU integration gate — pending-remote.** Qwen3.5/3.6 CUDA clamp triggering at
an over-large `--num-slots × --max-seq-len` (boots clamped instead of OOM) needs
a CUDA GPU; the H20 pod is the DSv4 lane and local is Mac. Tracked under the
unified-resource-budget plan; the local evidence (kernel unit tests + fitting-
regime byte-identity + new_slot_state mirroring by inspection) bounds the risk.

## Rule

When two backends solve the same resource problem (how many KV slots fit in
device memory), the *policy* — `budget = mem×fraction − fixed; affordable =
budget/per_unit; planned = min(requested, affordable)` — is identical and
belongs in one neutral, unit-tested kernel; only probing, cross-rank reduce, and
byte sizing are backend-specific. A backend with NO clamp (Qwen3.5/3.6) is not
"simpler" — it is the #60 OOM waiting at a larger shape.
