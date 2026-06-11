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
| Host RAM (T1) / SSD (T2) | hardcoded 4 GiB / 20 GiB constants | not machine-derived (**fixed C3**) |

## What Worked

**One neutral policy kernel in `infer-seam`** (the only crate both backends
depend on) — pure arithmetic, no I/O, no backend types, 10 unit tests:
`SlotBudget::{from_free,from_limit,fits_fixed,affordable}`,
`clamp_to_affordable`, `split_host_tiers`. Backends keep what is genuinely
backend-specific (memory probing, NCCL min-reduce, byte sizing, macOS paging
guards).

- **Phase B** (`15ec44cd`): the kernel + tests. Behavior-neutral (no consumers
  wired), no bench delta possible.
- **Phase B′** (`e5ad704f`): **Metal `plan_resource_budget` routed through the
  kernel** — `fits_fixed` guard, `from_limit` + `affordable()`,
  `clamp_to_affordable`. Byte-identical (`saturating_sub` == `-` after the
  guard; nested-floor identity for the token→page division; the legacy
  `.max(1)` a proven no-op after the `max_total_pages > 0` ensure);
  `MetalResourcePlan` fields and `describe()` output unchanged. Locked by a new
  unit test across fitting/clamped/tight regimes. Metal's probing
  (sysctl/vm_stat), dual reserve regimes, and paging guards stay Metal-side.
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
- **Phase C3** (`5353f17f`): **host RAM (T1) / SSD (T2) tiers machine-derived**
  instead of hardcoded. `kv_tier.rs` gains two probes — `/proc/meminfo`
  `MemAvailable` (dep-free) for RAM, POSIX `statvfs` (libc, unix) for free disk
  — feeding the same neutral `split_host_tiers` kernel. The hardcoded
  `DEFAULT_KV_TIER_BUDGET_BYTES` (4 GiB) / `DEFAULT_KV_SSD_BUDGET_BYTES`
  (20 GiB) are deleted; `default_t1_budget_bytes()` (executor constructor) and
  `default_t2_budget_bytes(root)` (the `--kv-ssd-path` attach in `loaded.rs`)
  replace them.
  - **Caps == the old constants** (`HostTierPolicy::default`): an ample host
    (the H20 pod, 256+ GiB RAM / TB disk) resolves to the exact 4 GiB / 20 GiB
    defaults — **byte-identical serving footprint**, no pod regression. A
    constrained host scales down (RAM ×0.25 with a 1 GiB floor, SSD ×0.5); a
    probe miss (Mac: no `/proc`) falls back to the cap, so it never
    over-shrinks. CLI overrides (`--kv-t1-budget-bytes` / `--kv-ssd-max-bytes`)
    are untouched — they short-circuit the probe.

With C3 + B′ the audit's last fragmented surfaces converge: **VRAM (DSv4 +
Qwen3.5/3.6), host RAM/SSD, and the Metal planner now all flow through the one
neutral infer-seam kernel.** No backend hand-rolls the budget/clamp policy.

## Evidence

- `infer-seam` unit tests: **10/10 pass** (kernel arithmetic, DSv4 fold
  byte-identity, tier caps/floors/fallback).
- `infer-cuda` `kv_tier` tests: **7/7 pass**, including C3's new probe tests
  (`statvfs` on a real dir reports Some + nonexistent path → None; T1 budget in
  `[1 GiB, 4 GiB]`; T2 budget ≤ 20 GiB cap).
- `infer-metal` `resource` tests: **10/10 pass**, including B′'s new
  `kv_budget_clamp_matches_legacy_inline_arithmetic` (kernel route reproduces
  the former inline `limit−fixed` / `budget/per_token/page` / `min().max(1)`
  byte-for-byte across fits/clamps/tight).
- Mac CUDA-Rust typecheck (`cuda,no-cuda`) + `clippy -D warnings`: **clean** for
  C1, C2, and C3 (both `cuda,no-cuda` and `no-cuda` paths); `clippy -D warnings`
  clean for B′ (`metal`).
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
