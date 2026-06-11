# Unified resource budget — VRAM/RAM/SSD convergence (post-#60 systematic fix)

## Context

#60 fixed one phantom allocation (per-slot×per-layer MoE decode scratch, one
live copy). The systematic follow-up audit
([plan](../../plans/unified-resource-budget.md)) found no second phantom — every
other per-slot alloc is a legitimate stateful cache — but exposed the real
systematic gap: **budget logic was fragmented and partly absent**.

| Surface | Before | Risk |
|---------|--------|------|
| DSv4 CUDA | bottom-up clamp `free×0.9 − shared`, NCCL min-reduce (dsv4.rs) | ok; missing reject-below-fixed guard (**fixed C4**) |
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
- **Phase C4** (`75530149`): **reject-below-fixed parity** — the last audit gap.
  Both CUDA `from_free` paths (DSv4 + Qwen3.5/3.6) dropped the per-rank
  `affordable().max(1)`, NCCL-min-reduce the **real** affordable count, then bail
  uniformly (`anyhow::ensure!(affordable > 0, …)`) when the reduced count is 0 —
  matching Metal's `fits_fixed` two-stage guard. Before C4 the CUDA paths
  admitted one slot when post-weights free VRAM couldn't hold even one, then
  OOM'd at arena/slot allocation; now they fail closed with a clear
  `affords 0 slots at max_seq_len N` message.
  - **Lockstep-safe:** every rank branches on the same *reduced* scalar, so the
    TP group rejects or admits as one — no half-OOM where one rank had room and
    another did not (the rule from `feedback_spec_decode_gate` / lockstep-state:
    a fail-closed bail must branch on the post-reduce uniform value).
  - **Byte-identical in every working config:** once the real affordable is ≥ 1
    on all ranks the former `max(1)` was a no-op, so the reduced count and clamp
    are unchanged. Behavior changes ONLY in the regime that previously OOM'd —
    a strict improvement (clean error vs device OOM), no regression in the
    fitting regime.

With C3 + B′ + C4 the audit's last fragmented surfaces converge: **VRAM (DSv4 +
Qwen3.5/3.6), host RAM/SSD, and the Metal planner now all flow through the one
neutral infer-seam kernel — for both the budget arithmetic *and* the reject
policy.** No backend hand-rolls the budget/clamp/reject policy.

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
  C1, C2, C3, and C4 (both `cuda,no-cuda` and `no-cuda` paths); `clippy -D
  warnings` clean for B′ (`metal`).
- C4 guard arithmetic: covered by the existing `infer-seam`
  `fits_fixed` unit test (`affordable == 0` ⇒ reject); the CUDA edit only drops
  a proven-no-op `max(1)` and adds the post-reduce `ensure!`.
- DSv4 C1 byte-identity: directly pod-validated against a HEAD-built binary —
  the new budget log reproduces the pre-convergence reference exactly (see the
  GPU-validation section). Not the old-binary #60 run; a fresh rebuild from this
  HEAD on the 8×H20 DSv4 lane.

**GPU validation — DSv4 lane DONE on the H20 pod (8×H20, TP=8, DeepSeek-V4-Flash,
release-fast binary built from this HEAD).** Deployed the 11 convergence files to
the pod build tree (`tn push` over the hostPath volume; all five critical files
hash byte-identical to HEAD; new binary carries the C4 reject strings — DSv4 ×2,
Qwen3.5 ×1 — and retains DeepGEMM-native, 399 symbols).

- **C1 byte-identity — CONFIRMED.** New binary's budget log is identical to the
  pre-convergence reference: `DSv4 KV budget: free 57035MB, per_slot 924MB
  (arena×2 784MB + rotated 21MB + state caches 118MB), shared DSA 36MB, shared
  MoE decode 0MB` (only the source line moved `1123→1129`, since C4 added the
  guard above — log *content* identical). 8-slot boot clean, no OOM; completion
  probe returned correct, coherent output (`" Paris.\nThe capital of France is
  Paris."`), so the kernel refactor did not perturb decode.
- **C4 reject — CONFIRMED (the one genuine behavior change).** Forced the
  previously-dormant regime with `INFER_DSV4_MAX_SEQ_LEN=2000000`: per_slot
  computed to 112802MB ≫ budget 48839MB → affordable 0. **All 8 TP ranks logged
  the identical fail-closed message** (`grep -c "affords 0 slots" = 8`) — the
  lockstep-safe uniform bail working exactly as designed (every rank branches on
  the same post-NCCL-min-reduce scalar; no half-OOM, no deadlock). Message:
  `DSv4 KV budget rejected startup: post-weights free VRAM affords 0 slots at
  max_seq_len 2000000 (per_slot ~112802MB + shared DSA 2492MB + shared MoE decode
  0MB exceed 0.9 of free). Lower --max-seq-len or free VRAM.` **Zero OOM / CUDA
  error / panic / core-dump** — the precise improvement over the old `max(1)`
  path (admit 1 slot → device OOM at arena alloc). Pod left clean (all 8 GPUs
  back to 0 MiB).

**Still inferred (not pod-exercised).** C2 + C4 on the **Qwen3.5/3.6 CUDA** path
share the now-validated kernel + NCCL-min-reduce + bail shape with DSv4, but the
DSv4 lane cannot exercise a Qwen serve; their per-slot mirroring of
`new_slot_state` is by inspection. C3 host RAM/SSD derivation is unit-tested and
resolves to the caps on the ample pod (byte-identical). A Qwen3.5 CUDA serve at
an over-large shape is the one remaining round-trip — risk is bounded by the
shared validated kernel.

## Rule

When two backends solve the same resource problem (how many KV slots fit in
device memory), the *policy* — `reject if mem < fixed; budget = mem×fraction −
fixed; affordable = budget/per_unit; planned = min(requested, affordable)` — is
identical and belongs in one neutral, unit-tested kernel; only probing,
cross-rank reduce, and byte sizing are backend-specific. The **reject** half of
that policy is as much a part of it as the clamp half: a backend that clamps but
admits one slot it can't fit (the CUDA `max(1)`, fixed C4) still OOMs — just one
shape later. A backend with NO clamp (Qwen3.5/3.6) is not "simpler" — it is the
#60 OOM waiting at a larger shape. For TP, the reject must branch on the
post-reduce uniform scalar, or the group half-OOMs.
