# Metal KV tier convergence — runtime demote/promote smoke PASS

## Context

The Metal disk tier (`MetalSsdTier` on the sharded-file substrate S3) was
converged onto the shared `KvTierStore` (moved from `infer-cuda` to
`kv-native-sys`, mmap substrate S2), and substrate S3 was deleted repo-wide
(plan). Functional gates
passed (cli 209 tests, kv-native-sys 24); this is the runtime demote/promote
smoke that was outstanding.

## What Worked (Apple Silicon, `arle serve --backend metal --kv-disk`)

Full demote→promote cycle on the shared store, `/v1/stats` verbatim:

```
kv_tier: demoted_pages 1490, promoted_pages 20, promote_failures 0, resident_blocks 345
kv_system: disk_pages 345, reuse_hit_disk 20, reuse_miss 10, demote_mset_count 369,
           promote_mget_count 1, prefix_match_full_blocks 20, fallback_recompute 0
```

- **Promote correctness**: a needle demoted to disk then re-sent recalled
  `738291` exactly (a corrupt promote gives wrong output) — the shared store's
  read path restores Metal KV bit-correctly.
- **Tier engages**: demote fired (1490 pages → disk), re-send promoted (20
  pages, one `promote_mget`, `reuse_hit_disk 20`), zero demote/promote failures.
- No panic, serve alive through the sequence.

Model: `Qwen3.5-0.8B-MLX-4bit` (the CLAUDE.md-sanctioned opt-out — the canonical
Qwen3.6-35B's 19 GB weights exceeded available memory on a swap-saturated box;
the convergence under test is backend-neutral KV plumbing
`kv_ssd.rs → kv_native_sys::KvTierStore → infer-core planner`, so the small
model exercises the identical code path).

## Sharp edge found (pre-existing, not the convergence)

`--kv-disk` ALONE on a disk with < 50 GiB free silently fail-closes the engine
(`"no usable page-addressable KV tier store"`): `nvme_l3_budget`'s
`reserve = max(50 GiB, 0.1×total)` saturates a small-free-disk budget to 0.
Workaround: pass `--kv-disk-limit`. Filed as an issue.

## Rule

A backend-neutral substrate convergence is runtime-verified by exercising the
shared code path (demote→promote→needle-exact + nonzero tier counters), not just
compile+unit gates — and the opt-out small model is a valid carrier when it hits
the identical plumbing and the canonical model can't load.
