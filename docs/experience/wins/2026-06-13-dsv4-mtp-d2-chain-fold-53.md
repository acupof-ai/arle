# DSv4 MTP d2 chain-fold — 44.5 → 53.3 tok/s B=1 (+20%), spec-depth design fixed

## Context

With the no-spec base at 44.5 (FP8 decode lane + automatic NUMA), the next
lever is speculative decode. The campaign's best was "chain-fold d2 = 38.04"
on the old 32.52 base (+17%); re-running on the new base — and fixing the
config design that buried it.

## The design fix (`1568f703`)

`spec_depth()` clamped the explicit `--mtp-draft-tokens N` to 1 unless
`ARLE_DSV4_MTP_UNCLAMP=1` — a CLI flag begging permission from an env var,
backwards. And it returned the request *unclamped*, a latent spec-ring
overflow for large N. Now `--mtp-draft-tokens N` is the single source of
truth, clamped to `[1, MAX_SPEC_DRAFT_DEPTH=8]` (safe-by-construction). Env
vars survive only as A/B opt-outs + diagnostics, never required opt-ins.

`commit_fold` (`5f48f90f`): flipped to **default ON** — it was opt-in
"until its own needle + perf gate licenses the flip"; this result is that
license, so `--spec-type mtp --mtp-draft-tokens 2` now yields the optimal
config with zero env hacks.

## What Worked

d2 **chain** decode (top-1, no tree — the campaign deleted tree width as
no-benefit): draft d0/d1 off the MTP head → one frozen batched-tree-attn
verify forward → accept the longest matching prefix + the free bonus token
→ **commit-fold** (re-ingest the accepted prefix from persisted verify rows,
no second forward). Drives 2-3 accepted tokens per verify forward at good
acceptance.

## Results (same worktree binary, same session)

| arm | B=1 tok/s | notes |
|---|---|---|
| no-spec base | 44.5 | FP8 lane + NUMA |
| d2 chain-fold (`COMMIT_FOLD=1` env) | 53.37 / 53.30 / 53.32 | ×3, σ≈0.04 |
| **d2 chain-fold (default-on, no env)** | **52.79 / 52.97 / 52.93** | ×3 — the flip confirmed |

**+18-20% over the no-spec base** (the with-env vs default-on delta is boot
noise — identical code path). The default-on run carries **no env at all**:
`--spec-type mtp --mtp-draft-tokens 2` is the whole config. Needle ×3 (the spec correctness gate, where
draft errors would corrupt output): 512 exact-DET, 6000 exact, 2048
partial-stable — identical to the locked envelope, no garbage, no miss. The
verify is teacher-forced so a wrong draft is rejected, never emitted; the
needle PASS confirms the accept/reject + commit-fold rollback are correct.

## Rule

- **A CLI flag must not need an env var's permission to take effect.** The
  flag is the explicit interface; env vars are opt-outs/diagnostics. Inverting
  that (the UNCLAMP gate) silently ignored the user's `--mtp-draft-tokens` and
  buried a +20% lever behind an undocumented env.
- **Re-measure validated levers on the new base.** d2 was +17% on the old
  32.52 base; on the FP8+NUMA base it's +20% (53.3) — the verify-cost /
  acceptance ratio shifted with the faster forward, so the old number was not
  the ceiling.
