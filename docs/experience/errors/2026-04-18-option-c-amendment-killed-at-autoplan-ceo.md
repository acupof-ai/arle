# Option C cross-process-shape amendment — killed at autoplan CEO phase

## Context

After reading the user's vision doc for a "production-native agent RL platform" (env × trajectory × verifier moat, multi-tenant tool-call → policy improvement), I drafted `docs/plans/2026-04-18-option-c-amendment.md` proposing to keep Phase 6's single-process implementation but pre-shape three interfaces (`TrajectoryRecord` with `schema_version: u32`, exportable `KvHandleRef` + `KvHandleProvider` trait, `#[async_trait] Verifier` with RPC-shaped request/verdict) so they were "cross-process compatible from day 1." Estimate: 2.5 person-days, zero milestone impact, absorbed into M2–M4.

Ran `/autoplan` Phase 1 CEO review with both outside voices (Claude subagent + Codex outside-voice).

## Root Cause

The amendment was the wrong shape at the wrong moment. Both voices converged independently:

- **Premise was asserted not argued.** "Multi-tenant by definition" was treated as axiomatic. The then-current reading of `agent-rl-self-evolving.md` still leaned on same-process Rust as the structural moat, so the amendment quietly downgraded the moat without justifying why.
- **`KvHandleRef` was designed for nobody.** `kv_tier` is a skeleton with zero production callers; current cache identity is private metadata + optional fingerprint + non-stable BlockId (`prefix_cache.rs:82`, `kv_tier.rs:48`). Freezing handle shape now would constrain the real RadixCache work that hasn't happened yet.
- **`#[async_trait] Verifier` was a bad cognition-phase trade.** Current `verifier.rs:8` is a sync borrowed-slice API. Forcing tokio for in-process math impl just to accommodate a hypothetical remote verifier is overhead in the wrong direction.
- **`schema_version: u32` was premature theater.** No second deployable producer/consumer pair exists.
- **2.5 person-days was founder math.** Realistic surface (per Codex enumeration): rewrite verifier from sync borrowed to owned RPC, lift trajectory from internal struct to scheduler output, design KV handle on a non-stable substrate. 5–8 days + coordination tax minimum.
- **The amendment picked the least decisive middle path.** Coherent strategies were either (a) defend the unified single-node moat until it dies, or (b) explicitly reframe agent-infer as a rollout/train substrate to veRL/ProRL. The middle path defended neither.

## Fix

Killed the amendment doc. Phase 6 ships as currently locked in `docs/projects/agent-rl-self-evolving.md`: unified Rust train/infer authority, LoRA-only, GRPO, single-node CUDA-first, with process shape treated as an implementation choice rather than the moat itself. Revisit the cross-process question only if M3's 6h closed-loop run exposes a real bottleneck — at which point the actual bottleneck shape will tell us what interface to design, instead of guessing now.

## Rule

**Don't pre-shape interfaces for hypothetical future consumers.** Trait contracts should be trailing indicators of real callers, not leading indicators of imagined ones. Specifically:
1. If the substrate is a skeleton with zero callers (kv_tier today), do not freeze its public handle shape.
2. If the existing surface is sync and owned-data, do not introduce `#[async_trait]` to accommodate a remote variant that does not yet exist — add a transport adapter when the remote variant ships.
3. `schema_version` belongs on schemas that already have a second producer or consumer; before that, it is theater.
4. When a doc proposes to "stay the course but change the shape," check whether it actually defends the original moat or quietly weakens it. If weakened, the doc owes a replacement moat.
5. Autoplan dual-voice convergence at Phase 1 = stop. Don't burn Eng/DX cycles on a doc both voices say should die.
