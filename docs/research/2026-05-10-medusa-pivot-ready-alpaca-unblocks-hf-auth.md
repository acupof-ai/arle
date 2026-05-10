---
title: Medusa Phase 1.A pivot ready — Alpaca dataset unblocks HF auth requirement (Path A from 2026-05-08 inventory)
date: 2026-05-10
type: research
status: superseded-for-qwen35-by-recurrent-rollback-blocker
related_tasks: [#28 (M_medusa scaffold, blocked on HF auth per ad14636)]
---

# Medusa Phase 1.A pivot ready — Alpaca dataset unblocks HF auth requirement

> **2026-05-10 later update**: Alpaca still resolves the HF-auth data
> blocker, but the active Qwen3.5 direction is blocked earlier by
> recurrent-state rollback. This document is now data-readiness context,
> not a pickup-ready Medusa implementation plan.

> **Purpose**: cross-link the 2026-05-08 Phase 1.A data inventory (which
> already documented public alternatives) to the 2026-05-10 `ad14636`
> HF auth blocker (which only blocks the gated `lmsys-chat-1m` dataset).
> Per 2026-05-08 Path A: `tatsu-lab/alpaca` is the recommended public
> alternative — NO HF auth setup needed. This makes Task #28 Medusa
> Phase 1.A pickup-ready for the case where PF8.5 license decision
> KILLs PF8 chain at bench v11.

## §1 The blocker (2026-05-10 ad14636)

`lmsys-chat-1m` is a gated HuggingFace dataset requiring:
- HF account
- `HF_TOKEN` environment variable
- License acceptance click-through on the dataset's HF page

User has not set up HF auth. This blocked Medusa Phase 1.A retry on
2026-05-10.

## §2 The resolution (already documented 2026-05-08)

`docs/research/2026-05-08-medusa-phase1a-data-inventory.md` §Path A
enumerated public alternatives explicitly:

> Use `crates/train/src/hub_dataset.rs` (existing infrastructure) to load:
> - `tatsu-lab/alpaca` (52k samples, ~10M tokens)
> - `vicuna_conversations` (70k samples, ~100M tokens)
> - `Qwen3 instruct datasets` (public on HF)
>
> Implementation:
> - `scripts/medusa_training_data.py`: HF dataset loader → Medusa format
>   (~50 LOC)
> - Calls existing `hub_dataset.rs` Rust trainer-side adapter
> - Output: safetensors with `(input_ids, target_ids, hidden_state_indices)`
>
> LOC: 50-80 LOC Python script + zero new Rust (reuses hub_dataset.rs)
> Risk: LOW (infrastructure exists)

`tatsu-lab/alpaca` is the simplest path:
- **Public** (no HF auth, no license click-through, no token)
- 52k samples, ~10M tokens — enough for first-pass Medusa-1 training
  per Cai et al. 2024 §4.1 paper
- Loadable via `datasets.load_dataset("tatsu-lab/alpaca")` with default
  Python `datasets` library, no special auth config

## §3 Why this wasn't the chosen path on 2026-05-10

Best guess (per ad14636 + scarce session memory): the 2026-05-10
session may have defaulted to lmsys-chat-1m without re-checking the
inventory. Per skill v1.12.0 #29 (default broken fixtures) +
candidate #36 (grep for variants before designing from scratch),
this is exactly the discovery pattern: BEFORE retrying a previously-
blocked path, audit the existing inventory for alternatives.

## §4 Codex pickup recipe (if PF8 KILLs at bench v11 → Task #28 pivot)

Per 2026-05-08 inventory Path A:

```bash
# 1. Verify existing infrastructure
grep -n "hub_dataset" crates/train/src/lib.rs
test -f crates/train/src/hub_dataset.rs && echo "hub_dataset.rs exists"

# 2. Write the loader (~50-80 LOC Python)
cat > scripts/medusa_training_data.py <<'EOF'
#!/usr/bin/env python3
"""Medusa Phase 1.A data prep — Alpaca → Medusa format.

Uses tatsu-lab/alpaca (public, no HF auth) to generate
(input_ids, target_ids, hidden_state_indices) safetensors for Medusa-1
head training.

No HF_TOKEN required. Resolves 2026-05-10 ad14636 blocker via
2026-05-08 inventory Path A recommendation.
"""
from datasets import load_dataset
# ... loader logic ...
EOF

# 3. Wire to existing trainer-side hub_dataset.rs adapter
# 4. Smoke-test: load 100 samples, verify safetensors output shape
# 5. Full data prep: 52k samples → ~10M tokens
# 6. Hand off to existing Medusa-1 training scaffold (`afdddec` plan)
```

LOC: ~50-80 Python (script) + zero Rust (reuses hub_dataset.rs)
Time: ~30 min codex (script) + ~30 min smoke test + ~1 hour data prep

## §5 Why this matters NOW (parallel preparation)

Per cron-loop directive "准备下 round 假设" (prepare next round
hypothesis): Task #28 Medusa is the documented pivot if PF8.5 license
decision KILLs PF8 chain at bench v11 (per 2e1e73a "both PF8 branches
converge on #28 Medusa P0").

If user runs `bash scripts/pf85_bench_v11_user.sh` (commit `ead46dc`)
and PF8 KILLs:
- Codex needs Task #28 ready to pick up immediately
- Without this cross-link doc, codex would re-discover the lmsys-chat-1m
  blocker, then re-search the inventory, wasting 30+ min
- WITH this doc, codex follows Path A directly → 30 min loader script
  → smoke test → data prep → training scaffold

This doc shaves ~30 min off the pivot critical path.

## §6 If user CAN set HF auth (keeping for completeness)

If user has time/willingness for HF auth setup, lmsys-chat-1m gives:
- 1M conversations vs Alpaca's 52k samples (~20× more data)
- Potentially better quality (real LMSYS arena conversations vs
  GPT-3.5-generated Alpaca instructions)
- Closer to Vicuna baseline distribution (Cai et al. 2024 §4.1)

But Path A (Alpaca) is sufficient for first-pass Medusa-1 per the
paper's stated convergence: "Loss converges in ~3 days on single A100
GPU" using ~100k token sequences (Alpaca's 10M tokens far exceeds).

## §7 Cross-references

- `2026-05-08-medusa-phase1a-data-inventory.md` (Path A source)
- `2026-05-08-medusa-phase1a-hf-download-blocker.md` (prior blocker session)
- `2026-05-08-medusa-phase1a-wget-workaround-success.md` (2026-05-08
  succeeded via wget; mismatch with 2026-05-10 ad14636 blocker)
- `ad14636` 2026-05-10 HF auth blocker (this resolution)
- `crates/train/src/hub_dataset.rs` (existing infra)
- `afdddec` Medusa-1 plan (training scaffold spec)
- Task #28 M_medusa scaffold
- `ead46dc` `pf85_bench_v11_user.sh` (the trigger event for Task #28
  pivot if PF8 KILLs)
- 2e1e73a "both PF8 branches converge on #28 Medusa P0"

## §8 Status

**Codex-pickup-ready** for the Task #28 pivot scenario. Resolves
2026-05-10 ad14636 HF auth blocker via 2026-05-08 inventory Path A.
~30 min critical-path savings if PF8 KILLs at bench v11.

NOT immediately actionable — gated on bench v11 outcome (LICENSE →
codex Task #47 H1' refactor / KILL → codex Task #28 Medusa via this
recipe).
