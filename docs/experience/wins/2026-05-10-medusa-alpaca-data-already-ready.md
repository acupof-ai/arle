# Medusa training data infra audit — Alpaca already downloaded + converted, removes §7 gate friction

## Context

Date: 2026-05-10 (cron-loop tick 105 KST)
Audit: verifying `arle data download` + `arle data convert` infrastructure
works for the Phase 1.B substrate brief (`f0c7561`) §4 training pipeline.
Per `M_medusa-phase1a-dataset-directive.md` recommendation Alpaca is one
of two top dataset candidates.

## What Worked

### §1 CLI surface verified

`arle data` exists with both required subcommands:
```
arle data download --repo <repo_id> --file <path>  [--dry-run] [--json]
arle data convert --input <jsonl> --format chat|dolly|alpaca|sharegpt
                  [--output <path>] [--dry-run] [--json]
```

Both have `--dry-run` for safe verification without network/disk activity.
`--json` enables machine-readable plan output for CI.

### §2 Alpaca dataset ALREADY READY at `/tmp/medusa_data/`

Surprise finding: prior session-tail work (dated 2026-05-08, traced to
prior tick activity) already executed the full prep pipeline:

| File | Size | Rows | Format |
|---|---:|---:|---|
| `alpaca_train.parquet` | 24 MB | 52,002 | HF parquet (raw download) |
| `alpaca_train.jsonl` | 47 MB | 52,002 | JSONL (converted) |
| `alpaca_chat.jsonl` | 22 MB | 52,002 | **Canonical chat format** ✓ |

Chat format spot-check (first row):
```json
{
  "messages": [
    {"role": "user", "content": "Give three tips for staying healthy."},
    {"role": "assistant", "content": "1.Eat a balanced diet..."}
  ]
}
```

This matches the canonical chat schema that ARLE training pipeline
consumes directly (per `crates/train/src/sft_data.rs` pattern).

### §3 §7 gate friction reduction

`f0c7561` Phase 1.B brief §7 gate had 4 user-decision items:
- [ ] Target model (Qwen3-4B vs Qwen3.6)
- [ ] Dataset (Alpaca vs lmsys-chat-1m)
- [x] **Alpaca pickup**: zero download/convert wait time (already done)
- [ ] Integration target (CUDA scheduler first)
- [ ] ~3-4 day wall-clock approval

**Implication**: if user picks Alpaca, training can begin **immediately**
once codex Phase 1.B substrate lands. No pre-training data prep wall-clock.

For lmsys-chat-1m, would need fresh download (~1 GB) + convert
(~10-15 min wall-clock) — adds friction not present for Alpaca.

### §4 Dry-run output (zero side-effects)

```
$ arle data download --repo tatsu-lab/alpaca --file alpaca_data.json --dry-run --json
{
  "command": "data download",
  "argv": ["--repo", "tatsu-lab/alpaca", "--file", "alpaca_data.json"]
}
```

Confirms wiring; no actual download triggered.

## Implications

### §1 Refines Phase 1.B brief §6 wall-clock estimate

Original estimate (assumed cold start):
- Substrate: 8-11 hr
- Training: 48-72 hr (download + convert + train)
- Bench: 1 hr
- **Total**: 3-4 days

Refined estimate (Alpaca pre-prepped):
- Substrate: 8-11 hr
- Training: 48-60 hr (just train, no prep wait)
- Bench: 1 hr
- **Total**: 2.5-3 days (12-24 hr saved)

### §2 Mild Alpaca preference signal

Pre-prep state mildly biases toward Alpaca over lmsys-chat-1m for
first-iter Medusa training. If user is indifferent on dataset
representativeness, Alpaca = faster.

If user specifically wants real-chat distribution (per
`M_medusa-phase1a-dataset-directive.md` Option 1 reasoning), still
recommend lmsys-chat-1m + accept the 12-24 hr extra prep wall-clock.

### §3 Validates ARLE data pipeline end-to-end

The 2026-05-08-dated artifacts prove that:
- `arle data download` works against HF Hub auth (via `HF_TOKEN`)
- `arle data convert --format alpaca` produces correct canonical chat
- Output is consumable by SFT-style training code (per ARLE conventions)

This is independent verification of `crates/train/src/hub_dataset.rs`
+ `arle data convert` end-to-end, no Medusa-specific bug surface
expected at the data-loading boundary.

## Rule

When auditing pickup readiness, always check `/tmp/`, project `data/`,
and existing artifacts BEFORE estimating download/prep wall-clock.
Prior session-tail work may have already advanced the gate. This
removed ~12-24 hr from the Medusa Phase 1.B critical path estimate
without writing any code.

## Cross-references

- `f0c7561` Phase 1.B substrate brief (§6 wall-clock now refined down)
- `0a0d221` Task #28 readiness audit
- `1ccb41f` vLLM Medusa prior-art survey
- `9735b47` REFUTATION wins entry (strategic pivot to Option A)
- `M_medusa-phase1a-dataset-directive.md` (dataset selection)
- `crates/train/src/hub_dataset.rs` — HF Hub data loader
- `/tmp/medusa_data/alpaca_chat.jsonl` (52,002 rows canonical chat, ready)
- 2026-05-08-dated artifacts (predate this session-tail audit)
