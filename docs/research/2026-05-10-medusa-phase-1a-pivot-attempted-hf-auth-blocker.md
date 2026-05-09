---
title: Medusa Phase 1.A pivot ATTEMPTED post-PF8.3 KILL — blocked on HF auth (HF_TOKEN not set, lmsys-chat-1m gated)
date: 2026-05-10
type: research
status: medusa-phase-1a-pivot-blocked-on-hf-auth
---

# Medusa Phase 1.A pivot ATTEMPTED post-PF8.3 KILL — blocked on HF auth (HF_TOKEN not set, lmsys-chat-1m gated)

> Per `0cde63d` PF8.3 RUNTIME KILL + `2e1e73a` decision matrix → KILL
> branch step 3 = pivot to #28 Medusa Phase 1.A. Phase 1.A unblocked
> per `8735361` (arle data download CLI verified). Pivot attempted
> THIS tick → discovered HF auth blocker.

## §0 Direct evidence (raw verification THIS tick)

### CLI works
```
$ target/release/arle data download --help
Download one dataset file from Hugging Face
Usage: arle data download [OPTIONS] --repo <REPO> --file <FILE>
```

### Auth status
```bash
$ test -n "$HF_TOKEN" && echo "set" || echo "NOT set"
HF_TOKEN NOT set

$ ls ~/.cache/huggingface/token
# (file does not exist)
```

### Download attempts
```bash
$ target/release/arle data download --repo lmsys/lmsys-chat-1m --file data.jsonl
[download_dataset] fetching 'data.jsonl' from dataset 'lmsys/lmsys-chat-1m'
# (timeout cut off; would likely fail on 401/403 if continued)

$ target/release/arle data download --repo lmsys/lmsys-chat-1m --file data/train-00000-of-00006.parquet
[download_dataset] error: failed to download '...': request error:
  io: unexpected end of file: io: unexpected end of file
```

`unexpected end of file` likely means the download started but server
returned an error response (HF returns 401/403 for gated datasets
without auth). The hf-hub client interprets the truncated response
as EOF.

## §1 Medusa Phase 1.A blocker chain

Per `M_medusa-phase1a-dataset-directive.md`:
> "License: requires HF account agreement (not free-anon access)"

Required steps for `lmsys/lmsys-chat-1m`:
1. User opens https://huggingface.co/datasets/lmsys/lmsys-chat-1m
2. Logs in OR creates HF account
3. Clicks "Agree and access repository"
4. Generates HF token at https://huggingface.co/settings/tokens
5. Sets `export HF_TOKEN=hf_...` OR runs `huggingface-cli login`
6. THEN `arle data download` works

None of these can be done from this Claude session — all require
user authorization at huggingface.co.

## §2 Alternative non-gated datasets

Per `M_medusa-phase1a-dataset-directive.md` Option 2-4:

| Dataset | License | Suitability |
|---------|---------|-------------|
| `lmsys/lmsys-chat-1m` | gated | RECOMMENDED but blocked here |
| `openai/openai_humaneval` | open | smaller, code-focused |
| `tatsu-lab/alpaca` | open | classic instruction tuning |
| `openai/gsm8k` | open | math reasoning |
| Custom synthetic | n/a | bypasses HF entirely |

For Medusa head training, lmsys-chat-1m matches the paper's
"real chat distribution" intent. Alternatives may produce different
acceptance rates.

## §3 Pivot decision tree (for next agent / user)

```
PF8.3 KILLed (gemm code 2 runtime failure)
└── 2e1e73a → KILL branch → Pivot to #28 Medusa Phase 1.A
    ├── Phase 1.A: dataset download
    │   ├── REQUIRES HF auth (HF_TOKEN + license accept)
    │   ├── Option A: user sets up HF auth → run download
    │   ├── Option B: use non-gated alternative (humaneval/alpaca/gsm8k)
    │   └── Option C: defer Medusa, pursue alternative axis
    │       ├── #35 cap=8 prefill warmup (concrete code work, codex own)
    │       ├── #30 Hybrid W4A16/W4A8 dispatch Phase 2-3 (substrate work)
    │       └── PF8.3 kernel fix investigation (code 2 root cause)
    └── Phase 1.B (1 week training, blocked on Phase 1.A)
    └── Phase 2 (integration, blocked on 1.B)
    └── Phase 3 (bench, blocked on Phase 2)
```

## §4 Recommended user direction (one-liner choices)

1. **"set up HF auth and download"** → user does steps 1-5 above
2. **"use humaneval instead"** → smaller dataset, no auth needed
3. **"defer Medusa, fix PF8.3 kernel"** → codex investigates code 2
4. **"defer Medusa, pivot #35 cap=8 prefill warmup"** → 100-150 LOC
   codex-doable per M_warmup directive
5. **"session pause, summarize"** → many commits accumulated, take
   stock

## §5 Cross-references

- `0cde63d` PF8.3 RUNTIME KILL (immediate antecedent)
- `2e1e73a` post-PF8.3 next-axis decision matrix (KILL branch)
- `8735361` Medusa Phase 1.A pickup chain (assumed CLI-only blocker;
  HF auth discovered THIS tick)
- `M_medusa-phase1a-dataset-directive.md` (dataset selection, license
  warning)
- `M_medusa-required-path.md` (Phase 1-3 plan)
- `M_warmup-prefill-pass-directive.md` (Task #35 alternative pivot)
- Task #28 [pending] Medusa scaffold
- Task #44 [in_progress] PF8 chain (substrate landed, kernel fix
  needed for license)

## §6 Status

Medusa Phase 1.A pivot ATTEMPTED, blocked on HF auth setup. User
direction needed to choose:
- HF auth setup + lmsys-chat-1m download
- Non-gated alternative dataset
- Different axis pivot (#35 / #30 / PF8.3 kernel fix)

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(HF_TOKEN env check + ~/.cache/huggingface/token ls + arle data
download error output — all THIS tick).
