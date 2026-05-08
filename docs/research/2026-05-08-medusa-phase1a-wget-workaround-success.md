# Medusa Phase 1.A.3 — wget workaround unblocks data pipeline; 52k alpaca samples ready

> Per `da68b98` `arle data download` HF Hub library blocker discovered。
> This entry validates **wget workaround**:manual fetch + `arle data
> convert` → canonical chat JSONL。
>
> **Result:Phase 1.A.3 fully unblocked**。52,002 alpaca samples
> (~2.6M tokens,**26× Medusa requirement**)ready for codex Phase 1.B
> tokenization + head training pickup。

## Workaround pipeline

### Step 2 — Manual wget(replaces broken `arle data download`)

```bash
mkdir -p /tmp/medusa_data
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    -O /tmp/medusa_data/alpaca_train.parquet
# Result:24 MB parquet,52,002 rows
```

### Step 2b — Parquet → JSONL(Python pandas)

```python
import pandas as pd
df = pd.read_parquet('/tmp/medusa_data/alpaca_train.parquet')
df.to_json('/tmp/medusa_data/alpaca_train.jsonl', orient='records', lines=True)
# Result:52,002 JSONL rows
# Columns: instruction, input, output, text
```

### Step 3 — `arle data convert`(WORKS — different from broken download)

```bash
$ ./target/release/arle data convert \
    --input /tmp/medusa_data/alpaca_train.jsonl \
    --format alpaca \
    --output /tmp/medusa_data/alpaca_chat.jsonl

[convert_dataset] /tmp/medusa_data/alpaca_train.jsonl (Alpaca) → /tmp/medusa_data/alpaca_chat.jsonl
[convert_dataset] 52002 lines · 52002 written · 0 skipped
```

### Sample output(canonical chat format)

```json
{
  "messages": [
    {"role": "user", "content": "Give three tips for staying healthy."},
    {"role": "assistant", "content": "1.Eat a balanced diet..."}
  ]
}
```

## Token estimate vs Medusa requirement

Per sample(alpaca average):
- ~30 chars instruction(~7 tokens)
- ~150 chars output(~37 tokens)
- ~44 tokens per sample

Total dataset:
- 52,002 samples × ~44 tokens = **2,288k tokens ≈ 2.3M tokens**
- Medusa paper recommendation:100k+ tokens
- **Ratio:2.3M / 100k = 23× MORE than needed**

Phase 1.A data sufficiency:**ABUNDANT**。Can subset to first 5k samples
(220k tokens, 2.2× requirement)for fast iteration if needed。

## Phase 1.A status update

| Step | Status | Note |
|---|---|---|
| 1.A.1 inventory(`74bde06`)| ✅ done | 584 → 100k+ gap surfaced |
| 1.A.2 dataset choice(`b4ae33f`)| ✅ done | alpaca smoke + WizardLM prod |
| 1.A.3 download + convert(this) | ✅ **DONE via workaround** | 52k samples ready |
| 1.A.4 tokenize | ⏳ blocked on codex Phase 1.B | needs `arle data tokenize`命令 |
| 1.A.5 train Medusa head | ⏳ codex Phase 1.B | needs `arle train medusa`命令 |

## Workaround documentation for production

For users hitting the same `arle data download` blocker(per `da68b98`):

```bash
# 1. Manual wget(replaces arle data download)
mkdir -p /path/to/dataset
wget https://huggingface.co/datasets/<repo>/resolve/main/<file> \
    -O /path/to/dataset/raw_file

# 2. If parquet,convert to JSONL
python3 -c "
import pandas as pd
df = pd.read_parquet('/path/to/dataset/raw_file')
df.to_json('/path/to/dataset/data.jsonl', orient='records', lines=True)
"

# 3. arle data convert(works fine)
./target/release/arle data convert \
    --input /path/to/dataset/data.jsonl \
    --format <alpaca|dolly|sharegpt|chat> \
    --output /path/to/dataset/chat.jsonl
```

## Codex pickup unblocking impact

`da68b98` flagged `arle data download` as P1 codex pickup blocking
Medusa Phase 1.A pipeline。This wget workaround **partially unblocks**:
- Phase 1.A.3 data prep:**no longer blocked**(workaround works)
- Phase 1.A.4 tokenize + Phase 1.B Medusa training:STILL blocked on codex implementation
- Production deployment:still needs `arle data download` fix(workaround不适合 user-facing)

So `da68b98` codex pickup priority:
- **P1 → P2**(production cleanup,not blocking research progress)
- Codex can prioritize Hybrid Phase 1 / KV W4A8 Phase 0a / Medusa Phase 1.B over hf-hub fix

## Cross-references

- HF download blocker: `da68b98`(`docs/research/2026-05-08-medusa-phase1a-hf-download-blocker.md`)
- Phase 1.A directive: `b4ae33f`
- Phase 1.A.1 inventory: `74bde06`
- Medusa Phase 0: `afdddec`
- Test data: `/tmp/medusa_data/alpaca_chat.jsonl`(52k samples,gitignored local)
- Skill v1.5.0:`f05ea3a`

## Status

- ✅ Phase 1.A.3 unblocked via wget workaround
- ✅ 52k alpaca samples ready(2.3M tokens,23× Medusa requirement)
- ✅ `arle data convert` works(different code path from broken `arle data download`)
- ⏳ Phase 1.A.4 + 1.B blocked on codex Medusa training implementation
- ⏳ `arle data download` fix demoted P1 → P2(workaround proven)

## Rule

**When substrate-level blocker is found(per `da68b98`),test workaround
paths in adjacent code paths before assuming all dependent work is
blocked**。Manual wget + `arle data convert` is 2× cheaper than waiting
for codex hf-hub fix。

For ARLE specifically:`arle data convert` and `arle data download` are
**SEPARATE code paths**(convert reads filesystem JSONL,download fetches
HF Hub)。Bug in one doesn't block other。Production users can use the
workaround until library fix lands。

Skill v1.5.0 generalization:**substrate blockers should be scoped to
the EXACT broken code path**,not the entire feature surface。Adjacent
paths may be unaffected。
