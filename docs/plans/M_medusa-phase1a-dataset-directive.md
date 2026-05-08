# M_medusa Phase 1.A directive — pick dataset + wire HF Hub loader

> Per `74bde06` Medusa Phase 1.A data inventory:584 tokens existing vs
> 100k+ target = 172× short。Path A(HF Hub integration)preferred per
> existing `crates/train/src/hub_dataset.rs` infra。
> This brief picks the specific dataset + writes invocation for codex
> pickup。

## Existing infrastructure

`crates/train/src/hub_dataset.rs`(read 2026-05-08):
- Thin wrapper over `hf-hub` sync API
- Returns local cached path to JSONL file
- Auth via `HF_TOKEN` or `~/.cache/huggingface/token`
- Plugs into `sft_data::load_jsonl`

Wire-up pattern(per docstring):
```bash
DATA=$(arle data download --repo <repo_id> --file <path>)
arle train sft --data "$DATA" ...
```

## Dataset candidates for Medusa-1 head training

### Option 1 — `lmsys/lmsys-chat-1m`(RECOMMENDED)
- 1M conversations from real LMSYS Chat usage
- Matches Medusa paper's "real chat distribution" intent
- Multi-turn,natural language,covers Q3-style tasks
- License:requires HF account agreement(not free-anon access)
- Format:JSONL,with `conversation` field per row
- Tokens:~500M+(vastly exceeds 100k requirement,can subset)

### Option 2 — `BelleGroup/multiturn_chat_0.8M`
- Chinese multi-turn dialogue dataset
- 800k turns
- May better fit Qwen3.6 multilingual training but English subset
  available
- License:open(CC-BY-NC-4.0,non-commercial OK for research)

### Option 3 — `WizardLM/WizardLM_evol_instruct_70k`
- 70k samples,instruction-following with evolved complexity
- Matches Medusa paper Vicuna baseline closely
- Format:JSONL with `instruction` + `output`
- Single-turn(simpler for first Medusa iteration)
- License:Apache-2.0

### Option 4 — `tatsu-lab/alpaca`
- 52k samples,instruction format
- Smaller(~10M tokens)
- Single-turn,simpler
- License:CC-BY-NC-4.0
- Quick smoke-test(fast download,fast training iteration)

## Recommendation:**Option 4 first(alpaca smoke)+ Option 3 production**

Phase 1.A.1(smoke test,1 hour wall):
- Use `tatsu-lab/alpaca`(52k samples)
- Validate end-to-end pipeline:download → tokenize → train Medusa head → eval
- Confirms infra works before scaling up

Phase 1.A.2(production training,~half day):
- Use `WizardLM/WizardLM_evol_instruct_70k`
- 70k samples ≈ Medusa paper baseline
- Matches single-turn instruction format(simpler than multi-turn for Medusa-1)

Phase 1.A.3(future,if accuracy plateau):
- Add `lmsys/lmsys-chat-1m` for multi-turn coverage(Medusa-2 prep)

## Phase 1.A.1 invocation(codex pickup)

```bash
# Step 1: HF login if not already
huggingface-cli whoami || huggingface-cli login

# Step 2: download alpaca via arle data CLI
DATA=$(arle data download \
    --repo tatsu-lab/alpaca \
    --file data/train.json)

echo "Dataset cached at: $DATA"

# Step 3: tokenize with Qwen3-4B tokenizer
arle data tokenize \
    --input "$DATA" \
    --tokenizer-model infer/models/Qwen3-4B \
    --output /tmp/alpaca-qwen3-tokens.bin \
    --max-seq-len 2048

# Step 4: train Medusa-1 head (small smoke run, 100 steps)
arle train medusa \
    --base-model infer/models/Qwen3-4B \
    --tokens /tmp/alpaca-qwen3-tokens.bin \
    --num-heads 4 \
    --steps 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --output /tmp/qwen3-medusa-head-smoke

# Step 5: eval Medusa head — accept rate on held-out
arle eval medusa \
    --base-model infer/models/Qwen3-4B \
    --medusa-head /tmp/qwen3-medusa-head-smoke \
    --tokens /tmp/alpaca-qwen3-tokens.bin
```

## Code gaps to address

Per `afdddec` Phase 0 + this directive:
- ✅ `crates/train/src/hub_dataset.rs` — HF Hub download(EXISTS)
- ✅ `sft_data::load_jsonl` — JSONL loader(EXISTS per docstring)
- ❓ `arle data download` CLI subcommand — verify exists,or add(~30 LOC if missing)
- ❓ `arle data tokenize` CLI — verify exists per existing tokenization infra
- ❓ `arle train medusa` CLI subcommand — Medusa-specific training entry,likely needs adding(~150 LOC per `afdddec` Phase 1.B scope)
- ❓ `arle eval medusa` CLI subcommand — Medusa accept-rate eval,likely needs adding(~50 LOC)

## Effort estimate update(per `afdddec` Phase 0 reconnaissance)

| Phase | Original(`afdddec`)| Updated(this brief)|
|-------|--------------------:|--------------------:|
| 1.A Training data prep | 50 LOC | **20-50 LOC**(orchestration,dataset choice resolved)|
| 1.B Head + training | 150 LOC | **150-250 LOC**(needs `arle train medusa`)|
| 1.C ARLE inference | 150 LOC | **150 LOC**(reuse speculative substrate)|
| 1.D Test gate | 50 LOC | 50 LOC |
| 1.E Bench gate | 50 LOC | 50 LOC |
| **Total Phase 1** | **500 LOC** | **420-550 LOC** |
| Wall time(codex)| 8-9 days | **8-10 days** |

Scope unchanged within ±10%。

## KILL criteria for Phase 1.A

- **Phase 1.A.1**:if alpaca smoke training crashes / loss diverges → KILL Phase 1.A
- **Phase 1.A.2**:if WizardLM 70k training accept rate < 30% on Qwen3-4B
  eval → KILL Medusa-1,re-evaluate(may need DFlash or DraftK instead)
- **Phase 1.A.3**:never executed unless 1.A.2 plateau

## Cross-references

- Medusa Phase 0 reconnaissance: `afdddec`
- Medusa Phase 1.A data inventory: `74bde06`
- Medusa plan main: `528844c`(Phase 3 corrected)
- Master strategy §7.4 P1.1 Medusa REQUIRED post-classical-DEAD: `5acbe94`
- HF Hub loader: `crates/train/src/hub_dataset.rs`
- Train substrate: `crates/train/src/`(causal_lm + dataset + sft_data + grpo + grad_accum)

## Status

Phase 1.A directive ready for codex pickup。Effort ~1-2 days for Phase
1.A alone(download + tokenize + smoke train + eval pipeline)。

Phase 1.A.2 production training adds ~3-4 days codex(WizardLM 70k full
run + accept-rate eval)。

Total Phase 1 complete:8-10 days codex,within original `afdddec`
estimate envelope。

When codex picks up:start with Phase 1.A.1 smoke(alpaca),verify
pipeline,then advance to Phase 1.A.2 production。
