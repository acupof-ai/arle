# Medusa Phase 1.A data inventory — current trace 584 tokens vs 100k+ target

> Per `afdddec` Medusa Phase 0 reconnaissance,Phase 1.A is "training
> data prep(~50 LOC,Claude scope)"。This entry inventories existing
> data sources and quantifies the gap to Medusa paper's 100k+ token
> training requirement。
>
> **Findings**:existing `scripts/data/agent_trace_default.jsonl` has
> only **584 tokens(6 sessions / 22 turns)**— **172× short** of
> Medusa paper recommendation(100k+ tokens)。Phase 1.A actually
> requires HF dataset integration via `crates/train/src/hub_dataset.rs`,
> not just script scaffolding。

## Existing data inventory

`scripts/data/agent_trace_default.jsonl`:
- Sessions:6
- Turns:22(mix user + assistant)
- Total chars:2,339
- Estimated tokens(chars/4):**584**

Format(per first session):
```json
{
  "session_id": "agent-001",
  "system_prompt": "You are a helpful AI assistant specialized in mathematics...",
  "turns": [
    {"role": "user", "content": "What is 17 * 23?..."},
    {"role": "assistant", "content": "I'll use the calculator to compute..."}
  ]
}
```

This is the EXISTING bench input for `bench_agent_trace.py` workloads —
adequate for inference benchmarking(replays 6 sessions × 2 turns × 8 conc = 256 turns)
but NOT Medusa training data。

## Medusa paper training requirement

Per Medusa-1 paper(Cai et al. 2024)§4.1:
- Vicuna 7B trained on ~100k token sequences from ShareGPT + Vicuna conversations
- 4 heads × 6.5M ResBlock params each = 26M trainable
- Loss converges in ~3 days on single A100 GPU

For Qwen3-4B Medusa-1(`afdddec` plan):
- Same heads × 4 architecture
- ~26M trainable params total
- Estimated training data:**100k+ token sequences**(>= Vicuna baseline)

**Gap**:584 / 100,000 = **0.58%** of Medusa requirement。

## Phase 1.A actual scope(refined post-inventory)

Per `afdddec` Phase 0:"50 LOC,Claude scope"。Refined post-inventory:
this assumes data already prepared。Need either:

### Path A — HF dataset integration(recommended)

Use `crates/train/src/hub_dataset.rs`(existing infrastructure)to load:
- `tatsu-lab/alpaca`(52k samples,~10M tokens)
- `vicuna_conversations`(70k samples,~100M tokens)
- `Qwen3 instruct datasets`(public on HF)

Implementation:
- `scripts/medusa_training_data.py`:HF dataset loader → Medusa format(~50 LOC)
- Calls existing `hub_dataset.rs` Rust trainer-side adapter
- Output:safetensors with `(input_ids, target_ids, hidden_state_indices)`

LOC:50-80 LOC Python script + zero new Rust(reuses hub_dataset.rs)
Risk:LOW(infrastructure exists)

### Path B — Synthetic teacher generation

Run Qwen3-4B on existing 6 sessions + variants:
- Each session gets 100+ rephrased prompts via teacher
- Total ~600 sessions = ~60k tokens
- Still ~half of paper recommendation but enough for first-pass training

Implementation:
- GPU work,~2 hours teacher inference at sm_89
- ~100 LOC orchestration script

LOC:100 LOC + GPU compute time
Risk:Medium(synthetic data quality may differ from real conversation distribution)

### Path C — Mixed(production-quality recommendation)

- HF dataset for bulk(50k samples Alpaca)
- Plus existing 22 agent turns(domain-specific)
- Total:50k+ samples,covers both general + agent shape
- Best for axis 2 production deployment per `aa00c6a` 4-KILL evidence

LOC:80-120 LOC orchestration + HF integration
Risk:LOW(combines proven sources)

## Recommendation

**Phase 1.A path**:Path C(mixed,production-quality)。

Specifically:
1. Phase 1.A.1(this entry):data inventory + gap analysis ✅
2. Phase 1.A.2(next,Claude or codex 0.5-1d):HF dataset integration
   via `hub_dataset.rs` adapter
3. Phase 1.A.3(Claude 0.25d):Medusa format conversion script(~50 LOC)
4. Phase 1.A.4(codex 0.5d GPU):synthetic agent-shape augmentation
   via Qwen3-4B teacher

Total Phase 1.A wall-time:**~2-3 days**(was estimated 2 days in `afdddec`)

## Cross-references

- Medusa Phase 0 reconnaissance: `afdddec`
- Existing trace: `scripts/data/agent_trace_default.jsonl`(used by `bench_agent_trace.py`)
- HF dataset infrastructure: `crates/train/src/hub_dataset.rs`
- Medusa paper:<https://arxiv.org/abs/2401.10774>
- Plan:`docs/plans/M_medusa-required-path.md`(`528844c`)

## Status

- ✅ Phase 1.A.1 data inventory complete(this entry)
- ⏳ Phase 1.A.2 HF dataset integration:NEXT(Claude or codex)
- ⏳ Phase 1.A.3 Medusa format conversion:after 1.A.2
- ⏳ Phase 1.A.4 synthetic agent augmentation:codex GPU work

## Rule

**Phase 1.A pre-flight inventory is mandatory before scoping data
prep work**。Naive estimate "50 LOC" assumes data exists;empirical
inventory shows existing data is 172× short → actual scope is
HF integration + augmentation(Path C ~100-150 LOC)。

For ARLE specifically:any new training axis(Medusa,EAGLE,LoRA)
should run Phase 0 inventory of(a)existing data corpus,(b)scale gap
to literature requirement,(c)available infrastructure for closing the
gap before committing LOC budget。

This methodology rule generalizes anti-pattern #14(upstream parser
correctness)to data corpus correctness:**when planning a training axis,
INVENTORY existing data BEFORE scoping prep code**。Naive single-file
estimates miss data scale gaps。
