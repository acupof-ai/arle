# Medusa Phase 1.B `arle train medusa` scope correction — empirical 600-1200 LOC vs my 150 estimate

> Per `b4ae33f` Phase 1.A directive estimated `arle train medusa` CLI
> at ~150 LOC。Direct code-grep of `crates/train/src/commands/train_sft.rs`
> reveals **1886 LOC for similar-shape SFT command**。Medusa training
> realistic scope is **600-1200 LOC**(2-3× my earlier estimate),
> updates Phase 1 total from 8-9 days → 10-14 days。

## Existing infrastructure(per code-grep)

`crates/train/src/commands/`(8 commands):
- `convert_dataset.rs`(dataset format conversion)
- `download_dataset.rs`(HF Hub wrapper,affected by `5889a8d` bug)
- `eval_lm.rs`(eval entry point)
- `pretrain.rs`(generic pretrain)
- `pretrain_dsv4.rs`(DeepSeek V4 specific)
- `train_grpo.rs`(GRPO RL command)
- `train_multi_turn.rs`(multi-turn SFT)
- `train_sft.rs`(**1886 LOC** — closest analog to Medusa)

`train_sft.rs:270 run_with_args` + `:281 run_with_family<F: SftFamily>`:
- CLI arg parsing(~200 LOC)
- Trainer config + family dispatch(~150 LOC)
- Training loop + control plane(~800 LOC)
- Eval + checkpoint logic(~400 LOC)
- Family-specific(Qwen3/Qwen3.5/DeepSeek)hook impls(~300 LOC)

→ SFT command is **1886 LOC of substantive logic**,not 150。

## Medusa training extensions over SFT

Medusa-1 training adds:
1. **Multi-head architecture**:N heads(typically 4-5)each predicting
   1-step-ahead token(~100 LOC for head module + integration)
2. **Loss function**:sum of N per-head cross-entropy(~50 LOC)
3. **Data preparation**:position-shifted labels for each head(~100 LOC)
4. **Optional tree attention**(Medusa-2):tree-aware forward pass + accept rate(deferred per `afdddec`,~300 LOC if added)
5. **Eval**:Medusa accept rate calculation(~150 LOC)
6. **Checkpoint**:save head + base model separately(~50 LOC)

Total Medusa-specific:~450 LOC NEW(vs SFT baseline)
Plus CLI/control reuse from SFT:~150-300 LOC could be shared

**Realistic Medusa Phase 1.B**:**600-1200 LOC**

## Effort estimate(corrected)

| Phase | Original `afdddec` | Updated `b4ae33f` | THIS empirical |
|-------|-------------------:|-------------------:|---------------:|
| 1.A data prep | 50 LOC | 20-50 LOC | 20-50 LOC |
| 1.B head + training | 150 LOC | 150-250 LOC | **600-1200 LOC** |
| 1.C ARLE inference | 150 LOC | 150 LOC | 150 LOC |
| 1.D test gate | 50 LOC | 50 LOC | 50 LOC |
| 1.E bench gate | 50 LOC | 50 LOC | 50 LOC |
| **Total Phase 1** | **500 LOC** | **420-550 LOC** | **870-1500 LOC** |
| Wall time(codex)| 8-9 days | 8-10 days | **10-14 days** |

Medusa Phase 1.B is the largest sub-phase。Empirical from SFT suggests
training commands are substantively larger than my Phase 0/1.A
estimates。

## Implementation strategy

**Option A** — separate `train_medusa.rs` command(~1200 LOC):
- Pros:clean isolation,doesn't affect existing SFT code path
- Cons:duplicates ~700 LOC of SFT control plane

**Option B** — add `--medusa N` flag to existing `train_sft.rs`(~600 LOC delta):
- Pros:reuses 1886 LOC of SFT scaffolding,smaller PR
- Cons:tangles Medusa-specific logic with SFT,risk regressing SFT

**Option C** — minimal `train_medusa.rs` calling shared SFT helpers(~800 LOC):
- Pros:isolation + reuse,medium PR
- Cons:requires factoring SFT into reusable modules first(could be its own ~200 LOC refactor)

Recommendation:**Option B initial** for Phase 1.B speed,migrate to
Option C if SFT regression risk surfaces。

## Decision implications

If user wants Medusa P1':
- Scope is 10-14 days codex(not 8-9)
- Larger PR than estimates suggested → review burden
- Consider whether B3 PrefixAwareAdmission(`a1965ab` ~350 LOC)is
  better ROI / lower-risk for axis 2 prep

## Updated pickup queue(corrected effort estimates)

- P0 Hybrid Phase 1b:155-175 LOC,0.75-1d(per `9dc32d6`)
- P0' M_warmup prefill pass:~150 LOC,1.5d(per `56dbd1c`)
- P1 B3 PrefixAwareAdmission:~350 LOC,~2-3 days(estimate)
- P1 KV W4A8 #33:**5-10 days codex**(~500-1000 LOC kernel)
- **P1' Medusa Phase 1.B**:**10-14 days codex**(~600-1200 LOC,corrected)

If user wants Medusa,it's **the longest single pickup** in the queue。
Hybrid + bimodal fix + B3 + KV W4A8 = ~2 weeks total。Medusa alone =
2 weeks。

## Cross-references

- `b4ae33f` original Medusa Phase 1.A directive(scope estimate)
- `afdddec` Phase 0 reconnaissance
- `crates/train/src/commands/train_sft.rs:270-281`(SFT entry,~1886 LOC)
- `4b5bb91` Phase 1.A.3 wget workaround(unblocks dataset)
- `5889a8d` hf-hub library bug(affects `download_dataset` cmd,workaround OK)

## Methodology lesson

Estimating "command line interface" as ~150 LOC is systematically
optimistic for substantive ML training commands。The CLI is just the
entry point;underlying logic dwarfs it。

**Rule**:when estimating new training/inference command scope,start
from existing analog command size(grep wc -l)not from intuition。

`b4ae33f`'s 150 LOC estimate was 8-12× too small。Empirical
correction shifted Phase 1.B from "1-2 days" to "1-2 weeks" — important
strategic realism for whether to greenlight axis 2 now or defer。

## Status

**This brief is the deliverable** — Medusa scope corrected,realistic
estimate documented。

If user picks Medusa P1' direction:expect 10-14 days codex,not 8-9。
Plan `b4ae33f` updated with empirical correction here。

If user wants faster axis 2 progress:consider B3 PrefixAwareAdmission
(2-3 days)as smaller-scope axis 1 win that frees codex bandwidth for
Medusa。
