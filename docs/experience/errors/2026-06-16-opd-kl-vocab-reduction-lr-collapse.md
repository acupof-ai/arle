# OPD KL loss `mean`-over-vocab silently collapsed the LoRA learning rate

## Context

The OPD distill loss `kl_distill_loss` (`crates/train/src/loss.rs`) reduced the
forward/reverse KL with `mean` over **all** logit elements — `[positions ×
vocab]` — i.e. `(sum_v t·log s / positions) / vocab`. A pure-KL 4B→0.8B run
(`--kl-direction forward --gkd-lambda 0.0`, GSM8K prompts, all-linear LoRA r32)
showed a "loss" of ~1e-5 that barely moved across 30 steps. The in-code comment
claimed "AdamW absorbs the constant `1/vocab` via its adaptive learning rate."

## Root Cause

The `1/vocab` (≈ 1/151936) rescale is **not** optimizer-invariant, and the
comment was wrong in this regime. Adaptive normalization `m̂/√v̂ ≈ ±1` only holds
when `√v̂ ≫ eps`. AdamW eps here is **1e-8** (`train_cli.rs:575/785/1064/1291`,
all examples). The per-logit gradient under the `mean` reduction is
`(s_j − t_j)/(positions·vocab)` ≈ `1e-3 / (64·151936)` ≈ **1e-10**, ~100× *below*
eps. So `√v̂ + eps ≈ eps` (eps dominates) and the update degenerates from
adaptive normalization to `m̂/eps`-scaled SGD → **effective LR collapses by up to
~vocab×**. The ÷vocab scale also (a) makes `--grad-clip 1.0` never fire (grad
norm ≪ 1) and (b) prints a loss `1/vocab` too small (~1e-5 = a real ~1.5-nat KL
divided by vocab).

Worse, the bug was **load-bearingly coupled**: `next_token_sft_loss_from_logits`
(`opd.rs:1571`) *deliberately* divided the GKD/ROPD hard-label CE anchor by
`vocab` to "match the KL internal normalization, otherwise lambda=0.3 would
dominate KL by roughly vocab_size." So in any λ>0 (ROPD) config, KL and CE were
both ÷vocab — internally consistent but globally mis-scaled, and the *whole*
blend ran in the eps-collapsed regime.

## Fix

Restore the canonical KD reduction = PyTorch `reduction='batchmean'` (sum over
vocab, mean over positions = `sum_v / positions`), which the PyTorch `KLDivLoss`
docs flag as the mathematically-correct KL (their `'mean'`, dividing by vocab
too, is the documented-wrong form). Implemented by multiplying the `mean` result
by `vocab` (constant scalar) — keeps the tested `mul_scalar`+`mean` device path,
no `sum_backward` scalar-broadcast (`loss.rs` forward + reverse + chunked
sibling, all three). Coupled half: **removed** the `1/vocab` rescale on the GKD
CE anchor (`opd.rs`), since `cross_entropy_loss` is already per-position;
`--gkd-lambda` now blends KL and CE at face value. Unit tests: 100/100 train lib
green; chunked-vs-baseline tolerance made hybrid abs+rel (the equivalence is a
relative property; the absolute bound was scale-fragile after ×vocab).

On-pod capability A/B (before=÷vocab vs after=batchmean, same config, GSM8K
direction + loss trajectory ~1e-5 → ~1.5) is **pending-remote** (.62, GPU3
lane); this entry lands the root-cause + code fix; the capability delta appends
when the run checkpoints.

## Rule

A constant rescale of a loss is **not** free under AdamW: if it pushes the
per-parameter gradient second moment `√v̂` near/below `eps` (1e-8), AdamW stops
normalizing and the effective LR collapses by that constant. Before assuming
"the optimizer absorbs it," check `gradient magnitude vs eps` arithmetically.
Default KD/distill reduction = `batchmean` (÷positions), never `mean` (÷positions
·vocab). When two blended loss terms are scaled to "match" each other, that
coupling is load-bearing — fixing one scale **must** re-derive the other
(`feedback_flag_silent_noop_passes_exit0_smoke`: the ÷vocab CE compensation
silently inverted the imbalance instead of removing it).
