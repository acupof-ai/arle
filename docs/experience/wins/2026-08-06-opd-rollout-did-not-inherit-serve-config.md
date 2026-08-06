# The OPD rollout engine did not inherit serve's config — CUDA, 2026-08-06

> Status: **Shipped** — `d925968f9` (stale seam default), `4850d0dd7`
> (`mem_fraction_static` 0.2 → flag, default 0.5).

## Context

`arle train agent-opd` builds an in-process rollout engine through the same
`infer_seam` traits that `arle serve` uses. Two settings that serve has shipped
for days never reached it, and neither failed loudly.

## Bug 1 — a licensed default that was still `false` in the seam

FlashQLA chunked GDN prefill went default-on 2026-08-02 after a full
correctness adjudication
([`wins/2026-08-02-flashqla-chunked-gdr-h48.md`](2026-08-02-flashqla-chunked-gdr-h48.md)).
The CLI flag flipped; `RuntimeFlags::default()` in
`crates/infer-seam/src/runtime_flags.rs` did not. Anything constructing flags
from the struct default instead of parsing serve's argv — the rollout engine —
kept running the recurrent scan.

Fix: `#[serde(default = "d_true")]` + `qwen35_gdr_chunked: true` in the seam,
mirrored as a `clap::ArgAction::Set` flag on the train CLI so the two paths
carry one shipped default and one override mechanism.

### What it is worth

1× H20 (GPU 6), ThinkingCap-Qwen3.6-27B-FP8, TP=1, c=16, 33K prompts, same
binary, only `--qwen35-gdr-chunked` differs:

| arm | completed | out tok/s | total tok/s | req/s |
|---|---:|---:|---:|---:|
| chunked (shipped default) | 16/16 | **21.1** | **2747.9** | 0.08 |
| recurrent (the stale default) | 16/16 | 14.8 | 1928.4 | 0.06 |
| Δ | — | **+42.6%** | **+42.5%** | — |

The DSpark decode tick is unchanged across the arms — 162.0 vs 162.9 ms at
rows=16 — so the whole 42% is prefill. Any workload dominated by long prompts
was paying it; agent-OPD rollouts are 21K-token prompts.

## Bug 2 — `mem_fraction_static: 0.2`, hardcoded

`train_cli.rs` built its `EngineLoadConfig` with a literal `0.2`, which reads
like "take a fifth of the card, leave the rest for the student".

It is not a share of free VRAM. It bounds the engine's share of **total**:

```
reserve = total × (1 − F)
rest    = free − reserve
tokens  = rest / cell_bytes          floored at PROFILE_KV_TOKENS_FLOOR = 4096
```

On the H20 with the 27B resident: total 97.5 GB, free 68 GB, weights 29 GB
(0.30 of total).

| F | reserve | rest | pool |
|---:|---:|---:|---|
| 0.2 | 78.0 GB | **−10 GB** | floor, 4096 tokens |
| 0.5 | 48.8 GB | 19.2 GB | real |

Any F below the weights' own share of total yields a zero pool by
construction. Admission then capped at 4096 tokens and **aborted every prompt
longer than that** — every agent prompt. The A/B that found this returned
`completion_tok=0` in both arms with no error at the train level.

Fix: `--rollout-mem-fraction`, default 0.5, and a `log::warn!` in
`infer-cuda`'s Qwen3.5 executor when the profile lands exactly on the floor —
the one place that has both the number and the model context.

This is the same knob's other cliff. At 0.9 it starved a co-resident draft
model of 356 MB
([`errors/2026-08-05-training-job-inherits-the-serving-kv-pool.md`](../errors/2026-08-05-training-job-inherits-the-serving-kv-pool.md));
at 0.2 it starves itself.

## Problems

The first A/B was uninterpretable — both arms produced zero completion tokens,
so the flag under test could not have shown a difference either way. The zero
was read as "the A/B is a wash" for one round before the arms were checked for
having done any work at all.

## Rule

**A default that ships behind a CLI flag has two homes, and flipping one is
half a flip.** The licensing bench flips argv; every non-serve constructor
reads the struct default. Grep for the field, not the flag.

**A run that produced no output tokens is not a measurement.** Before comparing
arms, assert the shared denominator is non-zero — completion tokens here, and
the same check applies to any A/B whose metric is a ratio.
