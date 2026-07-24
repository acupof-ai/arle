# PPL calibration shipped: FP8 quant loss on 27B is −0.25% vs bf16 (WikiText-2, ctx 2048) — no measurable degradation

> Status: Shipped (#174). Pod: 8×H20, GPU 2 only, `arle train ppl`,
> full WikiText-2-raw test = 146 windows / 296,907 scored tokens, greedy
> teacher-forced.

## Context

First run of the `arle train ppl` harness (landed 2026-07-20 code-only, never
GPU-verified). Goal: quantify FP8 quantization quality on the 27B lane.

## What Worked

| model | ppl | Δ vs bf16 |
|---|---|---|
| ThinkingCap-Qwen3.6-27B (bf16 anchor) | 6.6875 | — |
| ThinkingCap-Qwen3.6-27B-FP8 | 6.6708 | **−0.25%** |
| base Qwen3.6-27B-FP8 | 6.6709 | −0.25% |

- **FP8 verdict: no measurable quant loss** at ctx 2048 (FP8 marginally *lower*
  — within scoring noise, both shared-scale FP8).
- **Blocking bug found + fixed first** (`067849cf3`): `forward_token_logits`
  built a `new_linear_only()` transient slot, but the paged-KV migration never
  allocates contiguous `k_caches` — any full-attn layer panicked (index 0, len
  0, `qwen35.rs:5725`). Fix routes the transient forward through a free
  paged-pool slot (`Qwen35RecallForward`), freeing pages after. Same path
  serves the in-process OPD raw-logits teacher on hybrid models — that lane was
  silently broken too.
- Missing cells (honest): `-FP8-fixed` dir is weightless (29 MB, no
  safetensors); no base bf16 on pod. Scoring the fixed re-export is one command
  once the export exists.
- **Anomaly for owner check**: ThinkingCap-FP8 ppl ≡ base-FP8 ppl to 1e-6
  relative (6.670843 vs 6.670851) while the bf16 tune differs at 1.7e-2 — the
  "-FP8" dir's trunk is (near-)identical to base FP8 as WikiText scoring sees
  it. Filed #177.

Evidence: pod `/host/ppl/results.jsonl` + per-model logs; corpus
`/host/ppl/wikitext2_test.raw.txt`.

## Rule

- A "code-only, no GPU" harness landing is unverified until its first GPU run —
  budget the fix loop (here: a panic on the very first forward).
- Two independently fine-tuned checkpoints agreeing to 6 significant digits on
  held-out PPL means they share trunk weights — flag the checkpoint, don't
  average the anomaly away.
