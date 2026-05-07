# Bench — Qwen3.6 INFER_MOE_TOP_K sweep + canonical model audit — 2026-05-07

## Goal

Continue from
[`2026-05-07-bench-qwen36-moe-topk-runtime-knob.md`](2026-05-07-bench-qwen36-moe-topk-runtime-knob.md)
which validated `top_k=6` (−21.4% c=4 p50). This entry sweeps lower
values to find the sweet spot AND audits the Qwen3.6 canonical
checkpoint for prerequisite work (MTP tensor presence).

## Sweep results

| batch | top_k=8 (default) p50 | top_k=6 p50 | Δ vs 8 | top_k=4 p50 | Δ vs 8 |
|------:|----------------------:|------------:|-------:|------------:|-------:|
| c=4   | 28880 μs | 22694 μs | −21.4% | **19958 μs** | **−30.9%** |
| c=8   | 41108 μs | 37044 μs | −9.9%  | **34680 μs** | **−15.6%** |

→ **top_k=4 gives −30.9% c=4 p50 / −15.6% c=8 p50.** The reduction is
super-linear in c=4 (smaller than 8/4 = 2× pro-rata), suggesting the
encoder-overhead-per-expert is non-linear (smaller k is even
more efficient than the expert-dispatch ratio alone predicts).

## Canonical model audit — MTP prerequisite for M_e.7

Inspected `~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<sha>/`:

```
Total tensors:       2090
mtp.* tensors:       0
mtp_layers config:   absent
num_mtp_layers:      absent
model_type:          qwen3_5_moe
architectures:       ['Qwen3_5MoeForConditionalGeneration']
num_experts:         256  (was 128 in earlier surveys)
num_experts_per_tok: 8
hidden_size:         2048
num_hidden_layers:   40
moe_intermediate_size: 512
```

**Surprises**:
1. **MTP heads absent.** This was the prerequisite for M_e.7 (predicted
   ~40% c=1 win). M_e.7 is now blocked on switching canonical model
   or finding a Qwen3.5-MoE-text-only-with-MTP checkpoint. Plan doc
   updated with the blocker.
2. **256 experts, not 128** — earlier research surveys assumed 128.
   With 256 experts and top_k=8, only 8/256 = 3.1% of experts fire
   per token (vs the 8/128 = 6.25% assumed). Each token's MoE
   forward is "sparser" than thought; the SiLU+gather_qmm cost
   dominates per-expert in absolute terms.
3. **The model is actually multimodal (Qwen3.6-VL)** — `vision_tower.*`,
   `image_token_id`, `video_token_id` keys all present. ARLE has
   been benching just the text path, but the canonical includes the
   vision tower's weight footprint (auto-wired-limit of 20 GiB
   covers it; just noting).

These surprises don't invalidate any prior wins — auto-wired-limit,
oMLX-C v3, M_e.4 SwiGLU fusion, INFER_MOE_TOP_K all work on the text
forward path orthogonally to the vision components. But future plans
that depend on Qwen3.5/3.6 architectural assumptions should re-verify
against the actual checkpoint.

## Decision: don't flip default-on for top_k<8 yet

A separate quality eval is the gate. Without it, the −31% c=4 p50 win
at top_k=4 is a "cliff" we can't navigate safely. Today the env var
ships at default 8 (passthrough) with users opt-in for the latency
trade.

Options for default-on landing (multi-tick):

1. **Run a small MMLU/HumanEval subset** at top_k ∈ {4, 6, 8} and
   measure accuracy delta. Per the parallel research subagent on
   quality eval methodology (this date), the "deterministic
   regression" approach (greedy decode 50 prompts, assert text
   semantic equivalence) may be sufficient if we accept "matches
   target greedy output" as the quality gate.
2. **Per-deployment flag** — chat / latency-critical: top_k=6 default;
   eval / quality-critical: top_k=8 default. Document in AGENTS.md.

## Problems / observations

1. **MTP plan needs canonical-model swap or addition.** Updated plan
   doc with blocker section.
2. **Top_k=4 is huge but unvetted.** Until quality eval lands, only
   top_k=6 has the upstream cite ("~3% MMLU drop") to lean on.
3. **The encoder-bound diagnosis keeps validating.** Each primitive
   reduction (top_k=8→6→4) yields a roughly proportional ITL drop —
   exactly what
   `2026-05-07-bench-qwen36-encode-bottleneck.md` predicted.

## Learnings

1. **Always dump the canonical-model tensor manifest** before
   designing a plan that depends on tensor naming. This 5-line
   `safetensors.safe_open` + grep would have flagged the MTP-absent
   surprise during the M_e.7 design tick rather than during impl.
2. **Smaller top_k pays super-linearly.** −31% latency at half the
   experts (4 vs 8) is more than the 50% expert-dispatch reduction
   alone would predict. The constant overhead per gather_qmm
   dispatch is real and worth more than people guess.

## What worked / Rule

- Single-line env probe (already shipped commit `90942dd`) made the
  sweep a 5-minute exercise: stop server with `INFER_MOE_TOP_K=N`,
  start with new N, sweep.
- Phase-timing logs (already shipped) captured per-batch p50/p99
  with no new tooling.

## Rule

Before any plan that names specific tensor patterns or model
architectural keys, dump the canonical-model checkpoint's tensor
manifest and config:

```python
from safetensors import safe_open
import json
idx = json.load(open("path/to/snapshot/model.safetensors.index.json"))
keys = list(idx["weight_map"].keys())
print(f"Total: {len(keys)}; sample: {keys[:5]}; matches: {[k for k in keys if 'mtp' in k.lower()]}")
```

5 lines, runs in under a second, prevents L-effort waste on an
inapplicable plan.

## Next

- **Quality eval design** (research subagent dispatched in parallel
  this tick). When it returns, run the chosen eval suite at top_k ∈
  {4, 6, 8} on representative prompts. Lock in default top_k.
- **M_e.7 unblocked once canonical model swap lands** — flag for
  next session's discussion.

## References

- Predecessor (initial top_k=6 win):
  [`2026-05-07-bench-qwen36-moe-topk-runtime-knob.md`](2026-05-07-bench-qwen36-moe-topk-runtime-knob.md)
- Encoder-bound diagnosis explaining why this works:
  [`2026-05-07-bench-qwen36-encode-bottleneck.md`](2026-05-07-bench-qwen36-encode-bottleneck.md)
- M_e.7 plan blocker:
  [`docs/plans/M_e7-mtp-head-wiring.md`](../../plans/M_e7-mtp-head-wiring.md)
