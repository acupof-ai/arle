# M_e.8 — MoE quality eval scaffold (3-tier)

**Owner:** ckl · **Status:** designed 2026-05-07 (subagent), awaiting impl tick
**Track:** Quality / Metal · **Predecessor:** INFER_MOE_TOP_K runtime knob
  (commit `90942dd`)

## Goal

Provide the missing quality gate for `INFER_MOE_TOP_K=N` (and any
future MoE-related tradeoffs). Today the env var is opt-in because we
have no validated quality cost on Qwen3.6 35B-A3B-4bit. M_e.8 ships
the eval scaffold so future commits can flip `INFER_MOE_TOP_K=6` (or
`=4`) to default ON with confidence.

## Three-tier design (per parallel research subagent, 2026-05-07)

### Tier-1 — Deterministic regression (per-commit go/no-go, 30-90s)

50 fixed prompts at temperature=0, max_tokens=256. Run twice (configs A
and B), exact-match each output pair, aggregate to `(match_rate,
mean_first_divergence_token_index)`.

**New script**: `scripts/eval_topk_regression.py` (~120 lines, modeled
on `scripts/eval_ppl.py` server-launch pattern + the `assert_eq!`
greedy fingerprint pattern at
[`infer/tests/spec_decode_correctness.rs:162-191`](../../infer/tests/spec_decode_correctness.rs)).

```bash
./scripts/eval_topk_regression.py \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --prompts scripts/data/eli_agent_trace.jsonl \
  --top-ks 8,6 --max-tokens 256
```

**Pass criterion**: `match_rate ≥ 0.90 AND mean_first_divergence ≥ 32`.

**Use as**: per-commit gate for any change that touches MoE forward.
Detects regressions where commit X breaks top_k=6's previously-stable
output behavior.

**Caveat**: top_k=6 ≠ top_k=8 at the function level — divergence is
expected. This is a **fingerprint**, not a proof of equivalence.
Pair with Tier-2 for default-flip decisions.

### Tier-2 — HumanEval + GSM8K subset (default-flip gate, 15-20 min)

Extend `scripts/eval_ppl.py` with a `--moe-top-k 8,6` parameter axis
(replaces or augments the existing `--kv-cache-dtype` axis). Add a
generation+exec mode for HumanEval (subprocess-isolated, `signal.alarm(5)`,
standard openai/human-eval pattern). GSM8K already loaded
(`eval_ppl.py:39-47`).

```bash
./scripts/eval_ppl.py \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --datasets humaneval,gsm8k \
  --moe-top-k 8,6 \
  --max-tokens 256 --max-samples 100
```

**Pass criterion**: `humaneval_pass1[k=6] / humaneval_pass1[k=8] ≥ 0.97`
AND `gsm8k_acc[k=6] / gsm8k_acc[k=8] ≥ 0.97` (the "<3%" threshold from
the upstream vllm-mlx claim).

**Use as**: one-shot before flipping `INFER_MOE_TOP_K` default to
`6`. HumanEval + GSM8K cover the load-bearing axes (code reasoning +
arithmetic CoT) for chat / code workloads.

### Tier-3 — Full MMLU (pending-remote, reference)

Use `lm-evaluation-harness` against the OpenAI-compat endpoint. No
new ARLE code; just a reference run on the H20/CUDA box.

```bash
lm_eval --model local-completions \
  --model_args base_url=http://localhost:8765/v1/completions,model=default \
  --tasks mmlu --batch_size 8
```

**Pass criterion**: `mmlu_acc[k=6] − mmlu_acc[k=8] > −0.03` (absolute).

**Use as**: one-shot reference; cite in a `pending-remote` wins stub
when committing the default flip.

## What ARLE already has (don't reinvent)

- `scripts/eval_ppl.py` — server-launching, multi-dataset PPL via
  `/v1/completions`. **Server-lifecycle plumbing is reusable.**
- `infer/tests/spec_decode_correctness.rs:162-191` — `assert_eq!(plain,
  spec)` greedy fingerprint pattern. **Pattern is liftable.**
- `scripts/bench_eli_agent.sh` — boots `metal_serve`, replays JSONL.
  **Server-launch + replay plumbing is reusable.**
- gsm8k cached at `~/.cache/huggingface/datasets/openai___gsm8k`.
  HumanEval loaded by `eval_ppl.py:39-47`.
- OpenAI-compat `/v1/completions` exposes per-token logprobs
  (`infer/src/http_server/openai_v1.rs:560,669,756`).

## What ARLE doesn't have (build minimal)

- A "deterministic regression" runner — **build as Tier-1 above.**
- HumanEval generation+exec mode — **add to `eval_ppl.py`** as a
  `--mode=generate-exec` axis, ~30 LoC.
- An A/B-axis-parameterized server-launch loop. `eval_ppl.py:73-106`
  already has the shape; just generalize the param axis from
  `kv-cache-dtype` to a list of (env_var, value) tuples.

## Implementation steps

1. **Tier-1 first** (~120 LoC, 1 tick): `scripts/eval_topk_regression.py`.
   Boot `metal_serve` once with `INFER_MOE_TOP_K` unset, run 50
   prompts at temperature=0 max_tokens=256, save outputs. Boot again
   with `INFER_MOE_TOP_K=6`, run same 50 prompts. Diff and report.
   Token-level diff via the model's tokenizer for the "first
   divergence" metric.
2. **Tier-2 second** (~30 LoC diff to eval_ppl.py + subprocess
   exec helper for HumanEval, 1 tick): adds `--moe-top-k 8,6`.
3. **Tier-3 stub now**: `docs/experience/wins/2026-05-07-eval-mmlu-pending-remote.md`
   citing the future H20 run.
4. **Default-flip ticket**: only after Tier-2 passes, land a small
   commit changing `INFER_MOE_TOP_K`'s default behavior in
   `mlx_qwen35_moe_block.cpp` (or document recommended setting in
   AGENTS.md per-deployment-class).

## Risks

| ID | Risk | Mitigation |
|----|------|------------|
| R1 | Tier-1's "match_rate ≥ 0.90" threshold is arbitrary | Start at 0.90; tighten/loosen based on the first real run's distribution. The threshold is calibration, not foundation. |
| R2 | Tier-2 HumanEval execution sandbox not ready | Use `signal.alarm(5)` + `subprocess.run` per openai/human-eval — well-tested upstream pattern. |
| R3 | Server-launch overhead doubles the runtime (10 min becomes 20) | Acceptable for Tier-2 (one-shot); for Tier-1 (per-commit), consider hot-swapping `INFER_MOE_TOP_K` per-request — would require a small `metal_serve` patch (~10 LoC, accepting per-request env-var override header). |
| R4 | top_k=4 outputs diverge >10% even on chat workloads | Tier-1 detects this; we report the failure and downgrade k=4 from "pursue default" to "advanced opt-in only". |

## References

- `vllm-mlx` `--moe-top-k`: https://github.com/blaizzy/vllm-mlx
- Apple Recurrent Drafter exact-match acceptance:
  https://arxiv.org/abs/2403.09919
- HumanEval: https://github.com/openai/human-eval
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- MMLU: https://huggingface.co/datasets/cais/mmlu
- ARLE existing eval-PPL harness:
  [`scripts/eval_ppl.py`](../../scripts/eval_ppl.py)
- ARLE spec-decode greedy fingerprint precedent:
  [`infer/tests/spec_decode_correctness.rs:162-191`](../../infer/tests/spec_decode_correctness.rs)
- Predecessor (the knob being validated):
  [`docs/experience/wins/2026-05-07-bench-qwen36-topk-sweep.md`](../experience/wins/2026-05-07-bench-qwen36-topk-sweep.md)
