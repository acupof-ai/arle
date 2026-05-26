# P4 — V100 sm_70 perf bench (deferred)

**Status:** Deferred 2026-05-26 — vLLM upstream does not register the
`Qwen3_5ForConditionalGeneration` hybrid linear-attention architecture
that V100 P3 capability bench validated. Apples-to-apples comparison
needs a different model class.
Owner: ckl. Cross-link: [V100 sm_70 P3.1](../experience/wins/2026-05-25-v100-sm70-p3-1-capability-qwen35-4b.md) ·
[V100 sm_70 P3.2](../experience/wins/2026-05-25-v100-sm70-p3-2-capability-qwen35-9b.md) ·
[sm-coverage policy](sm-coverage.md).

---

## 1 · Why deferred

Original P4 acceptance criterion: **ARLE ≥ 80% of vLLM sm_70 baseline on
V100, same model, same prompt shape.**

Blocker uncovered 2026-05-26:

- ARLE V100 capability work (P3.1/P3.2) standardised on
  `Qwen/Qwen3.5-4B` (modelscope id), config
  `architectures: ["Qwen3_5ForConditionalGeneration"]`,
  `model_type: "qwen3_5"`, `layer_types: ["linear_attention",
  "linear_attention", "linear_attention", "full_attention", ...]`
  — hybrid linear-attention + full-attention stack, attn-output-gate, RoPE
  `head_dim=256`.
- vLLM 0.10.0 `vllm/model_executor/models/registry.py` only registers
  `Qwen3ForCausalLM` (vanilla GQA) and `Qwen3MoeForCausalLM`. No entry
  for `Qwen3_5ForConditionalGeneration`. Verified via
  `unzip -p vllm-0.10.0-*.whl vllm/model_executor/models/registry.py |
  grep Qwen3`.
- vLLM cannot load Qwen3.5 → no apples-to-apples on the model the P3
  capability work used.

Shipping P4 today would require either (a) downloading a vanilla
`Qwen/Qwen3-4B` to V100 and benching both engines on that, accepting
that the number does not describe Qwen3.5-on-V100 performance; or (b)
dropping the vLLM comparison entirely. Neither is the original P4
contract. Defer and revisit with a clear model choice.

## 2 · Acceptance options for revival

| Option | Apples-to-apples? | Validates V100 Qwen3.5 work? | Effort |
|---|:---:|:---:|---|
| **(A)** Two benches: ARLE Qwen3.5-4B canonical (V100-only narrative vs L4 reference) **plus** ARLE+vLLM Qwen3-4B (apples-to-apples) | yes (Qwen3-4B leg only) | yes (Qwen3.5 leg) | full; download Qwen3-4B ≈5 GB, run two sweeps |
| **(B)** Drop Qwen3.5, bench Qwen3-4B on both | yes | no | medium; one sweep per engine |
| **(C)** ARLE Qwen3.5-4B canonical only, no vLLM | n/a | yes | small; one sweep |
| **(D)** Wait for vLLM to register `Qwen3_5ForConditionalGeneration` upstream | yes (when it ships) | yes | zero today; unknown ETA |

**Recommendation when reopened:** (A). It gives both narratives the
honest answer they need — V100 Qwen3.5 perf in isolation, and ARLE vs
vLLM perf on a model both can serve. (B) sacrifices the V100 Qwen3.5
result that P3 paid for. (C) drops the competitive framing P4 was
designed to deliver. (D) is the only option that lets us bench Qwen3.5
directly against vLLM, but it depends on upstream timing we don't
control.

## 3 · V100 state already prepared

The P4 attempt 2026-05-26 left the following on V100 — reuse on revival:

- `~/bench_p4_venv` — Python 3.11 venv with
  `guidellm==0.6.0[recommended]` + transformers + httpx, installed via
  the project's `bench` extra. Smoke + quick presets validated against
  ARLE serve on `localhost:8000`.
- `/tmp/vllm_check/vllm-0.10.0-cp38-abi3-manylinux1_x86_64.whl` —
  downloaded but not installed; safe to delete or reuse.
- `jq` installed system-wide (`apt-get install jq`); required by
  `scripts/bench_guidellm.sh`.
- `scripts/bench_guidellm.sh` now respects `GUIDELLM_OUTPUTS` env var so
  offline boxes can skip the HTML finalize that hangs on V100
  (commit [`b0762414`](https://github.com/cklxx/arle/commit/b0762414)).
  Usage: `GUIDELLM_OUTPUTS="json csv" scripts/bench_guidellm.sh …`.
- `unset http_proxy https_proxy NO_PROXY no_proxy` required before any
  guidellm run — corp proxy env intercepts `localhost:8000` and fails
  `httpx.InvalidURL: Invalid port: ':'` on the validate-backend probe.

ARLE infer launch (proven working for P3 capability + P4 quick):

```bash
./target/release/infer \
  --model-path /home/chenkailun.c/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B \
  --port 8000 --num-slots 16 --max-seq-len 5120
```

vLLM launch sketch (untested — depends on which model option is chosen):

```bash
~/vllm_venv/bin/vllm serve <model-path-supported-by-vllm> \
  --max-num-seqs 16 --max-model-len 5120 --port 8001 \
  --dtype bfloat16
```

## 4 · Sweep findings already in hand

ARLE quick sweep 2026-05-26 (512-in/128-out, c=1,2,4,8, 60s each,
warmup 5s) on Qwen3.5-4B V100:

| c | succ/total | req/s | in_tok/s | out_tok/s | lat p50 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 24/24 | 0.418 | 214.5 | 53.5 | 2321.5 |
| 2 | 8/8 | 0.109 | 56.0 | 14.0 | 18145.3 |
| 4 | 12/16 | 0.218 | 111.9 | 27.9 | 19539.0 |
| 8 | 17/24 | 0.291 | 149.2 | 37.2 | 22835.6 |

Output throughput **degrades** c=1 → c=8 (53.5 → 37.2 tok/s) and c=2 is
the worst — opposite of healthy continuous-batching scaling on T1
hardware. Hypotheses to investigate on revival:

- Scheduler stall under Volta-specific kernel mix (sm_70 fallback paths
  may not pipeline the way T1 sm_80+ paths do).
- Hybrid `linear_attention` layers may have per-layer state that
  serialises across the batch dimension under the V100 path.
- 7 incomplete at c=8 → 60s ceiling is hitting the long tail; longer
  per-c duration may surface different steady state.

Raw artefacts on V100:
`~/code/agent-infer/bench-output/2026-05-26-v100-sm70-arle-quick/`
(`benchmarks.json` + `.csv` + `service_stats_trace.jsonl`).

## 5 · Why not now

- OPD Route B per-step optimization is higher-value next work — codex
  Phase 1 audit landed [`c2db68ec`](https://github.com/cklxx/arle/commit/c2db68ec)
  with a strong hypothesis (train-side linear-attention forces host
  fallback → likely 538 s backward main cause). Phase 2 needs V100 GPU.
- Even with option (A), P4 takes ~1.5 hours wall-clock and burns V100
  GPU that the OPD work needs.
- Without a model choice the user has signed off on, doing P4 now
  produces a wins entry that frames a different acceptance criterion
  than the original spec — likely will get reopened.

## 6 · Reopen trigger

Reopen when **any** of:

1. User picks option (A)/(B)/(C) on §2 explicitly — execute that
   leg on the prepared V100 environment.
2. vLLM upstream registers `Qwen3_5ForConditionalGeneration` (track
   `vllm/model_executor/models/registry.py` on vLLM main) — (D)
   unblocks.
3. OPD per-step optimization closes and we want a competitive
   inference number to publish alongside the OPD numbers.
