# Phase 3b PPL eval BLOCKED — `arle train eval` doesn't load multi-shard production models

## Context

Per `docs/plans/2026-05-10-rope-yarn-phase3b-ppl-eval-plan.md`,attempted
Phase 3b Bench A:vanilla Qwen3-4B 40k PPL baseline using `arle train eval`
on freshly-generated `bench-output/eval-longctx/eval-longctx-40960.tokenized.jsonl`(per
`scripts/gen_arle_longctx_eval.py` 0922e88,1.6M corpus tokens → 40k example)。

## What Failed

```
$ ./target/release/arle train eval \
    --model infer/models/Qwen3-4B \
    --data bench-output/eval-longctx/eval-longctx-40960.tokenized.jsonl \
    --seq-len 40960 --backend cuda

[ARLE train eval] error: failed to open safetensors file
  infer/models/Qwen3-4B/model.safetensors: No such file or directory (os error 2)
```

## Root Cause

`Qwen3-4B` ships as **multi-shard safetensors**:
```
infer/models/Qwen3-4B/model-00001-of-00003.safetensors
infer/models/Qwen3-4B/model-00002-of-00003.safetensors
infer/models/Qwen3-4B/model-00003-of-00003.safetensors
infer/models/Qwen3-4B/model.safetensors.index.json
```

`arle train eval`(per `crates/train/src/commands/train_multi_turn.rs:153`)
hardcodes `model.safetensors` (single file) — it's designed for ARLE's own
**trained checkpoints**(SFT / GRPO / multi-turn output dirs)which produce
a single `model.safetensors` after training。

For inference/serving,`infer/src/weight_loader.rs` handles multi-shard via
`model.safetensors.index.json` + per-shard load。But the **train eval path
doesn't reuse weight_loader**:it has its own simpler single-file loader。

This is a **legitimate train/eval-side limitation**,not a Phase 3 substrate
bug — production multi-shard model evaluation is outside `arle train eval`'s
intended scope。

## Alternative Phase 3b paths

### Path A — Convert multi-shard → single safetensors(one-time)

Use HuggingFace's `safetensors` Python tooling:
```bash
python3 -c "
from safetensors import safe_open
from safetensors.torch import save_file
import json, os
idx = json.load(open('infer/models/Qwen3-4B/model.safetensors.index.json'))['weight_map']
parts = {}
for name, fname in idx.items():
    if fname not in parts:
        parts[fname] = {}
shards = {}
for fname in set(idx.values()):
    shards[fname] = safe_open(f'infer/models/Qwen3-4B/{fname}', framework='pt')
weights = {n: shards[idx[n]].get_tensor(n) for n in idx}
save_file(weights, 'infer/models/Qwen3-4B/model.safetensors')
"
```

**LOC**:~10-20 LOC Python。**Wall-clock**:30-60s。**Risk**:doubles disk
usage temporarily(~16GB)。

### Path B — Use `infer` server logprobs API for PPL

Stream tokens through OpenAI completions API with `logprobs=True`,sum
per-token negative log-likelihoods,divide by token count → exp = PPL。

**Pros**:no disk ops,uses production multi-shard loader,can be done via
HTTP curl + Python jq。

**Cons**:streaming + completion forward ≠ teacher-forcing eval — token-level
losses depend on previously generated tokens,not ground truth conditioning。
Suitable for **generation PPL**,not strict **language modeling PPL**。

### Path C — Write `arle infer eval` surface(new code)

Reuse `weight_loader.rs` multi-shard loader path inside an eval-specific
binary:`arle infer eval --model <multi-shard-dir> --data <jsonl>`。

**LOC estimate**:200-400 LOC(new command + reuse forward path + per-token
loss extraction)。**Wall-clock**:codex pickup 1-2 days。

### Path D — Use Qwen3-0.6B or smaller single-file model

`infer/models/Qwen3-0.6B/`(if single-shard)— smaller model,faster eval,
but tests YARN math only on smaller scale。

## Recommendation

**Path A**(convert to single safetensors)— **fastest unblock**(30-60s)
+ no new code + tests YARN on production-shape model。Doubles disk briefly
but Qwen3-4B is only 8GB → 16GB temporarily,then can rm shard files if
single is preferred。

**Path B for production-grade PPL** — once Path A unblocks first eval pass,
Path B(server logprobs)gives a longer-term scalable eval path that
doesn't require multi-file conversion per model。

## Action items

- Path A unblock or defer Phase 3b execution to next user-direction tick
- Document this gap as `arle train eval` limitation(eval surface scope)

## Lesson

**Eval-surface ≠ inference-surface**。`arle train eval` and `arle infer
serve` use different loader paths(historical:train started single-file,
infer added multi-shard later)。**Document the limitation when scoping
Phase 3b** — should have caught this in the §6 "Memory feasibility check"
section of the plan(`eab591d`)by also noting "model loader compatibility"。

## Cross-references

- Phase 3b plan(now needs update):`docs/plans/2026-05-10-rope-yarn-phase3b-ppl-eval-plan.md`(eab591d)
- Eval data gen script:`scripts/gen_arle_longctx_eval.py`(0922e88)
- Phase 3a smoke PASS(production loader works):`docs/experience/wins/2026-05-10-phase3a-rope-yarn-server-smoke.md`(4efd30b)
- Eval lib path:`crates/train/src/eval_lm.rs`,`crates/train/src/commands/train_multi_turn.rs:153`

## 状态

Phase 3b first attempt blocked by `arle train eval` multi-shard
incompatibility。Path A(convert single safetensors)unblocks in ~1 min。
**Phase 3a substrate proven works**(server-side YARN inference);Phase 3b
quality eval just needs loader workaround,**not a YARN math issue**。
