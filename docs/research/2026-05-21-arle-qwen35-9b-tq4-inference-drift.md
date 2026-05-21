# Qwen3.5-9B-TQ4 Inference Drift Attribution

## Summary

The Qwen3.5-9B-TQ4 path is validated for OPD loop integration, but it is not
licensed for user-facing inference quality.

The decisive evidence is the 64-token greedy generation smoke in
`bench-output/2026-05-21-qwen35-9b-tq4-generation-quality/`: ARLE 9B-TQ4 loads,
serves, and completes requests, but the generated text is incoherent token
fragment noise. Therefore the previously observed full-model top-64 logit
relerr around `0.18` is functionally meaningful for inference, even though the
100-step OPD rollout-4 KL gate tolerated it.

Headline switch remains on hold. Commit `fc87bed` reverted the public
9B-TQ4 headline switch and recorded the KILL entry.

## Evidence

### OPD Functional Gate

The rollout-4 OPD bench passed:

- teacher: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4`
- student: `/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-0___8B-Base`
- LoRA: rank 16, `attention-qv`
- steps: 100
- rollout length: 4
- result: no OOM, no NaN, held-out KL monotonic

Held-out KL:

| Step | Held-out KL |
|---:|---:|
| 0 | 1.821073738029e-5 |
| 25 | 1.816574831537e-5 |
| 50 | 1.812112896005e-5 |
| 100 | 1.802543692975e-5 |

This licenses the train loop consuming the teacher path. It does not license
the teacher as a public inference-quality model.

### Generation Smoke

Serve command:

```bash
./target/release/arle serve \
  --backend cuda \
  --model-path /home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B-TQ4 \
  --port 8123 \
  -- \
  --num-slots 1 \
  --max-seq-len 256 \
  --chunked-prefill-size 128 \
  --max-num-batched-tokens 128
```

Request settings:

- endpoint: `/v1/completions`
- `max_tokens=64`
- `temperature=0`
- prompts: three short English prompts covering story, explanation, and code

ARLE completed all requests. The outputs were not coherent:

| Prompt | ARLE TQ4 output assessment |
|---|---|
| `Hello, world! Tell me a short story about a small robot.` | multilingual/token-fragment noise |
| `Explain on-policy distillation in two sentences.` | multilingual/token-fragment noise |
| `Write a Python function that returns the Fibonacci sequence up to n.` | multilingual/token-fragment noise |

Representative ARLE snippets:

```text
_CMD生根发电机演变价格的不完वन_MAP才会aris Martin搏击西城区Digits贾 coup thi involvement/theme...
```

```text
的阅读刮 ´ αυτή先用 qualche怀柔 jeu pointed initiative Gibbs DubaiJur提示营造发热ண本题...
```

```text
光源 symlink优越纠结ходя Giuseppe personalized协商先天也不会 SOCI arch致辞...
```

### PyTorch BF16 Reference Caveat

The PyTorch BF16 reference used the original
`/home/ckl/.cache/modelscope/hub/Qwen/Qwen3___5-9B` checkpoint with
`device_map=auto`. On the 16 GB card, Accelerate offloaded `layers.25-31`,
final norm, and `lm_head` to CPU.

The BF16 reference outputs were also weak on these raw prompts:

| Prompt | PyTorch BF16 output summary |
|---|---|
| story | repeated time/number fragments |
| OPD explanation | partial OPD wording plus repeated zeros |
| Fibonacci function | repeated exclamation marks |

This means the test is not a strong natural-language quality benchmark for the
base checkpoint. It is still enough to kill the ARLE 9B-TQ4 headline because
ARLE's outputs are visibly worse and not substantively close.

## Interpretation

The 0.18 full-model top-64 logit relerr should not be relaxed away for
inference. At generation time, small top-logit ordering shifts can compound
autoregressively: once greedy decode chooses a different token, every later
position is conditioned on a different prefix. The qualitative smoke shows that
the residual drift is not merely BF16-style numeric noise.

OPD is less sensitive in this specific experiment because its gate was
aggregate KL over a short prompt set and short rollout. That gate can decrease
even when free generation is unacceptable.

## Decision

Hold the public 9B-TQ4 headline switch.

Current user-facing headline should remain the validated Qwen3-0.6B LoRA/TRL
comparison plus the Qwen3.5-4B BF16 -> 0.8B cross-runtime OPD reference.

The 9B-TQ4 rollout-4 result remains useful as a functional fit result:

- it proves the server path fits on 16 GB in serve mode
- it proves the OPD loop can run against that teacher without OOM
- it does not prove inference quality

## Next Gate

Do not run another qualitative prompt smoke as the next root-cause step. Use a
token-level parity gate:

1. Run ARLE 9B-TQ4 and PyTorch BF16 on the same prompt.
2. At each generated position, compare top-64 logits and greedy argmax.
3. Identify the first generated token where argmax diverges.
4. At that exact prefix, run stage-local parity:
   - embedding
   - linear attention
   - full attention
   - MLP
   - final norm
   - lm head

License criterion for reconsidering the headline:

- greedy argmax matches PyTorch BF16 for the first 64 generated tokens on at
  least the three smoke prompts, or
- first divergence is explained and shown not to degrade a stronger external
  eval.

Until that gate passes, 9B-TQ4 remains an OPD-fit experiment, not the canonical
inference teacher.

## Optional 9B-Instruct Split Test Blocker

Follow-up proposal: download `Qwen/Qwen3.5-9B-Instruct` from ModelScope and run
the same three 64-token greedy prompts through ARLE BF16. This would separate:

- H1: ARLE's 24-layer / 4096-hidden Qwen3.5 forward path has a cumulative drift
  bug, so BF16 Instruct would also garble.
- H2: the base checkpoint is weak on raw chat-style prompts and TQ4 quantization
  pushes the teacher over the coherence threshold, so BF16 Instruct would be
  coherent.

Attempted command:

```bash
.venv/bin/python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download(
    'Qwen/Qwen3.5-9B-Instruct',
    cache_dir='/home/ckl/.cache/modelscope/hub',
    allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.jinja','merges.txt','vocab.json'],
))
PY
```

Result:

```text
Repo Qwen/Qwen3.5-9B-Instruct not exists on https://www.modelscope.cn,
will try on alternative endpoint https://www.modelscope.ai.
Repo Qwen/Qwen3.5-9B-Instruct not exists on either https://www.modelscope.cn
or https://www.modelscope.ai
requests.exceptions.HTTPError: <Response [404]>
```

So the proposed decisive split test is blocked as written: there is no
ModelScope `Qwen/Qwen3.5-9B-Instruct` repo available through `snapshot_download`
on this host.

Do not substitute the existing `Qwen/Qwen3.5-9B` BF16 source and call it
Instruct; that exact base checkpoint has already been used as the PyTorch BF16
reference above and produced weak raw-completion outputs.

Next-day alternatives:

1. Run the same smoke through ARLE and PyTorch using the checkpoint's chat
   template / non-thinking-mode prompt formatting instead of raw
   `/v1/completions` prompts.
2. Find a real ModelScope-available Qwen3.5-9B instruction LoRA or tuned
   checkpoint and repeat the BF16 serve smoke.
3. If neither is available, proceed with token-level parity on the existing
   BF16 base versus TQ4 path.
