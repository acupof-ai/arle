# 9B-TQ4 Generation Quality Gate KILL

## Context

The Qwen3.5-9B-TQ4 -> Qwen3.5-0.8B LoRA OPD bench passed the functional
rollout-4 gate:

- no OOM
- no NaN
- held-out KL monotonically decreased from `1.821073738029e-5` to
  `1.802543692975e-5`

Before using that run as the user-facing headline, we ran a multi-token
generation smoke on the same 9B-TQ4 teacher:

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

Requests used `/v1/completions`, `max_tokens=64`, and `temperature=0`.

Raw artifacts:

```text
bench-output/2026-05-21-qwen35-9b-tq4-generation-quality/
```

## Results

ARLE 9B-TQ4 completed all three requests without server crash:

| Prompt | Prompt tokens | Completion tokens | Assessment |
|---|---:|---:|---|
| `Hello, world! Tell me a short story about a small robot.` | 14 | 64 | incoherent multilingual/token fragments |
| `Explain on-policy distillation in two sentences.` | 10 | 64 | incoherent multilingual/token fragments |
| `Write a Python function that returns the Fibonacci sequence up to n.` | 13 | 64 | incoherent multilingual/token fragments |

Example ARLE snippets:

```text
_CMD生根发电机演变价格的不完वन_MAP才会aris Martin搏击西城区Digits贾 coup thi involvement/theme...
```

```text
的阅读刮 ´ αυτή先用 qualche怀柔 jeu pointed initiative Gibbs DubaiJur提示营造发热ண本题...
```

```text
光源 symlink优越纠结ходя Giuseppe personalized协商先天也不会 SOCI arch致辞...
```

PyTorch BF16 reference was also not a clean instruction-following baseline for
these raw prompts on this base checkpoint. It required CPU offload on the 16 GB
card (`layers.25-31`, final norm, and `lm_head` on CPU) and produced weak
outputs:

| Prompt | Completion summary |
|---|---|
| short robot story | repeated time/number fragments |
| OPD explanation | partial OPD words plus repeated zeros |
| Fibonacci function | 64 exclamation marks |

The PyTorch result means this smoke is not a strong natural-language quality
benchmark for the base 9B checkpoint. However, it still falsifies the proposed
9B-TQ4 headline switch: ARLE TQ4 output is visibly incoherent and not
substantively close enough to claim a user-facing generation-quality pass.

## Root Cause

The OPD KL gate and generation quality gate test different things. The 100-step
OPD run proves that the train loop can consume this teacher path and move held-
out KL monotonically on the configured prompt set. It does not prove the
teacher's generated text is good enough to headline.

The exact source of ARLE-vs-reference drift remains unresolved. Prior gates
showed:

- dense untied `lm_head.weight` module parity is fixed
- full-model logits top-64 relerr remained around `0.18`
- quantized projection GEMV parity was clean in tensor-local/layer-local tests

This smoke shows the remaining drift is functionally visible in free generation.

## Fix

Do not switch the public headline to 9B-TQ4 yet. Revert the user-facing
headline switch and keep the 9B-TQ4 rollout-4 bench as a functional OPD fit
result, not a generation-quality claim.

Next SOLID gate should be token-level generation parity, not another qualitative
smoke:

1. Run PyTorch BF16 and ARLE TQ4 on the same prompt and compare the top-64
   logits at every generated position.
2. Identify the first token where greedy argmax diverges.
3. At that token, run stage-local parity across embedding, linear attention,
   full attention, MLP, final norm, and lm head.

Only after that gate passes should 9B-TQ4 become the public headline teacher.

## Rule

For quantized teacher paths, OPD KL monotonicity is a functional train-loop
gate, not a substitute for generation-quality validation. Public headline
switches need a multi-token generation gate or a token-level parity gate first.
