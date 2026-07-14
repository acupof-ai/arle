# DSv4 DSpark acceptance and correctness

> Status: Shipped

## Goal

Restore coherent DSv4 DSpark decoding and non-zero acceptance on the production TP=4 path.

## Hypothesis

Three independent contract violations caused the failure: HC lanes were split instead of mean-reduced, BF16 Markov matrices were re-quantized to FP8, and target verification truncated committed state without folding the accepted prefix back into the recurrent rings.

## Params

- Model: DeepSeek-V4-Flash, DSpark draft
- GPU: 4× H20, TP=4, GPUs 3–6
- Draft block: 5, greedy, acceptance threshold 0
- Output: 128 tokens
- Prompt: `Write a brief history of computing.`
- Server log: `/host/dspark_tp4_final.log`
- Response: `/host/dspark_final_result.json`

## Env

- CUDA/NCCL release-fast build on the deployment pod
- DSpark opt-in; no default flip
- Target verify used the contiguous path because the deployed FlashMLA path does not support sparse chain verify

## Results

| Variant | Context rows for 10 prompt tokens | Accepted / opportunities | Output |
|---|---:|---:|---|
| Old lane split + FP8 Markov | 40 | 1 / 145 | Degenerate |
| HC lane mean only | 10 | 0 / 145 | Degenerate |
| Mean + BF16 Markov, stale target state | 10 | 38 / 110 | Degenerate |
| Final: mean + BF16 Markov + accepted-prefix fold | 10 | 61 / 170 (35.9%) | Coherent 128 tokens |
| No-spec control | n/a | n/a | Coherent 128 tokens |

Final accepted depths across 34 blocks:

`[2,2,0,3,2,0,1,3,2,1,1,1,2,1,2,1,2,4,0,3,3,0,5,0,3,2,5,0,1,4,4,0,0,1]`

The final output continued the requested history with early computers, programming languages, and the internet. A depth-zero control also produced coherent output, isolating accepted-prefix state restoration from draft quality.

## Problems

- Mean reduction was necessary but did not explain acceptance by itself.
- Aggregate acceptance initially hid target-state corruption; decoded per-step argmaxes exposed it.
- Sparse verify failed with `DSv4 chain verify requires the FlashMLA sparse prefill path`; contiguous verify now persists normalized rows, truncates, restores boundary rings, and folds only the committed prefix.
- Canonical GuideLLM throughput is deferred: this run licenses correctness only, not a performance claim or default flip.

## Learnings

DSpark requires all three contracts simultaneously: mean-reduced HC taps, native BF16 Markov weights, and recurrent-state equivalence after verification. Acceptance alone is not a correctness gate.
