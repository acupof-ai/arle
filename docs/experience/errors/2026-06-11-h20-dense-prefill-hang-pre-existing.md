# H20 dense-Qwen3 prefill hang (pre-existing) blocks runtime verification of prefix attach

## Context

Verifying the 2026-06-10 prefix-attach fix (`0f80fdd6`) on the 8×H20 pod with
Qwen3-0.6B (the only dense checkpoint present): engine builds in ~5 s and a
5-token completion works, but EVERY prompt ≥ ~500 tokens hangs the engine
thread at 100% CPU (spin-sync inside the step; server stops answering).
Bisect on the PRE-fix binary (`43571d6e`, so not caused by the fix), fresh
serve per size, 90 s budget each:

```
reps=50/100/150/190/204/210/225 (~500..2250 tok) → ALL HANG   (graph OFF)
same hang with INFER_CUDA_DECODE_GRAPH unset (graph on)        (default)
"Hello" (≈5 tok) → PASS                                        (both)
```

## Root Cause

OPEN — hypotheses, in order:
1. TileLang HD128 dense prefill-attention kernel mis-runs at seq beyond a
   small bound on sm_90a (the dense lane is V100-routed; H20 never exercised
   it — `project_infer_rewrite_and_verification_routing`).
2. Some other batch op in the prefill path spinning on H20.
KILLED: decode-graph capture at num_pages>1 (hang persists with
`INFER_CUDA_DECODE_GRAPH=0`); chunked-prefill continuation (sub-chunk 500-token
single-chunk prompts hang too).
Next cheap discriminators: bisect 5..500 (kernel tile bound would show a crisp
edge); `RUST_LOG=info,infer_core=trace` to see whether submit returns;
cuda-gdb/nsys on the spinning step.

## Fix

None yet. The prefix-attach runtime verification (crash-repro on the old
binary + TTFT gain on the new) moves to the V100 lane where dense Qwen is
routed; H20 keeps DSv4/Qwen3Moe verification.

## Rule

- A lane that "works" has only been proven on the SKU it is routed to; running
  it on another SKU is a fresh bring-up, not a formality — budget a smoke
  (tiny + mid + long prompt) before building experiments on top.
- When an engine thread burns 100% CPU with no log advance, suspect spin-sync
  inside a stuck device op; discriminate with a graph-off run and a prompt-size
  bisect BEFORE blaming the newest feature on the box.
