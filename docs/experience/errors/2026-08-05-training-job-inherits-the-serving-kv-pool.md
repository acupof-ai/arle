# A training job inherited the serving KV pool and OOM'd on 356 MB

## Context

First real run of `arle train spec-draft` on the H20. It died at step 0:

```
cuda htod copy failed: shape=[5120, 17408] bytes=356515840
  err=DriverError(CUDA_ERROR_OUT_OF_MEMORY)
```

A 356 MB upload failing on a 95.6 GiB card.

## Root cause

`EngineLoadConfig::single_sequence(512)` sets `total_pages = 32`, which reads
like a small KV budget. It is a **minimum-capacity floor**. The pool is sized
separately from measured free VRAM by `mem_fraction_static`, whose serving
default is 0.9.

Measured, engine alone, same config and model:

| Point | VRAM |
|---|---|
| after weights | 29065 MiB |
| after KV pool | 87691 MiB (0.896 × total) |

The trunk claimed 58.6 GiB of KV pool to run one 512-token sequence. The draft
then uploaded 10.9 GiB of weights and hit the ceiling.

The doc comment on `mem_fraction_static` says it is "wired for the dense Qwen3
CUDA pool; Qwen3.5/3.6 and DSv4 keep their per-slot sizing this phase". The
measurement says otherwise for Qwen3.6.

## Fix

`arle train spec-draft` passes `mem_fraction_static` (flag
`--trunk-mem-fraction`, default 0.45). A training job runs one short sequence
through the trunk; the pool is waste.

## Rule

Two wrong conclusions were reached before the measurement, both from
instruments unfit for the quantity:

1. **A sampler slower than the process it samples is not a clock.** Polling
   `nvidia-smi` through an SSH wrapper costs seconds per sample; the labels
   `t=Ns` were fiction, and a 12-second window "proved" a 150-second model load
   never happened. Sample on the box, or don't claim a curve.
2. **An arithmetic budget is a hypothesis, not a measurement.** The ledger said
   40 GiB and the card held 97.5 GiB. The gap was a default nobody in the call
   chain names.

`total_pages` naming both a floor and, apparently, a budget is the trap. When a
config field looks like a cap, check whether anything else sizes the same
resource before trusting it.
