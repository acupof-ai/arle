# Spec-decode concurrency gate — c=1 win kept, high-c loss gated out

> Phase exit (2026-07-26, `69560ae55` gate + env-deletion follow-up). Pod A/B
> PASS on both campaigns; `--spec-max-batch` default 1 is the shipped default.

## Context

MTP/DSpark speculative decode is a **c=1 win and a high-concurrency loss**,
measured on both models. The verify of B+1 draft positions is ~free while the
GPU is memory-bound (c=1) and costs ~(B+1)× step time once compute-bound (c≥4),
committing only ~2.5 tok either way. DSv4 already batches its spec verify and
still lost −48% at c=16 — batching harder cannot cross a compute-bound wall.
The dispatch was also three hand-rolled `dspark → mtp → plain` ladders (qwen35
rows==1, qwen35 rows>1 pure-serial `for row`, dsv4 B>1), none aware of the
crossover.

## What Worked

A pure `route_decode(spec_kind, n_rows, gate) -> {Plain,Mtp,Dspark}` in
`spec_decode.rs`; both executors call it. Speculate only at `n_rows ≤ gate`;
above it route decode to the plain batched path that scales. Gate rides
`--spec-max-batch` (default 1) through the existing CLI→`CudaRuntimeFlags`→atomic
path. Collapses the three ladders into one decision. `ARLE_DSV4_SPEC_DECODE`
env gate deleted in the follow-up — `--spec-type` is now the single opt-in.

### Campaign A — gate validation (DSv4-Flash-FP8, 4×H20 TP=4/EP=4, 128/128, max_tokens 128, seed 20260416, 60 s/pt, 0 err)

| c | no-spec | gate=1 (default) | Δ | gate=16 (spec always) | Δ |
|---|--------:|-----------------:|--:|----------------------:|--:|
| 1  | 42.4  | 44.7  | **+5.4%** | 44.6 | +5.2% |
| 4  | 78.9  | 78.1  | −1.0%     | 61.4 | −22.2% |
| 8  | 137.4 | 134.8 | −1.9%     | 76.7 | −44.2% |
| 16 | 174.3 | 173.8 | −0.3%     | 91.2 | **−47.7%** |

`spec_decode` chains at gate=1 stay flat (785→866, only c=1's chains) while
gate=16 scales with c (785→5188) — the gate routes c≥4 to plain, exactly as
designed. Raising the default to 4 would re-admit the c=4 −22% loss, so **1 is
the measured optimum**.

### Campaign B — #128 close (256-out champion-lineage set, max_tokens 256)

| c | no-spec | DSpark (gate=1) | Δ | accept_rate |
|---|--------:|----------------:|--:|------------:|
| 1 | 38.5 | 61.0 | **+58.4%** | 0.513 |
| 4 | 74.1 | 79.6 | +7.4% (gated → plain, noise) | 0.513 |

**#128 RESOLVED — the 07-20 +63.8% vs 07-25 +5% gap was the dataset, not a
second effect.** The 256-out set runs at accept_rate 0.51 vs the 128/128 set's
0.30 (~1.7× committed tok/verify step), so DSpark's c=1 win is ~11× larger
there. Same mechanism, different draft-friendliness.

## Rule

- **Speculative decode is a low-concurrency feature; gate it, don't batch it
  harder.** The only correct move at high concurrency is to stop speculating —
  a one-decision gate beats finishing a batched-spec-verify increment that would
  only help in the regime where spec loses anyway (YAGNI).
- **A spec-decode Δ is only comparable within one accept_rate.** The same code
  gave +5% and +58% on two datasets purely because draft-friendliness (accept
  rate) differs. Never compare spec gains across fingerprints.
- **One decision, one definition.** Three copies of a dispatch ladder is three
  places to forget the crossover; a shared pure fn is unit-tested without a GPU.
