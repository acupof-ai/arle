# Spec-decode concurrency gate — unified dispatch refactor

> Status: Proposed (2026-07-25). One design doc, `explanation` mode.
> Owner decision pending before code.

## Verdict first

Speculative decode (MTP / DSpark) is a **c=1 / low-concurrency win and a
high-concurrency loss** — measured, both models, root-caused. The fix is **not**
"make spec batch harder"; it is a **one-decision gate**: above a batch-size
threshold, route decode rows to the plain batched path that already scales.
The gate is the same knob for both backends, so the refactor also **collapses
three copies of the spec-dispatch ladder into one shared decision**.

- Zero new plumbing: `plan.decode_rows.len()` is already at the dispatch point.
- Zero new mechanism: threshold rides the existing `--mtp-adaptive`
  CLI→`CudaRuntimeFlags`→atomic path.
- Default `spec_max_batch = 1`: only true c=1 keeps spec; everything else is
  byte-identical to today's plain decode.

## Why (the measured wall, not a guess)

`6aa4ca6d1`, 128-in/128-out, `out tok/s`:

| c  | DSv4 no-spec | DSv4 DSpark | 27B no-spec | 27B DSpark |
|----|-------------:|------------:|------------:|-----------:|
| 1  | 42.4         | **44.5 (+5%)**  | 38.6    | **60.8 (+57%)** |
| 4  | 79.5         | 61.0 (−23%) | 75.1        | 52.1 (−31%) |
| 8  | 136.4        | 76.6 (−44%) | 126.0       | 52.5 (−58%) |
| 16 | 174.5        | 90.7 (−48%) | 150.9       | 53.0 (−65%) |

The crossover is one mechanism, not a bug:

```
each spec step:  draft block=B  →  target verifies B+1 positions  →  commit ~2.5 tok
                                          │
              ┌───────────────────────────┴───────────────────────────┐
        c=1: batch small                                        c≥4: batch full
        GPU MEMORY-bound                                        GPU COMPUTE-bound
        idle compute absorbs the                                B+1 positions cost
        B+1 verify positions ~free                              ~(B+1)× step time
              │                                                         │
        2.5 tok / ~1× time  →  WIN                       2.5 tok / ~6× time  →  LOSS
                                                        ratio 2.5/6≈0.42 (measured 0.52)
```

This is **physical**: DSv4 already batches its spec verify
(`dspark_decode_tokens_batched`) and still loses −48% at c=16. Batching the
draft/verify harder cannot cross a compute-bound wall. The only correct move at
high concurrency is to **stop speculating**.

## The bug being fixed, precisely

Two distinct problems, one gate closes both:

1. **qwen35 never batches spec** (`executor/qwen35.rs:2540-2553`): multi-row
   decode runs `for row in decode_rows { dspark_decode_row / mtp_decode_row }`
   — pure serial, one request per step. 27B MTP is flat 40 tok/s c1→c16
   (no cross-request batching at all). Comment admits it: *"batched spec verify
   is a later increment."*
2. **Even batched spec loses at high c** (dsv4, measured above). So "finish the
   later increment" is the wrong investment — the gate makes it unnecessary.

Both resolve identically: at `decode_rows.len() > spec_max_batch`, take the
plain batched decode path (`submit_decode_batch`) that already scales.

## Architecture — before / after

### Before: three hand-rolled dispatch ladders

```
                    infer-core Engine::step
                    builds ForwardPlan { decode_rows }
                              │  BackendExecutor::submit (seam)
              ┌───────────────┴───────────────┐
              ▼                                 ▼
   qwen35.rs submit()                  dsv4.rs forward_decode_batch_inner()
   ┌──────────────────────┐            ┌──────────────────────────────┐
   │ rows==1:             │            │ rows==1: forward_decode_row  │
   │   dspark? → mtp? →   │  ← ladder  │ rows>1:                      │
   │   plain   (copy #1)  │    #1,#2   │   dspark? → batched verify   │
   │ rows>1:              │            │   mtp?    → spec_step_batched│  ← ladder #3
   │   dspark? → FOR row  │            │   else    → plain batch      │
   │   mtp?    → FOR row  │  ← SERIAL  └──────────────────────────────┘
   │   else    → batch    │    (bug)
   └──────────────────────┘
   no concurrency gate anywhere — spec runs at every batch size
```

Three copies of `dspark → mtp → plain`, none aware of the crossover.

### After: one shared gate decides the route, backends execute it

```
                    infer-core Engine::step
                    builds ForwardPlan { decode_rows }
                              │  BackendExecutor::submit (seam)
                              ▼
        ┌─────────────────────────────────────────────────┐
        │  route_decode(spec_kind, n_rows, spec_max_batch)  │  ← ONE pure fn,
        │    n_rows > spec_max_batch  → Plain               │    device-neutral,
        │    spec_kind == DSpark      → Dspark              │    in spec_decode.rs
        │    spec_kind == Mtp         → Mtp                 │
        │    else                     → Plain               │
        └─────────────────────────────────────────────────┘
                    │            │            │
              ┌─────┘            │            └─────┐
              ▼                  ▼                  ▼
          Plain              Dspark              Mtp
   submit_decode_batch   dspark_decode_*    mtp_decode_*
   (already scales)      (c=1 win path)     (c=1 win path)
              ▲
   qwen35 + dsv4 both call route_decode() → same decision, one definition
```

`route_decode` returns an enum; each executor matches on it. The
`dspark→mtp→plain` priority ladder is written **once**, not three times.

## Change set (deletion-style, file:line)

| # | File | Change | Kind |
|---|------|--------|------|
| 1 | `infer-cuda/src/executor/spec_decode.rs` | Add `enum DecodeRoute { Plain, Mtp, Dspark }` + `route_decode(spec_kind, n_rows, gate) -> DecodeRoute`. Pure, no device types. | +add (the one new abstraction) |
| 2 | `infer-cuda/src/executor/qwen35.rs:2482-2553` | Replace **both** hand-rolled ladders (rows==1 and rows>1) with a `match route_decode(...)`. The serial `for row` spec loops stay reachable only for `Dspark`/`Mtp` routes (i.e. `n_rows ≤ gate`). | −collapse 2 copies → 1 |
| 3 | `infer-cuda/src/executor/dsv4.rs:2276-2301` | Route via the same `route_decode`; the `n_rows > gate` case falls straight to the existing plain batched decode. | −collapse copy 3 |
| 4 | `infer-seam/src/runtime_flags.rs:~102` | Add `pub spec_max_batch: usize` beside `mtp_adaptive`, default 1, with a `d_spec_max_batch()` serde default. | +field (mirrors existing) |
| 5 | `infer-cuda/src/runtime_flags.rs:~64,~128` | Add `SPEC_MAX_BATCH: AtomicUsize` + store in `apply_runtime_flags` + `spec_max_batch()` getter. | +field (mirrors `fa3_decode_splits`) |
| 6 | `cli/src/serve.rs` (ServeArgs + resolve) | Add `--spec-max-batch <usize>` (default 1), thread into `CudaRuntimeFlags`. | +flag (mirrors `--mtp-adaptive`) |

Net: **one new pure function + one config field threaded the standard way**;
**two of the three dispatch ladders deleted**. Lower entropy than today.

## Correctness invariants (must hold)

1. **Token cardinality switch is intended.** `dspark_decode_row`/`mtp_decode_row`
   emit `1..=B+1` tokens/row; `submit_decode_batch` emits exactly 1/row. Above
   the gate we deliberately switch to 1-token/step. The scheduler already
   handles variable tokens/step (spec path emits multiple today), so no
   stop/finish logic changes — but the smoke gate must confirm no dropped or
   duplicated tokens at the boundary.
2. **Default path byte-identical.** `spec_max_batch=1` + no spec flag → the
   `Plain` route is exactly today's plain decode; `route_decode` is a no-op
   rename for the non-spec case.
3. **No graph-capture perturbation.** The B=1 CUDA decode-graph lane is gated to
   `rows==1` plans and is *below* any sane gate; `route_decode` only redirects
   `rows>1` spec traffic, never touches the captured B=1 lane.
4. **Backend isolation.** `route_decode` takes a `spec_kind` enum + two `usize`,
   returns an enum — no CUDA types, lives above device code, safe to share.
5. **Mid-stream flip safe.** A slot may be spec at c=1 then plain at c=8 on the
   next tick. Plain decode reads the committed KV; the spec ring is only read by
   the spec path, so an unused ring tail is inert (self-heals on next spec step).
   Verify: needle ×1 after a concurrency ramp.

## Verification

- **Build/typecheck:** Mac cuda-lane (`cargo check -p infer-api … cuda,no-cuda`).
- **Correctness:** needle ×1 per backend, and a c=1→c=16 ramp (flip boundary).
- **Perf (the acceptance gate):** re-run the 6-arm 128/128 sweep. Expected:
  c=1 unchanged (spec still on), **c≥4 spec arms now == no-spec** (gated to
  plain), not net-negative. That IS the win — the loss columns disappear.
- **Bench entry:** `docs/experience/wins/` on the clean re-measure; `pending-remote`.

## What this explicitly does NOT do

- Does not implement batched spec verify for qwen35. The gate makes it moot —
  the only regime where batching would help (c≥4) is exactly where spec loses
  anyway. YAGNI.
- Does not change the c=1 spec paths. They win; untouched.
- Does not add a per-model threshold. One `spec_max_batch` for all; the crossover
  is at c≈2–4 for both models measured, so a single small default fits. Tunable
  if a future model's crossover differs.
