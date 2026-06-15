# Metal DFlash Qwen3.6 gate + scheduler-row reachability

## Goal

Rebuild the rewrite Metal DFlash side path for the canonical
`mlx-community/Qwen3.6-35B-A3B-4bit` target with
`z-lab/Qwen3.6-35B-A3B-DFlash`, fail closed with no target-only fallback, and
then let scheduler batched rows reach the DFlash lane.

## Hypothesis

A conservative first landing can be correct before it is fast:

- single-request DFlash owns draft load/compat, target hidden capture, draft KV,
  target KV/GDR snapshot, block verify, accepted-prefix match, rollback, and
  hidden commit;
- prefix-cache attach must be disabled for DFlash until target-hidden/draft
  snapshots are persisted with prefix state;
- scheduler batched rows can first enter DFlash as serial verified blocks, with
  explicit logs and no performance claim.

## Params

- Binary: local `target/release/arle`, built with
  `cargo build --release --no-default-features --features metal,no-cuda`.
- Compile/test gates:
  - `cargo check -p infer-metal --release --no-default-features --features metal`
  - `cargo check -p infer-api --release --no-default-features --features metal,no-cuda --lib`
  - `cargo clippy -p infer-metal --release --no-default-features --features metal -- -D warnings`
  - `cargo test -p infer-metal --release --no-default-features --features metal`
- Serve:

```bash
RUST_BACKTRACE=1 \
INFER_METAL_DFLASH_DRAFT_MODEL=z-lab/Qwen3.6-35B-A3B-DFlash \
INFER_METAL_DFLASH_MAX_ROWS=4 \
INFER_METAL_WARMUP=0 \
RUST_LOG=info \
./target/release/arle serve \
  --backend metal \
  --model-path mlx-community/Qwen3.6-35B-A3B-4bit \
  --bind 127.0.0.1 \
  --port 8127
```

- Correctness gate:

```bash
RAW=1 TEMPLATE=qwen3_nonthink PORT=8127 \
MODEL=mlx-community/Qwen3.6-35B-A3B-4bit \
python3 scripts/needle_gate.py 2000,8000 3 0.0
```

`TEMPLATE=qwen3_nonthink` uses the checkpoint ChatML non-thinking assistant
prefix. The default Qwen3.6 chat template emits `<think>` and burns the
16-token needle budget, which is a template artifact rather than a retrieval
verdict.

## Env

- Host: local Apple Silicon Mac, 48 GiB unified memory.
- Resource guard at final serve: `available=35.0GiB`,
  `gpu_working_set=37.4GiB`, `swap_used=3044MiB`, `memory_limit=29GiB`,
  `wired=20GiB`, `kv_capacity_tokens=95824`.
- Metal KV dtype: `int8`.
- DFlash runtime: `block_size=16`, `max_rows=4`,
  `target_layers=[1, 10, 19, 28, 37]`.

## Results

Canonical Qwen3.6 long-context gate passed on the final binary:

| length | runs | exact | partial | miss | determinism |
|---|---:|---:|---:|---:|---|
| 2000 | 3 | 3 | 0 | 0 | DET |
| 8000 | 3 | 3 | 0 | 0 | DET |

Raw output excerpts:

```text
SUMMARY len=2000 depth=0.00 exact=3 partial=0 miss=0 DET
SUMMARY len=8000 depth=0.00 exact=3 partial=0 miss=0 DET
```

Scheduler-row reachability passed with four concurrent Qwen3.6 DFlash requests
(`max_tokens=48`). All four returned HTTP 200 with `completion_tokens=48`.
Executor logs proved the row cap was actually opened and the DFlash lane saw
multi-row plans:

```text
Metal DFlash scheduler-mixed lane live: prefill_rows=2, decode_rows=2
Metal DFlash scheduler-row lane live: rows=4 (serial verified blocks)
```

This is a correctness/reachability result, not a performance result. The current
multi-row DFlash lane still serially verifies each row's block; true speedup
requires replacing the internal row loop with the existing C++ batched verify
entrypoint. The follow-up performance license was killed by same-prompt A/B
and accepted-token trace:
[`2026-06-14-metal-dflash-big-win-not-licensed.md`](../errors/2026-06-14-metal-dflash-big-win-not-licensed.md).

## Problems

- Repeated needle prompts initially failed after the first run because host
  prefix cache attached a target KV/GDR-only prefix directly into DFlash decode.
  DFlash also needs the target-hidden feature store and draft KV state, so the
  path now returns zero reusable prefix pages while DFlash is enabled.
- Concurrent requests initially failed on multi-prefill and mixed
  prefill+decode plans. DFlash now accepts those scheduler shapes with preflight
  fences and serial execution.

## Learnings

- DFlash opt-in must treat prefix reuse as an all-or-nothing side-path snapshot:
  KV/GDR-only prefix reuse is not safe.
- Opening scheduler rows before true batched kernels is still useful if it is
  labeled as reachability and remains fail-closed. It proves the frontend,
  engine-core planner, and executor row contract before optimizing the inner
  verify loop.
