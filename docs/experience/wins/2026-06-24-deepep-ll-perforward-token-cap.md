# deepep_ll per-forward token cap — long prompts under concurrency no longer crash

## Context

After the SPMD B fix made deepep_ll EP boot + serve
([entry](2026-06-24-spmd-multiproc-ep-fixed.md)), short-prompt concurrency worked
but a **long** prompt crashed all workers: `deepep.rs:605` asserts `owned_n` (the
forward's `seq_len`) `<= num_max_dispatch_tokens_per_rank` (the LL dispatch buffer
cap, default 256, ≤1024). A prompt longer than the cap — or a long-prompt mixed
forward — made the chunked-prefill chunk (default `chunked_prefill_size=2048`)
exceed the cap → assert → worker group dies → HTTP 500.

## Fix (commit `fda0c634`)

Cap the planner's per-forward token count at the executor's deepep_ll limit:

- `BackendExecutor::max_tokens_per_step()` (new seam method, default `usize::MAX`):
  max total plan tokens (decode rows + prefill chunk) per forward.
- CUDA executor returns the cap via `Dsv4Model` → `DeepEpTransport`
  (`max_owned_tokens_per_forward()` = `Some(num_max_dispatch_tokens_per_rank)` when
  low-latency, else `None`); non-DSv4 / non-LL → `usize::MAX`.
- Planner: `budget = prefill_step_budget().min(cap - decode_rows.len())`,
  `chunk_cap = prefill_chunk_size().min(cap)` → `decode + Σ(prefill chunks) ≤ cap`.
- `num_slots` clamped to `cap` so a pure-decode forward never exceeds it either.

**No-op for every non-deepep_ll path** (cap=`MAX` → saturating min/sub are no-ops):
TP=1, Metal, Qwen, allreduce/intranode DSv4 are byte-unchanged.

## What Worked (8×H20 pod, EP=4 GPUs 4-7, `max_tok=512`, arena 4096)

- **Long ~2000-tok prompts under concurrency: c=1/4/8 all ok=N/N** (was the
  "owned tokens 904 exceed 512" worker crash → HTTP 500).
- **Zero "owned tokens exceed" in the worker logs** — the planner chunks prefill to
  ≤ cap, so `owned_n` never exceeds the LL buffer.
- Needle correctness preserved (found at all lengths; partial/NONDET at 180/241 is
  the MoE non-determinism floor, within envelope).
- Boot READY ~18 s.

## Rule

A backend with a hard per-forward token limit must advertise it through the seam
(`max_tokens_per_step`) so the device-neutral scheduler caps the chunked-prefill +
mixed-batch token count — never let the executor assert/crash on a plan shape the
scheduler could have bounded. Gate the constraint so the default (`usize::MAX`) is a
byte-for-byte no-op on every other path.
