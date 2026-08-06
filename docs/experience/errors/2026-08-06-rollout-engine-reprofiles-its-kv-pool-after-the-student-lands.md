# The rollout engine re-profiles its KV pool after the student is resident

## Context

Re-running the agent-OPD rollout A/B after `4850d0dd7` raised
`mem_fraction_static` from a hardcoded 0.2 to a `--rollout-mem-fraction` flag
defaulting to 0.5.

The first task ran correctly — 2 samples, 10 and 13 agent turns, 26–29K
sequences, one clean group row:

```
gpu_busy_secs 96.995  rollout_secs 105.261  busy_frac 0.921
prompt_tokens 278892  completion_tokens 3622  rollout_tok_per_sec 34.41
```

Task 2 onward produced nothing. Each burned its full 183 s wall on
`API Error: 500 engine thread closed; cannot submit`.

## What actually happened

The rollout engine profiles its KV pool **twice**, and the two profiles see
very different cards.

| profile | when | free VRAM | reserve at F=0.5 | pool |
|---|---|---:|---:|---|
| 1 | engine construction, before the student loads | ~68 GB | 48.9 GB | real |
| 2 | prefix warm-up after round 0, student resident | **45.1 GB** | 48.9 GB | **floor** |

`reserve = total × (1 − F)` is a constant — 0.5 × 97871 MiB = 48935 MiB on the
H20. Free at the second profile is 46180 MiB (measured: 51691 MiB used of
97871). `46180 − 48935 < 0`, so the profile lands on
`PROFILE_KV_TOKENS_FLOOR = 4096` by construction, for the same reason 0.2 did
at the first profile. Raising the default moved the cliff; it did not remove
it.

The formula double-counts for a co-resident caller. `free_bytes` already
excludes everything resident — the student, its optimizer, the engine's own
weights. Subtracting a fixed share of **total** on top charges the engine for
memory it is not using, and the charge grows as its co-tenant grows.

Three defects chain from there:

1. **The re-profile is taken at the structurally worst moment.** Free VRAM is
   at its minimum right after the student lands, which is exactly when the pool
   gets resized.
2. **The floor path builds an inconsistent pool.** The device pool shrinks to
   the floor while the host pool keeps its requested size:
   `TokenKVPool::mirror_slot page 256 out of range 256 (host pool total_pages
   exceeds device pool budget?)`. That is a panic, not a degraded mode.
3. **agent-OPD treats a dead engine as recoverable.** It logs `prefix warm-up
   failed (continuing)` and continues into a state where no request can ever
   succeed. Three of four tasks ran to their full timeout producing zero
   tokens, and the run had to be killed by hand.

## Fix

`5ae88ee44` — **profile once, at construction; re-acquire at the recorded
size.** Neither candidate from the first analysis was right. Bounding against
`free` instead of `total × (1 − F)` changes sizing for every caller of
`profile_kv_pool_tokens`, serve included, which is a default flip needing its
own perf license. Not re-acquiring at all throws away the release, whose whole
point is handing that VRAM to the student.

The pool size is a decision, not a reading. It belongs to the moment the card
was in its known-good state:

- `alloc_full_attn_kv_pool(model, num_slots, pages, kv_format)` extracted — it
  allocates at an exact page count and does no profiling.
- `build_full_attn_kv_pool` profiles, then calls it, and returns
  `(pool, sized_pages)`.
- `kv_pool_sized_pages` replaces `kv_pool_mem_fraction_static` +
  `kv_pool_requested_pages`, which existed only to re-profile.
- `ensure_kv_pool` calls `alloc_full_attn_kv_pool` directly. An allocation that
  genuinely does not fit now errors, where it used to silently floor.

The host/device page mismatch needed no separate patch — it was downstream of
the shrink.

`b8d390bf3` fixes an unrelated fatality found while gating this: `list_dumps`
bailed on a missing dump dir while both callers already treat "no dumps" as a
skipped conversion. The dir defaults to `./dumps` relative to CWD, so anything
cleaning the build tree killed a run mid-round.

agent-OPD additionally bails when a group generates 0 completion tokens
(`5ae88ee44`).

### Gate

The fixed path only runs after `release_kv_pool`, which happens at the LoRA
sync — so a run that dies mid-round never touches it. The first attempt looked
clean (0 collapses, 0 engine deaths) and proved nothing: `re-acquired
full-attn KV pool` appeared **0 times**. The gate below forces the path and
asserts the size, not the absence of symptoms.

`--task-limit 2 --rounds 2 --samples-per-prompt 2`, `RUST_LOG=info`, own
`--eval-out-dir`, 1× H20 GPU 6, ThinkingCap-Qwen3.6-27B-FP8:

```
RUN_EXIT=0
construction_profile: sizing 19393 pages
re_acquires: 2
  re-acquired full-attn KV pool: 19393MB / 19393 pages
pool_collapsed: 0     engine_closed: 0     mirror_range: 0
groups=2   ZERO_TOKEN_GROUPS=0
  round=0 PyCQA__flake8    completion_tok=3150  gpu_busy=72.06  wall=78.31
  round=0 google__textfsm  completion_tok=2200  gpu_busy=66.84  wall=74.45
```

Both re-acquires land inside round 0, one per group's sync, so the pool was
genuinely released and rebuilt rather than short-circuiting on the idempotent
`full_attn_kv.is_some()` branch. 19393 pages restored against 19393 profiled;
before the fix the same call site produced the 4096-token floor (256 pages).
`groups=2` rather than 4 is task selection, not failure: `round 1:
task-selection ran=0/2 skipped=2` — both tasks were zero-variance in round 0.

## Rule

**A VRAM profile is only valid at the moment it is taken, and re-profiling a
long-lived pool re-runs the sizing arithmetic against a card that has changed
underneath it.** Any sizing formula that mixes a `free` reading with a `total`
constant will drift as co-tenants grow. Check what else is resident at every
point the profile runs, not just the first one.

**A gate that only counts absent symptoms passes on code that never ran.** The
first attempt reported 0 collapses, 0 engine deaths, 0 range errors — every
symptom of the bug, absent — while `ensure_kv_pool` was called zero times. The
run died before the round tail, which is the only place the fixed path is
reachable. Assert the treatment executed and produced the right value (19393
pages restored against 19393 profiled), never that the failure is missing.

**A warning that names a knob without printing the measurement cannot be acted
on.** The `KV pool collapsed to the 4096-token floor at mem_fraction_static
0.5` warn added in `4850d0dd7` fires at WARN while the free/total/cell numbers
sit in the adjacent INFO line, which is suppressed at the default log level.
The diagnosis needed `nvidia-smi` and arithmetic to recover numbers the process
already had. Put the operands in the message that fires.
