# Metal KV cache grows on decode overflow — long generations no longer crash

## Context

`arle` REPL on `Qwen3.5-9B-MLX-4bit` (metal), long agent turn:

```
ERROR infer_server::execution: execution.rs:151 infer-server engine step failed:
  K/V slice token range [1024, 1040) exceeds shape=[1, 4, 1024, 256]
Error: engine thread closed before request 5 completed
```

The Metal executor reserves the flat K/V cache **once** at prefill
(`reservation = total_tokens + 512`, rounded up to a 256 multiple — here 1024)
and never grew it. Decode advanced `cache_len` unbounded; once it crossed the
reservation the run corrupted then crashed.

## Root Cause

The C++ session writes each step's K/V with `slice_update`
(`mlx_qwen35_model.cpp:890`), which returns a **same-shape** array — so the
session's K/V capacity is frozen at `begin_session` time and never grows on its
own. Two asymmetries made this fatal:

- The **host KV pool grows page-by-page** for arbitrarily long sequences (the
  failing slice reached page 64 → the host had ≥65 pages), but the executor's
  `kv_flat` stayed at the prefill reservation. Host and executor capacities
  diverged.
- The **prequeue path** (`executor.rs:598`) correctly *bailed* at capacity, but
  the **cold decode path** (`step_session`, `executor.rs:492`) had **no capacity
  check**. Past capacity, `slice_update`'s out-of-range write was silently
  dropped (KV truncated → corrupt attention), and `publish_slot` hard-errored at
  the next page boundary — the observed crash, delayed one page (1024 → 1040)
  because `full_pages = cache_len / page_size` only ticks at a page boundary.

The earlier prequeue bound was a partial fix; it left the cold path as the hole.

## What Worked

`MetalSlotState::ensure_kv_capacity(model, needed)`: when
`cache_len + needed > capacity`, drain the session (it owns the arrays), extend
each `kv_flat` array's seq axis (index 2) with zeros via a new testable
`grow_kv_seq_axis` helper (geometric doubling: `round_up_capacity(max(required,
2*capacity))`), `eval` to materialize, then let the caller re-activate the
session with the grown arrays. Buffer enumeration (§0.1): `kv_flat` grows;
`gdr_flat` is sequence-independent recurrent/conv state → untouched (matches
`materialize_slot_from_prefix`); `cache_len`/`committed_len` keep advancing;
`session_active` reset by drain, re-set by `ensure_session_active`. Wired into
the cold decode path and (defensively) prefill — every KV-writing site now
ensures capacity first.

**E2E evidence** (`Qwen3.5-0.8B-MLX-4bit`, `--temperature 0`, greedy):
`prompt_tokens=117` → initial capacity **768** (`117 + 512` → round-up). Forced a
1–1500 enumeration: `completion_tokens=1600`, `total_tokens=1717` — `cache_len`
reached **1717**, far past the 768 reservation *and* the 1040 crash point.
Crossed three growth boundaries (768→1536→3072). Output stayed **coherent
sequential integers** the whole way (no garbage past 768 — proves growth
preserves + extends KV, not just avoids the crash). Exit 0,
`max_turns_reached=false` (stopped on `max-tokens`, not a fault).

Steady-state decode is **unchanged by construction**: `ensure_kv_capacity`
early-returns (no drain, no alloc) whenever the reservation suffices, before
`ensure_session_active` — the 99.9% common decode path is byte-for-byte the
prior code. Growth is a geometric, rare event (a few ms drain+concat+eval per
doubling, ~2–3 times across 1600 tokens).

Tests: `cargo test -p infer-metal --features metal` 8/8 incl. two new growth
guards (`grow_kv_seq_axis_preserves_tokens_and_zero_pads_tail`,
`..._is_noop_when_capacity_met`); `cargo clippy --tests -D warnings` clean.

## Rule

The Metal executor's `kv_flat` is an executor-owned reservation, **not** the
host KV pool — they grow independently. Any path that advances `cache_len` past
the prefill reservation must grow `kv_flat` first; the C++ `slice_update` session
never grows it for you, and an out-of-range `slice_update` fails *silently*
(corrupt output) long before the eventual `publish_slot` slice hard-errors.
Every K/V-writing call site (prefill, cold decode, prequeue) needs a capacity
guard, not just one.
