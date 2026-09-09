# qwen35 decode-graph invalidation converges on one policy, enforced by a hygiene gate

Date: 2026-09-10 · Lane: lane/d6 · Step 3a slice 2 (block 5 → `executor/device_sched.rs`, child module of qwen35)

## Context

The qwen35 executor invalidated its captured decode graph at six sites with
two operations: five whole-graph drops (`decode_graph = None`) and one
per-slot rebuild (`graphs[slot] = CudaGraphState::new(...)`). Each site
carried its own comment explaining why, and nothing enforced that a seventh
site would join the policy instead of writing `decode_graph = None`
directly. Block 5 of the qwen35 split moves this code into
`executor/device_sched.rs`; the move is the occasion to make the policy
explicit.

## What worked

- **Closed enum of reasons.** `DecodeGraphInvalidation` names the five
  whole-graph events (capture failure, weight offload, scratch release,
  LoRA re-merge, Markov-head swap).
- **The compiler, not the grep, enforces the field.** `decode_graph` is a
  `DecodeGraphSlot` newtype whose inner `Option` is private to
  device_sched; `invalidate(why)` is the only writer. `= None`,
  `.take()`, or an `&mut` handoff at any other site fails to compile —
  there is no next spelling to leak through. (A first cut enforced this
  with a grep for `decode_graph = None`; review found `Option::take()`
  — the standard way to drop an `Option` — sailed straight through.)
  Per-slot rebuilds still go through `rebuild_slot_graph`, since the
  reset touches `Qwen35DecodeGraph` internals the newtype cannot wrap.
- **The auto-guard stays where the drift is.** The workspace re-addresses
  on its own, so no event site can name that class: `stage_graph_step`
  compares the baked pointer set each step and rebuilds the one slot that
  drifted. The module doc states the two-shape split (explicit events vs.
  implicit drift) so the next editor knows the enum is complete by
  construction, not by omission.
- **The gate survives as the backstop.** `check_decode_graph_invalidation`
  still greps for per-slot `graphs[..] =` / `baked[..] =` resets (and the
  whole-graph spellings, now belt-and-suspenders) under
  `crates/infer-cuda/src`, excluding `executor/device_sched.rs`. Its
  selftest world proves the check can fail.
- **The `paged_decode_meta` clear stays at the capture-failure site.**
  `PageMeta::persistent_decode` allocates 8 device buffers of its own per
  slot, so weight offload and scratch release never dangle them; the clear
  pairs with the permanent `decode_graph_armed = false` disarm and is a
  resource reclamation, not a correctness fix. The comment now says so.

## Rule

A "the only way to do X is Y" invariant that lives in a comment degrades
to convention on the first bypass; a grep degrades on the next spelling
(`Option::take()` sailed through `decode_graph = None`). Enforce it with
a type whose privacy makes the bypass unrepresentable, and keep the grep
only for what the type cannot cover. When an invalidation policy has an
event class no call site can name (implicit state drift), put the guard
at the point that observes the drift and document why the enum is closed.

## Net

No baseline: refactor + policy enforcement, no measurement; test counts
and diff stat are mechanical facts.

`check_repo_hygiene.py --selftest` 9/9 green (new check fails on its
broken world); Mac CUDA clippy gate exit 0.
