# State Plane — the slot-extent propagation contract (`KvLayout` is not enough)

> Root-cause treatment of the state-plane gap the attention/KV doc names but
> does not solve ([`attention-kv-architecture.md`](attention-kv-architecture.md)
> lines 73-77 flag it OPEN). Every claim below is source-verified against HEAD
> `621c84c4` (2026-05-31). Companion:
> [`control-plane-scheduler-phase.md`](control-plane-scheduler-phase.md).
> Status: **approach-first artifact, awaiting sign-off.** Cross-cutting (>5
> files); per root `AGENTS.md` no runtime code lands until accepted. The contract
> trails a **real consumer** — the env-gated FP8 shared decode KV pool meeting
> `--num-slots>1` — not a hypothetical one
> (`memory/feedback_no_speculative_interface_shaping`).

## TL;DR — there are two state stores for one fact, with no contract between them

The "data state store" of a request is **not** one object. It is split, and the
split is correct — except for one missing wire. Source-verified:

- **`TokenKVPool`** (`crates/cuda-kernels/src/paged_kv.rs:34`) owns the
  **authoritative** per-slot extent: `seq_lens: Vec<usize>` (`:84`) and
  `page_indices: Vec<Vec<u32>>` (`:82`). It is mutated at exactly three sites —
  `alloc_tokens` (`:609`, `seq_lens[slot] += count`), `attach_pages` (`:681`,
  `= token_count`), `free_slot` (`:1168`, `= 0`) — and read via `seq_len(slot)`
  (`:1278`).
- **`DeepseekBatchDecodeBuffers.fp8_kv_pool`** (`infer/src/model/deepseek/
  batch_decode.rs:51`, `Option<CudaSlice<u8>>`) is a **shadow** store. Its byte
  offset (`:165-166`) is
  `(slot_idx * fp8_kv_layers + layer_idx) * (fp8_kv_slot_blocks * BLOCK_BYTES)`
  — computed from the **fixed per-slot capacity** `fp8_kv_slot_blocks`, **never
  from the live `seq_lens[slot]`**.

The offset arithmetic is stable (good — a slot's byte window never moves). The
gap is the **live extent**: how many tokens inside that window are valid. That
number lives in `TokenKVPool.seq_lens[slot]`, and the model's pack/decode
kernels (`weights.rs:1772` pack, `:3623` decode, gated by
`dsv4_shared_kv_pool_enabled` at `:6530`) need it to know how much to pack/read.
**There is no contract that hands the current `seq_lens[slot]` to the fp8
pool's pack/decode at the right time.**

Failure mode this produces (the reason `ARLE_DSV4_SHARED_KV_POOL` × multi-slot
is unvalidated): a slot freed by request A (`free_slot` → `seq_lens=0`) and
re-admitted as request B grows its extent in `TokenKVPool`, but the fp8 byte
window still holds A's stale bytes until B overwrites them token-by-token. Any
pack/decode that reads `capacity` instead of B's live extent reads **stale
cross-request KV**. Two stores, one fact, no sync → silent correctness risk
exactly at the multi-slot boundary.

## Why `KvLayout` (as drafted) does not close it

The attention/KV doc proposes `KvLayout` as a CUDA-bound **page-lifecycle**
trait — `alloc_tokens` / `attach_pages` / `alloc_detached_pages` / `free_slot` /
`budget_bytes_for_tokens` (`attention-kv-architecture.md:133-153`). That
correctly unifies the *pool surface* (scheduler stops reaching into
`TokenKVPool` directly). But every method on it is a **page** operation. **None
of them lets an external shadow owner learn that `seq_lens[slot]` moved.** The
doc itself says so at lines 73-75 ("no method to notify/query slot extent") and
defers it at line 166 ("Later"). So `KvLayout` collapses the alloc/attach/free
*duplication* — real, worth doing — but leaves the *cross-owner propagation*
untouched. That is the actual state-plane gap, and it is what blocks the only
state-plane improvement currently in flight.

## The grounded boundary — extend `KvLayout` with an extent contract

Add to `KvLayout` exactly the minimum that the FP8-pool consumer needs, and
nothing more:

```rust
trait KvLayout {
    // ... existing page lifecycle (alloc_tokens / attach_pages / free_slot / budget) ...

    /// Authoritative live token count for `slot`. Single source of truth.
    /// (Already exists on TokenKVPool as `seq_len(slot)`; the contract is
    /// making external owners read THIS, never a cached/derived copy.)
    fn slot_token_count(&self, slot: usize) -> usize;

    /// Monotonic generation stamp, bumped on every alloc_tokens/attach_pages/
    /// free_slot for `slot`. A shadow owner caches the stamp it last synced;
    /// `stamp != cached` ⇒ its derived per-slot state (fp8 byte window
    /// validity, packed length) is stale and must be re-derived before use.
    fn slot_epoch(&self, slot: usize) -> u64;
}
```

Two methods, both backed by data `TokenKVPool` already has (`seq_len` exists;
the epoch is one `Vec<u64>` bumped at the three existing mutation sites). This
is the FlashInfer-style discipline applied to **state**: the pool is the single
`plan`-time source of truth, shadow stores `run` against a stamp they validate —
the same "compute-once, validate-before-reuse" shape the compute plane already
uses, instead of two stores drifting.

| Today | Grounded form | Why |
|---|---|---|
| `seq_lens` in `TokenKVPool`, fp8 window validity implicit in `fp8_kv_pool`, no link | `slot_token_count` (read the one truth) + `slot_epoch` (cheap staleness check) on `KvLayout` | the fp8 pool's per-slot validity is *derived* state; derived state needs a freshness signal from its source, not an independent guess |

## What it collapses — the improvement that drives it

**`ARLE_DSV4_SHARED_KV_POOL` × `--num-slots>1`** (env-gated, landed `eb377eae`,
pod-validation-pending in the half-done scan). Today the coordination touches
**6-8 sites** that must each independently stay consistent with `seq_lens`,
source-verified:

1. `paged_kv.rs:609/681/1168` — the three `seq_lens` mutations (must signal change)
2. `scheduler/cuda/core.rs` — `install_restored_kv` (`attach_pages` call site)
3. `scheduler/cuda/decode.rs` — `free_slot` call site
4. `deepseek/batch_decode.rs:117` — `ensure_fp8_kv_pool` (re-checked per step)
5. `deepseek/weights.rs:1772` — pack dispatch (needs live extent)
6. `deepseek/weights.rs:3623` — decode dispatch (needs live extent)
7. `deepseek/forward.rs:88/328` — the two `dsv4_shared_kv_pool_enabled` branches

After the contract:

- **The three mutation sites bump one epoch counter** (3 one-line edits, behind
  `KvLayout`). **Pack/decode read `slot_token_count` + check `slot_epoch`** (2
  consuming sites). Every other site stops needing to "stay consistent with
  `seq_lens`" because there is now one authoritative read + one staleness stamp.
  The 6-8 must-stay-in-sync sites collapse to **3 producers (bump) + 2 consumers
  (read/validate)** with a typed contract between them — not N copies of "hope
  the offset still matches the extent."

This is the same N→1 collapse the compute plane gets from `oplib`, applied to
state: today every shadow-state user re-derives extent consistency by hand;
after, there is one source (`slot_token_count`) and one freshness signal
(`slot_epoch`).

## Migration — independently shippable, each revertible

- **Step 1 (PURE, CPU-tested, no behaviour change).** Add `slot_epoch: Vec<u64>`
  to `TokenKVPool`, bump at the three existing mutation sites, expose
  `slot_token_count` (alias the existing `seq_len`) + `slot_epoch`. CPU unit
  test: alloc/attach/free a mock pool, assert the epoch advances and
  `slot_token_count` tracks. No consumer wired yet — pure addition,
  bit-identical runtime.
- **Step 2 (one real consumer).** Make the FP8 pack/decode path
  (`weights.rs:1772/3623`) read `slot_token_count` and cache+check `slot_epoch`,
  re-deriving its per-slot view on mismatch. This is the first and only
  consumer; the contract trails it, not the reverse.
- **Step 3 (validate the consumer the contract was built for).** Run the
  blocked validation: 8×H20 TP=8 DSv4-Flash, `ARLE_DSV4_SHARED_KV_POOL` 0 vs 1,
  `--num-slots` 1/4/8 — OFF must equal main (parity), ON must be byte-identical
  at c=1/4 to OFF **and** no longer read stale KV at c=8 (needle-retrieval
  across a free/re-admit cycle). Only after this passes does the
  default-flip become discussable.
- **Later (gated, separate):** fold the `KvLayout` trait surface itself (the
  attention/KV doc's deferred page-lifecycle tranche) — now that the extent
  contract exists, `KvLayout` can carry it from the start, so the two land as
  one coherent trait instead of a surface that immediately needs extending.

## License-or-kill

- Step 1 is a deletion/legibility-class addition: gate is bit-identical runtime
  (the epoch counter is observed by nobody yet).
- Step 2-3 are licensed **only** by the multi-slot correctness consumer. The
  binding gate is the c=8 free/re-admit needle test: if the shared FP8 pool does
  **not** actually exhibit the stale-KV failure this contract prevents (e.g. the
  pack always overwrites the full window regardless of extent, making the gap
  benign), then the contract is **not licensed** — kill it and document that the
  shadow store is self-healing. Per `CLAUDE.md §0`, a root-cause hypothesis
  ("two stores drift") must itself be cheap-experiment-verified: **before
  building the contract, run the c=8 free/re-admit needle probe on current HEAD
  with `ARLE_DSV4_SHARED_KV_POOL=1` and confirm the stale read actually
  reproduces.** No repro → no contract.

## Honest gaps (not self-deception)

- The contract assumes the shadow store is **per-slot derivable** from
  `(slot_token_count, slot_epoch)`. That holds for the fp8 pool (fixed byte
  window + live length). A future shadow store whose validity depends on
  *cross-slot* state (e.g. a radix-shared prefix spanning slots) would need a
  richer signal; that is **out of scope** until such a consumer exists.
- This does **not** address the raw-`usize`-`slot_idx`-vs-typed-handle question
  (`attention-kv-architecture.md:77`, also deferred). A typed `SlotHandle` is a
  separate, larger refactor with its own consumer test; bundling it here would
  be scope creep. The extent contract works with the raw `usize` slot today.
- Tiering (T1 host-pinned / T2 NVMe / T3 RDMA — `kv_tier/`) is a **different
  axis** (which tier holds a page), orthogonal to this contract (how many tokens
  are live in a slot). This doc does not touch tier policy; see
  [`infer/src/kv_tier/AGENTS.md`](../../infer/src/kv_tier/AGENTS.md).
