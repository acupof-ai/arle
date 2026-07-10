# DSpark per-layer draft-KV cap — sliding-window ring, ~2560→~640 MB/slot

> Status: Active — pod license (needle + VRAM + c-sweep) pending-remote

## Context

DSpark reserved `cap = max_seq_len + block` draft K/V for **all** draft layers,
but 4/5 z-lab DFlash layers are `sliding_window`(2048) and, post-P2.5
([partial-ctx drafting](2026-07-10-dspark-partial-ctx-drafting.md)), only ever
read the last `window` keys. That over-reservation (~2560 MB/slot when ~600
would do) clamped concurrent slots on shared/loaded GPUs. Commit `23fa8f3e2`.

## What Worked

**Per-layer cache stride, scheme A (absolute-position ring).** Each draft ctx
K/V buffer is sized by layer type:

- **sliding (4×)**: `cap_sw = window + block`, addressed as an absolute-position
  ring — `row = pos % cap_sw`. Keeps only the last `window` keys the sliding
  attention ever reads.
- **full (1×)**: `cap_full = max_seq_len + block`, linear — `row = pos − ctx_base`
  (unchanged).

The RoPE cos/sin cache stays full `max_seq_len` (positions are absolute).

**No kernel rewrite, only a `ring_modulus` param.** The existing prep-write and
nonpaged-attention kernels already parameterize the per-head stride. Added a
`ring_modulus` arg to both `__global__` kernels + two new `*_ring_cuda` extern
wrappers; the existing wrappers pass `ring_modulus=0` → byte-identical for every
trunk / decode / autograd caller. Softmax is permutation-invariant, so a wrapped
ring read accumulates identically to a contiguous one — only the physical row
differs. Chosen over scheme B (per-layer advancing base): no per-layer base
state, no per-step compaction copies.

**Why `cap_sw = window + block` is the minimum (no live aliasing).** Within one
draft block the live span = `window` ctx keys + `block` noise keys = `cap_sw`.
Two distinct absolute positions `p ≠ q` with `|p−q| < cap_sw` never collide mod
`cap_sw`, so a noise write never overwrites a live ctx key still needed in that
block. One append launch writes `≤ cap_sw` rows (guarded by `ensure!`) → no
intra-launch write race; chunk sizes (≤32) sit far below `cap_sw`.

### §0.1 buffer table (draft K/V, per layer type)

| buffer | type | per-head stride | writer | index formula | read span | aliasing precondition |
|---|---|---|---|---|---|---|
| `k_ctx[li]`/`v_ctx[li]` | full 1× | `max_seq+block` | `prep_cuda` | linear `abs − ctx_base` | `[ctx_base, start+block)` | `abs−ctx_base < cap_full` (ctx grows ≤ max_seq) |
| `k_ctx[li]`/`v_ctx[li]` | sliding 4× | `window+block` | `prep_ring_cuda` | ring `abs % cap_sw` | `[max(ctx_base, q_pos−win+1), start+block)`, read via `nonpaged_ring` row `(lo+l) % cap_sw` | live span (win ctx + block noise) = `cap_sw`; distinct `p≠q`, `|p−q|<cap_sw` never collide mod `cap_sw`; one launch writes `≤cap_sw` rows |

**Byte-identity (memory fix, not accuracy).** For `ctx_base==0`, `abs < cap_sw`:
ring row `abs % cap_sw == abs ==` old linear row, same value; the per-row read
visits the same key set in the same ascending-abs order → attention output
byte-identical wherever the old linear code was correct (any `q` whose window
fits). For `ctx_end > cap_sw` the old buffer kept spare rows but the sliding read
still only took the last `window` — identical keys, still byte-identical; the new
code additionally frees the never-read prefix (and fixes the region where the old
linear buffer would have overflowed at `ctx_end − ctx_base > cap_full`).

### Per-slot draft VRAM

`before = 2·5·kv·hd·cap_full`; `after = 2·kv·hd·(4·cap_sw + cap_full)`.
Ratio `= (4·cap_sw + cap_full)/(5·cap_full) ≈ 0.25` at `max_seq≈32K`
(`cap_sw/cap_full ≈ 2050/32770 ≈ 1/16`). Documented baseline 2560 MB → ~640 MB.

| metric | before | after |
|---|---|---|
| per_slot draft bytes | ~2560 MB (pending-remote confirm) | ~600–640 MB (pending-remote) |

## Verification

- **Local gates (green):** `infer-api` cuda,no-cuda + `cli` metal,no-cuda
  typecheck; `clippy -D warnings` clean on `infer-cuda`+`cuda-kernels`;
  `arle` cpu tests pass. CUDA-on-Mac cannot compile the kernels — pod is the
  kernel-correctness gate.
- **Pod license (pending-remote, devops):** build `--features cuda --bin arle` +
  `strings … | grep dspark-sp`; needle 738291 ×3 exact + same-config-twice on the
  dspark greedy lane (`max_tokens 768`); per_slot draft bytes before/after;
  c-sweep c∈{1,4,8} dspark vs plain (the deferred #17 batching measurement).

## Rule

- Size per-slot caches by what each layer **reads**, not a uniform pool cap — a
  sliding layer needs only `window+block`; the ring modulus = that stride keeps
  the existing contiguous-cache kernels usable via a single `ring_modulus` param.
- `cap = window + block` is the minimum with no live-key aliasing: the live span
  (window ctx + block noise) exactly equals the modulus, so distinct positions
  within a block never collide.
