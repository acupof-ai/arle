# qwen4_exp device-side MoE group planning: the prefill's per-(layer,chunk) ids fence is gone — and the 80% claim is corrected by the A/B

## Context / Goal

Three independent receipts named ONE lever: `pf.moe.ids_fence` read 7.46 s of
the 9.27 s chunk-256 prefill (80%,
`2026-08-28-qwen4exp-chunked-gdn-prefill.md`), a 2-position verify chunk paid
the same 48 per-layer fences and killed MTP speculation's payoff
(`2026-08-28-qwen4exp-mtp-speculative-decode.md` post-merge verdict), and
decode had already gone fence-free for +8 ms/token. Task: move
`plan_moe_groups` (the host expert-grouping planner that forced a flush +
ids read-back per (layer, chunk)) onto the GPU, drive the class-specialized
grouped GEMVs with `vkCmdDispatchIndirect`, and hold the bit-exact
prefill=decode gate at 0.000e0.

## What Was Built

**Fixed-capacity class regions (`MoePlanLayout`) — the load-bearing trick.**
`NUM_COLS` is a pipeline specialization constant AND descriptor offsets must
be record-time; a densely-packed plan makes every class base a function of the
device-only block counts. Instead, each class `w` (1..=8) owns a worst-case
region computed from `(pairs, n_experts)` alone — full class `pairs/8`
blocks, remainder class `min(E, pairs/w)` (one remainder block per expert) —
so every bind offset is a record-time constant and ONLY the dispatch y extent
stays device-side. Cost: ~6x the live pair rows (~330 MB more HOST-CACHED
arena at chunk 256, not device heap; capacity gaps are never written or read).
The gather flips to a write-side scatter over live pairs
(`qwen4_block_scatter.comp`) so sparsity costs zero bandwidth; SwiGLU runs
over the capacity span (garbage-in/garbage-out on gaps, ~0.1 ms/layer-chunk).

**Three planner kernels, deterministic by construction, oracle-equal:**

- `qwen4_moe_plan_count`: one workgroup, atomic per-expert totals (order-free).
- `qwen4_moe_plan_scan`: ONE thread walks experts ascending — 8 running sums
  over 512 experts, microseconds — writing exclusive block-index prefixes,
  per-class totals, and the `VkDispatchIndirectCommand` triples (x = the
  record-time projection out-dim, y = the device block count, z = 1). Serial
  is what makes the block order a fact, not a race.
- `qwen4_moe_plan_emit`: closed-form scatter row per pair + the class-major
  id lists. The pinned intra-expert order is ARRIVAL order in the token-major
  scan (token asc, slot asc): rank = |{p' < p, same expert}| by direct rescan
  (pairs <= 2560, L1-resident, ~6.5M reads worst) — the order is a property
  of the INDEX. `plan_moe_groups` (kept as oracle + `_HOST_PLAN=1` fallback)
  pins the identical order and layout.

**Indirect dispatch plumbing** (`vulkan-sys` had none): `CommandRecorder::
dispatch_indirect` (+ stub arm), `barrier_indirect` (dst must include
DRAW_INDIRECT — a compute-only dst mask does NOT cover the args read; the
reverse WAR is already ordered because src masks extend to logically earlier
stages), and `INDIRECT_BUFFER` usage folded into every `alloc*` (flags are
free). The grouped GEMV needed NO shader change: its push words use
`n_blocks` only as an identity modulus (`expert_i0 % ne11`), so the class
CAPACITY is a valid record-time bound and the y extent rides the indirect
args. Empty classes dispatch y=0 (legal no-op).

## Proof Stack

- `tests/qwen4_moe_plan.rs` — ELEMENT-FOR-ELEMENT equality of device plan vs
  `plan_moe_groups`: scatter map (whole), all 8 block counts, live id-list
  prefixes, both indirect triples per class; on randomized rosters (t=1..256,
  hot experts, 32/512 experts, poisoned output buffers so unwritten lanes
  can't fake a zero) and on 16 REAL router rosters captured via `moe_ids` on
  the truncated fixture. Mutation check: moving the scan's exclusive prefix
  to inclusive fails it loudly ("scatter maps diverge") on the first
  hot-expert roster.
- The truncated prefill=decode gate: **0.000e0** at chunk 7, chunk 24
  (fence-free default) AND chunk 24 under `_HOST_PLAN=1` — both modes lay
  the same fixed regions, so they record byte-identical GEMV work.
- Full-scale parity (`ARLE_QWEN4_PREFILL_PARITY=1`): **max rel 0.000e0, max
  abs 0.000e0, argmax 198 == 198**.

## Measured (full scale, 512 tokens, one ~70 GiB load, same sitting; powercfg
## scheme read Balanced `381b4222`, NOT the Performance fingerprint `27fa6203`
## of earlier sittings — trust the in-sitting ratios, not cross-sitting absolutes)

- chunk 256: fenced host plan **62.7 tok/s** (8.16 s; `pf.moe.ids_fence`
  7137.8 ms / 288 fences) → fence-free device plan **64.7 tok/s** (7.92 s;
  ids_fence phase GONE, `pf.moe.plan` records at 1.8 ms host / ~0.9 ms GPU).
  **+3.2%, not the hoped-for half of 80%.**
- chunk 64 fence-free: **58.7 tok/s** (8.72 s) — where the chunked-GDN A/B's
  side receipt had read 0.6 tok/s with 654 s of ids fence. The pathology at
  high chunk COUNTS is simply gone (the curve still rises with width, 58.7 →
  64.7).
- Verify-chunk probe (`ARLE_QWEN4_SPEC=1 ARLE_QWEN4_SPEC_KS=1`, next load,
  same box/mode): k=1 verify 69.0-123.3 ms/cycle, speedups 0.63x / 1.08x /
  0.94x (factual-qa / code / chat) — statistically the SAME as the MTP
  verdict's 68-119 ms and 1.09x-best. The verify chunk records fence-free
  through the shared `record_moe` now, so its cost is proven to be its own
  GPU work (a full 48-layer forward for 2 positions), not fences: speculation
  stays shelved, and un-shelving it needs verify-side GPU cost cuts, not
  more fence surgery.

**The premise correction that matters:** the "ids fence = 80% of the wall"
receipt double-counted. The fence phase was where ALL of a layer's recorded
GPU work DRAINED — `prof`'s wall attribution booked GPU execution against the
stage that happened to flush. Removing the fence moves that wall to the
end-of-chunk drain (`prompt` 6.45 s in the fence-free arm) and recovers only
the true fence OVERHEAD: per-layer submit + host-bubble serialization, ~0.24 s
of 8.16 s at chunk 256 (~3%), plus the entire chunk-count-scaling pathology at
small chunks. The prefill wall is now ~81% GPU execution in one submit — the
next levers are kernel-side (the NVFP4 dequant-ALU floor, linattn occupancy),
not fence-side.

## Rules

- A drain wall booked under a fence phase is NOT fence cost: decompose with
  GPU timestamps BEFORE sizing a structural rewrite by it (`prof`'s host
  table attributes drained GPU-busy to whichever stage flushed — the
  fence-free A/B is what finally split "fence overhead" from "GPU work that
  drained at the fence").
- Record-time descriptor offsets + device-computed dispatch extents compose:
  fixed worst-case regions per specialization class turn "the plan lives on
  the GPU" into "only the y extent lives on the GPU" — no vendored-shader
  surgery, no bit-exactness risk, memory (host-cached, bounded ~6x a small
  region) as the only price.
- Determinism on GPU is free when order is a function of the INDEX (serial
  scan for prefixes, rank-by-rescan for arrival order) — atomics only where
  order is irrelevant (totals).
