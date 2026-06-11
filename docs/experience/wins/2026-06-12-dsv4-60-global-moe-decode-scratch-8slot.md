# #60 — DSv4 8-slot OOM fixed: per-slot×per-layer MoE decode scratch → one model-wide shared instance

## Context

`arle serve --backend cuda --num-slots 8` on DSv4-Flash (8×H20, TP=8/EP=8)
OOM'd at engine build — `Dsv4GroupedDecodeScratch::out_padded` (moe.rs) was the
first failing alloc, 2.26 GiB short. 4-slot booted; 8-slot did not.

Root cause (SOLID, not inferred): `Dsv4MoeDecodeScratch` was allocated
**per-slot × per-layer, unconditionally** — 8 slots × 43 layers × ~114 MiB
≈ 39 GiB resident — but only **one** copy is ever live (it is pure
overwrite-before-read scratch, dsv4.rs:711, never crosses step/layer), and the
**default path (no GPU router) allocated it but never used it**. The 4→8 slot
step doubled a 39 GiB phantom that nothing read.

Mirrors the #67 fix for the DSA selector scratch (`Dsv4DsaSharedScratch`): one
shared instance on `Dsv4KvAdapter`, single budget deduction.

## What Worked

**One model-wide shared instance on `Dsv4KvAdapter`**, allocated only when a
consumer is actually live:

- `attention.rs` (+42): `Dsv4KvAdapter` gains `moe_decode_shared:
  Option<Dsv4MoeDecodeScratch>` + `shared_expert_out: Option<HiddenStates>`
  (the latter always-Some, hidden_size×1 BF16 ≈ 14 KB, used every decode);
  split-borrow accessor `moe_decode_shared_mut()`.
- `dsv4.rs` (+72/−33): deleted the per-slot `Vec` scratch fields + alloc loop;
  forward + decode-graph paths source the scratch from `kv_adapter` instead of
  `slot`; budget `kv_budget_num_slots` folds `moe_decode_shared_bytes` as a
  fixed term (no longer in per-slot).
- `moe.rs` (+98): `Dsv4MoeDecodeScratch::device_bytes()` mirrors `::new` for the
  budget (MUST-mirror discipline, like `dsv4_dsa_shared_scratch_bytes`).

**Allocation predicate caught a regression mid-validation.** First cut gated the
scratch on `ARLE_DSV4_GPU_ROUTER` alone. But the decode-graph path
(`ARLE_DSV4_DECODE_GRAPH=1`, FlashMLA-decode, designed "no gpu_router
dependency", dsv4.rs:1254) runs the masked-MoE tail through this **same**
scratch unconditionally (dsv4.rs:3063/3078). Router-only gating made
decode-graph-without-router error at dsv4.rs:3032 where it used to work (it read
the always-allocated per-slot Vec). Fix: `needs_moe_decode_shared =
use_gpu_router || dsv4_decode_graph_enabled()`, mirrored in the budget. This is
exactly the §0.1 "enumerate every consumer of a mutated buffer" check — the
decode-graph consumer was invisible until the path was traced.

### Evidence (pod, 8×H20, CUDA 12.9, sm_90a, release-fast)

Env: TP=8, INFER_DSV4_MAX_SEQ_LEN=16384, MOE_BACKEND=allreduce,
EXPERT_BACKEND=deepgemm (native), num-slots=8, kv-cache-dtype=auto (FP8).

| Config | 8-slot boot | budget "shared MoE decode" | needle ×3 @ {115,300,446,2k,8k} |
|--------|-------------|----------------------------|----------------------------------|
| **default** (no router/graph) | ✅ no OOM | **0 MB** (per-slot ×39 GiB reclaimed) | 14/15 exact (1 miss @300 = MoE non-det, NONDET) |
| **decode-graph** (no router)  | ✅ no OOM | **114 MB** (one instance, was 8×43×114 MiB) | **15/15 exact** |

- nvidia-smi after full KV alloc (default): **62 887 MiB / 97 871 MiB** used per
  GPU → ~35 GiB headroom (pre-fix: OOM, 2.26 GiB short).
- KV budget log (default): `free 57035MB, per_slot 924MB (arena×2 784 +
  rotated 21 + state caches 118), shared DSA 36MB, shared MoE decode 0MB` — the
  per-slot term no longer carries the moe scratch.
- **Concurrent-slot correctness (the design's "serial replay safe" claim,
  now with evidence not inference):** 8 needle requests in flight at once →
  **8/8 retrieved the code**. Default concurrent decode loops `forward_decode_row`
  per row serially (executor.rs:1602; true batching is the opt-in
  `INFER_DSV4_BATCHED_DECODE` lever), so the single shared buffer is reused
  strictly serially — no corruption under genuine 8-way concurrency.

### Throughput (8-way concurrent, default per-row-loop decode)

`scripts/dsv4_concurrent_probe.py 18188 8 4 128`: 32 reqs, 4096 out tok,
126.3 s wall → **aggregate 32.4 tok/s**, per-req mean 30.7 s (p50 31.6, p99
36.9), single-stream-equiv 4.2 tok/s.

**Δ% vs baseline:** no prior 8-slot baseline exists — pre-#60 it crashed at
build (0 usable). The comparable fixed-config baseline (4-slot) is
**byte-identical** by construction: the default-path forward compute is
unchanged (routed-MoE `decode_scratch` was `None` before and after — the gating
`use_moe_decode_scratch = use_gpu_router && …` pre-existed; #60 only deleted
unused allocations + folded budget terms). Aggregate ≈ single-stream because the
default decode loop is serial — raising 8-slot *throughput* is the separate
batched-decode lever (memory: batched +57% @c=8), not #60's scope. #60 unlocks
the **capability** to run 8 slots correctly.

Canonical guidellm sweep: **pending-remote** — guidellm absent on the offline
pod; the aiohttp concurrent probe above is the on-pod stand-in (real tok/s +
concurrent correctness). Ticket #60.

## Rule

A scratch buffer allocated per-slot × per-layer when only one copy is ever live
is a phantom — size = slots × layers × per-copy, and the OOM scales with the
slot count even though nothing reads the extra copies. Before sizing scratch
per-slot, prove it crosses a step or slot boundary; if it's overwrite-before-read
within one forward, it is one shared instance gated on its **actual consumers**
(enumerate **all** of them — the decode-graph consumer was invisible until the
path was traced, and router-only gating silently broke it). Verify the shared
buffer under **concurrent** load, not a serial gate: serial retrieval can't
surface cross-slot aliasing; 8 needles in flight can.
