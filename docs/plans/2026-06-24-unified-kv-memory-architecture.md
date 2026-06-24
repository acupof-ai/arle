# Unified KV-memory architecture — one shared-pool design for all backends

Status: design, awaiting sign-off. Global convergence (architectural, all backends)
→ approach-first per the agent contract. Grounded in a full per-backend map
(`kv-memory-global-map` workflow, 2026-06-24).

## The invariant (one sentence, every backend)

`weights + KV_pool = mem_fraction_static × total_VRAM`; KV is **one shared
dynamic-draw token pool** sized by `profile_kv_pool_tokens` from measured free VRAM
after weights; `num_slots`/`max_running_requests` is a **zero-HBM soft index cap**
that NEVER multiplies the KV pool; per-slot×max_seq scratch is **deleted** and
re-expressed as **batch-bounded (b=max_running) shared scratch** living in the
`(1 − mem_fraction_static)` headroom.

This is SGLang's design ([research](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)):
shared `token_to_kv_pool` (max_total_tokens) + small `req_to_token_pool` index +
workspace/CUDA-graph/activations in the headroom.

## The seam + core are ALREADY this — FREEZE them

The map's key finding: the device-neutral layer needs **no change**.
- `infer_seam::HostPagedKvPool` = shared LIFO page free-stack, empty per-slot vecs,
  dynamic `alloc(slot, tokens)` draw (`host_paged_kv_pool.rs:35,137-155`).
- `infer_seam::profile_kv_pool_tokens` = the SGLang sizing formula
  `reserve = total×(1−frac); tokens = (free−reserve)/cell` (`resource.rs:146-161`).
- `num_slots` allocs only `Vec<u32>` bookkeeping — **zero HBM** (`host_paged_kv_pool.rs:49-61`).
- core `admit_waiting` gates on `kv.free_pages()`; `num_slots = min(num_slots,
  max_rows_per_step, max_tokens_per_step)` (`infer-core lib.rs:947,396-408`).

All convergence work is **executor-side** (the 4 backends that call the seam).

## Reference = CUDA dense Qwen3 (zero delta)

`executor.rs:748-823`: probe free VRAM → `profile_kv_pool_tokens` → `sized =
profiled_pages.max(requested_pages)` — `requested_pages` is a pure floor, **no
num_slots multiply**; `num_slots` reaches `PagedKVPool` only as the index-vec count.
Every other backend converges onto exactly this shape.

## The deltas (file:line)

- **Δ-A — CUDA Qwen3.6 MoE (small):** drop the `num_slots × total_pages` pool floor
  (`executor.rs:3207,3217,3235`) → floor at pure `total_pages` like dense. Full-attn
  KV is already shared (`per_slot_kv_bytes` returns 0, `qwen35.rs:1668-1680`); the
  ~61 MB/slot recurrent state (gdr+conv) is genuinely per-slot, seq-independent —
  keep. Delete the dead legacy contiguous `k_caches/v_caches` lane (`qwen35.rs:404-437`).

- **Δ-B — CUDA DSv4 EP (the real work, TWO parts):**
  1. **Shared token pool (Stage B) — KERNEL-BLOCKED.** The MLA latent `TokenKVPool`
     is profiled but **pre-allocated per-slot up-front** (Stage A identity layout:
     `total = tokens_per_slot × num_slots` + a construction `alloc_tokens(slot,…)`
     loop, `attention.rs:750-808`), so num_slots still multiplies the pool. Going
     dynamic (drop the per-slot loop + `×num_slots`, size purely from profiled tokens,
     runtime `alloc/free/mirror`) is **blocked on the device pack/index kernels
     consuming a device page table instead of band-base addressing** (the
     `contiguous_page_table_byte_range` identity gate, `paged_kv_table.rs:90-117`) —
     the load-bearing kernel work that let Stage A ship byte-identical. **This is the
     piece that actually unlocks DSv4 high concurrency.**
  2. **Per-slot DSA scratch → batch-bounded headroom (doable now).** `dsa_rotated_per_slot`
     + `state_caches_per_slot` (selector/compressor/indexer caches, max_seq-scaled ×
     num_slots, `dsv4.rs:1573-1644`) move to a model-wide b=max_running shared scratch
     in the headroom — mirroring how FlashMLA/MoE decode scratch is already done
     (`dsv4.rs:196-201`). This was the missing step in the earlier (killed) refactor.

- **Δ-C — Metal Qwen3.6 (small):** drop the `× num_slots` fold (`resource.rs:270-273`)
  → one num_slots-independent token budget; adopt `mem_fraction_static` (retire the
  `args.rs:495` carve-out), keeping the macOS anti-swap clamp on `memory_limit` BEFORE
  the fraction. (Metal is scalar today: `max_rows=1` → num_slots clamped to 1, so the
  fold is moot in practice but must change for correctness/uniformity.)

- **Δ-D — HIP DSv4 GGUF (early lane):** replace `num_slots × max_seq.div_ceil(page)`
  (`infer-hip executor.rs:185`) with a ROCm free/total probe → `profile_kv_pool_tokens`
  → one shared pool, mirroring CUDA. Nonconformant today (stub-ish lane).

- **Naming:** introduce `max_running_requests` as the public SGLang-named soft cap;
  `num_slots` stays the internal field (`SchedulerConfig.num_slots`). Doc/CLI only.

## Honest scope split

- **Contained (no kernel work):** Δ-A, Δ-C, Δ-D, naming, **Δ-B.2** — unifies the
  SIZING + most workspace across Qwen3.6/Metal/HIP/DSv4. Good global hygiene; partial
  DSv4 help.
- **DSv4 high-concurrency unlock needs Δ-B.1 (Stage-B kernel work)** OR the orthogonal
  **EP=8** (weights halve → headroom for more slots, no kernel work). The contained
  deltas alone do NOT lift DSv4's slot ceiling — Stage A still multiplies by num_slots.

## Implementation order + verification matrix

1. Δ-B.2 (DSv4 DSA scratch → batch-bounded) — addresses the earlier OOM; pod-verify
   DSv4 num_slots>4 fits.
2. Δ-A (Qwen3.6 floor) — pod-verify Qwen3.6 num_slots scales.
3. Δ-C (Metal), Δ-D (HIP), naming.
4. Δ-B.1 (DSv4 Stage-B kernel) — the load-bearing remainder; separate effort.

Verify each: **Qwen3-dense, Qwen3.6-MoE-FP8, DSv4-EP4, Metal-Qwen3.6** — boot + needle
exact + a num_slots-scaling c跑 (the per-backend throughput proof).

## Risks
- The map is source-survey (hypothesis), not measured. Each delta needs a pod A/B
  (boot + needle + the num_slots c跑) before it's "shipped" per the bench rule.
- Δ-B.1 device-page-table kernel change is the highest-risk item (the reason Stage A
  shipped identity-layout); treat as its own project with the kernel-correctness gate.
