# Unified shared-paged KV cache — complete plan (all models, SGLang-aligned)

**Goal (ckl):** 扩充 KV cache 用量 — fit more/longer requests in the same HBM, via **one
model-agnostic shared paged token pool** for ALL models (not a per-model hack), each model
verified. Grounded in the ARLE allocation map + an SGLang best-practice study (2026-06-24).

## 0. The principle — one pool, model-agnostic
KV memory is **one shared, paged, profile-sized token pool** living above the seam
(`infer-seam`/`infer-core`); a model is just an attention kernel that reads/writes it through
a page table. This is already true for dense Qwen3; the work is bringing Qwen3.6 and DSv4 onto
the same substrate and adopting SGLang's sizing/admission/reuse discipline — **uniformly**, so
adding a future model means writing a paged-attention kernel, not a new memory system.

## 1. The finding (both sides grounded)
ARLE's **dense Qwen3 is already SGLang-shaped** (shared paged pool + radix + chunked prefill +
on-demand admission). The waste is **Qwen3.6/DSv4 only**: per-slot contiguous KV sized to
`max_seq_len` per slot at build → ~**4× more VRAM** than a shared pool for the same model, no
prefix reuse, static admission.

| | dense Qwen3 | Qwen3.6 | DSv4 | SGLang target |
|---|---|---|---|---|
| pool | shared `PagedKVPool` ✓ | per-slot `k/v_caches` max_seq_len (`qwen35.rs:374-379`) | per-slot MLA arena max_seq_len (`dsv4.rs:835-934`) | shared paged `TokenToKVPool` + `ReqToToken` |
| sizing | `total_pages` | `total_pages×num_slots` (`executor.rs:2908`) | `max_seq_len×num_slots` | profiled: `rest=free−total×(1−0.9)`; `max_tok=rest/cell` |
| prefix reuse | radix ✓ | none (`lib.rs:391`→0) | none | radix + `lock_ref` + leaf-LRU |
| admission | `retract_decode_to_fit` | static | static | forecast `ratio×max_new` + `new_token_ratio` backoff + retract |
| chunked prefill | yes (full-prefix) | bypassed by recall prefill | 4096 cap | yes, full-prefix bit-equiv |

## 2. Unified architecture (the model-agnostic contract)
- **Device pool** `PagedKVPool` (`cuda-kernels/src/paged_kv.rs`) — page-grained K/V tensors,
  `set_kv_buffer`-style scatter writes at a page-index array, `alloc`/`evict`/`reinstate`.
  Already model-agnostic in shape `(num_layers, kv_heads, head_dim, dtype, page_size)`.
- **Host index** `HostPagedKvPool`/`KvAllocator` (`infer-seam/src/host_paged_kv_pool.rs`) —
  the `ReqToToken` analogue: per-slot logical page list, global free stack, on-demand `alloc`.
  Model-agnostic.
- **Per-model wiring** = a `PageMeta` (the page table + start_pos) handed to the model's paged
  attention kernel. Dense uses `PageMeta::for_slot`; Qwen3.6 recall uses the same; DSv4 needs
  an MLA-paged variant. **This is the only per-model surface.**
- **Scheduler/admission/radix** (`infer-core`) — already model-agnostic; just gated off for
  Qwen35/DSv4 today (`lib.rs:391`).

## 3. Profile-based pool sizing (model-agnostic; SGLang #1)
Replace per-slot `max_seq_len` and the `cuda_admission_total_pages` per-slot multiply with a
single profiled budget, computed **once after weights load, from measured free VRAM**:
```
reserve_frac = 1 − mem_fraction_static     (default 0.10; ≥5-8 GB)
rest_bytes   = free_vram_after_weights − total_vram × reserve_frac
cell_bytes   = num_kv_heads · head_dim · num_layers · 2(K+V) · sizeof(kv_dtype)   // per token, per pool
max_total_tokens = rest_bytes / cell_bytes
total_pages = max_total_tokens / page_size
```
Lands in a new `infer-seam` helper `profile_kv_pool_tokens(free, total, cell_bytes, mem_frac)`
used by every backend's executor build (replaces the per-kind branches in
`loaded.rs:1266-1287`). A `--mem-fraction-static` CLI flag (default 0.9). One shared pool,
not `× num_slots`.

## 4. Per-model migration

### 4a. Dense Qwen3 — DONE (the template)
Shared `PagedKVPool`, radix, chunked prefill, on-demand admission. Switch its sizing to the §3
profiler (drop any hardcoded `total_pages`). Verify unchanged correctness + sizing.

### 4b. Qwen3.6 (canonical; pod-testable first)
The paged full-attn pool already exists as `recall_kv`. Make it the **default** full-attn KV:
1. Always route Qwen3.6 full-attn through the shared `PagedKVPool` (the `full_attention_paged`
   read-swap becomes default, not `--kv-recall`-gated). Drop per-slot `k_caches/v_caches`
   (Track C `free_full_attn_caches` generalized to never-allocate). Keep linear-attn `gdr/conv`
   per-slot (MB-scale, recurrent — not pageable).
2. Size the pool via §3 (shared, profiled), not `num_slots × max_seq_len`.
3. Wire the paged prefill through the planner's chunked prefill (`planner.rs:41-83`) so peak is
   bounded + each chunk attends the full resident prefix.
4. Flip `reusable_prefix_blocks()` (`lib.rs:391`) on for Qwen35 → radix prefix reuse.

### 4c. DSv4 (MLA; bigger, after 3.6)
MLA KV is FP8 latent per-layer in a per-slot arena (`dsv4.rs:835-934`). Migrate to a paged MLA
pool: a `PagedKVPool` variant whose cell is the MLA latent (`kv_lora_rank + qk_rope` bytes),
the FlashMLA decode/prefill kernels reading via the page table (FlashMLA already supports paged
KV — wire ARLE's page table to it). Keep spec-decode ring snapshots per-slot. This is its own
sub-plan (the MLA paged-attention wiring is the real work); flip radix on after.

## 5. Cross-cutting SGLang adoptions
- **Radix + lock_ref**: `page_attach_count` (paged_kv.rs) is the `lock_ref` seed — make in-flight
  pages provably non-evictable (inc on batch-enter, dec on exit). Recursive leaf-LRU eviction
  already in `radix.rs`.
- **Forecast admission**: over `retract_decode_to_fit`, reserve `new_token_ratio × max_new`
  (init 0.7, min-factor 0.14, decay 600 steps, bump on retract) instead of static reserve.
- **page_size**: keep 16 where the TileLang/FlashMLA kernel needs it (per-backend init, not
  global); don't chase SGLang's page_size=1 (kernel-constrained here).

## 6. Verification matrix (ALL models — the "都验证好" requirement)
Each model, on the pod, before its phase is "done":

| check | how | pass bar |
|---|---|---|
| **VRAM drop** | `nvidia-smi` idle, paged vs per-slot build | Qwen3.6/DSv4 idle HBM drops ≈ `num_slots × per-slot-max_seq_len` |
| **Correctness** | needle-retrieval + coherent generation (correct-inference gate, same-config-twice, NOT byte-identity) | matches the per-slot build's quality |
| **Capacity (the goal)** | max concurrent requests / max context that fits before OOM | strictly more than per-slot (target ~4× for 3.6) |
| **Throughput** | `bench_guidellm.sh` TTFT/ITL/tok-s vs the per-slot baseline | no decode regression (Δ within noise) |
| **Prefix reuse** | repeated-prefix workload, `/v1/stats` radix hit | reuse > 0 (was 0) |
| **Default-off / parity** | dense Qwen3 unchanged; non-migrated path byte-identical | identical |

Models: **dense Qwen3 (V100/H20), Qwen3.6 (H20), DSv4 (8×H20 TP=8)** — each gets the full row.
A `wins/` entry per model (pending-remote until pod numbers land).

## 7. Order + gates (each phase lands in main, pod-verified, before the next)
1. **§3 profiler** (model-agnostic helper + `--mem-fraction-static`) — lands first, dense uses it,
   verify dense unchanged. Foundation.
2. **§4b Qwen3.6 default-paged** + profiled sizing — the 4× win; verify VRAM↓ + needle + capacity.
3. **§4b radix + chunked-prefill wire** for Qwen3.6 — verify prefix reuse + bounded prefill.
4. **§5 admission refine** — verify no over/under-admission, retraction rate sane.
5. **§4c DSv4 MLA paging** — its own sub-plan; verify on 8×H20.
6. **Tiering/recall** (the durable NVMe store `92faf97a` + the gap-injection blueprint) layer on
   top of the shared pool — for beyond-HBM. Separate track.

Gate every phase: `cargo check` + `cargo test -p infer-core -p infer-seam` green, default-off
byte-identical, the §6 row for the affected model green on the pod, a `wins/` entry.
