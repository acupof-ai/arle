# Expanding KV-cache capacity: migrate Qwen3.6/DSv4 to a shared paged pool (SGLang-aligned)

**Goal (ckl):** 扩充 KV cache 用量 — fit more/longer requests in the same HBM. Grounded
in the ARLE allocation map + an SGLang best-practice study (2026-06-24).

## The finding in one line
ARLE's **dense Qwen3 is already SGLang-shaped** (shared paged pool + radix + chunked
prefill + on-demand admission). The waste is **Qwen3.6/DSv4 only**: they reserve
`num_slots × max_seq_len` per-slot contiguous KV at build → ~**4× more VRAM** than the
shared paged pool for the same model, no prefix reuse, static admission.

## ARLE today vs SGLang (both grounded)

| | ARLE dense Qwen3 | ARLE Qwen3.6 / DSv4 | SGLang |
|---|---|---|---|
| Pool | shared global `PagedKVPool` | **per-slot contiguous, max_seq_len each** (`qwen35.rs:374-379`, `executor.rs:2908`; dsv4 `dsv4.rs:835-934`) | one shared paged `TokenToKVPool` + `ReqToTokenPool` index |
| Alloc time | build-time pool, **on-demand pages/req** (`host_paged_kv_pool.rs:138`) | build-time, **static per-slot** | startup pool sized by **measured** free VRAM, on-demand slots |
| Pool sizing | `total_pages` | `total_pages × num_slots` (conflates global budget w/ per-slot depth) | `rest = free − total×(1−mem_frac_static0.9); max_tok = rest/bytes_per_tok` |
| Prefix reuse | radix ✓ | **none** (`lib.rs:391` → 0) | radix + `lock_ref` refcount + recursive leaf-LRU |
| Page size | 16 (TileLang) | 16 | **1** default, page-align per-backend |
| Chunked prefill | yes (`planner.rs:41-83`, full-prefix) | yes, but **per-slot peak unaffected**; recall prefill bypasses it | yes (SARATHI; chunk's KV stays resident → later chunks attend full prefix, bit-equiv) |
| Admission | `retract_decode_to_fit` (`planner.rs:94`) | static | forecast `ratio×max_new`, adaptive `new_token_ratio` (0.7→floor over 600 steps, bump on retract), retraction safety valve |
| Tier (overflow) | RadixCache demote→host; recall write-through | recall opt-in | HiCache L1/L2/L3, **off decode hot path** (layer N+1 prefetch while N computes), `write_through_selective`, `timeout` prefetch |

## The plan — migrate Qwen3.6 full-attn KV to the shared paged pool (the 4× win)

The paged pool already exists for Qwen3.6 (`recall_kv`, used under `--kv-recall`). The
migration = **make it the default KV path, shared + profile-sized**, not gated on recall.
Qwen3.6 is hybrid: only the **full-attn** KV migrates to paging; the small linear-attn
recurrent state (`gdr/conv`) stays per-slot (it's MB, not GB). Track C already proved the
full-attn caches are a separable, dead-under-recall Vec — this generalizes that to always-paged.

1. **Default-paged full-attn KV for Qwen3.6** — route full-attn through the shared `PagedKVPool`
   always (not just `--kv-recall`); drop the per-slot `k_caches/v_caches`. The read-swap path
   (`full_attention_paged`) becomes the default, not an opt-in.
2. **Profile-based pool sizing** (SGLang #1) — size the shared pool from **measured free VRAM**
   after weights load (`rest = free − total×(1−mem_frac)`, default mem_frac 0.9, 5-8 GB reserve),
   not `num_slots × max_seq_len`. Replaces `cuda_admission_total_pages`'s per-slot multiply.
3. **Radix prefix reuse for Qwen3.6** — paging enables it; flip `reusable_prefix_blocks()`
   (`lib.rs:391`) on for Qwen35 once full-attn is paged.
4. **Chunked prefill on the paged path** — the recall/paged prefill writes each chunk's KV to
   the pool and attends the full resident prefix (SGLang #6; ARLE's planner already does this for
   dense — wire Qwen3.6's paged prefill through it).
5. **Forecast admission refine** (later) — adopt `new_token_ratio` backoff over the existing
   `retract_decode_to_fit` so we don't statically reserve full `max_new`.
6. **DSv4 (MLA) — later** — its arena is FP8 latent per-layer; paging it is a separate, bigger
   change (the MLA KV layout). Do Qwen3.6 first (canonical, pod-testable).

## What we DON'T rebuild (ARLE already has the SGLang pieces)
`PagedKVPool` + `HostPagedKvPool` (= TokenToKVPool + the index), `RadixCache` (radix + tier
demote), chunked prefill (`planner.rs`), `retract_decode_to_fit`, the write-through tier
(`CudaKvTierStore`, now durable+NVMe per `92faf97a`). The migration is **wiring Qwen3.6 onto
the dense path's proven substrate**, not new infrastructure.

## Order
Qwen3.6 default-paged + profile-sizing (the 4× win, pod-verifiable: idle VRAM drops, more
concurrent/longer requests fit) → radix-for-3.6 → chunked-prefill wire → admission refine →
DSv4 paging → (then tiering/recall for beyond-HBM, on top of the shared pool).

## Best-practice deltas to copy verbatim
- page_size: SGLang defaults **1**; ARLE's 16 is a TileLang-backend constraint (per-backend init,
  not global — keep 16 where the kernel needs it, don't globalize).
- `lock_ref` refcount = the *provable* non-evictable mechanism for in-flight KV — adopt the
  inc/dec-on-batch-enter/exit discipline (ARLE's page_attach_count is the seed).
- Admission: init `new_token_ratio` 0.7, min-factor 0.14, decay 600, retract-every-20.
- Tier prefetch: `timeout` policy (protect TTFT), `write_through_selective` (hit-gated), layer-wise
  N+1-while-N overlap — for the recall/tier track.
