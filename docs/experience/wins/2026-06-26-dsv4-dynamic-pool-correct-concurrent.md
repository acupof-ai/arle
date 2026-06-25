# DSv4 dynamic KV pool — correct + concurrent (c=32), root cause + full cascade fixed

Status: VERIFIED on pod (TP=4, DeepSeek-V4-Flash-FP8). The dynamic fragmented FlashMLA
KV pool decodes correctly, at long context, concurrently to c=32. Caps the multi-day
DSv4 dynamic-pool arc.

## Root cause (one root, whole cascade)
DSv4's FlashMLA band readers used **contiguous band-base addressing** (`page == first +
logical`); the dynamic free-stack pool gives **fragmented pages**. Every bug in the
cascade (C1 batched stride, C2 prefill alloc/free, budget double-spend, full-band draw,
band cursor, contiguity, batched-scratch width) was the same root — fragmented pages
through contiguous-only readers. The incremental-draw direction (`02e179f4`) was
architecturally mismatched with the band's fixed sliding-window-ring + compressed layout.

## The complete fix (commits, this arc)
- `ffea0aaf` — 10 adversarial-review bugs (C1 batched build_indices stride, C2 prefill
  free-then-alloc, H1 Qwen3.6 budget, H2 MTP seq_len, H3 host page_size 16→64, M1-3, L1-2).
- `73169fe0` — O-LoRA dense-BF16 grouped path → run at TP≠o_groups (TP=4 verify unblocked).
- `4f6193ea` — band cursor = LOGICAL position (unbounded), not bounded by band capacity
  (the ring wraps / compressed appends; bounded by max_seq at ingress, not the band).
- `31117139` — compressed-delta pack → device page table (the last MODEL1 contiguity
  holdout: `arle_dsv4_fp8_kv_pack_completed_compressor_row_start_pos`, kernel routes
  `block_id = page_table[logical]`, FFI/wrapper/caller in lockstep).
- `f2271899` — batched-decode `page_table_batched` scratch sized for the WIDEST layer's
  band (CSA total_blocks), not layer_shapes[0] (SW-only=2); check `==`→`<=` capacity.

## Verified (pod, TP=4)
- sanity: `" Paris. The capital of Italy is Rome..."` — correct.
- long-ctx (~1300 tok): retrieves the planted code `8472` — correct.
- needle_gate len=115/180/241/300: **exact=3** every run (NONDET label = MoE run-to-run
  non-determinism floor, not a miss).
- concurrency: **c=1/8/16/32 → 1/1, 8/8, 16/16, 32/32 ok**, 26→63 tok/s. num_slots
  ceiling lifted ~4 → 32+ vs the pre-refactor per-slot×max_seq reservation.

## Rule
A fixed-layout KV band (SW ring + compressed, addressed by slot-LOGICAL block id) is
fundamentally incompatible with a fragmented dynamic pool UNLESS every reader/pack kernel
routes logical→physical through a device page table. Convert ALL of them (write-pack,
build_indices, compressed-delta, batched decode) — one contiguous holdout corrupts the
whole path. Verify with a needle (exact retrieval) + a concurrency c-sweep, never just
boot/throughput (boot passed for months while decode was silently wrong on >1-page prompts).

## Remaining (separate)
- Throughput (63 tok/s @ c=32) is serial-forward-bound (MoE all-to-all + MLA + DSA + MTP),
  not slot-bound — a separate optimization (batched decode / graph / DP-attn).
- max_seq=1M (the compressor `.compressed` cache, head_dim-wide, still per-slot×max_seq;
  ckl's indexer ring `80aae0fa` bounded the indexer half) — a separate pooling effort.
