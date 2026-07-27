# FA3 paged decode makes 32k cost what 66 tokens cost — 2.76× at c=1

## Context

The first long-agent baseline put decode at 72.1 ms/token at 32k, c=1 — 13.9
tok/s on a model whose short-context step is 26.6 ms. The whole gap was the
context term.

Finding it took one command and three wrong turns. `ARLE_QWEN35_PROFILE=1`
attributed the step directly:

| label | ms/step |
|---|---:|
| `qwen/full_paged/attention` | **50.84** |
| `qwen/linear_attention` (48 GDN layers) | 20.35 |
| `qwen/dense_ffn` | 16.37 |
| everything else | ~7.8 |

Attention was 53% of the step at 3.18 ms/layer — 134 MB of KV per layer, so
**42 GB/s against 4 TB/s of HBM**. It resolves through `resolve_paged_attn_v1`
to the TileLang AOT kernel `batch_decode_paged_hd256_q24_kv4`, which:

- pads `BLOCK_M = 64` around **one** real query row (98.4% of every GEMM),
- runs one CTA per **query** head over a per-**KV**-head cache (6× the traffic),
- has no split-KV: at c=1 the grid is 24 CTAs × 128 threads, ~2% occupancy,
- walks 32k serially in 2048 `BLOCK_N=16` iterations, where 16 exists for L4's
  99 KB shared cap.

## What Worked

The vendored FA3 hopper units already ship exactly this shape —
`flash_fwd_hdim256_bf16_paged{,_split}_sm90.cu`, both **PackGQA-only**: one CTA
per KV head serving the whole GQA group, split along KV to fill the SMs. The
shim needed the page table wired; nothing else.

Single-request ITL probe (no bench harness, no queueing), 32k prompt, 96 tokens,
GPU 2, 3 trials:

| | ITL p50 | decode tok/s |
|---|---:|---:|
| TileLang | 72.1 ms | 13.9 |
| **FA3 paged split-KV + PackGQA** | **26.1 ms** | **38.3** |

**2.76×**, all three trials within 0.4%. The number lands where the byte ledger
says it should: the short-context step measured **26.6 ms** on the same binary,
so **the context term went from 45.5 ms to ≈0**. Short context is unchanged —
37.96 tok/s vs `baselines.md`'s 38.62, inside the ±3% drift band.

Needle gate (`needle_gate.py 512,4096,16384,32768 3 0.0`, `qwen3_nonthink`, RAW):
**exact=3 miss=0 DET at every length**, 32k included.

Three integration facts, each checked against the vendored source rather than
assumed:

- **No relayout.** `paged_kv.h:117` builds the K tensor from an arbitrary CuTe
  stride tuple over dims `(page_size, head_dim, head, page)`, so the HND pool
  `[page, h_k, page_size, d]` is expressible directly.
- **`pagedkv_tma = false`.** The `page_size % kBlockN == 0` assert
  (`mainloop_fwd_sm90_tma_gmma_ws.hpp:537`) is gated on `!PagedKVNonTMA`, so
  qwen35's 16-token pages are fine on the gather path.
- **The 4th K/V dim IS the page dim** when a page table is set
  (`flash_fwd_launch_template.h:98-100`), so its stride is the pool's page
  stride. The shim was deriving it from `batch_extent()`, the contiguous-KV
  formula — 8.3M elements against the correct 16384, and the first paged decode
  step died on `CUDA_ERROR_ILLEGAL_ADDRESS`. That was the only real bug.

## Problems

- **Scoped to sm_90 + BF16 + batch 1 decode.** FA3 hopper is Hopper-only, so
  every other target keeps TileLang permanently — a capability split, not two
  implementations of one job. Batch > 1 needs a padded `[b, max_pages]` page
  table and still runs the TileLang kernel; c≥4 is unmeasured and is the next
  tranche.
- **Three wrong turns preceded the profile**, all avoidable by running it first:
  a hand-written GQA/warp-per-key/scaled-splits rewrite of
  `fused_gqa_attention_decode_batched_kernel` (−20.5%, reverted `fcf709e0f`);
  `--qwen35-fa3-decode`, which moved nothing (72.4 vs 72.1 ms); and a four-way
  batched/per-row × FA3 sweep that landed within 0.4% because **none of the four
  paths executes** — they are all in the non-paged lane.
- **The deletion is not done.** Five full-attention decode implementations still
  exist and exactly one runs. `fused_gqa_attention_single_token_kernel` was
  deleted (`5dc0d28e7`, 185 lines, zero callers); the batched kernel + reduce +
  their FFI + scratch + `--qwen35-batched-decode-attention` + its per-row arm
  remain, as does `--qwen35-fa3-decode` now that FA3 is not a flag.
- No `ncu` before/after. The 2.76× and the needle gate are the evidence; the
  per-kernel bandwidth number is not measured.

## Rule

**Profile before touching a kernel, and prove the kernel you are editing is the
one that runs.** This model has two full-attention lanes; every serve flag
reaches only the unused one. Four configurations agreeing to 0.4% is not
"the parameter does not matter" — it is "none of these is executing", and that
reading was available three attempts earlier.

And when a portability constant sets a performance ceiling — `BLOCK_N = 16` so
the cubin loads on L4 — name the device that pays. Here it cost the H20 serving
path a 4× longer inner loop for a GPU that is not in the support matrix.
