# Quantized paged attention: one CTA dequantizes K/V for the whole GQA group — 1.4–1.6× at the kernel from B=4

> Status: Landed (`04e3b4e3e` + `e3b9b0f81`; the first commit lost the `.cu`
> to the fmt hook and shipped shifted FFI args — the second lands it).

## Context

`paged_attention_quantized_fa3_partial_kernel` ran one CTA per (q-head, split).
On Qwen3.8-27B (24 q-heads, 4 kv-heads) the 6 q-heads of a kv-head each
re-read and re-dequantized the same fp8 K/V bytes. ncu at c=16/32: 80 % of
decode in this kernel, DRAM 5.9 %, L1 hit 87 % — issue-bound on the dequant.

## What Worked

The kernel is templated on `H` = q-heads per CTA (`heads_per_cta`, one of
1/2/3/4/6/8, must divide the GQA ratio). Each token's K and V are loaded and
dequantized once per lane and applied to `H` q-heads (`H` online-softmax
states, `H` partial outputs; the cross-warp merge assigns heads round-robin).
Grid `[num_q_heads / H * num_splits, batch]`. Per-lane arithmetic order is
unchanged, so `H=1` is bit-identical to the previous kernel.

Standalone microbench, H20 (sm_90), 24/4 heads, head_dim 256, fp8 KV, ctx
32768, page 16, 30 launches, output compared bitwise against the old kernel.
`s` = splits (old: `max(8, ceil(78 / (B·4)))` capped at 16; new: ×H, capped
at 16):

| B | old µs (s) | H=1 | H=2 | H=3 | H=6 |
|---|---:|---:|---:|---:|---:|
| 1 | 520.8 (16) | 1.05× bit-exact | 1.00× | 0.98× | 0.88× |
| 4 | 1223.3 (8) | 0.62× (s=8) | 1.13× (s=16) | 1.11× | **1.44×** |
| 16 | 4715.1 (8) | 0.93× | 1.41× | 1.42× | **1.47×** |
| 32 | 9199.5 (8) | 0.97× | 1.41× | 1.49× | **1.59×** |

Same splits → bit-exact. Where the split count changes (8→16) the max diff is
4.88e-4, one bf16 ulp from the merge order. Registers (HD=256, fp8): H=1 56,
H=2 78, H=3 94, H=6 128, no spills; H=8 spills 16 B and is never selected
for this model.

At B=1, H=6 leaves 64 CTAs for 78 SMs; the Rust side picks the largest `H`
with `batch · (q_heads / H) · 16 ≥ 2 · sm_count` — H=2 at B=1, H=3 at B=2,
H=6 from B=4 (`qwen35_attention.rs`, decode quantized-pool branch).

End-to-end, Qwen3.8-27B-NVFP4, 1×H20 (GPU 0, container), fp8 KV, MTP on,
32 K agent prompts ×32, 214 output tokens, two interleaved trials per arm,
base = HEAD before this change:

| arm | c=1 decode tok/s | c=16 decode tok/s | c=32 decode tok/s |
| --- | ---: | ---: | ---: |
| base t1 / t2 | 82.7 / 84.1 | 7.9 / 7.9 | 4.5 / 4.5 |
| new t1 / t2 | 83.3 / 83.7 | 10.0 / 10.2 | 6.0 / 6.1 |

Per-request decode c=16 +27–29 %, c=32 +33–36 %, c=1 wash (H=2 at B=1, the kernel was 1.00×
there). Needle ladder ×3 at 512/4096/16384/32768 on the new binary: 12/12
exact, DET. 200-item GSM8K-train greedy eval (same recipe as the 0.5.8
release): 179/200, against 179–180/200 before the change.

## Rule

For GQA decode over a quantized cache, the unit of work is the kv-head, not
the q-head: dequantize once per CTA and fan out over the group. Scale the
group size with batch so the grid still fills the machine.
