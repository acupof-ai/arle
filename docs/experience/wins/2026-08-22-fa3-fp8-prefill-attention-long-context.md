# FA3 fp8 prefill attention from 64K tokens — 220K TTFT −17 %, short context unchanged

> Status: Landed (`8f48ff6b4` + `7d58850dc`). Closes
> [the FA3 fp8 prefill plan](../../plans/2026-08-22-fa3-fp8-prefill-attention.md).

## Context

At 180K tokens `full_paged/attention` is 50 % of the prefill and sits on
FA3's bf16 compute floor (≈43 s of 46 s). The fp8 tensor-core rate is the
only lever below that floor. The quantized-pool prefill shim dequantised
pages into a bf16 temp and ran the bf16 kernel.

## What Worked

The shim has two operand forms routed on shape. For a prefill chunk
(`seqlen_q ≥ 256`) over a long KV (`seqlen_k ≥ 65536`) it requantises the
pages the table names into an e4m3 temp with one descale per (row, kv_head)
— the row's largest per-token scale, tight because the per-token quantiser
puts one element at full range — quantises Q the same way and runs FA3's
e4m3 hdim256 paged kernels (two vendored instantiations added). Everything
else keeps the bf16 form. Output is bf16 either way; Rust args unchanged.

Route measurement, H20, Qwen3.8-27B-NVFP4, fp8 KV, TTFT of one request, fp8
form vs bf16: 33K +10 %, 66K −1 %, 132K −11 %, 220K −16 %. At 32K the fp8
form's attention was 1564 → 2644 ms (profile), and on the few-row spec-verify
step at 220K it cost −21 % decode tok/s — hence both floors.

Gate on the routed binary (`7d58850dc`), base `1df0acf68`, same GPU:

| | base | new |
|---|---:|---:|
| 32K c=1 TTFT p50 / ITL p50 | 5.93 s / 21.28 ms | 5.90 s / 21.26 ms |
| 32K c=16 TTFT p50 / ITL p50 | 1.52 s / 37.65 ms | 1.61 s / 37.53 ms |
| 220K TTFT | 129.7 s | **108.2 s (−17 %)** |
| 220K decode tok/s (MTP on) | 49.2 | 49.2 |

Needle ×3 at 512/4096/16384/32768 and ×1 at 200000: all exact, DET; the
220K request reproduces the passphrase at 50 % depth. The 200-item eval uses
short prompts and never reaches the fp8 form (unchanged code path).

## Rule

Route an operand-precision change by the shape that made it pay: the fp8
kernel wins only where the O(L²) term dominates, and the same kernel loses
on short q tiles. Measure the crossover on both axes (q rows and KV length)
before picking the threshold; one axis was not enough here.
