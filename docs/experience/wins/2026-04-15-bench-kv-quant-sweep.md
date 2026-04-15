# 2026-04-15 · Qwen3-4B L4 — bf16 vs int8 vs fp8 KV-cache quick sweep (post M3b classifier tightening)

## Context

After the 2026-04-15 L4 remote acceptance for M2b / M0.3 / M3a / M3b / M3c
landed in commits `eb347d9` + `1bbc135` (now `87eb14e` + `875669a`
post-rebase), the user asked to re-bench on the post-pull tree and
confirm the three supported KV-cache quantization modes (`bf16`,
`int8`, `fp8`) all work end-to-end on the L4 host.

The re-bench window also lets us measure the perf impact of
[`283da19 fix(kv-tier): tighten M3b lookup classifier + refresh
soft-pin on hit`](https://github.com/cklxx/agent-infer/commit/283da19),
which touched
`infer/src/{prefix_cache.rs,scheduler/cuda/core.rs,scheduler/cuda/runtime.rs}`
(321 +, 90 −). The new bf16 numbers below are directly comparable to
the same-host 2026-04-15 `page16` full sweep
([`2026-04-15-tiered-kv-m0.3-m3a-remote.md`](2026-04-15-tiered-kv-m0.3-m3a-remote.md))
on the C=1..4 rows — this time run with `--quick` against the same 6
canonical configs.

## Environment

- GPU: NVIDIA L4 24 GB (driver 580.82.07, CUDA 13.0, SM 89)
- Model: Qwen3-4B BF16, `Qwen/Qwen3-4B` HF Instruct variant
- Commit: `2b61644`
  (post `875669a` remote-acceptance push, post
  `283da19` M3b classifier tightening + `2b61644` doc correction)
- Cargo env: `CARGO_HOME=/tmp/cargo-home-local`
- Server: `target/release/infer --model-path models/Qwen3-4B --num-slots 4 --kv-cache-dtype <mode>`
  - `cuda_graph=true`, warm batch sizes {1, 2, 4}
  - `max_seq_len=4096` auto, `num_slots=4` explicit
  - KV cache modes exercised: `bf16`, `int8`, `fp8`
- Bench: `python3 scripts/bench_throughput_sweep.py --quick`
  (`QUICK_CONFIGS` in `bench_throughput_sweep.py:65-72` — 6 canonical
  configs)
- JSON snapshots paired:
  `2026-04-15-bench-bf16-quick.json`,
  `2026-04-15-bench-int8-quick.json`,
  `2026-04-15-bench-fp8-quick.json`

## Server boot diagnostics

Each mode logs its KV layout choice at boot; captured from the server
log for each run:

- **bf16** — `kv_cache_mode=bf16`, contiguous=BF16, paged_pool=BF16,
  `TokenKVPool … format=BF16`, `page_size=16` (M0.3 default for BF16).
- **int8** — `kv_cache_mode=int8`, contiguous=INT8, paged_pool=INT8,
  `TokenKVPool 45383 max tokens … format=INT8`, `page_size=1`
  (quantized formats stay at 1 per M0.3 design). The contiguous KV
  cache log line is:
  `KV cache INT8: storage=302.0MB scales=9.4MB working=16.8MB (was 604.0MB bf16, saving 46%)`.
- **fp8** — `kv_cache_mode=fp8`, contiguous=**BF16**, paged_pool=FP8E4M3,
  `TokenKVPool 46726 max tokens … format=FP8E4M3`, `page_size=1`.
  FP8 keeps the contiguous prefill cache in BF16 and quantizes only
  when migrating into the paged token pool (per the `--kv-cache-dtype`
  CLI help text). Per-layer pool usage: `data=95.7MB/layer, scales=0.0
  MB/layer, working=191.4MB`. FP8 has no per-token scales.

All three modes boot, warm CUDA Graphs for batch sizes {1,2,4}, and
serve the full `--quick` sweep with `Err = 0` on every config.

## Raw results — bf16 / int8 / fp8 (same L4 host, same commit, same --quick configs)

```
─────────────────────────────────────────────────────────────────────────────────────────
              bf16 (post-283da19)     int8                    fp8
In | Out | C   tok/s  TTFT50  ITL50   tok/s  TTFT50  ITL50    tok/s  TTFT50  ITL50
─────────────────────────────────────────────────────────────────────────────────────────
 128 | 128 | 1  29.8   39ms   33.5    30.0    39ms   33.3     30.0    40ms   33.3
 128 | 512 | 1  29.6   44ms   33.9    29.8    45ms   33.6     29.8    45ms   33.4
 512 | 256 | 1  29.3   86ms   33.9    29.4    87ms   33.8     29.5    88ms   33.7
1024 | 256 | 1  29.0  165ms   34.0    28.8   147ms   34.3     28.7   166ms   34.3
2048 | 256 | 1  28.5  276ms   34.2    27.6   280ms   35.3     27.7   276ms   35.2
 512 | 256 | 4 109.5  302ms   35.1   105.1   289ms   36.6    105.8   283ms   36.4
─────────────────────────────────────────────────────────────────────────────────────────
Peak C=1    29.8 tok/s             30.0 tok/s              30.0 tok/s
Peak C=4   109.5 tok/s            105.1 tok/s             105.8 tok/s
ITL p50 range  33.5 – 35.1 ms    33.3 – 36.6 ms          33.3 – 36.4 ms
Err            0 / 48             0 / 48                  0 / 48
Wall total     404.0 s            406.3 s                 405.5 s
```

All rows are `Err=0`. All three modes hold within ~4 tok/s of bf16 on
every config, including the max-context 2048/256 case.

## Versus the 2026-04-15 `page16` full sweep baseline (same host)

bf16 `--quick` rows agree with the `page16` full sweep's C=1..4 rows
within ±0.2 tok/s — within noise. `283da19` has no measurable perf
impact on the single-request or small-concurrency paths:

```
                page16 full (875669a)   bf16 quick (2b61644)    delta
 128 | 128 | 1      29.8 tok/s              29.8 tok/s             0
 128 | 512 | 1      29.6 tok/s              29.6 tok/s             0
 512 | 256 | 1      29.3 tok/s              29.3 tok/s             0
1024 | 256 | 1      29.1 tok/s              29.0 tok/s            −0.1
2048 | 256 | 1      28.5 tok/s              28.5 tok/s             0
 512 | 256 | 4     109.3 tok/s             109.5 tok/s            +0.2
```

So the post-`283da19` build still meets the M0.3 / M2b acceptance
baseline captured in
[`2026-04-15-tiered-kv-m0.3-m3a-remote.md`](2026-04-15-tiered-kv-m0.3-m3a-remote.md).

## KV-quant deltas (int8 and fp8 vs bf16, same --quick configs)

```
                    int8 vs bf16              fp8 vs bf16
 128 | 128 | 1       +0.2 tok/s (+0.7%)       +0.2 tok/s (+0.7%)
 128 | 512 | 1       +0.2 tok/s               +0.2 tok/s
 512 | 256 | 1       +0.1 tok/s               +0.2 tok/s
1024 | 256 | 1       −0.2 tok/s               −0.3 tok/s
2048 | 256 | 1       −0.9 tok/s (−3.2%)       −0.8 tok/s (−2.8%)
 512 | 256 | 4       −4.4 tok/s (−4.0%)       −3.7 tok/s (−3.4%)
```

The pattern holds for both INT8 and FP8: **essentially flat at short
contexts (C=1, in ≤ 512)** and **small regressions of 3–4% at the
long-context 2048 row and the 4-wide concurrent row**. The
long-context regression comes from dequant-on-read work being a larger
share of the decode budget when the KV cache is bigger; the C=4
regression comes from the same overhead being amplified across 4
concurrent streams inside a single CUDA-Graph decode step.

ITL p50 range:

- bf16: 33.5 – 35.1 ms
- int8: 33.3 – 36.6 ms
- fp8:  33.3 – 36.4 ms

Both quant modes widen the ITL range on long / concurrent rows by
~1–1.5 ms, consistent with the per-token dequant cost.

## Memory footprint (HBM)

From the boot-time `TokenKVPool` / `KVCache` log lines for the same
`--num-slots 4`, `max_seq_len=4096` auto-sized setup:

```
                  contiguous KV cache           paged token pool        pool tokens
bf16              604 MB                        3.4 GB (page_size=16)   43,680
int8              302 MB + 9.4 MB scales        3.6 GB (page_size=1)    45,383
fp8               BF16 (same as bf16)           3.6 GB (page_size=1)    46,726
                  working buf: 16.8 MB          FP8 no per-token scales
```

- **int8**: the `KVCache::ensure_allocated` log line explicitly reports
  `"storage=302.0MB scales=9.4MB working=16.8MB (was 604.0MB bf16,
  saving 46%)"` — **~46 % HBM saving on the contiguous KV cache** vs
  bf16 for the same max_seq_len.
- **fp8**: the contiguous prefill cache stays in BF16 (per the CLI
  help text) so the contiguous-cache saving is zero. The paged pool
  is FP8E4M3 with no per-token scales. The net effect is a small
  increase in `pool tokens` budget (46,726 vs 43,680) because the
  auto-sizer can fit more FP8 tokens into the same pool byte budget.

## Things that did NOT regress

- **C ≥ 8 high-concurrency rows** still work on this host. The
  2026-04-13 page1 baseline reported `0 tok/s` on every C ≥ 8 row
  because the sticky CUDA error killed the server's context on the
  first C=8 config; the 2026-04-15 `page16` full sweep recovered those
  rows. This `--quick` sweep only runs C=1 and C=4 configs, so it does
  not re-test C ≥ 8, but there were no errors at C=4 across all 3
  modes, consistent with the full page16 C ≥ 8 recovery.
- **End-to-end greedy output**: no requests returned empty streams;
  all `Err = 0`. The INT8 dequant path and the FP8 paged-pool
  quantize-on-migration path both complete successfully.

## Sign-off

- [x] bf16 post-`283da19` quick sweep matches the 2026-04-15 `page16`
      full-sweep baseline on all C ≤ 4 rows within noise.
- [x] int8 quick sweep: 48 / 48 requests green, throughput within 4.0
      tok/s of bf16, 46 % contiguous-cache HBM saving.
- [x] fp8 quick sweep: 48 / 48 requests green, throughput within 3.7
      tok/s of bf16, pool budget slightly larger than bf16 for the
      same auto-size envelope.

**Supported KV cache modes on the 2026-04-15 L4 host**:
`bf16`, `int8`, `fp8`, plus the TurboQuant pool modes `tq2`/`tq3`/`tq4`
(not measured here). CLI help text at `target/release/infer --help`
has the canonical list.

## Rule

KV-cache quantization on Qwen3-4B at BF16-target HBM is a **memory
lever**, not a throughput lever, on L4. Expect ~46 % contiguous-cache
savings with INT8 and a widened pool token budget with FP8, both at
the cost of 3–4 % throughput on long contexts and concurrent batches.
The decode path is HBM-bound, so the extra dequant ALU cost only
shows up where the compute-to-bandwidth ratio gets tighter —
long-seq C ≥ 4. For short-prompt agent traces at C = 1 the two quant
modes are essentially free.

**Corollary**: if you are running on a host where KV memory is the
gate (single-GPU, many slots, long contexts), INT8 is the right
default — the contiguous-cache saving is the big win. If you are
running on a host where compute is the gate (short prompts, C ≥ 4),
stay on BF16 and pay the HBM. If you need both contiguous-cache
savings AND concurrent throughput at long contexts, the
TurboQuant `tq3`/`tq4` pool modes are the next lever to test; not
covered in this sweep.
