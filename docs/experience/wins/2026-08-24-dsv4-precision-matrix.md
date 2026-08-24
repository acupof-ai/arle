# DSv4 precision matrix: FP8 experts win c=1, quantized KV is not wired — CUDA, 2026-08-24

> Status: Characterized. FP8 experts + BF16 KV is the c=1 champion (59.5 vs
> NVFP4 44.4 tok/s, MMLU wash); the four quantized-KV arms cannot load by
> design. The matrix's FP8 26.5 was a census-contaminated measurement.

## Goal

Bench and eval DSv4-Flash across every precision configuration the runtime
admits: {NVFP4, FP8} experts x {bf16, fp8, int8} KV.

## Parameters

```bash
arle serve --backend cuda --model-path <ckpt> --tensor-parallel-size 4 \
  --kv-cache-dtype <kv> --spec-type none --port 8300
python3 scripts/bench_throughput.py --concurrency-grid 1,8 \
  --requests-per-concurrency 16 --max-tokens 256 --temperature 0 \
  --prompts-jsonl bench-agent-32k-16x8.jsonl
arle_capability_eval_local.py --tasks mmlu --n-samples 200 --concurrency 1 --seed 0
```

- Binary `gatemiss-v3`; 4xH20 (GPUs 0,1,2,4), TP=4, `--comm-backend nccl`
- Checkpoints `/data00/DeepSeek-V4-Flash-0731` (NVFP4 experts, 156 GB) and
  `-0731-FP8` (287 GB)
- Prompt p50 28568 tok, completions 256 exact

## Results

| experts | KV | status | captures | c=1 decode tok/s | c=1 ITL p50/p99 ms | c=8 decode tok/s | MMLU (200) |
|---|---|---|---:|---:|---|---:|---|
| NVFP4 | bf16 | OK | 76 | **44.4** | **22.2 / 42.0** | 27.0 | 0.855 |
| FP8 | bf16 | contaminated, re-measured below | 0 | ~~26.5~~ | ~~37.0 / 74.3~~ | 25.7 | 0.860 |
| NVFP4 | fp8 | LOAD_FAILED | — | — | — | — | — |
| NVFP4 | int8 | LOAD_FAILED | — | — | — | — | — |
| FP8 | fp8 | LOAD_FAILED | — | — | — | — | — |
| FP8 | int8 | LOAD_FAILED | — | — | — | — | — |

FP8 c=1 re-measured on `fp8recheck-v1` (includes `ebb1dd89b`), same GPUs,
workload and flags, `ARLE_GRAPH_NODE_CENSUS` unset, 16 requests per arm:

| experts | arm | captures | c=1 decode tok/s | TTFT p50 ms |
|---|---|---:|---:|---:|
| FP8 | graph | 72 | **59.5** | 8256 |
| FP8 | eager | 0 | 51.9 | 8215 |

Both agree with the 2026-08-23 entry (59.5 / 52.4). Accuracy is a wash: 0.855
vs 0.860 on 200 items is one item. At c=1 the FP8 checkpoint is 1.34x NVFP4
(59.5 vs 44.4) although it reads 1.8x the expert bytes; the M=1 lane is the
W4AFP8 GEMV kernel (`moe/dsv4.rs` `dsv4_moe_forward_w4a16` with INT4 dequant)
versus the FP8 GEMV lane, so the NVFP4 decode lane is kernel-bound, not
bandwidth-bound.

FP8 c=8 needs a slot budget the default flags do not give it. The per-slot
attention state at the default `max_total_tokens` 1048576 is 6982 MB
(`[dsv4-slot-ledger]`: compressor 5567, indexer 1344, dsa_rotated 42,
fused_wqkv 23, sw_window 5), so the FP8 checkpoint (72 GB weights/rank) is
clamped to **`num_slots=1`** and the matrix's "c=8" row was 8 requests
serialized through one slot. With `--max-total-tokens 131072` the slot is
1049 MB; the planner then admits 18 slots and c=8 dies with CUDA OOM at
step 4432 on every rank (no reserve for prefill transients), and with
`--max-running-requests 8` on top:

| FP8, 128K total tokens, 8 slots | c=1 | c=8 |
|---|---:|---:|
| decode tok/s per request | 60.0 | 21.9 |
| TTFT p50 ms | 8197 | 1996 |
| MMLU (200) | 0.860 (172/200) | — |

Prefill at c=1 is 28568 tokens in 8.2 s = 3.5k tok/s per request at TP=4
(server-side prefill busy: 485,660 tokens in 135 s = 3.6k tok/s); NVFP4 is the
same (TTFT 7870-7926 ms), so the expert format does not touch prefill.

## Problems

**The four quantized-KV arms cannot load, by design.** Every rank bails at
engine construction:

```
--kv-cache-dtype fp8 is not supported for Dsv4; only Qwen35 supports quantized paged KV
```

`crates/infer-api/src/loaded.rs:2054` admits a non-BF16 KV dtype only for
`CudaModelKind::Qwen35`. `docs/support-matrix.md` claimed "INT8/FP8 paged
quant-KV dispatch landed opt-in, correctness licensed (#68)", which is what
made this a six-arm matrix instead of a two-arm one; the entry is corrected in
the same change. `crates/cli/src/args.rs:944` had it right all along ("wired
one mode at a time under #68 T3").

The coordinator hides this: all four workers die before the boot-ping, so the
only surfaced error is `RelayCoordinator write envelope to worker rank 0:
relay write payload: Broken pipe`. The real message is in the worker stdout,
interleaved four ways because all four ranks write it concurrently.

**The FP8 arm's 26.5 tok/s was a measurement artifact.** `16857e541` made an
allocating capture fatal; the FP8 checkpoint recorded 86 alloc nodes per
capture, so every capture was rejected, and the matrix ran with
`ARLE_GRAPH_NODE_CENSUS=1`, so every decode step re-attempted the capture and
walked the node census. That is 2x slower than plain eager (26.5 vs 51.9).
`ebb1dd89b` restores the warn; the re-measure above is the truth. DeepEP and
DeepGEMM were not a factor: both checkpoints ran the default `allreduce`
transport and the M=1 GEMV lanes.

**The 86 alloc nodes: the FP8 checkpoint's `wo_a` is dense BF16.** The first
explanation written here ("the O-LoRA staging resizes per layer") was a guess;
a probe on that resize branch fired 0 times while the capture still recorded
86. The node census places each pair (`ALLOC 8192` + `ALLOC 2048`) between
`dsv4_update_window_cache_start_pos_ptr_kernel` and
`dsv4_oproj_group_gather_kernel` followed by a cuBLAS `nvjet` GEMM: the
per-group cuBLAS lane `dsv4_wo_a_grouped_linear`, which the FP8
re-serialization forces (its low-rank `wo_a` is left unquantized, so there is
no DeepGEMM cache) and which at TP=4 < `o_groups`=8 allocated its `[4096,1]`
gather and `[1024,1]` scatter scratch on every call. NVFP4's block-scaled
`wo_a` takes the DeepGEMM lane, which already staged through the fixed
buffers. `7fa06218a` shares that staging with the cuBLAS lane:

| build | FP8 captures | FP8 alloc nodes | FP8 c=1 tok/s | NVFP4 alloc nodes | NVFP4 c=1 tok/s |
|---|---:|---:|---:|---:|---:|
| `a4bd9a8a2` | 72 | 86 | 60.3 | 0 | 45.2 |
| `7fa06218a` | 72 | **0** | 60.4 | 0 | 45.1 |

Decode is unchanged, as expected: the allocations were legal under
`AUTO_FREE_ON_LAUNCH`; the fix is what lets the strict alloc audit hold across
the family.

## Learnings

**The FP8 checkpoint is the c=1 configuration for DSv4-Flash**: 59.5 vs 44.4
tok/s with no accuracy difference on MMLU. NVFP4 buys 131 GB of weight memory
(156 vs 287 GB) at a 25% c=1 decode cost; the gap is the W4AFP8 GEMV decode
kernel, which is the lever if NVFP4 must serve at c=1.

**Never bench with `ARLE_GRAPH_NODE_CENSUS=1` armed.** It is a diagnostic that
runs on every capture attempt; with rejected captures that is every step.

**The census is the only evidence for an alloc node's origin.** Two
same-sized buffers (`normed` [4096,1] and the O-LoRA `latent` [1024,1]) were
first "identified" from sizes alone and moved to graph buffers (`6469d1668`,
reverted): node count and alloc count did not move, because the captured c=1
lane is the slot lane, whose activations were already persistent. Read the
kernel neighbours in the census before touching code.

**The DSv4 slot budget has no reserve for prefill transients.** 18 slots of
1049 MB filled the 19.7 GB budget and c=8 OOMed mid-prefill. Until the planner
reserves the prefill working set, `--max-running-requests` is the guard.

**A matrix arm that cannot load is worth running anyway.** The four failures
took under two minutes each and produced the finding: a support-matrix claim
that had been wrong long enough to shape an experiment design.

**A strict audit verified on one checkpoint is not verified for the family.**
The alloc-node gate was measured at zero on NVFP4 and shipped as a default;
the FP8 checkpoint of the same model, on the same code path, still allocated
86 nodes. Nothing about "DSv4 captures cleanly" generalized across the
quantization axis, and the run that would have caught it was the one this
matrix ran.

Open: the compressor's BF16 full-history retention (5.5 GB/slot at 1M): decode
already attends the FP8-packed rows in the paged pool, but
`flashmla_prefill_attention` reads the BF16 history, so a ring there is a
prefill numerics change behind the needle gate, not host bookkeeping. The
prefill-transient reserve landed as `94a15d415` — see
`2026-08-24-dsv4-budget-prefill-reserve.md`.

## Artifacts

- `/host/arle-ops/runs/c1g/matrix/matrix.tsv`, `bench-*.json`, `mmlu-*/`
- `/host/arle-ops/runs/c1g/kvprobe/serve-{fp8,int8}.log`
- `/host/arle-ops/runs/c1g/recheck/bench-fp8-nocensus-{graph,eager}*`, `serve-{graph,eager}.log`
- `/host/arle-ops/runs/c1g/c8/bench-fp8-mtt131072-mrr8*` (c=1,8), `eval8/mmlu/` (MMLU 200), `census2/rank0.txt` (node census), `fix/bench-{fp8,nvfp4}*` + `serve-{fp8,nvfp4}.log` (`7fa06218a`)
