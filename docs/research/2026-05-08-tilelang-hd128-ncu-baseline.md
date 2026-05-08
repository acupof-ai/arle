# TileLang HD128 prefill+decode ncu baseline (sm_89 Ada)

> Status: spec + execution protocol. Run AFTER codex 0:0 P0.1 graph-capture
> bench releases the GPU. Output baseline data unblocks E1 (decode BLOCK_M
> re-tune) + E2 (prefill BLOCK/STAGES/THREADS A/B).
>
> Master strategy ref:
> [`docs/projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md)
> §10 不确定性 "FlashInfer paged prefill vs TileLang HD128 kernel-time" 解除条件。

## 1. Goal (type: diagnosis)

Establish ncu baseline for `tilelang_prefill_paged_hd128_*` and
`tilelang_decode_paged_hd128_*` kernels on **sm_89 RTX 4070 Ti SUPER**
under the same 4k longctx workload codex benches for P0.1. Output: a
single populated table that says — for each kernel family — whether
sm_89 occupancy is the binding constraint, what's the smem/register
budget headroom, and what tunable change has highest expected ROI.

This is **not** an optimization run. No code changes here. Output
feeds E1+E2 plans with grounded reasoning.

## 2. Hypothesis

Current tile defaults (`BLOCK_M=64, BLOCK_N=64, NUM_STAGES=2,
NUM_THREADS=128`) carry the comment "chosen as Hopper defaults; tuned
during the H100 spike" — explicitly Hopper-tuned, never re-tuned for
Ada. Differences that should matter:

| Hardware param | sm_90 Hopper | sm_89 Ada (4070 Ti SUPER) | Expected impact |
|---|---:|---:|---|
| smem / SM | 228 KB | 100 KB | NUM_STAGES ≥ 3 may overflow on Hopper-default |
| register file / SM | 64 K | 64 K | unchanged |
| max threads / SM | 2048 | 1536 | thread-block sizing may need lower NUM_THREADS for higher occupancy |
| FP16/BF16 tensor cores | ✓ | ✓ | unchanged |
| L2 cache | 50 MB | 48 MB | unchanged |

Specific predictions before running:

- **Prefill HD128 (BLOCK_M=64, BLOCK_N=64, STAGES=2, NTHREADS=128)**: smem
  per stage ≈ `(64×128 + 64×128 + 64×128) × 2B = 48KB` × 2 stages = 96KB.
  This already exceeds sm_89's 100KB usable budget by ~zero margin. **Predicted
  occupancy ≤ 1 block/SM**, register-bound or smem-bound. STAGES=3
  expected to fail compile or fall back to register spill on Ada.
- **Decode HD128 (BLOCK_M=64 padded, BLOCK_N=16=PAGE_SIZE, STAGES=2)**:
  qo_len=1 so only 1/64 rows is real — predicted **massive thread idle**
  (occupancy looks high but useful work is 1/64 of capacity). Reducing
  BLOCK_M to 16 or 8 should multiply effective utilization by 4-8×
  before any compiler-level changes.
- **PV matmul vs QK^T**: predicted bottleneck shift by stage — QK^T
  smem-bound, softmax compute-bound, PV matmul HBM-bound for 4k context.

If hypothesis is wrong (e.g. occupancy is 2 blocks/SM already), E2 is
likely a smaller win than predicted and we deprioritize it; E1 stays a
win since utilization argument is independent of occupancy.

## 3. Environment

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB GDDR6X, sm_89
- CUDA: query at run time via `nvidia-smi --version` and
  `nvcc --version`; record exact versions in §5.
- Model: `Qwen/Qwen3-4B` (32 q-heads, 8 kv-heads, head_dim=128)
- ARLE commit: record at run time (`git rev-parse HEAD`)
- Feature set: `--release --features cuda` default (no FP8, no graph
  capture — baseline is the eager BF16 path the kernel is most often
  exercised on)
- Workload: **match codex 0:0 P0.1 default-4k bench params exactly** so
  kernel-time numbers correspond 1:1 to the wins entry's TTFT. As of
  spec write, that's likely `--data prompt_tokens=4096,output_tokens=256
  --concurrencies 4 --max-seconds 60 --warmup 5` — verify by reading
  codex's `bench_guidellm.sh` invocation log before launching.

## 4. Method

### 4.1 Wrapper

Use the canonical `scripts/profile_ncu_guidellm.sh` (already exists) —
attach mode against a running ARLE server.

### 4.2 Sequence

1. Start ARLE serving in one shell, **same args as the codex P0.1 bench
   default arm** (no `INFER_PREFILL_GRAPH=1`):

   ```bash
   cargo run --release --features cuda --bin infer -- \
       --model-path models/Qwen3-4B --port 8000
   ```

2. In another shell, launch the ncu wrapper twice — once per kernel family:

   ```bash
   # Prefill HD128
   scripts/profile_ncu_guidellm.sh hd128-prefill-baseline \
       --target http://localhost:8000 \
       --model Qwen/Qwen3-4B \
       --kernel-family attention \
       --kernel-regex 'regex:tilelang_prefill_paged_hd128_h32_kv8' \
       --section-set full \
       --launch-skip 5 --launch-count 1 \
       --data prompt_tokens=4096,output_tokens=256 \
       --concurrencies 4 --max-seconds 30

   # Decode HD128
   scripts/profile_ncu_guidellm.sh hd128-decode-baseline \
       --target http://localhost:8000 \
       --model Qwen/Qwen3-4B \
       --kernel-family attention \
       --kernel-regex 'regex:tilelang_decode_paged_hd128_h32_kv8' \
       --section-set full \
       --launch-skip 5 --launch-count 5 \
       --data prompt_tokens=4096,output_tokens=256 \
       --concurrencies 4 --max-seconds 30
   ```

3. Stop ARLE, hand GPU back.

### 4.3 Sections to read

`--section-set full` captures all. Headline metrics to extract for §5
table:

| Metric | Extract for | Why |
|---|---|---|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | both | Occupancy floor — sm_89-vs-Hopper key |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | prefill (softmax) | Compute-bound check |
| `dram__bytes.sum.per_second` (% of peak HBM) | both | HBM-bound check (PV matmul, decode KV reads) |
| `smsp__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | prefill (QK^T, PV) | Tensor core utilization |
| `launch__shared_mem_per_block_static + dynamic` | both | smem budget vs 100 KB cap |
| `launch__registers_per_thread` | both | register pressure → blocks/SM |
| `gpc__cycles_elapsed.avg` | both | absolute time (ground truth) |
| `lts__t_sectors_aperture_device.sum.per_second` (L2 BW) | decode | KV cache locality |
| `smsp__inst_executed.sum` per warp | decode | predict thread idle (BLOCK_M=64 padded for qo_len=1) |

### 4.4 Watch list (§5 of bench-spec)

- ncu attach must NOT race ARLE startup — wait for `/v1/stats` 200 OK first.
- ncu serializes kernel launches → bench latency numbers from the
  guidellm side are NOT reliable while ncu is attached. Don't compare
  to wins-entry TTFT. Use ncu absolute kernel time only.
- `--launch-skip 5` skips warmup variance.
- If `launch__registers_per_thread > 128`, expect occupancy degradation
  on Ada — flag for E2.
- `sm__warps_active` < 25% with `dram__bytes < 50% peak` = neither
  occupancy nor HBM bound = compute-bound (likely tensor core
  underutilized) → E2 should try larger BLOCK_M to amortize.

## 5. Results table (FILL IN AT RUN TIME)

| Kernel | Occupancy | Smem/block | Reg/thread | HBM % peak | Tensor % peak | Avg cycles | Bound by |
|---|---:|---:|---:|---:|---:|---:|---|
| `tilelang_prefill_paged_hd128_h32_kv8` (BM=64 BN=64 ST=2 NT=128) | TBD | TBD KB | TBD | TBD% | TBD% | TBD | TBD |
| `tilelang_decode_paged_hd128_h32_kv8` (BM=64 BN=16 ST=2 NT=128) | TBD | TBD KB | TBD | TBD% | TBD% | TBD | TBD |

Raw `.ncu-rep` artifacts: `bench-output/<label>/`.

## 6. Interpretation rules → E1/E2 actions

| Observation | Implies | E1 (decode) action | E2 (prefill) action |
|---|---|---|---|
| Decode `sm__warps_active` < 20% | Thread idle dominates (qo_len=1 padded) | **A/B BLOCK_M ∈ {16,8}**, expected proportional speedup | n/a |
| Prefill smem/block ≥ 96 KB | Hopper-tuned exceeds Ada budget | n/a | **A/B BLOCK_N=32 or NUM_STAGES=1** to reclaim headroom |
| Prefill occupancy ≤ 1 block/SM | Register or smem bound | n/a | **A/B NUM_THREADS=64**(smaller block, higher concurrency) |
| Prefill HBM > 80% peak | Memory-bound (PV matmul) | n/a | DEFER E2 — kernel-time floor near hardware limit |
| Prefill tensor % < 40% | Tensor cores underutilized | n/a | **A/B BLOCK_M=128**(larger fragment, more matmul work per launch) |
| Decode `lts__t_sectors` low | KV cache eviction | **A/B page-size impact**(out of E1 scope, file separate task) | n/a |

## 7. Δ vs baseline

First run.

## 8. Cross-references

- Master strategy §3.3 R1 finding (kernel-impl difference is one of
  several non-graph closure paths)
- Master strategy §10 unkonwn (FlashInfer vs TileLang HD128 kernel-time)
- M_pf-graph-prefill-capture plan §Phase 0 (codex 0:0 owner)
- bench-and-trace-spec.md §3 (internal info sources) §6 (auto-iterate)
  §7 (protocol rules)
- TileLang prefill kernel: `crates/cuda-kernels/tools/tilelang/batch_prefill_paged_hd128.py`
- TileLang decode kernel: `crates/cuda-kernels/tools/tilelang/batch_decode_paged_hd128.py`

## 9. After this runs

1. Fill §5 table with real numbers.
2. Apply §6 interpretation rules to pick E1/E2 specific tunables.
3. File E1 plan (`docs/plans/E1-tilelang-decode-hd128-blockm-retune.md`)
   and E2 plan (`docs/plans/E2-tilelang-prefill-hd128-sm89-retune.md`)
   with grounded acceptance criteria.
4. Cross-link this doc from both plans.
5. Update master strategy §10 to mark "FlashInfer vs TileLang HD128
   kernel-time" unknown as **partially resolved** (we now know the
   ARLE-side baseline; FlashInfer-side comparison is a separate later task).
