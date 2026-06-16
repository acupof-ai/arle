# Qwen3.6 FP8 Decode Vectorized Dequant

## SLO-shape probed?

N. This entry closes the c=1 decode regression gate on the 4095/256 shape. It
does not claim a default flip or a high-concurrency throughput verdict; c=8/32
was blocked by a dirty pod GPU memory state.

## Roofline check

| Op | Metric | Before | After | Verdict |
| --- | ---: | ---: | ---: | --- |
| FP8 SwiGLU decode | DRAM / SM peak | 10.9% / 33.5% | 12.3% / 25.5% | latency-bound, not bandwidth-bound |
| FP8 down decode | DRAM / SM peak | 4.60% / 19.87% | 5.07% / 19.62% | latency-bound, not bandwidth-bound |
| Dense FP8 block GEMV | DRAM / SM peak | n/a | 17.56% / 42.08% | still not HBM-saturated |

The root cause was not HBM bandwidth. At B=1 the FP8 decode kernels were
latency/ALU-bound, so halving weight bytes did not pay back the scalar FP8
decode tax. The fix reduces that tax by using four-wide FP8 conversion and
folding the block scale outside the per-element FMA.

## Step 0 decode-graph recheck

After this patch, a load-bearing recheck tested whether the remaining
~22 ms/token was mostly eager host launch overhead. It was not.

Evidence: `/tmp/arle-step0-nsys-eager-full/trace.sqlite`, Qwen3.6 FP8,
4K prompt, c=1, graph env unset, no `ARLE_QWEN35_*_PROFILE`.

| Component | Mean ms/token | Share |
| --- | ---: | ---: |
| attention/KV kernels | 13.0965 | 57.0% |
| dense GEMV/norm kernels | 2.5912 | 11.3% |
| MoE kernels | 4.0415 | 17.6% |
| linear-attention kernels | 0.6421 | 2.8% |
| sampling | 0.0798 | 0.3% |
| other kernels | 0.6374 | 2.8% |
| total GPU kernel active | 21.0885 | 91.8% |
| inter-kernel gap / CPU idle remainder | 1.8762 | 8.2% |

The same trace saw 1194 kernel launches/token and no CUDA graph runtime events.
So graph was definitely off, but the measured wall was not 97% host launch gap.

Graph A/B then used the same rebuilt binary and same shell:
`/tmp/arle-step0-graph-ab-1781596033`.

| Mode | max1 wall | max257 wall | Completion tokens | Slope ITL |
| --- | ---: | ---: | ---: | ---: |
| eager | 1.5997 s | 7.3568 s | 1 -> 257 | 22.489 ms |
| `ARLE_QWEN35_DECODE_GRAPH=1` | 1.6003 s | 6.9368 s | 1 -> 257 | 20.846 ms |
| graph delta | +0.0% | -5.7% | same | -7.3% |

Graph correctness passed: `/tmp/arle-step0-graph-needle-1781596452/result.json`
retrieved `BLUE-73-MANGO` under graph-on, and server logs showed one capture plus
100+ replays with no fallback. Graph is correct and modestly faster, but this is
below the documented >=10% default-flip threshold and is not the 4-10x lever the
host-launch-only hypothesis predicted.

Current conclusion: do not spend the next tranche on a MoE/dense GEMV roofline
rewrite by default. The measured c=1 decode wall is dominated by full-attention
KV work plus many tiny kernel/sync boundaries; that needs its own RCA before a
new optimization target is licensed.

## Goal

Optimization: make Qwen3.6 FP8 c=1 decode no slower than BF16 after the fused
decode dispatch fix still left FP8 at ITL 27.93 ms vs BF16 24.41 ms.

## Hypothesis

If the c=1 gap is mostly scalar FP8 dequant and per-element scale overhead, then
vectorized FP8 conversion plus scale-folding should erase most of the 14% decode
regression without changing the higher-level dispatch path.

## Command

```bash
# Isolated per-kernel timing.
cd /data01/arle-qwenfp8-smoke
CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDARC_CUDA_VERSION=12090 \
  NVCC_CCBIN=/usr/bin/g++ RUSTFLAGS="-C link-arg=/tmp/ssl_peer_cert_compat.o" \
  ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1 ARLE_CUDA_KERNEL_SET=dsv4_flash \
  ARLE_CUDA_DISABLE_FLASHMLA=1 \
  cargo build --release --features cuda -p infer-cuda --example fp8_decode_probe
INFER_CUDA_DEVICE=3 ARLE_FP8_DECODE_PROBE_ITERS=1000 \
  ./target/release/examples/fp8_decode_probe

# NCU bound check.
INFER_CUDA_DEVICE=3 ARLE_FP8_DECODE_PROBE_ITERS=1 \
  ncu --target-processes all --kernel-name-base function \
  --kernel-name "regex:dsv4_fp8_grouped_down_decode_kernel" --launch-count 1 \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  ./target/release/examples/fp8_decode_probe
```

The e2e gate used the same rebuilt `target/release/arle` binary for FP8 and
BF16, `--num-slots 1 --total-pages 272 --page-size 16`, and a 4-run streaming
OpenAI-compatible 4095/256 client. Raw JSON is under
`/tmp/arle-fp8-decode-fix/`.

## Environment

- Backend: CUDA
- Model: `/data01/models/Qwen3.6-35B-A3B-FP8` vs `/data01/models/Qwen3.6-35B-A3B`
- Hardware: NVIDIA H20, CUDA 12.9 toolchain
- Commit: `26014f4e`
- Feature set: `cargo build --release --features cuda --bin arle`
- Non-default env: `ARLE_CUDA_ENABLE_DEEPGEMM_NATIVE=1`,
  `ARLE_CUDA_KERNEL_SET=dsv4_flash`, `ARLE_CUDA_DISABLE_FLASHMLA=1`

## Results

### Isolated kernel probe

| Stage | Before FP8 ms | After FP8 ms | BF16 ms | After delta vs BF16 |
| --- | ---: | ---: | ---: | ---: |
| dense FP8 block GEMV B=1 | 0.0169 | 0.0052 | 0.0061 | -15.7% |
| MoE fused SwiGLU decode | 0.0319 | 0.0278 | 0.0243 | +14.4% |
| MoE fused down decode | 0.0375 | 0.0346 | 0.0353 | -2.0% |
| MoE scatter/combine | 0.0051 | 0.0053 | 0.0053 | 0.0% |
| MoE fused total | 0.0746 | 0.0677 | 0.0650 | +4.2% |

### 4095/256 c=1 e2e

| Backend | TTFT median | ITL median | Latency median | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 1793.98 ms | 24.35 ms | 8.003 s | 31.99 |
| FP8 | 1811.68 ms | 22.38 ms | 7.508 s | 34.10 |
| FP8 delta | +1.0% | -8.1% | -6.2% | +6.6% |

This meets the requested c=1 decode gate: FP8 ITL is now below BF16.

## Problems

- The FP8/BF16 text preview for this synthetic latency prompt was degenerate in
  both arms (`"the the ..."`), so this e2e run is a latency A/B only, not a
  correctness gate. The CUDA-level numeric guard for this patch is the new
  `dsv4_fp8_grouped_decode_matches_reference` test, plus the existing FP8 block
  GEMV reference test.
- c=8/32 e2e was not trusted. One discarded attempt set
  `--total-pages 8704`, which inflated per-slot max sequence to 139264 and
  clamped slots. After killing the serves, all H20s still reported about 45 GB
  used with no visible ARLE/guidellm process, so a high-c sweep would be
  confounded by pod state. No high-c throughput verdict is recorded here.
- The kernels are still not HBM-saturated, but the later Step 0 trace shows a
  MoE/dense GEMV roofline rewrite is not the next load-bearing lever for c=1
  ITL. The remaining wall is mostly attention/KV plus tiny-kernel orchestration,
  not the FP8 dequant path alone.

## Learnings

- For Qwen FP8 B=1 decode, scalar e4m3 conversion plus per-element scale can
  erase the byte-size win before the kernel reaches the bandwidth-bound regime.
- The block-scaled dense GEMV fast path must stay shape-gated on
  `K % 16 == 0 && block_k % 16 == 0`; odd shapes keep the scalar fallback.
- `--total-pages` is per-slot sequence capacity in this serve path, not a global
  page budget to multiply by slot count.
- SGLang's `/workspace/sglang/sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu`
  uses CUTLASS grouped tensor-op FP8 for grouped-M GEMM. That is the right
  upstream reference for prefill/larger-M grouped FP8, but it is not a direct
  replacement for this row-compact w8a16 B=1 decode kernel.

## Delta vs baseline

- Baseline diagnosis: [`2026-06-16-qwen36-fp8-decode-fused-root-cause.md`](2026-06-16-qwen36-fp8-decode-fused-root-cause.md)
- Prior e2e regression before this patch: FP8 ITL 27.93 ms vs BF16 24.41 ms
  (+14.4% FP8 slower).

| Metric | Prior FP8 | Now FP8 | BF16 anchor | Verdict |
| --- | ---: | ---: | ---: | --- |
| c=1 ITL | 27.93 ms | 22.38 ms | 24.35 ms | FP8 now -8.1% vs BF16 |
| c=1 output tok/s | 28.78 | 34.10 | 31.99 | FP8 now +6.6% vs BF16 |

## Verification

```bash
cargo fmt --check
CUDARC_CUDA_VERSION=12090 cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib
CUDARC_CUDA_VERSION=12090 cargo clippy -p infer-api --release --no-default-features --features cuda,no-cuda --lib -- -D warnings
CUDARC_CUDA_VERSION=12090 cargo check -p cuda-kernels --release --no-default-features --features cuda,no-cuda --tests
CUDARC_CUDA_VERSION=12090 cargo clippy -p cuda-kernels --release --no-default-features --features cuda,no-cuda --lib -- -D warnings

# Pod CUDA tests:
cargo test --release --features cuda -p cuda-kernels fp8_block_scaled_gemv_matches_reference -- --nocapture
cargo test --release --features cuda -p cuda-kernels dsv4_fp8_grouped_decode_matches_reference -- --nocapture
```

`cargo clippy -p cuda-kernels --tests -- -D warnings` is still blocked by
pre-existing unrelated test lints in `kv_quant.rs:1606`, `kv_quant.rs:1695`,
and `tensor.rs:3597`; those files were not changed in this tranche.

## Artefacts

- FP8 e2e JSON: `/tmp/arle-fp8-decode-fix/e2e-fp8-decodefix.json`
- BF16 e2e JSON: `/tmp/arle-fp8-decode-fix/e2e-bf16-decodefix.json`
- Pod smoke tree: `/data01/arle-qwenfp8-smoke`
