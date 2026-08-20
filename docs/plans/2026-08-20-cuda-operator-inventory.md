# T0 — CUDA operator inventory

> Parent: [CUDA operator organization](2026-08-20-cuda-operator-organization.md).
> Census date 2026-08-20, checkout `80d8ded87`. Families 1–10 per the parent
> plan's "Operator families" section.

## Raw-FFI consumption outside cuda-kernels

243 raw call sites in 37 files (197 file×symbol pairs); 146 distinct symbols
consumed raw, 140 without a typed wrapper (6 wrapper-bypass). 15 example/bench
files hold 26 sites; the 22 production files hold 217. `check` (nccl.rs:173) is
a safe helper, not a kernel — 12 of the sites.

Heaviest production consumers:

| Consumer | Sites | Dominant families | Owner |
| --- | ---: | --- | --- |
| `infer-cuda/src/attention.rs` | 68 | 4 attention, 5 KV, 3 gemm, 1/2 prep | dsv4 + qwen3 |
| `infer-cuda/src/qwen35_attention.rs` | 35 | 4 attention, 6 recurrent, 2 norm | qwen35 |
| `infer-cuda/src/ops.rs` | 14 | 1 embedding, 2 norm/elementwise, 3 gemm, 8 argmax | shared |
| `infer-cuda/src/tp.rs` | 12 | 9 custom all-reduce (`arle_car_*`) | shared |
| `infer-cuda/src/qwen35/dspark.rs` | 10 | 8 sampling, 4 ring prefill | qwen35 |
| `infer-cuda/src/attention/flashmla.rs` | 9 | 4 FlashMLA decode | dsv4 |
| `infer-cuda/src/ops/quant_linear{,_fp8,_fp4,_int}.rs` | 21 | 3 quant linear | shared (T2 in flight) |
| `infer-cuda/src/{hc,dsv4/mtp,dsv4/dspark,qwen35*,loader}.rs` | 28 | 2, 6, 8, 3 | dsv4/qwen35/shared |
| `autograd/src/backend_cuda/*.rs` | 9 | 6 recurrent, 4 ring attention | autograd |

Wrapper-bypass (typed wrapper exists, still called raw): `arle_fa2_sm70_attention_cuda`,
`arle_fa3_fwd_hd256_bf16_cuda`, `arle_fa3_real_kernel_marker_cuda`
(qwen35 attention) and `dsv4_deepgemm_{fp8_gemm_nt,fp8_paged_mqa_logits_fused_cache,pack_quantize_bf16_to_fp8}_cuda`
(attention.rs calling moe.rs wrappers' symbols).

Fully wrapper-routed already (zero raw external consumers): `ffi/moe.rs`
(34 symbols, consumer `infer-cuda/src/moe.rs`) and the NCCL collective path.

## Typed-wrapper coverage in cuda-kernels

349 extern fn decls in 14 `ffi/*.rs`; 115 wrapped, 234 production symbols
unwrapped:

| ffi file | total | wrapped | note |
| --- | ---: | ---: | --- |
| gemm.rs | 71 | 36 | GEMV/Marlin/dequant set unwrapped (T2 tranche 1B target) |
| misc.rs | 50 | 1 | FlashMLA, DSA, MHC, compressor, prepare_qk — all raw (T3 target) |
| attention.rs | 47 | 23 | prefill/decode prep + ring set unwrapped (T3) |
| recurrent.rs | 40 | 0 | GDR/conv1d/FlashQLA (T4) |
| ffi/moe.rs | 34 | 20 | 14 EP-transport symbols unwrapped (T5) |
| nccl.rs | 20 | 17 | wrapped via `collective.rs` |
| comm.rs | 16 | 0 | custom all-reduce, consumed by `tp.rs` (T6) |
| kv.rs | 14 | 13 | one gap |
| elementwise.rs | 10 | 1 | T1 target |
| norm.rs | 9 | 0 | T1 target |
| embedding.rs | 8 | 0 | T1 target |
| sampling.rs | 8 | 0 | T6 target |
| quant.rs | 5 | 3 | |
| gemm_tests.rs | 17 | — | test-only externs |

## Registry gaps

`operators/registry.toml` binds one semantic operator
(`qwen.fp8_dense_projection`, 3 implementations, 1 policy);
`benchmarks/operators/optimal.json` binds only it. Families 1, 2, 4–10 and
most of family 3 have production launches and zero registry binding. Each
tranche T1–T7 adds its family's binding; T8 closes.

## Autograd classification (29 files)

12 backward, 9 forward, 2 rollout, 1 optimizer, 1 bridge, 2 fused
attention fwd+bwd, 2 uncertain (`attention_decode_online.cu`,
`linear_attention.cu`). Serving-side equivalents exist for 7
(`embedding`, `rms_norm` fwd, `bridge` casts, `fp8_block_scaled` dequant,
`add_into`, `silu` ≈fused, `rollout` argmax/embedding); 18 are NVRTC-only
training math. T7 sharing surface is those 7 plus the 2 uncertain after
numerical-contract proof.

## T1 scope (embedding / norm / elementwise)

16 launch sites in `infer-cuda`, 15 raw ffi, 9 concentrated in thin `ops.rs`
helpers (`embedding_batch`, `rms_norm_batch`, `add_batch`, `silu_mul`,
`split2`, `split_qkv`, `silu_mul_fused`, `lora_scaled_add_into`); the rest in
`attention.rs` (2× `rms_norm_batched_cuda`), `qwen35.rs` (2 offset variants),
`qwen35_attention.rs` (`rms_norm_gated_cuda`), `qwen35_lora.rs`
(`add_scaled_row_cuda`). T1 = relocate the `ops.rs` helper layer behind
typed launchers in `cuda-kernels` and route the stragglers through them.

Constraint: 9 family symbols (`embedding_decode_cuda`,
`q{4,5,6}k_embedding_*`, `fused_add_rms_norm*`, `add_assign_cuda`,
`add_bf16_into_f32_cuda`, `add_scaled_row_segment_cuda`,
`rms_norm_batched_f32_in_cuda`) have no infer-cuda caller — they are consumed
only by the HIP lane via mirrored decls in `crates/hip-kernels`. T1 wraps only
CUDA-lane consumers and leaves HIP-only symbols untouched.

## T0 exit

Every production launch above maps to one family and one owner; registry gaps
are explicit per family. Full per-site tables live in the session transcript;
this document is the binding summary.
