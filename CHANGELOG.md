# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog should record more than feature additions. It should also record:

- breaking changes
- deprecated surfaces
- support-matrix changes
- migration notes when user action is required

Related governance docs:

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)

## [Unreleased]

- **FIX (accept) — w2s no longer round-trips the FP8 base through host** (2026-08-13; `62017ec8a`, `bc96d29ec`, `f77ca2eb5`, [errors](docs/experience/errors/2026-08-13-w2s-fp8-base-offload-roundtrip-was-lossy.md)). Offloading the 27B `CudaFp8BlockScaled` base dequantized it to f32 host and re-uploaded it as bf16, doubling 27.9 GB to 54 GB (OOM on a 95.2 GB H20) and leaving π_base for the global KL no longer equal to the checkpoint. The mechanism existed for a "27B aux" that is actually 0.8B, so it is deleted along with `upload_frozen_bf16_from_host` and `rope_cache_ids` (net -271 lines). Base stays FP8-resident: 27.9 GB after base+student, 33.5 GB after four 0.8B aux, 61.7 GB free. Added the missing flashqla AOT GDN geometry H=16/Hg=16 for the 0.8B aux forward. `--steps 2` reaches `RUN_EXIT=0` with step 0 loss=25.158342, consistency=0.7372.

- **FIX (accept) — prefix-cache metrics report actual restored work** (2026-08-13; `c112b81de`, [bench](docs/experience/wins/2026-08-13-kv-prefix-metrics-and-oversubscription-slice.md)). Raw and backend-licensed radix matches are counted at lookup; hits, tokens, pages, and resident reuse are counted from the restored token boundary. A Qwen3.6 8191-token common prefix with no sidecar now reports zero hit and one fallback, while the 8192 boundary reports 8192 restored tokens and 512 pages. Needle retrieval passed 3/3 at 512, 4096, 8192, and 12000 tokens. The existing whole-slot minimum decode slice is configurable; its default remains 8 because formal 32K A/B trials were blocked by external GPU-process termination.

## [0.5.5] - 2026-08-13

- **FEATURE (accept) — batched paged decode for FP8/INT8 KV pools** (2026-08-13; `ff33bdb77`). Removed three BF16-only gates that blocked quantized KV from batched decode (batch > 1): the `for_rows` batch==1 restriction on quant metadata, the `for_decode_batch` format check, and the `submit_decode_batch` `paged_kv_bf16()` gate. Multi-row `new_token_rows` concatenates per-slot; `quant_decode_meta` uses all slots. The `decode_attention_varlen_quantized` kernel already accepted batch_size, so no kernel change was needed. Verified: 4 concurrent needle requests with FP8 KV all passed, needle ladder 115-2000 tok 3/3 exact.

- **REFACTOR (accept) — unify qwen35 FP8/INT8 KV on the NHD split-KV kernel, delete the dead TileLang FP8 path** (2026-08-13; `64be73980`). The TileLang `paged_attn_fp8_v1` kernel read HND layout but the FP8 quantizer writes NHD, so every FP8 paged decode read garbage. qwen35 FP8 now routes through the NHD split-KV varlen kernel (same as INT8), selected by an `int8_kv` bool; `decode_attention_varlen_int8` is renamed `decode_attention_varlen_quantized`. The dead TileLang FP8 path is deleted (3 .py sources, the build.rs emitter, kernels.toml ABI + 11 entries). Net: -880 lines.

- **INFRA — build/sync hardening** (2026-08-13; `64be73980`). `CARGO_BUILD_JOBS=32` in pod-build-env.sh (the 180-way nvcc build OOMs the shared box); `tar --no-xattrs` in pod.sh sync (silences macOS provenance pax warnings).

- **BENCH — KV dtype comparison on H20, ThinkingCap-Qwen3.6-27B-FP8** (2026-08-13). BF16 / INT8 / FP8 KV compared at c=1/4/8, 16 requests per level, fixed 214-token outputs. All three formats completed 16/16 at every level with 0 errors and 0 correctness failures. Throughput (out tok/s): BF16 53.4 / 125.8 / 185.7, INT8 — / 123.9 / 175.0, FP8 49.8 / 126.6 / 185.9 at c=1/4/8. TTFT p50: BF16 48 / 347 / 352 ms, FP8 49 / 344 / 363 ms. ITL p50: BF16 15.8 / 20.2 / 21.3 ms, FP8 17.3 / 20.3 / 21.3 ms. e2e p50: BF16 1.87 / 3.14 / 4.13 s, FP8 2.00 / 3.15 / 4.09 s. FP8 scales near-linearly (49.8 → 185.9 tok/s, 3.73x at c=1→8) and is statistically tied with BF16 at c=4/8; INT8 trails both by ~6% at c=8. Concurrent needle (4 batched requests, ctx=2000, pool=4096): 4/4 passed on all three formats — each request recovered its own per-request code, so batched decode does not cross-contaminate requests under any KV format. Raw JSON: `/tmp/bench-{bf16,int8,fp8}.json` on the pod.

## [0.5.4] - 2026-08-12

- **FEATURE (accept) — INT8 KV cache support for Qwen3.5 paged attention** (2026-08-12; `b20859520`). Qwen3.5's paged attention path now supports `KVFormat::INT8` alongside BF16 and FP8E4M3. Quantization uses `quantize_paged_kv_single` (per-token per-head symmetric INT8, scale = absmax/127); attention reuses the FP8 split-KV varlen kernel `decode_attention_varlen_fp8` with `int8_kv=true`. The kernel gained runtime head_dim dispatch (128/256) since Qwen3.5 uses head_dim=256. Verified on Qwen3.6-27B-FP8: needle gate exact at 115/300/1000/4000/8000 tokens (deterministic, matches BF16), temp arm PASS (no glued repetition), GSM8K 17/17 = 100%.

- **PERF (accept) — FP8 dequant GEMV floor lowered to M>=2: WMMA GEMM replaces cuBLAS for small batches** (2026-08-12; `b20859520`). `QWEN_FP8_DEQUANT_GEMM_MIN_M` dropped from 100000 to 2: M=1 decode stays on the batched GEMV, M>=2 uses the WMMA GEMM which avoids the 2× memory blowup of cuBLAS for small batches (DSpark verify, MoE expert routing).

## [0.5.3] - 2026-08-11

- **PERF (accept) — DSv4 whole-slot KV tier serialization simplified: swap_out/swap_in only persist mutable fields, FP32 carry skipped via `fp32_carry_stale`** (2026-08-11; `3d499a4fb`). The whole-slot park image previously serialized every per-(slot,layer) buffer including constant-after-init fields and FP32 carry accumulators. `swap_out` now captures only `fp8_kv_sw_bootstrapped` and `fp8_kv_comp_packed_rows` from FlashMLA (the rest are init constants); the compressor's FP32 carry buffers are skipped when `fp32_carry_stale` is set, and on `swap_in` the flag is propagated so the next forward reseeds FP32 from the bf16 carry. `copy_pages_to_host_no_sync` removes the per-layer sync barrier in swap_out (one global sync at the end), and `flashmla_ensure_band(zero: false)` skips zeroing on swap-in since pages are overwritten. Verified on H20 TP=4 with `--kv-oversubscription`: 1184 park/promote cycles, 0 errors, correct output ("Paris", "42") after promote. Park latency ~57 ms, promote ~100 ms for seq_len 18–137 (fixed overhead dominates; KV data is ~38–80 KB).

- **FIX (accept) — DeepGEMM native build fixes for CUDA 12.9** (2026-08-11; `64ffa8dcf`). `mega_moe.cuh`'s `Data` constructor was `constexpr` with an assert body, which CUDA 12.9 rejects ("statement may not appear in a constexpr constructor"); removed `constexpr` and replaced `DG_UNIFIED_ASSERT` with `assert`. Call sites in `sm90_fp8_mega_moe.cuh` and `sm100_fp8_fp4_mega_moe.cuh` changed `constexpr auto` to `const auto`. The DeepGEMM JIT flag `-std=c++20` broke the flashmla CUTLASS `is_any_of` fold expression (`... || std::is_same_v<T, Us>` → "pack expansion does not make use of any argument packs"); changed to `-std=c++17` (the build itself uses c++17; without `-fconcepts`, c++17 libstdc++ stays `requires`-free).

- **VERDICT (reject) — FlashQLA `block_DV=32` improves wave count but fails numerical parity** (2026-08-10; `3582c881a` + `e2a837ff6`, [error](docs/experience/errors/2026-08-10-flashqla-block-dv32-numerical-kill.md)). `ncu` confirmed the shipped Q=2048/H=48 `fq_fwd` is dependency-stalled: 92.67% of cycles have no eligible warp, long scoreboard is 87.2% of the issued-instruction interval, DRAM is 2.80%, and the 96-CTA grid gives 1.23 waves over 78 H20 SMs. The first 32-wide build exposed a separate AOT defect: the kernel expected four value tiles per head while the wrapper hardcoded `2 * H`, leaving half the output domain unlaunched. The wrapper now derives its grid from the kernel module's `FQ_DV/FQ_BLOCK_DV`; the shipped 64-wide grid remains 96 and the corrected candidate is 192. On that real 192-CTA path, 790 in-forward FlashQLA-vs-recurrent metrics exceed the 5% error budget; at layer 0/Q=2048, state max error grows 6.676→63.542 and output max error 0.21875→2.75. The candidate is killed before performance timing; `FQ_BLOCK_DV=64` remains shipped.

- **VERDICT (reject and revert) — unmeasured CUDA split, fast-math, and GEMV changes regressed correctness** (2026-08-10; `17c60435e` + `9a6ca91ac9`, [error](docs/experience/errors/2026-08-10-unmeasured-cuda-micro-optimizations-regressed-correctness.md)). The original run did not preserve operator outputs or error metrics, so the numerical cause is unknown. The rollback restores all ten CUDA files exactly to their pre-change versions. An isolated 1xH20 CUDA release build at `9a6ca91ac9` completed, Qwen3.6-27B RAW needle passed 3/3 at 2K/8K/16K with zero misses, and the sampled coherence arm completed 200/200 tokens with zero glued repetition. Kernel optimization acceptance now requires an operation-class numerical contract before timing: bit identity for data movement, an output-ULP bound for pointwise math, reference-error regression bounds for reductions, and the exact-binary model gate.

- **CALIBRATION — the anchor's `nsys` window over-states prefill kernel shares by ~2x, and `pack_quantize` delivers 5.12x in situ** (2026-08-09; `nsys` at `5cfe8494f`, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md)). The `pack_quantize` rewrite predicted ~5.4% of wall from a 7.75% window share and measured **−2.98%**; a capture on the new binary settles what the 48% shortfall was. **Not partial engagement:** same 30 s steady-state window, same analyzer, wall 29,642 / 29,693 ms and GPU busy 96.5% / 96.4% across the two trees — the kernel goes **2216.19 → 441.24 ms, 7.75% → 1.54% of kernel time, 141.78 → 27.70 us per launch = 5.12x**, and every one of 15,931 launches carries the new `/8` grid, the `gridX` distribution mapping 1:1 onto the baseline's at exactly one eighth with no launch at an undivided grid. **The shortfall is window placement, and it is now a number**: a term of run-level share `s` sped up `f` times returns `s(1 - 1/f)` of wall, so `2.98% / (1 - 1/5.12) = 3.70%` run-level against `7.75% x 96.4% = 7.47%` in-window — **2.02x, in a 1.7-2.4x band given the A/B's 0.5% BASE spread.** **Every `c_prefill` share must be halved before predicting end-to-end value**; the perf chain's model section carries the factor and every projection in it was re-derived. Mechanism inferred, not measured: the capture sits 31-38% into the run with all 16 slots saturated on prefill, while the ramp and drain are decode-heavy at low batch. **The corollary is larger than the calibration — roughly half this workload's wall clock is not set by kernel time at all** (128 requests over 16 slots, TTFT p90 73-184 s), so every remaining kernel lever combined is worth **~11% of wall** against the 22.7% the kernel-time column suggests, and scheduling is worth more and is unmodelled. Same capture re-ranks the open items: **`fq_fwd` (FlashQLA) is now the largest non-GEMM term at 1708 ms / 5.97%**, and its floor was derived from `tools/tilelang/flashqla_gdr.py` plus the trace grid — grid `(96, 1, 512)` is `ceildiv(DV 128, block_DV 64) x H` so **H = 48 heads** (cross-checked against `rms_norm_gated`'s grid putting the gated-delta output at 48 x 128 = 6144), six GEMMs per 64-token chunk per CTA give `3.405e13` FLOP, floor **230 ms against 1708 measured = 13.5% of the bf16 peak**. **But it is `T_stall`, not `T_inflation`, and that bucket has never been measured**: 96 CTAs on 78 SMs caps utilization at `96/(2x78) = 62%` (lifting the floor to 371 ms; `block_DV = 32` would give 192 CTAs and 82%, a concrete falsifiable A/B), and a 2048-token segment is **32 serial chunks of six dependent 64x128x64 GEMMs** — Hopper `wgmma`'s minimum M — running at **18% per chunk step**. Its 1478 ms nominal headroom is the same size as the tail's remaining 2248 ms and is **not comparable to it**: one is inflation with a demonstrated 5.12x remedy, the other a latency structure whose reclaimable fraction is unknown until `ncu` splits the warp stall reasons. **Two numbers from this capture that must not be used**: the per-kernel "where did the 1775 ms go" comparison is cross-tree rather than cross-kernel (the baseline predates the draft-attention change, whose `nonpaged_prefill_attention_kernel` went 16 args to 18 and 270 launches to 30), and the two profiled runs' 11,587 against 13,760 tok/s is cross-day, cross-GPU and profiler-loaded, not a regression.

- **PERF (accept) — `pack_quantize` at 16 B loads: 5.13x, still bit-identical, and the reciprocal shortcut rejected** (2026-08-09; `5cfe8494f`, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md)). Follow-on to `554173b36`. **16 lanes per 128-element quantization block** instead of 32, so each lane holds 8 bf16 from one `uint4` 16 B load, `__shfl_xor` offsets stay below 16 so each half-warp reduces its own block, and `__nv_cvt_float2_to_fp8x2` converts two values per instruction. `ncu`: **7.95 M instructions against the original 46.61 M (5.87x fewer), 20.6 us against 100.7 (4.89x), DRAM 5.3% -> 25.7%** — the speedup tracks the instruction count at every step, confirming the kernel was never memory-bound. Microbench 4.98x / 5.13x / 5.06x at the three traced shapes. **Rejected: multiplying by a reciprocal instead of dividing is a further 1.43x (7.35x total, 2.5 TB/s) and is NOT bit-identical** — against full-mantissa random bf16 it shifts **3987 / 13335 / 4974 elements, 3.8e-4**, by one e4m3 ulp. Worth ~130 ms, **0.44% of anchor wall**, against permanently losing the strongest gate this kernel has. **The structured test pattern reported 0 mismatches for that same code** — a bit-identity claim is only as strong as its input's coverage of the rounding boundaries. **The offline gate was widened to the grouped MoE path, which the first one missed**: the change rewrites the thread→work mapping that `active_experts` / `active_offsets` / `active_counts` index through, and the original gate only ran `active_count = 1`. Eight shapes, `active_count` 1/3/4/8, ragged per-expert counts, experts in reverse order, sentinel-fill spill check on rows past each count — **0 value, 0 scale, 0 spill**. That gate's own first run failed at `active_count = 3` because the harness mapped all three groups to one expert id; the harness, not the kernel. Serve gate `GATE_EXIT=0`, all 5 default rungs 3/0/0 DET, identical to BASE rung for rung, serve log confirming the changed kernel was exercised. **No A/B is claimed**: the delta over the shipped 4/lane form is ~171 ms of a 409 s wall, **0.42%, below the anchor bench's own 0.5% BASE spread**, so the run would be a null by construction. Still instruction-bound at SM 69.2% against DRAM 25.7%, at ~0.83 instructions per element; the remaining path is fusion into the producing kernel's epilogue, not more tuning.

- **PERF (accept) — `pack_quantize` was instruction-bound, not memory-bound; one warp per quantization block gives 3.67x on the kernel and −2.98% anchor wall** (2026-08-09; `554173b36`, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md)). The kernel that converts bf16 activations to FP8 blocks for DeepGEMM is **7.8% of prefill kernel time** (2205 ms of 28,168) and moved 593 GB at **0.27 TB/s — 7.6% of achievable bandwidth**. It gave one 128-thread block to each 128-element quantization block: **one `uint16_t` per thread**, a shared-memory reduction with two `__syncthreads`, and a second pass re-reading the input to scale it. **`ncu` refuted the obvious diagnosis** — the kernel was never starved on memory: **DRAM throughput 5.2% against SM throughput 81.3%**, occupancy 89.8%, IPC 3.30. It was saturating the issue pipeline with address arithmetic, reduction, and synchronization while the memory system sat idle. Rewritten to one **warp** per quantization block, four bf16 per lane via `ushort4`, values held in registers so `amax` and the scaled write share a single read, `__shfl_xor` instead of shared memory: **11.81 M instructions against 46.79 M (3.96x fewer), 27.6 us against 101.4 (3.67x), DRAM 19.2%** — the speedup equals the instruction reduction. `extern "C"` signature, the `cols % 128 == 0` precondition, and every Rust caller unchanged; only the grid is divided by 4. **Bit-identical**: 0 value and 0 scale mismatches driving the real entry point against a CPU reference at 2048x{5120, 17408, 6144} plus the ragged 336x5120 and 54x17408; microbench 3.53x / 3.70x / 3.62x. Gate clean on both arms — `lever_gate.sh` **default** ladder (115 / 300 / 446 / 2000 / 8000, the short rungs straddling the 241 boundary included), 3 runs per rung, **3/0/0 exact and deterministic everywhere, `GATE_EXIT=0`**. Anchor A/B, dataset sha matching `baselines.md`, one fresh serve per trial, order new-base-base-new-new-base, both models `mlock`ed so every serve reached ready in 10 s: **wall −2.98% (BASE 408.37–410.27 s at 0.5% spread against NEW 394.08–400.89), total tok/s +3.34%, TPOT −5.34%, out tok/s +6.07% — the three-trial ranges do not overlap on any of the four, the worst NEW trial beating the best BASE trial on each.** Both TTFT metrics are inside noise; this is not a TTFT lever. **The SOTA row does not move**: the A/B runs one fresh serve per trial at c=16 while the row is a single ascending c=1→16 sweep inheriting a warm cache, so the two report different quantities (TPOT 257 ms here against the row's 110.52) and re-measuring the champion needs the sweep on both arms. **The return is ~55% of the ~5.4%-of-wall prediction and the shortfall is unexplained** — dilution by decode and queueing is the likely reading, but an `nsys` capture on the new binary is what settles it. Method: **a kernel at 7.6% of bandwidth is not necessarily bandwidth-bound**, and had the bandwidth diagnosis been trusted, TMA / async copy / warp specialization would all have returned nothing here.

- **MODEL (supersede) — the anchor window is now an exact partition, prefill arithmetic is at the hardware floor, and the data-prep tail is the last kernel lever** (2026-08-09; `70760bc09`, [bench](docs/experience/wins/2026-08-09-anchor-window-partitioned-exactly-prefill-arithmetic-is-finished.md)). The perf chain's model was a sum of point estimates, `c_prefill = 251.9 us/token`, closing **79%** of the window with a 5.9 s residual read as one unexplained effect. Replaced by a partition: every kernel is assigned to prefill or decode by whether its start falls in one of the seven decode windows, so the ledger sums to the window by construction — **28,601 ms kernel = 28,168 prefill + 433 decode, over 90,208 prefill and 349 decode tokens, `c_prefill` = 312.3 us/token**. The residual was three ordinary things: small-`M` GEMM variants, the elementwise tail, and a `pack_quantize` count low by 40% (15,631 launches, not 8,463). **FA3 is closed at 140.8 TFLOPS = 95.1% of the bf16 peak**, which the doc had called its highest-value open measurement and declared unscorable because the persistent (78,1,1) grid hides sequence lengths — recovered from the *slope* instead: joining each launch to its preceding `split_qkv` gives `Q`, and the `Q = 2048` launches form a ladder stepping **732 us per 2048 tokens of KV depth, constant to ±0.8% over nine rungs**, one rung being a known `4 x 2048 x 2048 x 24 x 256` rectangle with no unknown in it. Mean effective KV depth **18.0K**, range 5.9K–30.5K. The "33 vs 61 chunk count" that blocked this for a day was not a contradiction: 33 is full 2048-token chunks, 61 is *segments*, since a chunk spanning two requests issues as two launches — **44 chunks, 55 prefill segments, 6 decode passes, `977 = 16 x 61`**; the doc's earlier "176 chunks" was wrong. Every GEMM resolved by `(K, M)` with `N` from the consumer's grid, correcting `qkv`'s N from 6144 to 14336 and pinning `in_proj`'s at 16384: **`in_proj` 95.1%, `gate_up` 92.0%, `qkv` 88.8%, `down_proj` 86.3%, `out_proj` 86.1% of FP8 peak**. **GEMM + attention are 74.6% of prefill time with 1797 ms of headroom — 6.1% of wall — so no kernel lever remains in prefill arithmetic.** **The lever that does remain is data preparation:** the nine non-GEMM non-attention kernels are 4544 ms moving 2111 GB at **0.06–1.03 TB/s against 3.5**, floor 603 ms, **headroom 3940 ms = 13.3% of wall and 2.2x the entire arithmetic headroom** — `pack_quantize` alone 2216 ms at **7.6% of HBM bandwidth**. One mechanism explains the column (`csrc/gemm/dsv4_deepgemm_ops.cu:89`): one 128-thread block per 128-element quantization block, **one `uint16_t` per thread**, a shared-memory block reduction per 128 elements, and a second pass that re-reads the input to scale it — 2-byte accesses where H20 needs >=8, and double the necessary traffic. Total identified headroom rises **12% -> 22.6% of wall, 59% of it in kernels that do no model arithmetic at all**. Still unmeasured: the `ncu` warp-stall split that says how much of the 3940 ms returns, and a floor for `fq_fwd` (1618 ms, 5.7% of prefill). Measurement only, same 08-08 capture, no new GPU time.

- **FIX (root cause confirmed; gates 2–3 pending) — the Qwen3.6 trunk's final RMSNorm applied `w` instead of `(1+w)`, and every eval on this model before today is a floor** (2026-08-08; `694245eec`, [entry](docs/experience/errors/2026-08-08-qwen36-final-norm-missing-offset.md)). ARLE's greedy argmax disagreed with **both** sglang 0.5.13 and HF transformers 5.6.0 on an identical 130-token prefix — they pick `328` at p1 0.768 / 0.7639, agreeing to 0.004 across FP8 and bf16, while we picked `5289`, their **rank 3, 2.25 nats down**, at p1 0.350 and entropy 7.147; downstream the trajectory degenerated into 48 repetitions of a partial-codepoint token, which is **~22% of all 500 GSM8K items**. Qwen3.5/3.6 uses the `(1+w)` offset convention, `norm.cu:666` already carried the kernel family for it, and the 64 in-layer trunk norms already called it — so this was a **call-site selection error, not a missing implementation**, which is exactly why in-layer norms were bit-exact (`attn_in` relL2 1.7e-09) while the final one was **2.101×** off (norm weight RMS 0.9715, so `(1+w)` ≈ 1.99). 14 sites swapped; **two instances, not one** — the trunk's final norm plus **every norm in `qwen35/dspark.rs`**, the same weights the trunk offsets, which bears on spec-decode and which a trunk-only investigation would never have surfaced. Gate 1 passes: top-3 is now `328 > 348 > 5289`, the exact ordering both external runtimes give, logits 27.875 / 26.25 against HF's 28.0 / 26.0. **Method, at the cost of a day:** eleven kernel-level exclusions and a full component parity harness all came back clean **because the trunk is correct** — the final norm had a tap before it (L63 `resid`) and nothing after it until the logits, making it the only transform in the model no comparison could see. A parity harness that puts every component at its floor has *told you something*: it closes "some kernel is wrong" and makes the untapped segment the answer by elimination. Three of my own errors are recorded in the entry, all caught by measurement: a **bf16 floor applied to an e4m3 GEMM** manufactured a 7× "defect" that was FP8's own cost (the check that catches it is the **third comparison**, reference-A against reference-B — if the references differ from each other as much as we differ from either, nothing was measured); an argument that **per-GEMM error compounds across layers**, refuted by the depth profile saturating then decreasing (8.7e-03 → 4.7e-02 → **2.9e-02**); and an activation-clipping mechanism refuted by reading the kernel. Also settled along the way: sglang uses the **identical** FP8 recipe (per-token-group-128, `amax/448`, DeepGEMM on H20, no bf16 fallback), so a recipe difference was never the answer, and the position is **not** ill-conditioned — perturbing HF's L63 hidden at relL2 0.029 through 0.8, 12 draws each, never moved the argmax. **Gates still open:** the GSM8K item end to end, needle ×3 against the baseline envelope, then the capability re-measure on the identical grid — this changes the logits of every request, so those license it rather than one position. **MMLU 0.8693 ± 0.0170 and GSM8K 0.8981-at-27.4%-invalid were measured through the defect and are floors; the delta is the result.**

- **VERDICT (close the lever) — the anchor's FP8 GEMM is 57.7% of all kernel time and runs at ~90% of FP8 peak** (2026-08-08; `70760bc09`, [bench](docs/experience/wins/2026-08-08-anchor-fp8-gemm-is-at-90-percent-of-peak.md)). The single largest cost on the workload ARLE targets was priced from §1.1's "199 / 189 TFLOPS, 64–67% — leave alone", measured at **33K cold, single request**. Decomposed at the served shape off the existing `nsys` capture with **no new GPU time**: the launch sequence is exactly periodic, so each GEMM is named by the kernel it sits between and by the dimension of the `pack_quantize` feeding it, and 14,094 launches resolve into **four GEMMs per layer over 176 chunks of 2048 tokens** — `out_proj` 503.7 µs, `gate_up` 2646.8 µs, `down_proj` 1409.0 µs, linear-attn `in_proj` 1256.7 µs, with `silu_mul`'s (68, 2048) grid pinning `intermediate_size` 17408 and confirming `gate_up` is the fused 2×17408 projection. Measured against the 296 TFLOPS peak: **`gate_up` 275.9 TFLOPS = 93.2%, `down_proj` 259.1 = 87.5%, `out_proj` 255.8 = 86.4%.** MLP alone is 69.7% of FP8 GEMM time and **~40% of all GPU kernel time**. **The lever is closed because it is already at the hardware floor** — on the anchor the dominant cost cannot be made faster, only smaller, so the remaining levers are prefix-cache hit rate, sparsity, and effective context length rather than kernels. §1.1's verdict was right and its number was wrong by 20–29 points. This is the third instance in two days of a number measured at one shape governing a decision at another, and the first in the direction of *overstating* a lever; the draft-attention entry below is the same defect with the opposite sign. Remaining prefill kernel candidates are FA3 prefill (16.3%) and `pack_quantize` + norms (12.3%), neither decomposed, together half the size of a line at the floor.

- **PERF (accept) — DSpark draft attention was launched once per slot at 192 blocks; batching the slot axis gives ITL mean −10.4%** (2026-08-08; `3a8f99b1f`, [bench](docs/experience/wins/2026-08-08-dspark-draft-attention-slot-batched.md)). The kernel that is 30.5% of a decode tick was never slow at arithmetic. All 39,690 launches in the decode-shaped window carry grid **(32, 6, 1) = 192 blocks**, serialized on one stream 7.5 µs apart — 16 slots × 5 draft layers, one launch each — which is ~2.5 blocks per SM on an H20's 78. `dspark_draft_blocks` batched the draft GEMMs and left the ring kernels per-slot, as its own doc comment stated, because each slot owns a separate `df.k_ctx[li]` ring allocation. `nonpaged_prefill_attention_kernel` now takes a per-slot ring-pointer array and promotes `blockIdx.z` to a slot axis, so the grid is (32, 6, 16) = 3072 blocks and the launch count per layer drops 16 → 1. Pinned-shape A/B, **output bit-identical in every arm: −71.1% / −71.4% / −69.3% / −68.4% at `kv_len` 512 / 1024 / 1376 / 2048** — flat across the window, so the win is structural. Serve A/B, two binaries from the same tree with the arms swapped across GPU 0 and GPU 1: **ITL mean 31.05 → 27.81 ms (−10.4%), total tok/s +14.1% on each device independently**, TTFT a wash as expected for a decode-side change, `accept_rate` agreeing closer than BASE's own device-to-device spread. Needle ladder ×3 at depths 0.0 and 0.5, all four arms: **3/0/0 exact and deterministic at every length, 72/72 runs** — the check that mattered, since the harness bit-identity covers the kernel math and the failure mode here is caller-side slot indexing. **This also explains the two 2026-08-01 reverts**: both were tuned by `ncu` at **3072** blocks pinned from the model config, where the kernel genuinely is ALU-bound, while the serve ran at 192 where it is occupancy-starved — so pin a microbench's shape from the trace's grid dims, not the config. Open: **only 57% of the projected win transferred** (the decomposition gives −18.2% on tick span against −10.4% measured on ITL); cause unknown, to be settled by an `nsys` capture on the new binary. Recorded alongside it: the trunk's verify attention has the identical per-slot shape at **24 blocks** over 126,948 launches, but costs 1.6% here and does not yet earn the fix. **Scope — on the 32K long-agent anchor the change is a null, and the champion row does not move.** 3 trials per arm, counterbalanced, dataset sha matching the one pinned in `baselines.md`: TPOT **+0.8% (NEW slower)**, out tok/s −0.7%, total tok/s +3.3% — deltas pointing opposite directions, each inside the arm's own trial spread (BASE spans 3.8% on TPOT), and BASE's best total tok/s exceeds NEW's median. Predicted before the run from the tick decomposition (all decode is 2.1–6.1% of GPU time on a 279:1 prompt:output workload) and confirmed by it. **So this is a real structural defect, correctly fixed, worth +14.1% on a decode-shaped workload and nothing measurable on the one the baseline is defined over** — the kernel's share, not its inefficiency, was always the governing number. Correctness is clean across the board: **11/11 needle rungs matched on both arms including the 241 boundary** (the short rungs were run after an initial ladder wrongly omitted them), and an **MMLU smoke with 0 per-question disagreements across all 50 items**, `accept_rate` in agreement on both workloads.

- **VERDICT (accept) — agent-opd rollout concurrency: `--prompts-per-update` + the cp fleet; production config is cp=4 × G=2** (2026-08-08; `7aef20557`, `f996e6826`, `5b1cd473d`, [bench](docs/experience/wins/2026-08-07-agent-opd-rollout-fleet.md)). The cc-rollout lane trained one task group per policy update, so only `samples_per_prompt` sessions were ever in flight and the cp fleet's slots ran at **29% utilization** — one group's 600 s straggler idled its siblings. The round loop is now windowed the verl way (G groups roll concurrently under ONE policy version, then a single update over their merged batch; staleness stays 0), and every cp rank serves rollouts. Five-arm sweep on subset16: raising per-engine pressure buys throughput and loses session health (G=4 = 8 sessions/engine: wall −51.7% but per-sample walls +79% and 9/64 samples pinned at the `--cc-timeout` cap), and **a 1200 s cap is strictly worse** — capped samples fall 9 → 3 but realized concurrency collapses 9.05× → 6.11× because stragglers hold slots. Adding *engines* instead wins: **cp=4 × G=2 (2 sessions/engine × 4) gives the lowest sum-of-sample-walls in the campaign (9350 s, below even the uncontended baseline's 10621 s), 1/64 at the cap, the highest aggregate throughput (88.7 tok/s), 2 GB lower VRAM per card, and a 37% better total round wall than G=1.** Correctness verified at 4 ranks: 212 writeback DONE lines in 53 four-way identical groups, all followers at `end of stream`, every group carrying exactly 4 records. Two method results: **pass counts do not resolve at one rep** (64 samples at a ~11% rate carry a binomial SD of ≈2.5, so the sweep's 9/5/5/10/10 is not a ranking — only the mechanistic axes are), and **a wall-clock rollout cap is the wrong control under variable contention**, since it makes the training signal a function of scheduling load. Concurrency defaults stay G=1 (safe at cp=1, where G>1 *is* the contended regime). Caught pre-run by review: concurrent groups shared the `<model>#s<sample>` tag that dump attribution keys on, which would have labelled one group's requests with another's reward (`5b1cd473d`). New bottleneck, measured: realized concurrency is 3.63× of 8 slots, so the rollout phase is host-bound (sandbox boot / tool exec / pytest), not engine-bound.

- **BASELINE — decode re-anchored on a decode-shaped workload: draft attention is 30.5% of a tick, not the 4.3% it was priced out at** (2026-08-08; `nsys` at `70760bc09`, [bench](docs/experience/wins/2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md)). Every decode number in the perf chain was priced on the long-agent 32K anchor, where all decode together is 2–6% of GPU time. A 16 × 1-turn workload at ~1.1K prompt / 1.8K generated (**1 : 1.62**, inverted from the anchor's 279:1), 32 requests so there is no queueing ramp, c=16 with **16.0 rows/tick by four independent counts**, gives the first decode baseline: **draft attention 30.5%, FP8 GEMM 28.8%, GDN 21.0%, dense GEMM 12.1%, norms 5.9%, full-attn paged 1.8%** of a 96.88 ms tick GPU-busy; row 407.97 out tok/s, ITL mean 31.08 ms, `accept_rate` 0.475. **The window was reconciled against run totals before any share was quoted** — 8.070 ticks/s in-window against a run-level 7.559 from the empirical chain length, +6.8%, and below the 11.05/s `steps` bound — the check the anchor capture failed. **A decode lever's share is workload-dependent by 17×**: full-attention is 1.8% of a tick at 2.5K context and 42.5% at 32.5K, so "priced out" is meaningless without the shape attached and the register carried verdicts without one. Consequences: **DSpark draft attention is re-opened as #1** (three earlier kernel rewrites failed to transfer because they were sized against 4.3%, capping a −33% microbench win at −1.4%); FP8 GEMM on the verify shape and the GDN lane at short context are open and never decomposed there; **13.8% of tick span is idle with 61.4% of it in 53 gaps over 1 ms**, where the anchor's decode ticks had nothing above 20 µs, cause unknown. Also corrected: `nonpaged_prefill_attention_kernel` **does fire on serve** — 39,690 launches in a 60 s window — as the DFlash draft's attention, since the draft has no paged full-attn KV pool; a standing note said it never runs outside the training-forward path, which is true only of the trunk.

- **VERDICT (reject the ranking, mechanism confirmed) — FA3 decode-verify is 29.2% of roofline and 0.39% of GPU time; the anchor is a prefill benchmark** (2026-08-08; `nsys` at `70760bc09`, [entry](docs/experience/errors/2026-08-08-anchor-is-a-prefill-benchmark-decode-levers-ranked-off-it.md)). `docs/perf-qwen36-27b.md` had made FA3 decode-verify KV bandwidth its #1 open item from a derivation, because **no capture in that document was taken above batch 1**. The first c=16 capture confirms the mechanism — 9 rows × 32550 tok × 65536 B = 19.20 GB in 18.79 ms/tick = **1.02 TB/s, 29.2% of achievable**, cross-checked per-layer, and 42.5% of a decode tick's GPU-busy — and destroys the size: in a 30 s window at 96.7% GPU-busy all decode ticks together are 0.98% of GPU time and FA3 decode-verify 0.39% — but **that window undersamples decode 2–6×** (the bench has a queueing ramp, TTFT p50 4.2 s against p99 164 s, and the window sits at bench elapsed 118–148 s), so the run-level bound from 15,981 generated tokens over 414 steps is **decode 2.1–6.1% and FA3 decode-verify 0.86–2.5%**. Removing it entirely moves the row 1–3%. The 29.2% bandwidth figure is internal to a tick and unaffected. The cause is the dataset shape: the anchor is **4,452,150 prompt tokens against 15,965 output — 279:1**, or **38.6:1** counting only real prefill work after the measured 0.876 prefix-cache hit rate — so long-agent 32K × 8 turns is by GPU work a prefill benchmark, and every decode number and lever in the perf chain was ranked against it. The dataset is not the mistake: it models measured coding-agent traces (TraceLab, 4,265 Claude Code / Codex sessions), and for the workload ARLE targets prefill genuinely dominates. Pricing decode levers on it was the mistake. The latency view hid this (`itl_p50` 0.0197 ms against `itl_mean` 176.99 ms with 128 requests on 16 slots is queueing, not decode). Same capture also corrects: **rows/tick is 9, not the nominal 16** (three independent counts); `gated_delta_rule_prefill_recurrent_kernel` is **live on the decode path** at 7.75 ms/tick despite being retired from prefill on 08-02; `accept_rate` is **0.478** against 0.31 recorded; decode ticks have no launch-gap problem (all 11,422 gaps in 0–5 µs). The perf chain now carries a **provenance table** for all thirteen measurement tables — date, batch, context, stale-or-current — after an audit found its two foundational budgets (§1.1 prefill, §2.1 plain decode) both predate the default flips that followed them, §2.1's #2 row being a kernel deleted on 08-03. Preceding this, four adversarial reviews of `b3198fb22` retired a three-factor performance frame that never closed arithmetically, whose 12× ceiling double-counted, and whose central claim — that a batch-independent intercept is what adding rows fails to amortize — was inverted.

## [0.5.2] - 2026-08-21

- **BASELINE — corrected Qwen3.6-27B DSpark anchor is complete** (2026-08-10; runtime `9b38ba6c0`, runner `c98c4e0b2`, [bench](docs/experience/wins/2026-08-10-qwen36-27b-corrected-baseline.md)). The canonical c=1/2/4/8/16 sweep completed 128/128 at every point with zero errors, fixed 214-token outputs, prompt p50 +8.84% from the 32K target, and concurrent needle 78/78 exact. DFlash acceptance is 26.90-27.81% at c=2-16 after the norm fix. The isolated warmup restores the expected 112/128 c=1 prefix hits. At c=16 the row records 162.60 output tok/s, 26590.69 total tok/s, TTFT p50 939.0 ms, and ITL mean 92.27 ms. This fingerprint has one valid sweep; matched A/B is required until repeat drift is measured.

- **FIX — benchmark warmup no longer primes a measured prefix** (2026-08-10; [error](docs/experience/errors/2026-08-10-benchmark-warmup-contaminated-cold-session.md)). The first 16-session × 8-turn point should have 112 prefix hits but reported 113 because the runner warmed with dataset prompt zero. The warmup now prepends a dedicated marker to the same production-length prompt, preserving the execution shape while making its cache key disjoint. The contaminated sweep is diagnostic only and must be rerun.

- **FIX — DFlash draft norms restore Qwen3 plain-weight semantics** (2026-08-10; `9b38ba6c0`, [error](docs/experience/errors/2026-08-10-dflash-draft-norm-offset.md)). The Qwen3.6 target final norm correctly uses `(1+w)`, while the Qwen3 DFlash draft uses plain RMSNorm. Seven draft call sites inherited the target convention in `694245eec`, scaling checkpoint norms by 1.4–1.9x and reducing acceptance to 0.334%. The draft feature, layer, and final norms now use the existing plain kernel in both single-slot and batched paths. Draft q/k prep keeps its deliberate `w-1` storage, and target verify keeps the offset kernel. CUDA release build passed; concurrent needle passed 78/78 exact, every baseline point completed 128/128, and acceptance recovered to 26.90-27.81% at c=2-16.

- **FIX — the canonical fixed-output benchmark now forces `ignore_eos=true`** (2026-08-10; [error](docs/experience/errors/2026-08-10-fixed-output-benchmark-allowed-early-eos.md)). The current corrected Qwen3.6-27B model legitimately emitted EOS as its first token on one canonical prompt, so the c=1 baseline aborted at 127/128 even though its contract fixes every output at 214 tokens. Streaming, non-streaming, and DSpark-off probes isolated the behavior to target generation. With `ignore_eos=true`, the same request completes 214 tokens with non-empty output. The runner now sends and records that parameter; the empty-output gate remains strict, and the aborted performance numbers are invalid.

- **PERF — V100 (sm_70) prefill: W4A16 dequant→FP16 GEMM + GDR/FA2 tuning** (`df77f7668`). On Volta the W4A16 prefill path ran an on-the-fly dequant FP32 batched GEMM (no tensor cores, 15 TFLOPS). Prefill now dequantizes 4-bit weights to FP16 once per projection and runs a cuBLAS FP16 GEMM on tensor cores (125 TFLOPS); small K/V projections are cached. GDR prefill kernel drops from 5 to 3 `__syncthreads` per token (fused q/k norms) and overlaps exp_g/beta with the norm pass. FA2 sm70 uses Br=16/Bc=64. Qwen3.6-27B-W4A16 on V100: 1K-token prefill 30s→3s (10×), 21K-token 130s→43s (3×).

## [0.5.1] - 2026-08-07

- **VERDICT (accept, end-to-end null) — the prefix sidecar serialized 146.8 MiB per element; bulk copy is −9.5% on the operation and 0.9% of wall** (2026-08-07; `d626a1b03`, [bench](docs/experience/wins/2026-08-07-prefix-sidecar-serialize-bulk-copy.md)). `Qwen35RecurrentSnapshot::to_bytes` walked the whole recurrent state one f32 at a time — 37M four-byte `extend_from_slice` calls per snapshot — and `from_bytes` rebuilt it with `chunks_exact(4).map().collect()`. The payload is fixed at **146.8 MiB** (48 linear layers × 3 MiB gdr + 60 KiB conv) regardless of context length, and it is written at **every stride boundary of every prefill**: 578 snapshots and **83 GB of host serialization per 512 s bench**. Counterbalanced A/B (D,E,E,D) with the timing log in **both** arms, identical n and payload in all four: **84.45 → 76.40 ms, −9.5%**, halves agreeing at −8.0/−8.1 ms with no overlap between arms. End to end it is a **null on every metric** (c=16 itl_mean +2.6%, itl_p99 −9.1%, tok/s −0.6%, all inside the 2.7% within-arm spread) because the serialize is only **9.4% of wall**. Kept, not reverted: strictly less work, cannot regress. **Two predictions were wrong and both are recorded** ([error](docs/experience/errors/2026-08-07-named-a-call-site-whose-gate-was-off.md)). The per-element loop was predicted at ~half the event and is 9.5% — 146.8 MiB in 76 ms is 1.9 GB/s, so the residual is allocating and first-touching fresh heap, and the way to make it cheap is to not copy at all. And the mechanism was predicted at ~45% of wall from an nsys window of 19.92 s in which 8.98 s sat in 79 stalls — **a window aimed at decode reports decode's share of the window, not of the run**. The same capture killed the CUDA-graph lever properly: all **430k launch gaps total 0.59 s, 3% of the window**, against 91% of the idle in those 79 stalls, so the earlier "graphs are the only big lever left" was priced off kernel-count arithmetic instead of a binned distribution. A payload signature was also mis-attributed to the whole-slot park, whose two routes are both unreachable in a default serve (`--kv-oversubscription` defaults off; the other needs `kv_tier_capacity() == 0`) — `a546ba80a` fixes the same shape there and adds park/promote timing logs, unmeasured because the path is default-off. Open, and larger than what was fixed: the sidecar's restore hit rate is unmeasured, so whether 83 GB per bench is earned is unknown.

- **FIX — agent-opd cp>1: rank 0 owns rollout, followers mirror the update stream** (`9da8ff777`). With context-parallelism > 1 the rollout phase could deadlock the writeback collective because follower ranks issued the update stream out of order with rank 0. Rank 0 now owns the rollout and followers mirror its update stream.

- **VERDICT (confirmed) — agent-opd cp=2 fix validated; the cc-rollout training loop closes end-to-end under the new defaults** (2026-08-07/08, pod GPUs 4+5, [error entry](docs/experience/errors/2026-08-07-agent-opd-cp2-rollout-divergence-deadlock.md)). Two subset16 × 4-sample × 3-round runs, both `RUN_EXIT=0`. Single-GPU: per-update losses improve monotonically (0.113/0.148/0.091 → 0.077/0.071/0.077), the hard task sqlparse `9b5ribm7` climbs 1/4 → 3/4 → 4/4, GRESO trims 16 → 3 → 6 groups, writeback walls flat across rounds, VRAM peak 76.9 GB. cp=2: the former deadlock point completes in ~30 s; 46 writebacks, rank-0/follower losses identical to all printed decimals, clean follower exit, mesh dir removed. Side finding: cp=2's round-0 update wall is **7.1× faster** than single-GPU (1119 vs 7892 s; backward 28–30 vs 213–252 s at the same ~21 K seq — the ~10.5 K/rank shard stays out of the checkpoint-offload regime), making cp=2 the preferred operating point for this lane rather than only a parity mode.

- **PERF (partial) — agent-opd cp rollout fleet: every rank serves, rank 0 keeps the harness** (`7aef20557`, [bench](docs/experience/wins/2026-08-07-agent-opd-rollout-fleet.md)). Follower ranks were idle through the rollout phase (75% of the round wall), so each rank now runs a rollout serve and rank 0's harness spreads a group's samples round-robin across the fleet. Measured on subset16 cp=2: per-sample latency **−23%** and aggregate throughput **+43%** on the groups that finish under the `--cc-timeout` cap, but the round wall moves only **−5%** — a group's wall is `max` over its samples, and 9 of 96 samples sit at the 600 s cap. Root cause of the shortfall, measured: only one group (4 samples) is ever in flight against the fleet's 8 slots, and the group barrier idles the fast samples — **rollout capacity utilization 29%**. Follow-up `f996e6826` adds `--prompts-per-update G` (G groups roll concurrently under one policy version, then one update over the merged batch; staleness stays 0); its A/B is pending-remote.

## [0.5.0] - 2026-08-07

Progress spine. Entry classes recorded here the day they land: phase exits,
default flips, and accept-or-reject verdicts (AGENTS.md §Docs lifecycle &
progress spine).

- **VERDICT (accept) — the c≥4 DSpark decode regression is CLOSED; anchor re-anchored on `70760bc09`** (2026-08-07; `70760bc09`, [bench](docs/experience/wins/2026-08-07-dspark-rollback-replay-batched.md)). Second of two per-row host loops: `dspark_rollback_batch` routed to a per-slot `replay_linear_only` whenever `--qwen35-gdr-chunked` was on, leaving `replay_linear_only_batched` — whose own doc names the cost it avoids — dead in every shipped configuration since that flag defaulted on in `c2eb5de9e` (08-02). Partial accept fires on most ticks at accept 0.30, so the loop cost rows × 48 layers × ~6 launches ≈ **4608 a tick at 16 rows against 144**. Counterbalanced A/B (B,C,C,B — position was worth up to 6.3% within an arm): **c=16 TPOT −11.4%, itl p99 −16.9%, total tok/s +7.7%**, gain monotone in concurrency. Day total at c=16: 124.99 → **110.52 ms (−11.6%)**, now **0.9% ahead of the 07-30 champion** it had been 21.5% behind. **Both of the day's wins are the same bug — a default flip turning an `if flag { per-row }` minority branch into the only path** — and neither failed, had a test, or was findable by reading code; a pure-decode kernel-instance ledger exposed the "four kernels per (row, layer)" signature. Rule: a default flip is a routing change, so grep the flag's call sites, not just the feature it names. **Also settled: decode and prefill are opposite regimes** — pure decode is **71.5% GPU idle at 19157 launches/s** while prefill is ≥76% busy with the dominant GEMM at 92.7% SM throughput (ncu). Three earlier nsys windows all landed in prefill and were used to kill the launch-bound hypothesis that turned out to be correct ([error](docs/experience/errors/2026-08-07-measured-prefill-concluded-about-decode.md)); the same entry records the confounded `--chunked-prefill-size` A/B, the nsys CLI traps, and two changes that sizing-before-building dropped (prefill-prefill fusion at 3%; the dominant GEMM already saturated). `--max-num-batched-tokens` added in `ed92c6d8c` — the per-tick token budget was a hardcoded 16384 with no CLI path, ~250× past this box's ~65-token roofline ridge, and `itl_p99` 758.8 ms against a 110.52 ms mean is the decode row waiting behind it.

- **VERDICT (accept) — the DSpark verify linear core is batched; long-agent anchor re-anchored on `4933e1bf4`** (2026-08-07; `4933e1bf4`+`9119ebcbb`, [bench](docs/experience/wins/2026-08-07-dspark-verify-linear-core-batched.md)). `LinearCore::Rows` ran a per-row host loop, so a verify tick issued five launches per row per linear layer (conv1d + FlashQLA prep/cumsum/kkt/fwd) on a 7-token chain — 80 launches per layer at c=16 against 5 at c=1, which is the shape of the c≥4 regression the 08-06 audit opened. The batched kernels already existed: `replay_linear_only_batched` drives `conv1d_prefill_varlen_cuda` + `gated_delta_rule_prefill_recurrent_varlen_cuda` through pointer tables, and their `s * len` row stride **is** the trunk's ragged packing when every row has the same length, which a verify tick guarantees. Routing uniform rows ≤64 tokens (one FlashQLA chunk) through them is worth **c=8 TPOT −12.7%, c=16 total tok/s +10.0%, c=1 a null at +0.3%** — counterbalanced A/B, two sweeps per arm in both orders, single-commit delta. c=8 is now within 1.4% of the 07-30 champion; **c=16 is still 5.5% above it and unattributed.** Two method results came out of it: the arm that runs second on a shared box loads cold and measures 3–6.5% slower, larger than the whole c=16 effect, so single-order sweeps on this row are invalid; and the standard needle ladder is single-request, so it runs the path this change does *not* touch — added `scripts/needle_concurrent.py` (a distinct needle per row, which is what catches a pointer table handing row `i` row `j`'s state) and gated both arms 6/6, 24/24, 48/48 exact at c=2/8/16. Acceptance drops 1.2% (0.3036 → 0.2996, stable within each arm) because the batched recurrent core is not bit-identical to per-row FlashQLA; recorded on the row rather than treated as the gate.

- **DEFAULT FLIP + VERDICT (accept B / reject C) — `--checkpoint-reload-device` on by default; pinned checkpoint pool stays off** (2026-08-06; `5cec66ea3`..`d1870526f` + this flip, [bench](docs/experience/wins/2026-08-06-checkpoint-reload-and-pinned-offload.md)). Lever 2 pod A/B, one binary at `d7ecbbcee`, cp=2 seq=81920: reload-to-device cuts the 80K OPD backward **304.6 → 121.8 s (−60.0%, 19× the 9.8 s spread)**, step wall 372.1 → 192.1 s, loss bit-identical 4.537510, grad_norm 7.970384 in-envelope, peak VRAM *lower* than baseline (−2.9 GiB cp0) and host RSS −3.9 GiB — the mechanism is one HtoD up front instead of the recompute-forward repacking host `Vec`s and re-uploading per op. The pinned arm (+8 GiB budget) is a wash over B (−6.5 s, inside the spread) and drove cp1 peak to 97.4/97.9 GB with no engagement probe to prove it even fired: **rejected**, `--checkpoint-pinned-offload-bytes` default stays 0; re-adjudication needs an engagement counter + 3 reps. Adversarial review (codex ×2 + 13-agent workflow) landed four fixes before the measurement: pinned-slot 64 MiB granularity, stream drain before pool free, parked-input-only re-park, and a `slice_host_eager` full-source clone removal.

- **WASH — reshape/rmsnorm backward heal is correct but a no-op for the profiled cost** (2026-08-06; `7da312d0d` kept, not reverted, [error](docs/experience/errors/2026-08-06-healed-the-wrong-reshape-backward-not-recompute-forward.md)). Probe C put 21% of on-CPU backward self-time in a `reshape`/`rmsnorm` → libc `memcpy`; I healed `reshape_backward`/`rmsnorm_backward`'s missing `ensure_device` (mirror `matmul_backward`). Pod-measured: **parity PASS** (loss 4.537510 exact, grad_norm in-envelope), **step wall WASH** (backward 315.6 → 315.7 s), **re-profile NULL** (reshape 16.4% → 16.2%). Probe D resolved the caller frame: all 50,958 reshape samples are `train::lora::LinearWithLora::forward` — the forward `reshape` replayed under checkpoint-recompute on a host-resident (offloaded) activation, NOT `reshape_backward`. The heal is a real latent-trap fix (kept — reverting a correct fix to restore a bug is wrong) but off the hot path. The 16.2% reshape + 35% `upload_slice` HtoD are two symptoms of one thing — recompute-forward on offloaded activations — which **Lever 2 (pinned+async checkpoint offload) subsumes**. Rule: a flamegraph names a symbol, not a call path; read the resolved caller frame before fixing the op it labels.

- **VERDICT (reject) — OPD_SEQ_CHUNK 4096→8192 is a null; the backward wall is total-work CPU, not per-chunk overhead** (2026-08-06; pod-only, local git untouched, [error](docs/experience/errors/2026-08-06-opd-chunk-knob-null-backward-is-total-work-cpu.md)). An `nsys -t cuda,nvtx,osrt` host trace of one 80K backward (305.4 s) attributed the wall: **131.7 s (43%) is on-CPU Rust host compute NOT in any CUDA API** (Tape/HashMap/forward-recompute across 640 nested sub-backwards); launch-gap orchestration is only 18.6 s (6%), copy-engine DtoH+HtoD 70 s / 429 GB, GPU kernels 84.6 s. Critical-path threads block 100% on `ioctl`, zero `futex` — genuine on-CPU work. Doubling the chunk (640→320 iters) was the hypothesised lever; matched A/B: backward 315.6 → 307.1 s (−8.5 s) sits **inside** the 4096 run-to-run spread (305.8↔315.6 = 9.8 s), peak VRAM flat (+1.6 GB, not +16 GB), loss bit-identical 4.537510. **KILL** — the 131.7 s scales with total op count (the ~61k per-op host dispatches), not chunk count, so the knob was orthogonal by construction. Two structural levers survive, both orthogonal to chunk count: **CUDA-graph capture** of the backward (graph replay issues the DAG with one host call, re-running none of the per-op Rust orchestration between launches — amortizes the 131.7 s to ~0 on steps 2..N; blocked today by the non-static per-iter alloc/free/park, needs a capture-safe pool) and **op-count reduction / fusion**. Decisive next step before building capture infra: a pod `perf record -g` flamegraph to split the 131.7 s into launch-orchestration (graph-fixable) vs genuine host-math. Rule: name what the dominant cost scales with before picking a knob; and an effect the size of the baseline's own variance is not a result.

- **VERDICT (reject) — native FP8 training forward halves the GEMM cluster but moves the step wall 1%** (2026-08-06; `cafda607c`+`3c021aead`, [error](docs/experience/errors/2026-08-06-native-fp8-forward-optimized-the-wrong-17-percent.md)). `--fp8-native-gemm` copies serve's DeepGEMM dense FP8 call into the training forward (frozen-weight projections), replacing dequant→cuBLAS-bf16. Matched A/B at SEQ=81920 cp=2: the mechanism works — nvjet bf16 GEMM cluster **45.1% → 15.2%** (−30pp), `fp8_block_scaled_to_bf16` dequant **16026 → 4052 launches** (−75%), forward wall −10.4%, forward-value parity 4.14e-4. But the forward is ~17% of the step and the backward (~84%) is bf16 straight-through, so net step **363.6 → 359.8 s (−1.0%)** — below any win bar. During the backward GPUs 4,5 idle at 0–11%: the wall is host-orchestrated chunked-scan CP + nccl SendRecv, not GEMM. Flag stays **off**; feature and the stream-guard fix (dense DeepGEMM entry accepts CUDA's null default stream, which autograd runs on) remain — the fp8 forward is correct and available. Rule: rank optimizations by `phase_share × speedup` and check per-phase GPU utilization first; a kernel-precision change cannot move a host-bound wall.

- **FIX — REPL/OCR load caps slots at 1: Qwen3.5-9B now fits in 48 GB** (2026-08-06; [win](docs/experience/wins/2026-08-06-repl-single-slot-load.md)). `LoadedInferenceEngine::load` used `EngineLoadConfig::default()` (`num_slots=256`); Qwen3.5's GDR per-slot recurrent state made `static_state` 12.5 GiB for the 9B model, pushing the fixed requirement to 21 GiB and tripping the Metal resource guard on memory-constrained boxes. `load()` now sets `max_running_requests: Some(1)`, collapsing `static_state` to 49 MiB. The serve path is untouched (it uses `load_with_config` with the serve-derived slot budget). Verified: Qwen3.5-9B-MLX-4bit loads in the REPL at 4.8 GiB peak RSS; bash + python agent tools execute.

- **FIX — CUDA serve auto-downloads HF model ids, matching Metal** (2026-08-06; [win](docs/experience/wins/2026-08-06-cuda-serve-auto-download.md)). `cuda_serve_handle` passed `model_path` straight through, so a HF id failed with "config.json not found". It now calls `infer_util::hf_hub::resolve_model_path` (the same resolver `arle model download` uses) before tokenizer / engine load. `infer-util` added to `infer-api` deps. Local paths still short-circuit.

- **DEFAULT FLIP — FlashQLA GDN chunkwise backward is default-on: 80K training step 1.99×, backward 2.14×** (2026-08-05; `bb5561649`, [win](docs/experience/wins/2026-08-05-flashqla-gdn-backward-default-on-2x.md)). The 71% `linear_attention_chunked_scan_backward_f32` row (the whole 80K step, prior characterization below) is replaced by the QwenLM FlashQLA SOTA kernel. Matched A/B at seq=81920 cp=2, same harness, only the flag varies: backward **670.28 → 312.64 s (2.14×)**, step **752.96 → 378.72 s (1.99×)**, forward also 1.26× (native GDR chunk-prepare rides the chunkwise path). Loss 4.537510, grad_norm 7.976866, RUN_EXIT=0; the 71% kernel row is gone from the nsys profile. `--gdr-chunkwise-prefill` now defaults true (CLI + `AutogradRuntimeFlags`); `--la-backward-mono` keeps the recurrent arm for A/B. The two bf16 backward paths move grad_norm at the bf16 floor (32K liveness 2.15 vs 2.26); the f32 anchor is `qwen36_fp8_lora_fd_gate --gdr-chunkwise` (arm-internal analytic-vs-FD), pending-remote. Unblocked by the sm_90a target fix (`4b85750e4`) and the CP head-geometry table (`1b913e31e`).

- **CHARACTERIZATION — an 80K training step is one kernel: GDN chunked-scan backward is 71%; FA3 is worth 3.54x at 80K, not 2.17x** (2026-08-05; [win](docs/experience/wins/2026-08-05-80k-training-step-is-one-kernel.md)). Every prior ranking rested on a seq=8192 pre-FA3 capture. At the real target: nsys on a cp=2 seq=81920 step puts `linear_attention_chunked_scan_backward_f32` at **71.0%** (707 s of GPU time, 90 launches) and `gated_delta_rule_prefill_recurrent` at 6.7% — 77.7% on the two GDN rows, with FA3 attention backward down to 1.5%. The earlier note that "GDN is O(s) so its share drops at 80K" was wrong in direction. Matched FA3 A/B at 81920: step **2670.06 -> 752.96 s (3.54x)**, fwd 6.35x, bwd 3.21x, VRAM and host RSS a wash. Single card does NOT fit 80K — forward clears (3972 s, the `merge_grad` in-place fix holds) and backward dies on `cuda alloc_zeros failed` with no op/shape/bytes logged, so cp=1 x dp=8 is unavailable and the bf16-tape "one card holds the sequence" argument has an unnamed price to beat. The 11.6% grad-norm gap between the two cp=2 arms reproduces the 14% seen at 32768: #85 is not sequence-specific.

- **DEFAULT FLIP — FA3 is the unconditional CP ring path; `ARLE_CP_RING_FA3` deleted** (2026-08-05; `15caff0d0`, [win](docs/experience/wins/2026-08-05-80k-training-step-is-one-kernel.md)). The route now gates on hd256 + sm90 + the real-kernel marker alone; scalar kernels remain the non-sm90 / non-hd256 fallback. Rationale: the flag bought no correctness. #85 is a CP-vs-single grad divergence **both** cp=2 arms show (32K: cp=1 3.745, FA3 cp=2 2.265, scalar cp=2 1.984 — the FA3 arm sits *closer* to the single-card anchor), so keeping FA3 off only made an equally suspect path 3.54x slower. Cost: the scalar-vs-FA3 A/B lever is gone from a single binary. **Gate returned, flip stands.** The decisive run — real 27B, cp=2, seq=32768, production shape — reproduces the standing FA3 reference to 6-decimal loss (10.871086) and 0.06% grad-norm (2.263385 vs 2.264733), both ranks post-all-reduce. `cp_hidden_parity` at head_dim 256, where FA3 now actually engages (the original block was a gate running at hd128 where it never did): `cp_vs_cpu_f32` 3.16e-2 <= `single_vs_cpu_f32` 3.49e-2, CP tracking the f32 anchor slightly better than single-card. `nd_parallel_parity` at cp=2 and cp=4: CP's signed deviation from f32 is smaller than single-card's and in the same direction, no world-size bias. The 0.8B cp1-vs-cp2 residual that looked like a problem is resolved: on one binary at `4846f8046` (which still carries the env var, so the flag is the only variable), 3 reps per cell across cp=1/2/4 show FA3 inert at cp=1 (identical loss, grad_norm within spread) and its deviation from cp=1 **shrinking** with ring-step count — 1.419e-3 at cp=2 to 1.80e-4 at cp=4, inside the 1.7e-4 noise floor — while the scalar path's grows (1.085e-3 to 1.655e-3). A mis-paired block or mis-merged LSE compounds with ring steps; this does not. The earlier cp=1-referenced sign flip was real but cp=1 is a third numerical path, not ground truth. Also retired a stale comparator: the "8.5e-5 CE floor" cited in earlier entries was measured at hd128 pre-FA3 and is not a valid same-config bound.

- **VERDICT — the prefill gap was a stub build: FlashQLA was never compiled into the pod binary, TTFT 31.08 → 25.01 s** (2026-08-05; `101d68b91`+`6e3f68fac`, [win](docs/experience/wins/2026-08-05-flashqla-was-never-compiled-into-the-pod-binary.md)). A `--cuda-graph-trace=node` ledger of one cold 33K prefill per stack splits the 10.63 s TTFT gap with no residual: quantized GEMM is identical (Marlin 18.660 vs 18.675 s, 8448 launches each), and the whole gap is **linear attention 7.231 s vs 0.314 (+6.92)** plus **GPU idle 3.877 s vs 0.190 (+3.69)**. `--chunked-prefill-size` is not a lever on either stack (2048 vs 4096 vs 8192 all inside 0.07 s). Root cause was in the boot log the whole time — `FlashQLA chunked GDR unavailable (stub build)`: the flag defaults true and the kernel has been default-on since `c2eb5de9e`, but `scripts/pod-build-env.sh` never set `ARLE_CUDA_ENABLE_FLASHQLA_GDR`, so `build.rs` skipped every flashqla row and prefill silently ran the serial recurrent scan. **Every W8A16 prefill number from 2026-08-02 through 08-04 was measured on that binary.** Two failures hid behind it: the pod tree was a mixed state (git at `b7fecaa5d`, files overwritten piecemeal by `tn push`) missing ckl's `4b85750e4` sm_90a targeting, and tilelang 0.1.12 renames C++-keyword params (`do` → `do_1`), breaking `gdr_fq_bwd` codegen. Re-measured, 2 reps/arm: **TTFT p50 31.08 → 25.01 s (−19.5%), prefill tok/s 1068 → 1329 (+24.4%), ITL unchanged at 16.69 ms**. Against SGLang 1.48× → **1.19×**, GPU-busy now within 0.93 s. **The remaining prefill gap is 3.8 s of GPU idle**, not kernels.

- **VERDICT — the decode step reaches parity with SGLang; the gap is now entirely prefill** (2026-08-04; `17fdb6aab`+`e1017b40d`, [budget](docs/experience/wins/2026-08-04-w8a16-decode-step-kernel-budget.md)). A whole-step CUDA graph makes `nsys --trace=cuda` report one kernel — `--cuda-graph-trace=node` is mandatory on a graphed path. With it, ARLE's 33K W8A16 decode step is **Σ kernel 16.651 ms vs SGLang 0.5.13's 16.527 (+0.7%)** on the same GPU, same int8 values, same `gptq_marlin`: Marlin matches to 0.8%, FA3 is 5% faster here, and the one row behind (conv1d, two kernels) is closed by fusing the ring update into the decode kernel (−0.079 ms, needle ladder to 33691 tokens `exact=3 miss=0`, [entry](docs/experience/wins/2026-08-04-conv1d-decode-fusion.md)). From 1.57× behind on 2026-08-02. Re-anchored end to end (2 reps/arm, [baselines](docs/baselines.md)): **ITL p50 ARLE 16.66 ms vs SGLang 17.14 — ARLE leads decode by 2.8%**, the first lead since the comparison began, carried by the host tail (0.061 ms vs ~0.6) rather than by kernels. **TTFT p50 31.08 s vs 21.03 = 1.48× behind, unchanged from 2026-08-02's 30.5 s**: every accepted optimization since then landed on decode and prefill has not moved. Prefill is the next front.

- **DEFAULT FLIP — FA3 decode split ceiling is derived from the SM count: −11.2% decode step at batch 1** (2026-08-04; `574045dc1`+`53f9c5143`+`0e750fa18`, [win](docs/experience/wins/2026-08-04-fa3-decode-splits-fill-the-sms.md)). `--qwen35-fa3-decode-splits` caps FA3's own `num_splits_dynamic`; the constant 8 bound it at batch 1, where `pack_gqa` leaves `kv_heads × splits` = 32 tiles for 78 SMs. Default is now `0` = derive `sm_count / (batch × kv_heads)` floored at 8 — 20 at batch 1, unchanged from batch 4 up. 33K W8A16 27B, two reps: **18.989 → 16.859 ms**; the batch 4/8 rows are identical configurations and measure the 0.09 ms noise floor. Needle ladder to 33691 prompt tokens, both arms `exact=3 miss=0`. The kernel is still 8× off its 618 µs roofline; DRAM traffic is the next probe.

- **ACCEPT (perf) — FA3 replaces the scalar CP ring-attention kernels: 2.17× per training step; default OFF pending grad parity** (2026-08-04; `2fe12a2fe`+`df75a1da2`+`a15d3ec75`+`d293fcc74`, [win](docs/experience/wins/2026-08-04-fa3-ring-attention-2x.md)). Vendored FA3 hopper hd256/bf16/sm90 backward substrate + a zigzag (q_run, k_run) pair decomposition (diagonal causal, past full, future skipped); FA3's normalized (o, lse) folds into the existing flash-2 accumulators, so tape and finalize are untouched. G2 cp=2 seq=32768 same-binary A/B: fwd 2.71×, bwd 2.07×, step 460→212 s, losses at the bf16 floor with the FA3 arm *closer* to the cp=1 anchor than the scalar ring. **Default stays OFF** (`ARLE_CP_RING_FA3=1` opts in): the correctness A/B ran at hd128 where FA3 never engages — a gate on the scalar path wearing an FA3 label — and post-backward grad norms disagree across configurations (cp=1 3.744990, scalar cp=2 1.984009, FA3 cp=2 2.264733) while losses agree to 7.5e-5. That divergence predates FA3: `nd_parallel_parity` runs the full step but only ever compared the loss, so **the CP backward has never been checked against single card**. Both CP parity examples now run at the production head_dim 256 and gain a three-way grad-norm comparison against the CPU f32 arm.

- **ACCEPT — GDR chunk-prepare native CUDA: 289× per launch, −10% training fwd wall, losses bit-identical** (2026-08-03; `3d80dd473`, [win](docs/experience/wins/2026-08-03-gdr-prepare-native-289x.md)). The nsys mystery `kernel_kernel` (12.3 s/step, 13%) was the TileLang GDR chunk-prepare: the lowering replicated the full q/k row into every thread's registers (local-memory spill, 128× redundant loads, 66 ms vs the ~0.1 ms roofline). Native warp-per-token replacement (solve-stage precedent): ncu 66 ms → 228 µs, G2 fwd 102→91.7 s, bwd −8.8 s, losses 6-decimal identical. Attribution follow-up also dissolved the 15.9 s HtoD wall (13.5 s = first-16 s weight-upload capture artifact; steady state ~2.4 s = GDN backward's host-resident saved qkv/z) and G3 exposed a reported-loss ÷dp bug (numerator replica-local over the dp-global denominator; fixed `8277ff6fe`, re-gate queued).

- **PHASE EXIT — CP×DP mesh verified end-to-end; 131072 cp=4 runs clean; the training step is attributed** (2026-08-03; `4aa6e5e02`+`00e482f50`+`a644adab8`+`e57c59793`+`3cae75304`, [win](docs/experience/wins/2026-08-03-cpxdp-verified-and-training-step-attributed.md)). Composed CP×DP trains on 4 GPUs (ncclCommSplit seq subgroup; ports fixed to world-rank offsets; DP count reduce fixed — every cp rank contributed the same replica count, losses came back exactly ÷world). **G4: real 27B cp=4 seq=131072 completes a full step in ~3100 s with host RSS 170.4 GiB total (~44.6 GB/rank) — half the cp=2 343 GB wall: host RSS scales with the per-rank shard, so the 131072 OOM is solved by cp=4 and 256K extrapolates to cp=8.** nsys attribution (seq=8192) re-ranks the perf campaign: ring-attention kernels 31% + GDN chunk backward 26% + 75k HtoD uploads (15.9 s) are the walls; GEMMs 10%, idle 6% (graph capture worthless), elementwise NOT dominant — the bf16-tape time thesis is demoted to a VRAM play; the knife goes to ring attention, the HtoD source, and GDN backward.

- **ACCEPT — per-token O(cached-pages) scan → O(1) counter: −6.0% decode ITL (cumulative −29.4%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-resident-page-scan-per-token.md)). `resident_evictable_pages()` walked the whole page-ref map (~20k entries on a warm prefix cache) three times per decode token — planner tick plus two `publish_counters` calls — for ~1.2 ms/token that no cold bench or CUDA profile could see. Now a counter maintained at the four page-state transitions; also dropped the redundant post-admission counter publish, cached the per-step `ARLE_STEP_DIAG` getenv, and early-returned `admit_waiting` on an empty queue (after the TP collective, so per-rank collective counts are unchanged). c=1 ITL 20.19→18.98 ms, p99 21.32→19.52; greedy byte-identical.

- **ACCEPT — T6 GDN decode kernel: −2.8% decode ITL (cumulative −24.9%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t6-gdn-decode-kernel.md)). The decayed state tile stays in registers between the decay and rank-1-update passes (one DRAM read + one write per step instead of two of each) and the grid goes 48→96 blocks so a 78-SM H20 stops idling 30 SMs. c=1 ITL 20.77→20.19 ms; arithmetic order unchanged so greedy is byte-identical. Corrects the 2026-08-01 step-budget note that dismissed grid width as a lever — true for prefill varlen, false for c=1 decode, which has no token axis left to shorten.

- **ACCEPT — T5b lm_head GEMV → cuBLASLt: −2.8% decode ITL (cumulative −22.7%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t5b-lmhead-cublas.md)). `ops::gemv` (lm_head decode logits) issues an N=1 cuBLASLt GEMM; the hand-written kernel read the 1.5 GB lm_head at ~1.1 TB/s vs nvjet's 2.2. The `gemm_small_n_uses_gemv` guard + loop are deleted — all bf16 N≤4 GEMMs ride cuBLASLt. c=1 ITL 21.37→20.77 ms, graphed lane, greedy byte-identical (argmax stable). Remaining vs SGLang 17.07: host tail ~2.8 ms + GDN 0.5 + FA3 0.2.

- **DEFAULT FLIP — `--qwen35-decode-graph` ON (serve + seam default)** (2026-08-03). License: −7.9% ITL matched A/B (23.21→21.37/21.41 ms flagged/default-flags), greedy byte-identical to eager across page boundaries, MMLU 84/100 (invalid=0) vs the 80-81 gdr-chunked battery baseline, 17 captures / 4100+ replays counted per run. Escape hatch + same-binary A/B arm: `--qwen35-decode-graph false`. The train/rollout mirror flag stays OFF pending the OPD co-residency license.

- **ACCEPT — T4 whole-step decode graph under paged KV: −7.9% decode ITL (cumulative −20.5%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t4-paged-decode-graph.md)). The graph lane, a documented no-op under the paged serving default since 2026-08-01, now captures the paged decode step: per-slot fixed-capacity `PageMeta::persistent_decode` refreshed outside the graph, FA3 `seqlen_k` pinned to capacity (device-side `prepare_varlen` re-derives real scheduling per replay), TileLang fallback hard-refuses capture (bakes `num_pages`). c=1 ITL 23.21→21.37 ms with 17 captures / 4100+ replays counted in-log; greedy byte-identical to eager across page boundaries. Opt-in via `--qwen35-decode-graph` pending the MMLU parity default-flip.

- **ACCEPT — T2 qkv + qkvz row-fusion: −2.5% decode ITL (cumulative −13.7%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t2-qkv-row-fusion.md)). Full-attn q/k/v and linear-attn qkv/z load as row-fused marlin matrices via the loader generalized to N parts with per-part shard specs (`load_matrices_row_fused`); one GEMM + split per group, ~80 marlin launches/step removed. c=1 ITL 23.80→23.21 ms; reasoning-channel greedy byte-identical to T5. Closes the marlin launch-count delta vs SGLang (357→~277 vs 270). Remaining program: T4 whole-step decode graph (~3.6 ms) + T6 GDN (~0.7) + T7 FA3 (~0.5).

- **ACCEPT — T5 small-M bf16 GEMV → cuBLAS: −5.1% decode ITL (cumulative −11.5%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t5-small-m-gemv-to-cublas.md)). The gemv guard gated on N and K but never M: the T3-fused `[96,5120]` in_proj_ba ran 52 µs/launch on a 6-block grid (78 SMs idle, ~19 GB/s); requiring `M >= 4096` routes it to cuBLASLt split-K (~8 µs) while lm_head-class shapes keep the hand-written kernel. c=1 ITL 25.08→23.80 ms. Gate = f32 anchor (max err 0.033, bf16 scale) — accumulation-order change makes md5 inapplicable; also fixed `bench_throughput_chat.py` to count `reasoning_content` deltas as ITL events. Full module ledger vs SGLang (both stacks nsys-decomposed, columns sum to measured ITL) re-ranked #196: T4 graph 3.6 ms → T2 qkv 0.9 ms → T6 GDN 0.7 ms → T7 FA3 0.5 ms.

- **ACCEPT — T3 in_proj_b+a row-fusion: −4.7% decode ITL (cumulative −6.7% with T1)** (2026-08-03; `4952f0df5`, [win](docs/experience/wins/2026-08-03-t3-in-proj-ba-fusion.md)). The two per-layer `[48×5120]` bf16 micro-GEMVs (~44 µs each, <10 µs of it data) fuse into one `[2*Vh, hidden]` GEMM + a `split_halves` kernel writing the existing b/a scratch — downstream consumers unchanged. c=1 ITL 26.31→25.08 ms; greedy byte-identical. T1+T3 removed 160 launches/step for 1.8 ms, matching the 4.9 µs/launch nsys gap. Remaining vs SGLang's 17.07 ms is structural: ~5.7 ms of idle across ~1000 still-eager launches → T4 (whole-step decode graph under paged KV) is the only lever left.

- **ACCEPT — T1 gate+up row-fusion: −2.1% decode ITL; Marlin fixed-grid correction re-ranks #196** (2026-08-03; `3e383c082`, [win](docs/experience/wins/2026-08-03-t1-gate-up-fusion.md)). gate/up load as one row-fused matrix (W8A16 fuses INT8 pre-repack; bf16/FP8 device-side), dense MLP = 2 launches instead of 3, LoRA merges via row windows. c=1 ITL 26.88→26.31 ms; greedy byte-identical to unfused. Key learning: Marlin's m=1 grid is fixed at `sms×1` blocks and iterates tiles internally — wider N does NOT lift the 12.5% occupancy, so projection fusion only recovers launch overhead. The 9.8 ms gap's real composition: ~4 ms in_proj_a/b micro-GEMVs (T3, next) + ~5 ms launch-gap idle (T4 whole-step graph).

- **VERDICT — W8A16 matched A/B vs SGLang: same kernel, same weights, SGLang decodes 1.57× faster; the gap is our runtime, not the GEMM** (2026-08-02, [entry](docs/experience/wins/2026-08-02-w8a16-sglang-matched-ab.md)). The ARLE W8A16 checkpoint was mechanically repacked to GPTQ v1 (identical int8 values, kU8B128 semantics) so SGLang 0.5.13 serves the exact same weights through the exact same gptq_marlin kernel. Same H20, same 32k×256 c=1 protocol: decode ITL p50 **ARLE 26.9 ms vs SGLang 17.1 ms**. ARLE reproduces its own record precisely — the ~9.8 ms/step (36 %) delta is non-GEMM wall (SGLang decodes inside a whole-step captured graph; ARLE launches eagerly). Decode #1 lever is now measured, not hypothesized: whole-step decode graph / launch-gap elimination, entry ticket = an nsys diff of one decode step per stack.

- **ACCEPT — device-native cat: matched A/B verdict, strict win (−10.6 GB host RSS, ~5.6× faster)** (2026-08-02; `7276fa081`, [win](docs/experience/wins/2026-08-02-device-cat-ab-strict-win.md)). The device-lazy `ops::cat` was suspected of causing the 343 GB host-RSS OOM at seq=131072; a matched A/B at seq=32768 (real 27B cp=2) proved the opposite — device-cat 46.2 GB peak RSS / fwd 102.6 s / bwd 386.5 s vs host-cat 56.8 GB / 670.0 s / 2054.1 s, losses identical. The 131072 host-RAM wall is inherent seq scaling (superlinear), a separate memcg/offload decision.

- **PHASE EXIT — W8A16 Marlin tensor-core GEMM: bf16-class decode at half the weight VRAM** (2026-08-02; `3ca42b44a`, [win](docs/experience/wins/2026-08-02-w8a16-marlin-tensorcore.md)). The 2026-07-31 wiring lit the W8A16 path on a scalar warp-per-row GEMV (compute-bound, ncu SM 69%); this ports SGLang's `gptq_marlin` (kU8B128) to reach the tensor-core ceiling. **Copy target = Marlin, not Machete**: Marlin runs sm_80→sm_120 as one binary (`__CUDA_ARCH__>=800`, `mma.sync.m16n8k16`, no wgmma) — covers the sm_120 G4 box; Machete is Hopper-only and would abandon half the fleet. Vendored near-verbatim (Apache 2.0), pruned to kU8B128; the two TVM-FFI host wrappers replaced by a plain-CUDA `extern "C"` shim + a ~10-symbol `marlin_compat.cuh` (no TVM-FFI/dlpack/HIP). Symmetric per-group int8 (scale=amax/127, gs=128) → uint8b128 (+128) repacked to the Marlin tile layout at load, scales transposed + `scale_perm`-permuted, kept bf16; SM+shape gated (major≥8, K%16, N%64, gs∈{32,64,128}), scalar/dequant fallbacks kept for the gated-out shapes. c_tmp+workspace scratch alloc-once, SM-sized, never grown → graph-capture safe. **Pod-verified sm_90 H20:** compiles clean sm_80→sm_120; parity 18/18 (ratio 1.00, f32-anchored). **Bench (sanctioned 32k agent workload, graph-captured fast path, matched binary): c=1 ITL marlin 26.9 vs bf16 46.5 ms = 1.73× faster** — pure decode, under the 2× weight-bandwidth ceiling (int8 1 B/param vs bf16 2); c=4/8 prefill-contended on one H20, not cited. **VRAM (P1): freeing the int8 source after repack dropped 27B weights 53→30 GB (0.58× bf16), KV pool 1.74×** — before, both the int8 source and marlin layout stayed resident (the "half VRAM" claim was a lie). Not a prefill/large-batch win (bf16-rate mma). Next wall: sm_120 fleet-coverage run + multi-H20 c-sweep.

- **PHASE EXIT — real 27B 256K CP training runs end-to-end; the VRAM wall is measured at 94.2 GB/GPU (cp=2 fits)** (2026-08-02; `fd8e38e5c` + `b41b130e5` + `1734c69cc`, [win](docs/experience/wins/2026-08-02-linear-attn-cp-a2a-reorder-256k-runs.md)). The hybrid 27B's 48 linear-attention layers use a seq↔head all-to-all CP transport, not the ring — two gaps `nd_parallel_parity` (no linear-attn layers) never saw: (1) `all_to_all_device` world>1 was an `Err` stub (crash at `linear_attention.rs:429`); (2) once a2a ran, zigzag+a2a interleave the 2N seq blocks `[c0,c_{2N-1},c1,...]` but the sequential scan needs true global order — the ring's per-row-position mask tolerates any order, the scan does not. Fixed: NCCL a2a transport (no new kernel) + a 2N-block reorder in `linear_attention_core_cp` (pure slice+cat, backward free). **Verified (pod, GPUs 2/3):** f32-anchored gate with layer 0 = LinearAttention (the prior gate had zero linear-attn layers, proved nothing about the reorder) — `ce_cp_vs_cpu=8.5e-5` at the bf16 floor, `cp_vs_cpu_f32 ≤ single_vs_cpu_f32`, PASS; real 27B cp=2 seq=131072 completes a full fwd+bwd+optimizer step, no OOM. **Backward VRAM peak 94,175 MiB/GPU (~96.6%, ~3.3 GB headroom) → cp=2 is sufficient for 131072, cp=4 not needed; the wall is activations, not the 34 GB of weights.** Also landed: the live agentic `update()` no longer hardcodes `CpContext::single()` (`3c61cff42`, the 2nd cp split-brain). Known follow-up: step wall-clock ~4.3h — the reorder's `ops::cat` host-forces every input (#74, gated behind this PASS).

- **ACCEPT — FA3 for batch==1 prefill (−4%) and the driver-context thread-lottery fix** (2026-08-02; `b0368426a`, [win](docs/experience/wins/2026-08-02-fa3-batch1-prefill-and-ctx-bind.md)). The 2026-07-28 FA3-prefill kill was the per-request-launch cost on ragged c=8 batches; batch==1 is one launch either way, so the paged FA3 route now admits it (split-KV forced to 1 for long q): 33K cold prefill 20.51/20.27 → 19.72/19.47 s, greedy identical, needle 9/9 on the combined default — **33K TTFT-cold now 19.5 s vs the 28.9 s 2026-08-01 baseline (−32%)**. The bigger find: the TileLang AOT dispatch resolves SM/module via the calling thread's driver context, making chunked-GDR availability a per-thread lottery (`CUDA_ERROR_NOT_SUPPORTED` from a binary whose sibling process served fine) — the true mechanism behind this morning's mis-attributed serve failures; fixed with `bind_to_thread` in the probe and branch.

- **DEFAULT FLIP — `--qwen35-gdr-chunked` ON, licensed by the chat-format battery** (2026-08-02; `c2eb5de9e`). Chat GSM8K 100: **95/100 both arms, zero per-item disagreements**; chat MMLU 100: 80 vs 81 (3 disagreements, noise); 33K cold prefill −26/−28%; needle 9/9 ×2; stub-build probe fallback verified. Named trade from the verdict below: raw-completion few-shot can flip knife-edge boundary tokens; chat/agentic serving — the canonical workload — is parity.

- **VERDICT — the chunked-GDR GSM collapse adjudicated: bf16 drift on a knife-edge harness, kernels correct, chat-format quality identical** (2026-08-02; probe `aa03e0566`, [error](docs/experience/errors/2026-08-02-gdr-chunked-gsm-collapse-was-a-knife-edge-harness.md)). The full chain: standalone chained/masked-tail/slow-decay oracles all ≤4.5e-3; the in-serve `ARLE_FQ_PARITY` probe green on the failing request itself (96 layer×segment pairs, state ≤3.7e-3, o ≤5.7e-3); teacher-forcing continues perfectly; temp-0.7 sampling shows the **recurrent arm itself EOSes 3/12** at the failing boundary (raw few-shot on a thinking model = knife edge) and chunked shifts EOS ≈25%→≈50%; **chat template: 14/15 both arms**. Fast-math exonerated by A/B. Default stays OFF pending a chat-format battery; the raw-completion degradation is real and must be a named trade.

- **REVERT — `--qwen35-gdr-chunked` default back to OFF: GSM8K 11/100 vs 46/100** (2026-08-02; flip `2e2ab667c`, revert `715c37a0c`). The flip's license (needle ladder ×3 9/9 exact both arms, greedy-64 byte-identity, 33K prefill −27%) was **insufficient**: a 100-sample GSM8K greedy A/B on the same binary scored chunked **11/100 vs recurrent 46/100**, 35 disagreements all one-directional. Short-recall gates don't catch a prefill-state error that long-form generation compounds; needle runs 2–3 are full prefix-cache hits that never re-enter the chunked path at all. Kernels stay in-tree and opt-in; root cause under investigation (prime suspect: continuation from a prefix-cache-restored h0, the one path the needle ladder never exercised — its fresh-prefill run 1 was exact even at 8k across 4 chained segments). Lesson: an accuracy eval (≥100 long-form samples, same-binary A/B) joins the needle ladder as a default-flip precondition for anything touching sequence state.

- **ACCEPT — FlashQLA chunked GDR generalized to head geometry and made real: 33K prefill −27%** (2026-08-02; `778fef873` + `5b851d193`, [win](docs/experience/wins/2026-08-02-flashqla-chunked-gdr-h48.md), [error](docs/experience/errors/2026-08-02-pod-b64-arg-truncation-stale-binary.md)). Backlog #1 from the step budget. The chunked path was dead twice over: shape-guarded to `v_heads == 32` (the 27B has 48) and **never compiled anywhere** — no `generated/` artifacts existed, `fq_fwd` used an undefined `bhg`, and tilelang 0.1.11's TMA lowering emits `*_desc` wrapper params the AOT surface can't build. H/Hg are now per-instantiation AOT parameters (a `kernels.toml` row triple per geometry; build.rs unchanged) with runtime dispatch on `(k_heads, v_heads)` and recurrent fallback — a new GDN geometry is 3 toml rows + 1 match arm, per the model-universal goal. Measured (H20 GPU 6, same binary, two distinct 33K prompts, cold): recurrent **28.95/28.64 s** → chunked **21.63/20.65 s = −26/−28%**; greedy-64 byte-identical; nsys path probe shows `fq_fwd/kkt/prep` × 1152 at **1.06 s total vs the 9.37 s serial scan (−8.8×)**. New prefill #1 is TileLang full attention (3.99 s / 25%) → backlog #3 (FA3 promotion) is the next lever. Flag stays opt-in; default flip pending the needle ladder ×3.

- **PHASE EXIT — the 27B step is profiled end to end; the backlog is re-ranked off share, not off ideas** (2026-08-01; [win](docs/experience/wins/2026-08-01-prefill-and-decode-step-budget.md), baselines row in [docs/baselines.md](docs/baselines.md)). Two `nsys` captures on a GPU-idle H20 (ThinkingCap-Qwen3.6-27B-FP8, dense, 64 layers = 16 full-attn + 48 linear-attn, 31.2 GB). **Decode, 25 ms/step:** 13.8 ms in `fp8_gemv_batch_kernel` (400 launches) + 4.3 ms in `gemv_handwritten_kernel` (97) = **87%**, `gated_delta_rule_decode` 4%, full attention 1%, and ~4 ms of GPU idle across 1094 launches/step. Against 31.2 GB and H20's *measured* 3.5 TB/s achievable read the GEMVs run at **49% of achievable**, reproducing the 51% attributed on [2026-07-10](docs/experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md). **Prefill, 33K in 28.6 s:** `gated_delta_rule_prefill_recurrent` **9.37 s = 33%**, DeepGEMM 8.33 s = 29%, TileLang full attention 3.93 s = 14%, `pack_quantize` 1.50 s = 5%, plus 2328 `cuMemcpyDtoH` costing 1.58 s inside the prefill loop. Per-part ceilings kill the "30% MFU" framing: DeepGEMM `gate_up`/`down` hit **199 / 189 TFLOPS = 64–67% of the FP8 peak and need no work**, full attention is 54 TFLOPS = 36% of the BF16 peak on the TileLang kernel rather than the FA3 decode already uses, and the recurrence is not compute-bound at all but a latency chain — **5.9 µs per token per layer**, ~6 dependent `__syncthreads` each, with no free parallel axis left (the block is already 512 threads = `val_dim 128 × j_slice 4`, and the token axis *is* the recurrence). Its `<<<48, ...>>>` grid starves a **78-SM** GPU only at c=1; the varlen launcher uses `grid(num_value_heads, batch)`. Shortening the chain (chunked matmul form) is the lever; widening the grid is not. This re-ranks the week: the draft attention three rewrites chased is **1.5 ms of a 35 ms step (4.3%)**, so its −33.2% microbench win was Amdahl-capped at −1.4% before it was written. Also recorded: the same 33K prompt twice runs 35.1 s then **0.525 s** (prefix cache), and two ready-looking flags are inert — `--qwen35-decode-graph` (zero `cuGraph*` calls) and `--qwen35-gdr-chunked` (shape-guards `local_linear_v_heads == 32`; this model has 48, and the CUDA chunked path in `csrc/recurrent/gdr_prefill_{batch,solve}.cu` has an FFI decl and no call site). `docs/baselines.md` cut to champion rows + the step budget; rejected arms and analysis stay in the linked entries.

- **REJECT — `--qwen35-decode-graph` is a no-op under paged KV; the decode step's 87% is weight GEMV, not any kernel** (2026-08-01; not landed, [error](docs/experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md)). Profiling where a Qwen3.6-27B-FP8 decode step goes (`nsys`, 59 steps, H20) put **13.8 ms/step in `fp8_gemv_batch_kernel` (400 launches) + 4.3 ms in `gemv_handwritten_kernel` (97) = 87% of GPU time**, with `gated_delta_rule_decode` at 4%, full attention at 1% (only 16 of 64 layers), and **~4 ms of GPU idle across 1094 launches/step**. Against 31.2 GB of weights and H20's *measured* 3.5 TB/s achievable read (not the 4.02 spec sheet), the GEMVs run at **~49% of achievable** — independently reproducing the 51% attributed on [2026-07-10](docs/experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md), where the per-row activation load+convert tail was named as the whole gap and two kernel fixes were killed. That reframes the day's other REJECT: the DSpark draft attention is **1.5 ms of a 35 ms step (4.3%)**, so its −33% microbench win was Amdahl-capped at −1.4% before it was ever built. The ~4 ms of launch gaps looked like a free flag flip — `--qwen35-decode-graph` exists and its help says "opt-in until the pod license" — but the flag is inert: `try_graph_decode`'s one call site sits below an unconditional `if self.full_attn_paged() { return decode_row_paged_default(..) }`, and paged KV is the serving default, so the lane is live only on the legacy contiguous path. Two `nsys` captures, flag off then on, give **1094 vs 1076 `cudaLaunchKernel` per step and zero `cuGraph*` calls in both**; a four-arm GPU-swapped serve A/B produced only noise (between-GPU spread 87.6–97.8 tok/s at c=8 exceeded the on/off spread). The startup log still prints `decode graph ARMED`. Verify decomposes as **22 ms intercept + 2.48 ms/row** (5.18 ms/row at 33k), and the intercept equals a plain non-spec step — verifying 8 speculative tokens costs what decoding 1 costs, so spec decode is working and the intercept is the wall. Next levers, in order: a Marlin-style tensor-core W8A16 port (the answer already named in July, kernel wiring landed [2026-07-31](docs/experience/wins/2026-07-31-w8a16-kernel-wiring.md)), and making graph capture reachable over a growing page table.

- **ACCEPT — CP training now actually rings: fixed a `self.cp` split-brain, pod-verified FAIL→PASS** (2026-08-01; fix `3d9bc3717`, docs `b5ad2a136`, [error](docs/experience/errors/2026-08-01-cp-split-brain-forward-read-self-cp-not-arg.md), [win](docs/experience/wins/2026-07-31-zigzag-ring-device-kernel-per-row-positions.md)). `masked_writeback_step` sharded the sequence by its `cp` **argument**, but the forward decided whether to run the ring by reading a **different** source — the model field `self.cp`, set only by `set_cp`, whose sole non-test caller was the `cp_hidden_parity` diagnostic. No cli/production path called it, so `self.cp` was always `single()`: **every CP training run silently ran plain attention on each shard, never the ring** (#59's "256K liveness" proved a step completes, not that it's correct). The f32-anchored `nd_parallel_parity` gate caught it — loss_cp 3.2425 vs f32 3.0729, `cp_vs_f32=5.5e-2` = 3700× the single-card bf16 floor (1.5e-5). Fix: thread `cp` as a forward argument (beside `positions`), delete the `self.cp` field + `set_cp` setter — `tp` shards weights at load so it's a ctor arg; `cp` only routes attention at forward time (weights replicated across the cp group) so it belongs with the call, and `layer.forward` already took it. Pod re-verify (HEAD 3d9bc3717, seq=16, GPUs 1,3): **PASS, `cp_vs_f32` 5.5e-2 → 2.4e-4** (~bf16 floor, 83× under the 2e-2 margin), RUN_EXIT=0. The 256K rung (`ARLE_ND_SEQ=131072`, cp=2, local shard 65536 = the >65535 ring path) then completes a full forward+backward+optimizer step: `loss_single=3.232068` vs `loss_cp_sum=3.232163` (two ranks 1.629638 + 1.602526), `rel_err=2.958e-5` ≪ `bf16_tol=1e-1`, no OOM, no hang — **the ring actually fires at 65536**. Corrects the earlier "no bug, gate miscalibration" verdict — the gate was right; I dismissed a real failure via a diagnostic that reconstructed the CE on a correct hidden while skipping the broken forward.

- **REJECT — the draft attention is ALU-bound, but removing the IDIV only pays in a microbench** (2026-08-01; not landed, [error](docs/experience/errors/2026-08-01-draft-attention-idiv-win-is-microbench-only.md)). `ncu` on a pinned-shape standalone harness (`crates/cuda-kernels/tools/nonpaged_attn_bench.cu`, shape from `dspark-fr-native/config.json`) finally characterizes `nonpaged_prefill_attention_kernel`: **Compute (SM) 80.15%, ALU pipeline 61.9%, FP32 at 11% of peak, L2 hit 99.58% with L2 only 7.29% busy, DRAM 0.06%** — issue-bound on integer work, and the GQA-re-read hypothesis is dead. The integer work is `(ring_base + abs_pos) % ring_modulus`, a runtime-modulus IDIV run per key per thread, twice per key; every caller bounds the walk by `kv_len <= ring_modulus`, so normalizing the base once at entry leaves a conditional subtract. Microbench: **−33.2% at 96 rows** (3.794 → 2.535 ms), bit-identical output, ncu duration 4.12 → 2.74 ms. Serve: **+2.7%** (draft attn 7.46 → 7.66 ms at a full ring, output 231.8 → 230.3 tok/s, accept identical 5407/11525), reproduced with the GPUs swapped between arms. The gap is the operating point — the serve runs at `kv_len ≈ 600`, not the config's 2048 window, so the per-key IDIV is a small share of the loop while the entry-normalization IDIV is paid by every thread. Reverted; the harness stays, and its `FastRing` template parameter is a free bit-identity gate for the next attempt.

- **REJECT — the DSpark draft attention's per-key reduction was not its cost** (2026-08-01; reverted in `aa4d2a6ec`, [error](docs/experience/errors/2026-08-01-draft-attention-reduction-axis-was-not-the-cost.md)). Phase splits put 33 ms of the 63 ms draft forward in `nonpaged_prefill_attention_kernel`, whose QK loop gives every key its own block-wide `warp_reduce_sum` — 2048 dependent reductions per row. Swapping the axis (one warp per key, lane strides `head_dim`) is a wash to a regression on the workload: matched A/B against a baseline built from the same HEAD with only the `.cu` reverted (H20, ThinkingCap-27B-FP8 + `dspark-fr-native` block 6, 128 reqs @ c=16) gives attn +5/+11/+15/+2/−1% at 72/78/84/90/96 draft rows, and end-to-end TPOT −1.6% at c=1 / +1.5% at c=16 with accept unchanged (0.412, 0.395 — the reassociated softmax is numerically safe, it just does not pay). At 96 rows the grid is 3072 blocks, far past what is needed to hide a reduction. The linear-in-rows curve (2.4 ms at 12 → 28.9 at 96) says only that the kernel is *saturated*, and the "bandwidth / GQA re-read" reading it suggested was itself wrong — counters later put L2 at 99.58% hit and 7.29% busy. See the follow-up REJECT above for what the kernel is actually bound on.

- **ACCEPT — ISO-Merger grafts one RL skill onto another, same-lineage, data-free** (2026-08-01; `aec0b17c7`, `a84cdfea9`, `a1940eee6`, [win](docs/experience/wins/2026-08-01-iso-merger-same-lineage-27b-graft.md)). The correct ISO-Merger (arXiv:2607.19331 — freeze base Σ₀, merge only the singular *frames* the RL fine-tune rotated), not the spectrum-flattening Iso-CTS that collapsed the MoE ([error](docs/experience/errors/2026-07-30-moe-expert-merge-collapse.md)). Qwen3.6-27B base + ThinkingCap (reasoning RL) + Huihui (de-censoring RL) → one merge, `W* = U* Σ₀ V*ᵀ`, per-tensor retention coefficients from a Gram-solve (no λ sweep, no data). The `c*` split *is* the mechanism: TC near-unit on attention+MLP (0.99/0.93 = reasoning trunk), Huihui sparse (74% of tensors c*≈0 = de-censoring injected only where its frame dominates). Result, n≈100 seed 0 bf16/H20: MMLU iso **0.808** vs TC 0.798, GSM8K iso **0.949** ≥ TC 0.920 — reasoning not just kept but ≥ source — and the lock-pick prompt TC **refuses**, the merge **answers**. Three capabilities stacked, zero dilution. Tool `scripts/iso_merger.py` (fp32 SVD, `--shard-mod/-idx` data-parallel, 140→29 min on 4 GPUs); GSM8K eval fixed to score thinking models via the chat endpoint (invalid 91-97% → 0-10%, root cause verified by single-question probe then n=5 smoke before the full run). dev-only tools, bench-entry gate N/A. FP8 experts need dequant→merge→requant first — not implemented, no verified run. **Supersedes** the symmetric-merge REJECT's implication that data-free merge doesn't work on these models: it does, when the method matches the objective (anchored skill-graft, not symmetric soup).

- **PHASE EXIT — CP correctness core complete: ring full-attn + zigzag load-balance + linear-attn all-to-all-to-head, all CPU-gated** (2026-07-31; `8b3571973`, [win](docs/experience/wins/2026-07-31-cp-zigzag-seqshard-per-row-positions.md), [win](docs/experience/wins/2026-07-31-linear-attn-cp-all-to-all-to-head.md), [win](docs/experience/wins/2026-07-30-cp-ring-attention-and-all-to-all.md)). All three CP build units land, calibrated against Megatron-Core. **(1) Ring full attention** (`cp_causal_sdpa` + `ring_block_attention.cu`) replaces the deleted option-B all-gather — peak O(seq/N·hd), the fix for the >65535 `slice_bwd` OOM. **(2) Zigzag `SeqShard`**: `CpContext::shard` splits the sequence into `2N` chunks and gives rank r the pair `{r, 2N−1−r}` (front+back) so every rank carries equal causal work; the ring now masks by per-row absolute position (`q_pos`/`k_pos` arrays, `cp_causal_sdpa` gains `positions: Option<&[usize]>`) since zigzag columns aren't monotonic. `DpContext::batch_shard` stays contiguous (batch items independent). **(3) Linear-attn CP = all-to-all-to-head** (`linear_attention_core_cp`): each rank runs the full-sequence gated-delta recurrence for 1/N of the value-heads, exact and cross-rank-independent — a source read of Megatron's GDN refuted a `linear_attention_core` interface refactor and confirmed our fused-qkv + packed-conv1d shape *is* the canonical CP contract. CP transport lives in autograd (model-agnostic), not the model forward; qwen35 derives positions from the mesh view (`cp.shard(...).local_rows()`), not a threaded param. `cp.size==1` byte-identical throughout. Gated by zigzag disjoint-cover/target-partition/shard-union parity, ring fwd+bwd vs full-softmax under the new mask, head-split recurrence reconstruction, world==1 vs `causal_sdpa_recompute`; full autograd+train no-cuda suite + clippy + Mac CUDA typecheck green. **Not an ACCEPT**: the device per-row-position ring kernel (zigzag on GPU) + all multi-rank NCCL transport are **pending-remote** — the CUDA path errors loudly on `positions.is_some()` rather than silently mis-attending. The one binding pod gate is the >65535 local-seq CP parity run ([#59](https://github.com/cklxx/arle/issues/59)) + a zigzag-vs-contiguous load-balance c-sweep.

- **DEFAULT FLIP — DSpark static confidence truncation deleted; the head now drives the paper's goodput budget** (2026-07-30; [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md)). `--dspark-conf-threshold` is gone — sglang's deployed DSpark has no such switch (their eval defaults it off too; it is the prior art §3.2.2 replaces). In its place, the sglang-aligned scheduler core: per-position confidence → survival `cumprod` → `qwen35_spec::dspark_verify_lens` picks the batch verify budget maximizing `(R + Σ top-B survival) / (bias + row·(R+B))` and admits globally by survival, with the B=0 arm seeding the search so it structurally cannot lose to no-spec. The additive cost model ships as `--dspark-sps-bias-ms`/`--dspark-sps-row-ms` (defaults = the measured H20 c=16 decomposition, 211 ms + 0.53 ms/row). Both DSpark stacks converge on the one host function: Qwen3.5/3.6 (`dspark_verify_keeps`, batch-global R=b on the batched path) and DSv4 (`dspark_verify_keep`, R=1 per-slot). Simplifications vs sglang, both structural not behavioral: no lag ring (their lag serves overlap scheduling; this engine is synchronous, so current-step survival is available before verify — the paper's Algorithm 1 form), and no STS yet (no fitted temperatures exist for these checkpoints; raw sigmoid until calibrated). Host-only unit gate (`budget_scales_with_survival`). **GPU A/B measured 2026-07-31** (H20, ThinkingCap-27B-FP8, block 6, 128 reqs/point, needle ladder exact=3 DET at 512/4k/16k/32k on both spec arms): against the fixed-depth DFlash control the goodput arm runs **TPOT 20.93 vs 21.77 ms at c=1 and 153.96 vs 162.50 ms at c=16 (−3.9 / −5.3%)**, and it gets there the way the model says — at c=1 rows are cheap so it admits the full block (depth 5.00), at c=16 rows are dear so it cuts the mean chain to 3.79 rows and accept rises **0.275 → 0.402**. It is also the first arm to blunt the c=1→c=16 accept collapse (−0.14 vs the control's −0.23) rather than just absorb it. **The ragged-chain penalty that was supposed to block this does not exist** — isolating A/B at zero code (`--dspark-sps-row-ms 0` makes rows free, so the same binary/weights/prompts emit uniform blocks instead of ragged ones): uniform 160.75 ms vs ragged 153.96 at c=16, and 20.99 vs 20.93 at c=1 where the two arms are the same configuration by construction (0.3% = the campaign's noise floor). Ragged is *faster*, by about what its 1.2 fewer rows/request are worth. The earlier 3.1× was confounded — every "ragged" arm was a *thresholded* arm, and thresholding is what cuts chains to the bare anchor, which used to drop them out of the batched verify into a serial per-slot forward (`dspark_warm_decode_row`, one trunk forward each — the only mechanism in this path at the observed 56 ms/step magnitude; workspace-slot realloc is 40× too small at ~1.5 ms, and the decode graph covers `rows==1` only so a verify is always eager either way). `7358cb06c` keeps those chains in the batch. sglang's `round_up_grid` is not needed here. Spec still costs 2× TPOT at c=16 vs c=1 — that gap is now unattributed and needs its own decomposition. **This A/B first ran against a broken convention discriminator** ([error](docs/experience/errors/2026-07-30-dspark-convention-keyed-on-wrong-config-field.md), fixed `13a5e5dd0`) which collapsed accept to 0.026 — lossless, gate-green, and invisible to everything but the accept counter.

- **DEFAULT FLIP — OPD seq-chunked recompute is unconditional; verdict still PENDING** (2026-07-30; `110632738`, `730ce7f31`, `bbd544f72`, `8970528d3`, [win](docs/experience/wins/2026-07-30-seq-chunk-bake-in-and-dparam-offload.md)). MLP and full-attention recompute each shipped behind a `total_rows ≥ 40961` gate plus an `ARLE_OPD_{MLP,ATTN}_SEQ_CHUNK` override — two knobs, two code paths, and the un-chunked path was the one never exercised past 40960. Both collapse into `runtime_flags::OPD_SEQ_CHUNK = 4096`. Chunking here is **exact, not a tradeoff** (MLP is position-wise; attention is FlashAttention q-tiling with `q_start` and position-sliced RoPE), and a threshold on an exact transform only doubles the paths that can rot. Sub-threshold cost is one input copy. CP untouched — the attention dispatch still sits behind `!cp.is_enabled()`. Three memory walls cleared en route, all on the single-GPU 256K writeback path (ThinkingCap-Qwen3.6-27B-FP8, 1×H20 97871 MiB): full-seq f32 `d_k`/`d_v` (1.5 GiB each at 65536) parked on host between chunks; `checkpoint_seq_chunked`'s `cat_seq` fold, O(seq²/chunk) with old+new co-live, replaced by one preallocated buffer written by disjoint rows — **131072 forward peak 94580 → 85899 MiB, OOM → no OOM**; and eight hand-rolled grad folds across five files converged onto two in-place accumulators (`SeqAccum` for the row axis, `ChunkSum` for the reduce axis), which also fixed a silent zero-gradient class by deriving `requires_grad` in `TapeEntry::record`. **Not an ACCEPT**: 49152 and 57344 complete (`RUN_EXIT=0`, loss 7.2336 / 6.1978), but there is **no completed step at 65536 or 131072** — the 131072 re-run was stopped by hand at 99 min, still in forward. The verdict needs a serial 65536 + 131072, plus 40960 to license deleting `trim_after_checkpoint_replay` ([#188](https://github.com/cklxx/arle/issues/188)). The remaining O(seq) slope is the 48 un-chunked linear-attention layers, which is what blocks 256K on one card ([#189](https://github.com/cklxx/arle/issues/189)). Two cleanup follow-ups fell out of the review and are filed rather than bundled: an in-place accumulate with no ownership probe ([#190](https://github.com/cklxx/arle/issues/190)) and the chunk-accumulator tranche ([#191](https://github.com/cklxx/arle/issues/191)).

- **DEFAULT FLIP — `--dspark-conf-threshold` 0.5 → 0: the shipped default made spec decode slower than no spec decode** (2026-07-30; [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md)). With the no-spec denominator measured, the old default lost outright at c≥8: block 6 TPOT 84.67 vs no-spec 83.34 at c=8 and 159.68 vs 124.86 at c=16, block 16 worse still (91.92 / 172.63). At `0` the same arms run 60.41 and 104.73 — **−13.9/−28.7/−34.4% at c=1/8/16** (block 6), −15.9/−22.6% at c=1/16 (block 16). Five comparisons, one sign, against 2.7% run-to-run noise; only the c=1 pairs match on point order. `0` now also skips the head (`dspark.rs:1541`) — it can never truncate, so its GEMM, D2H and sync were dead work. Gate exact=3 DET at 512/4k/16k/32k. **This is not a rejection of DSpark's Algorithm 1** (arXiv 2607.05147 §3.2.2) — it is the argument for it. We diverge from the paper three ways: thresholding the per-position conditional `c_k` rather than the cumulative survival `∏c_i`, no temperature calibration, per-slot rather than batch-global against a profiled `SPS(B)` curve. A static threshold is the prior art §3.2.2 names and replaces, and the paper's premise — an extra verified token is near-free under light load, costly under concurrency — is measured true here: `dflash16` beats `dflash6` at c=1 (9.16 vs 9.80 ms) and loses at c=16 (144.37 vs 109.43), so the optimal depth moves with load while `--dspark-block-size` does not. In `Θ = τ·SPS(B)` at c=16 there is a real maximum (block 6 = 146.2 tok/s, no-spec 128.1, block 16 **110.8 — under no-spec**), and Algorithm 1 initializes `Θ_best ← R·SPS(R)` and breaks when throughput stops rising, so it structurally cannot ship the failure this flip fixes. **Looked blocked on one thing** (since disproved 2026-07-31, see the entry above): it emits unequal `ℓ_r` by construction and this engine appeared to charge 3.1× for ragged chains (`fr16` threshold 0 vs 0.15: 354.1 vs 1095.7 ms at 253.9 vs 256.0 rows) — a confound, since the penalized arm is also the arm whose truncated chains fell out of the batched verify. Fix that first; then profile `SPS(B)` and index depth on running-request count — per-request allocation is the second increment.

- **ACCEPT the batching, REJECT the confidence truncation — DSpark markov+confidence checkpoints now speculate at concurrency, and the head's verdict (not the head) is what makes them lose** (2026-07-30; `de58404b1`, `51985031d`, [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md)). The 2026-07-29 flip gated its batched draft on `batchable_draft()` (`markov.is_none()`), so a markov head sent the arm back to `--spec-max-batch 1` — **superseding that entry's "markov heads stay clamped"**. I justified the gate as unverifiable for want of a checkpoint; `dspark-fr-native` (markov rank 256 + confidence head, same 5-layer/5120/32-head geometry as `Qwen3.6-27B-DFlash`) was on the host the whole time. The settle's `[248320, 256]` GEMM is weight-bound, so B slots re-read 127 MB B times with an argmax D2H ending every round; one settle over `[vocab, b*block]` with slot-major `prevs` runs every slot's rounds together (a settled slot reproduces its own tokens, so looping to global agreement returns what B separate settles would), and the confidence prefix's per-slot sync + `block` D2D copies become two `batched_copy` launches and one sync. `dspark_block_greedy` and `dspark_confident_prefix_len{,_at}` collapse into `dspark_settle_rows` / `dspark_confident_keeps`, which the b=1 row path also calls; `batchable_draft()` is deleted. Shipped arm unmoved (TPOT +0.3 / −0.07 / +1.4% at c=1/8/16 vs `d05d0aee6`, inside the ±3% band — the new code is a path DFlash never enters), gate exact=3 DET at 512/4k/16k/32k, `cargo test -p infer-cuda` 123/0. Nine arms then priced the drafts at 128 req/point on one binary. `dspark-aeon` and `dspark-fr-native` share geometry, `markov_rank` 256 **and `layer_types`** (5×`full_attention`, all forced to the 2048 window), so they differ only in `enable_confidence_head`; DFlash alone keeps one true full layer. **The batched markov settle is nearly free** — `aeon6` and `dflash6` verify the same 96 rows at the same depth 5.00 and step 269.9 vs 262.7 ms, so the markov head is **≥+7.2 ms/step (+2.7%)**, a lower bound because DFlash's true full layer over 35k should make *its* base step the dearer one; at c=16 it ties the champion (TPOT 108.33 vs 109.43, tok/row 0.415 vs 0.400) — the first measurement of a markov checkpoint speculating above c=1 at all. **The confidence head's execution is free; acting on its verdict costs +162 ms/step.** `--dspark-conf-threshold 0` keeps every GEMM, copy, embedding lookup, D2H and sync and removes only the truncation: `FR t0` steps in 268.7 ms against `aeon6`'s 269.9 at 96 rows/depth 5.00 (**−0.4%**) and 354.1 against `dflash16`'s 352.3 at 256 rows/depth 15.00 (**+0.5%**), while `FR` at 0.5 costs 430.8 ms on **20 fewer rows**. `t=0.15` and `dflash16` verify the same work (253.9 vs 256.0 rows, depth 14.87 vs 15.00) and step **3.1×** apart, 1095.7 vs 352.3 ms. Only how much the chain length *moves* differs — root cause open; the leading hypothesis is that `HiddenSlot`/`VecSlot`/`SliceSlot::get` (`workspace.rs:56`, `:109`, `:135`) reuse a cached buffer only on an **exact** size match, so a varying verify row count re-allocates and re-zeros every buffer including `[248320, rows]` logits (47 MB), and the one within-arm test of it was inconclusive. **REJECT the truncation**: every arm that acts on the verdict falls below *not speculating* (FR 6.3/5.8 vs no-spec 8.0) even though truncating lifts tokens per verify row 33% (0.568 vs 0.428 at c=16) — the head does its job and the wall clock of a variable-length chain eats ~4× what the saved rows are worth. **But `FR t0` block 6 is a champion candidate**: at matched 96 rows and depth 5.00 it beats the shipped `dflash6` by **−4.3% TPOT at c=16 (104.73 vs 109.43), +5.1% decode at c=1 (107.2 vs 102.0), +7.0% tok/row (0.428 vs 0.400)**, and beats `aeon6` too (9.5 vs 9.2 tok/s), with the correctness gate exact=3 DET at 512/4k/16k/32k. That is the FR *weights*, not the head and not the batching. Not flipped: needs the third trial, the c=8 point, and a decision on whether shipping a checkpoint with its own head switched off is the shape we want. **This supersedes the first version of this entry**, which put the cost at +127 ms/step paid once per step, block-independent, and named `gemm_batch` routing a quantized `conf.weight` into `quant_linear::gemm_batch` as the first suspect — the weight is BF16 `[1, 5376]` so that path is never taken, `rows/step`/`step_ms` were miscomputed (81.1/222.2 where the counters give exactly 96.0/262.7), and the excess scales per row, not per step. Two further findings: head-free block size wants **opposite values at the two ends** (block 16 wins c=1 at 109.1 vs 102.0, block 6 wins c=16 at 9.1 vs 6.9) while `--dspark-block-size` is static; and **accept halves at concurrency on every draft** (0.509 → 0.280) with every c≥8 chain drafting on a rebased context (`partial_ctx_chains/chains` 0.75 → 1.00) even as prefix reuse improves — `df.rebase()` leaves the draft suffix-only after a prefix-cache restore. `docs/baselines.md` re-anchored: champion row re-measured, an **accept/tok-per-row column added** (two drafts at one TPOT are not one result), and the rule-4 anchor audit run — the archived `a956f69b1` binary reproduced its own row to −0.03 / +0.40 / +2.26%, bounding accumulated drift across five accepted DSpark updates and the default flip.

- **REJECT — data-free MoE expert merge (Qwen3.6-35B-A3B, 256→N)** (2026-07-30; [error](docs/experience/errors/2026-07-30-moe-expert-merge-collapse.md)). Merging routed experts to cut MoE size destroys the model at every useful ratio. All 4:1 variants (iso64/cal/coact, and router-preserving preserve256) generate repetition garbage, not text ("月食是月食是…"); 2:1 (iso128) recovers grammar but not semantics and still loops. `preserve256` keeps the original 256-way router yet still collapses → the cause is expert weight averaging, not router repack. The MMLU ~30% that looked like "weak" was an eval artifact — A/B/C/D first-token guessing masks a broken generator; only open-ended generation exposed it. Low-rank distillation (all-linear LoRA rank 16, forward-KL from the 256 teacher) held flat at 30/28/30% over 50 steps — low-rank increments can't undo high-rank averaging loss. Storage-only remap (256-logical/64-physical) is output-bit-identical to preserve256 → also collapsed. Two runtime fixes landed en route and stay (both pending-remote): FP8 grouped-MoE LoRA re-merge via BF16-resident experts (`227790953`), and OPD rollout speed — raw-bytes teacher transport (`a501c1f24`) + decode-graph wiring (`2cc72cf8a`, ~2× step). Lesson promoted to method: inspect generation before trusting any MCQ score.

- **DEFAULT FLIP — `--spec-max-batch` 1 → 16: Qwen3.5/3.6 DSpark now speculates at concurrency** (2026-07-29; `6eada66df`, `7ceb39eb6`, [win](docs/experience/wins/2026-07-29-dspark-varlen-replay-c16-win.md), [win](docs/experience/wins/2026-07-29-dspark-batched-draft-across-slots.md)). Spec decode was a c=1-only feature because the draft and the partial-accept rollback both ran per slot. Two batchings fixed that: the draft head's MLP GEMMs are weight-bound at 6 rows (52 µs each, 88.8% GPU-busy — not launch-bound), so B slots re-read the same weights B times; and the rollback replayed 48 conv/GDR layers per slot, 1,536 launches of 10-60 µs per tick. One forward over `B*block` rows, and a varlen pointer-table form of both prefill kernels (1,536 → 96 launches). Matched 5-point sweep, ThinkingCap-Qwen3.6-27B-FP8 + 27B-DFlash, 1×H20, block 6: **c=1 +35% / c=2 +77% / c=4 +71% / c=8 +28% / c=16 +8.5%** tok/s, TPOT −63/−48/−44/−21/−6.1%. Gate exact=3 DET at 512/4k/16k, 0 errors. Shipped after two further batchings of the tick's D2D piles ([snapshot](docs/experience/wins/2026-07-29-dspark-batched-state-snapshot.md), [capture](docs/experience/wins/2026-07-29-dspark-batched-linear-capture.md)): **+37 / +88 / +83 / +37 / +17%** tok/s, TPOT −64 / −51 / −48 / −26 / −13%. Per-request decode (`1000/TPOT`, the SLO the aggregate column hides on this 32k-prompt set) goes 34.6 → 97.3 tok/s at c=1 and 9.8 → 11.3 at c=16. **DSv4 and MTP keep the c=1 gate** — their drafts still run per slot (−47.7% at c=16 in the 2026-07-26 campaign) — as do markov heads, sampling, and quant-KV pools, each clamped at its own call site.

- **ACCEPT — context-parallel N=2 OPD writeback runs end-to-end** (2026-07-29; `b8e2ad96b`, `f55c883a3`, [win](docs/experience/wins/2026-07-29-context-parallel-n2-writeback-works.md), [error](docs/experience/errors/2026-07-29-cp-nccl-wedge-is-hashmap-param-order.md)). First N>1 run with correct loss, not just lockstep collectives. Two ranks (H20, seq 8192, cp-size 2) shard the sequence and DP-sum the replicated weight grads post-backward: grad_norm 1.575355e0 **bit-identical across ranks**, shard losses 4.77+5.73=10.50 vs N=1 10.61 (**1.1%**, within MoE nondeterminism) — parity holds and incidentally confirms RoPE/q_start absolute-position alignment. The wedge was a per-process `HashMap` param order: each re-exec'd rank iterated `adapter_name_map()`'s 2-entry map differently, pairing lora_A `[16,5120]` against lora_B `[12288,16]` into one NCCL collective (no size rendezvous → GPU spins, CPU races to DONE). Fixed at the source (`adapter_ordered()`, fixed A-then-B) plus a fail-fast cross-rank layout guard. **Next: 256K seq-ladder on 8×H20 — gather-prefix-KV load imbalance (rank N-1 does N× rank 0's attention) is the expected next wall.**

- **ACCEPT — Qwen3.5/3.6 paged full attention launches once per layer, not once per row** (2026-07-28; `978c55e09`, `e628be4d3`, [win](docs/experience/wins/2026-07-28-fa3-one-launch-per-layer.md)). An nsys capture at c=16 found 34,212 FA3 launches over ~218 decode steps against 10 full-attention layers — one per row, 16 CTAs on 78 SMs, 52% of the step. The loop's stated reason (FA3 zeroes the page stride under `seqused_k`) is wrong: only `cu_seqlens_k` drops the K/V batch strides. The shim now takes `cu_seqlens_q` + `seqused_k` + a rectangular page table, collapsing decode and spec verify into one call. Matched A/B on `Qwen3.6-35B-A3B-FP8`: ITL p50 c=16 94.61 → 60.90 ms, TPOT c=16 109.42 → 73.74 ms, launches → 3,049 (10.0/step). c=1 unchanged. needle gate exact=3 DET. **MoE expert decode is now 53.9% of GPU time and the open item.**

- **ACCEPT — Qwen3.5/3.6 converges onto the host-authoritative KV page mirror** (2026-07-28; `1fad68524`, [win](docs/experience/wins/2026-07-28-qwen35-host-authoritative-kv-mirror.md)). The arm ran its own device allocator, so a radix hit was only a token match and the recurrent sidecar round-tripped the prefix's full-attention KV through the host every turn. All 23 sites now `mirror_slot` from the host pool, as `executor/qwen.rs` already did, and the KV blob is deleted. Matched A/B on `Qwen3.6-35B-A3B-FP8`, 1×H20, c=1: warm TTFT 3.020 → 0.175 s at 33k and **flat in prefix length**; needle gate exact=3 DET at 512/4k/16k/32k. Seam additions: `KvAllocator::reinstate_slot_page`, grow-to-target for speculative extras. Withdraws the "c=8 wall is scheduler queueing" reading — TPOT moves 224.29 → 61.40 ms at c=8 with no scheduler change.

- **ACCEPT — training-system correctness program, Phases 1–5** (2026-07-27; commits `7bf66b90d`, `a48ebbc02`, `fb066003a`, `986d52d9e`, `2e67bd68e`). Five independently-verifiable tranches, each correctness-complete with local gates; every runtime effect that needs a GPU is `pending-remote`.
  - **P1 — ratio-weighted objectives are the clipped surrogate** ([win](docs/experience/wins/2026-07-27-grpo-family-clipped-surrogate.md)). GRPO/DAPO/Dr.GRPO use the token-level sign-aware clipped surrogate `-min(rA, clip(r)A)` (zero gradient in the saturated branch); GSPO the sequence-level form; CISPO's clamped-IS weight unchanged. Scalar + fused oracles for both advantage signs × below/inside/above.
  - **P2 — one finite-step transaction** ([win](docs/experience/wins/2026-07-27-finite-step-transaction.md)). All CE/PG/GKD/critic/DSpark mutation routes through `finite_optimizer_step`; a non-finite loss or grad norm clears pending grads and advances no parameter/moment/schedule/baseline/artifact.
  - **P3 — public algorithm contracts fail fast** ([win](docs/experience/wins/2026-07-27-algorithm-contracts-failfast.md)). `--gkd-entropy-weight != 0` and `--teacher-topk` reject before model load; Dr.GRPO's fixed-budget normalizer pinned separate from group averaging.
  - **P4 — DAPO dynamic sampling** ([win](docs/experience/wins/2026-07-27-dapo-dynamic-sampling.md)). Dead rollout groups (zero-variance / all-truncated) are refilled from the round's unscheduled tasks, not filtered post-hoc; `RefillBudget` terminates an impossible corpus deterministically. Default off; end-to-end gate `cfg(cuda)` → pending-remote.
  - **P5 — paper-faithful ISO on the DSpark head** ([win](docs/experience/wins/2026-07-27-iso-paper-faithful-dspark-head.md)). Replaces the projection prototype with explicit orthonormal frames `U,V` + frozen `Σ₀`, reconstructed on the tape (chain-rule `G_U/G_V` unit-checked), polar-retracted per publish. Implementation correct; **premise since falsified** (see REJECT below) — default off, no propagation to Agent-RFT.

- **CLOSE (Phase 7a — long agent writeback)** (2026-07-28; forward-peak win `e736c485a`, [win](docs/experience/wins/2026-07-28-opd-writeback-forward-peak-freed.md), [decomposition](docs/research/2026-07-27-opd-writeback-wall-decomposition.md)). Move 1 (free dead MLP+LoRA transients in the tape-disabled forward) shipped and cleared the **forward** wall — seq=40960 all-linear now completes 64/64 forward groups, numerically a no-op. The wall then moved to backward and was measured, not modeled: the mempool A/B is closed (a 35 GB free swing left the identical `add [40960,17408]`=2720 MiB OOM ⇒ live-bytes, not hoard), and the binding chunk is the **attention forward-recompute inside the checkpoint backward (+24 GB before MLP begins)**, pinned to `tape.rs:897-926`. Cutting that peak means seq-chunking the attention forward-recompute in the backward replay — a standalone autograd-op + CUDA-kernel + Metal project, **deferred**: 40960 single-card lossless writeback is not a current product need. 7b (DSpark objective/head capacity) was dropped with DSpark's concurrent-serving deprioritization — **that premise is withdrawn** (see the WITHDRAWN entry above; DSpark is 3.06× at c=1 and positive at c=8), so 7b's drop needs re-deciding on its own merits rather than inheriting this. Phase 7 closes the training-system correctness program.

- **REJECT (premise) — ISO near-isospectral premise fails on the DSpark head** (2026-07-28, `e7e33ff3b`; [errors](docs/experience/errors/2026-07-28-iso-premise-fails-on-dspark-head.md)). Warm head + PM-fix resolved the 2026-07-27 confounds; the clean 3-arm sweep (Qwen3.6-27B, single-GPU, live PM) gives w1 spectrum_drift **2.6e-6 (pure PG) / 3.67e-6 (0.5) / 4.21e-6 (pure dense PM)**. The α=1 positive control — dense supervision, which the paper says moves the spectrum ~100× — is only ~1.6× the PG arm, so near-isospectral drift is a property of this low-rank head under every objective, not evidence RLVR uniquely preserves the spectrum. ISO stays default-off; #32 (Agent-RFT ISO) stays gated and cannot inherit a license here — Agent-RFT ISO, if revisited, must re-establish the premise on its own full-weight modules.

- **WITHDRAWN — "DSpark is net-negative once decode is fast"** (filed 2026-07-27 as a REJECT, `92175f3d5`; withdrawn 2026-07-28, `55bf627bc`). The A/B was rigged: `full_attention_paged` gated FA3 on `meta.seq_len == 1`, and a DSpark verify carries 17 query rows, so **FA3 reached the no-spec arm only** ([win](docs/experience/wins/2026-07-28-fa3-covers-every-query-length.md)). The −6.3% / −7.1% at c=8/16 measured a fixed path against an unfixed one. Re-measured on one binary with the predicate widened to per-request `seqlen_q`, 9 points, 0 errors ([champion row](docs/baselines.md)): DSpark is **3.06× per token at c=1** and still **+6.6% total tok/s at c=8**, +5.2% at c=16. The win decays with concurrency and never turns negative — speculation converts idle capacity, and a full batch has none. What survives from the original entry: the machine saturates at c=8 (all three arms, spec and no-spec, dense and MoE, land within 0.15 ms of each other at ITL p50 66 ms), so the remaining throughput is in the scheduler, not the kernels.

- **ACCEPT — Qwen3.6-35B-A3B MoE is the faster serving target on 1×H20** (2026-07-28, `55bf627bc`; [champion row](docs/baselines.md)). Same FA3 paged kernel, same dataset, same session as the dense 27B, no MoE-specific work needed — the two models share `full_attention_paged` and diverge only at the FFN, and the MoE exercises GQA 8 against the dense model's 6, which PackGQA had not been checked at (needle gate exact=3 DET at 512/4k/16k/32k). At c=1: decode **62.8 vs 34.8 tok/s (1.80×)**, prefill **10991 vs 4252 tok/s (2.6×)**, total **3174 vs 1403 tok/s (2.26×)**, warm TTFT **4.8 vs 14.2 s**. No drafter exists for it, so it has no spec arm; the dense DSpark arm's 106.6 tok/s is not a comparison against this row.

- **ACCEPT — sm_90 paged decode attention routes to vendored FA3** (2026-07-27, `7a275d8ce` + `585e49337`; win: [2026-07-27-fa3-paged-decode-32k-2.76x](docs/experience/wins/2026-07-27-fa3-paged-decode-32k-2.76x.md)). `ARLE_QWEN35_PROFILE` put full attention at 50.84 ms of a ~95 ms decode step at 32k c=1 (3.18 ms/layer, 42 GB/s against 4 TB/s), inside the TileLang `batch_decode_paged_hd256` — which pads `BLOCK_M=64` around one real query row, gives one CTA per query head over a per-KV-head cache, and has no split-KV (~2% occupancy at c=1). The vendored FA3 `paged`/`paged_split` units are PackGQA-only, exactly the wanted shape. Measured ITL p50 **72.1 → 26.1 ms (2.76×, 13.9 → 38.3 tok/s)**, 3 trials within 0.4%; the context term goes from 45.5 ms to ≈0 against the 26.6 ms short-context step. Needle gate exact=3 miss=0 DET at 512/4096/16384/32768. Scoped to sm_90 + BF16 + batch 1; every other target keeps TileLang (FA3 is Hopper-only) and c≥4 is unmeasured.

- **ACCEPT — DSpark markov path batches by speculating its own chain** (2026-07-26, `ffc9ea652` + `0ade41244`; win: [2026-07-26-dspark-markov-chain-self-speculation](docs/experience/wins/2026-07-26-dspark-markov-chain-self-speculation.md)). `bias = w2·w1[prev]` made greedy row selection serial and cost 22.5% throughput just for having a head installed. Guessing every row's predecessor from the base argmax, correcting all rows in one batched pass, and re-running only on disagreement collapses the draft `argmax` sub-phase 8.97 → 0.13 ms/tick (draft total 11.68 → 2.85 ms, post-simplify) with bit-identical greedy output (sha256 match on 4×300 tokens across all three binaries) and unchanged `k_mean` (3.823 → 3.827). Not a default-baseline move: the path only runs with `--dspark-markov-init`.

- **ACCEPT — Agent RFT uses generation-time behavior probabilities** (2026-07-26; win: [2026-07-26-agent-rft-sidecar-denominator](docs/experience/wins/2026-07-26-agent-rft-sidecar-denominator.md)). Ratio-weighted fresh, stale, experience-replay, and offline-replay updates now share one immutable `gen_logprobs` denominator; malformed evidence fails during shared preflight before model work, while CE/GKD remain sidecar-free. Isolated H20 CUDA build and two-epoch offline replay passed; missing/misaligned sidecars failed before model initialization. Fresh online GRPO (`rft-toy08b-g2`) trained 672 tokens with finite IS. The final combined gate (`rootcause-g65-markers-gpu1-20260727e`) trained a real `staleness=1` group for 1,756 tokens, completed five age-1 replay updates of 1,756 tokens each with finite IS (`final mean=0.965304`, `max=4.949177`), recorded `replayed_groups=5`, and exited 0.

- **DEFAULT FLIP — OPD carry GDN backward routes through the device chunked
  path** (2026-07-26, `d6ae52dc1` + `c4709d348`;
  bench: [2026-07-26-carry-gdn-device-reroute-tranche2](docs/experience/wins/2026-07-26-carry-gdn-device-reroute-tranche2.md)).
  `linear_attention_core_with_carry_taped` no longer forces the host
  full-sequence recompute (`state_history` ≈ 86 GB at seq=40960); it seeds the
  carry into `chunk_state[0]` and reuses the seq-independent device chunked
  backward. Host recompute demoted to CPU/unsupported fallback. Live device-carry
  gradcheck PASSED on pod (`5fbf38e4e`); dq 1.74e-3 / dconv 6.29e-3 are bf16
  rounding vs the f32 oracle (A/B-confirmed), not logic bugs — the bf16 gradient
  is the correct adjoint of the bf16 forward. Trainable seq (single H20)
  24576 → 40960 (1.67×; the 64× arithmetic was retracted, `274af1271`).
  Loss parity CONFIRMED (cross-commit A/B vs `a03bf04f2` host baseline, replay lane,
  12 epochs × 3 runs/arm): cross-arm mean Δ ≤ 2e-4/epoch, under arm A's own ≤5e-4
  run jitter — device is statistically indistinguishable from host, well inside the
  <1e-2 bf16-grad bar. Perf license: device chunked backward is +2.6% slower
  (565.2 s vs host 550.6 s median, seq=24576) — a deliberate VRAM-for-time trade,
  paying ~2.6%/step to unlock the 1.67× trainable-seq wall.

- **REJECT (current form) — online markov-head self-RL cannot reach training
  scale, and the markov path taxes 22.5%** (2026-07-26, `14669ec33`;
  bench: [2026-07-26-markov-head-online-selfrl-cannot-reach-scale](docs/experience/errors/2026-07-26-markov-head-online-selfrl-cannot-reach-scale.md)).
  On DeepSpec's own dataset (`mlabonne/open-perfectblend`, target-regenerated
  answers) 1500 prompts yielded 220 sidecar steps × batch 8 = 1760 samples
  against a 127M-parameter head; the learned bias spans 0.052 logits against an
  O(1) top-2 gap, and k_mean is flat (3.831 → 3.837). Turning the head on costs
  −22.5% tok/s (203.8 → 158.0) because it leaves the batched-argmax path. Batch
  the markov gemm first, then train offline. Also fixes
  `--dspark-markov-init` on a DFlash backbone, which had no head slot to
  install into without `--dspark-train`.

- **FINDING — the DSpark draft is a good ranker and a bad argmax** (2026-07-26,
  `d420d894e`;
  bench: [2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax](docs/experience/wins/2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax.md)).
  At the position that breaks the chain the trunk's token is inside the draft's
  top-2 47.0% of the time (top-4 73.3%, top-8 87.8%; rank median 2). Width-2
  candidates project `E[k]` 2.19 → ~5.1. Draft-side cost is ~zero — DSpark's
  non-causal block means row r's logits do not depend on the token picked at
  rows < r — so only verify pays. Cashing it needs tree attention; outlined, not
  started.

- **AMEND — DSpark block size is a lever at concurrency, not only an
  instrument** (2026-07-26;
  bench: [2026-07-26-dspark-block-size-is-a-lever-at-concurrency](docs/experience/wins/2026-07-26-dspark-block-size-is-a-lever-at-concurrency.md)).
  The 2026-07-25 verdict ("keep the default at 16, the flag is not a lever") was
  measured at c=1 and generalized past its regime. At c=8, block 8 beats 16 by
  6.8% (3/3 trials) for the same k_med = 2 accepted drafts: verify 62.1 → 39.1
  ms, tick −30%. No default flip — that needs the long-agent re-measure.

- **ACCEPT — one ragged-window launch per DSpark draft layer** (2026-07-26,
  `9a27eda4b`;
  bench: [2026-07-26-dspark-ragged-window-draft-attention](docs/experience/wins/2026-07-26-dspark-ragged-window-draft-attention.md)).
  Each draft layer launched 16 single-row non-causal attentions, leaving the GPU
  95% idle per launch; `nonpaged_prefill_attention_ring_varlen_cuda` takes device
  per-row window arrays and runs one grid. Measured: draft `attn` 1.53 → 0.19
  ms/slot, draft −32% and tick −6.2% at c=8, aggregate tok/s +4.2% / +2.3% /
  +1.9% at c=1/4/8 (medians of 3 interleaved trials per arm; AFTER wins all 9
  paired points). Acceptance unchanged.

- **ACCEPT (mechanism only, no serving delta) — one batched argmax per DSpark
  tick** (2026-07-26, `308c8b247`;
  bench: [2026-07-26-dspark-batched-argmax-tick](docs/experience/wins/2026-07-26-dspark-batched-argmax-tick.md)).
  The greedy accept scan read argmax row by row through launch + sync + D2H, up
  to 128 pipeline drains per c=8 tick; `argmax_rows()` does the whole verify
  output in one launch. Measured: draft argmax 0.56 → 0.04 ms/slot, tick
  −3.9% at c=8, aggregate tok/s within noise (+1.1% / +1.2% / −0.4% at
  c=1/4/8). Kept for the removed work, not claimed as a speedup. The predicted
  −18% did not materialize: commit's cost is the rejection rollback, not the
  argmax D2H.
- **WORKLOAD — bench workload is multi-turn agent sessions at the TraceLab
  medians** (2026-07-26, `08e1f10f8`;
  bench: [2026-07-26-long-agent-32k-is-the-workload](docs/experience/wins/2026-07-26-long-agent-32k-is-the-workload.md)).
  One-shot unique 32k contexts could never hit the prefix cache; real
  coding-agent serving hits it 95.7% (arXiv:2606.30560). `f4f419629`'s
  "~89% is prefill" reading is withdrawn — at the trace medians decode is
  ~60% of per-step wall clock.
- **PHASE EXIT — spec-decode concurrency gate; three dispatch ladders → one
  `route_decode`** (2026-07-26, `69560ae55`;
  win: [2026-07-26-spec-decode-concurrency-gate](docs/experience/wins/2026-07-26-spec-decode-concurrency-gate.md)).
  MTP/DSpark speculate only at decode batch ≤ `--spec-max-batch`; above it,
  where spec is a compute-bound loss, decode routes to the plain batched path.
  One pure `route_decode(spec_kind, n_rows, gate)` shared by both CUDA
  executors replaces the qwen35 (rows==1 + serial rows>1) and dsv4 (B>1)
  ladders.
- **DEFAULT — `spec_max_batch = 1`** (2026-07-26, `69560ae55`). Pod A/B PASS:
  gate=1 keeps DSpark's c=1 +5.4% (128/128) / +58.4% (256-out) and pins c≥4 to
  ≈ no-spec (±2%); gate=16 reproduces the old c=16 −47.7% loss. Raising to 4
  re-admits the c=4 −22% loss, so 1 is the measured optimum.
- **VERDICT — #128 DSpark accept-or-kill: KEEP as a c=1 feature; the 07-20
  +63.8% vs 07-25 +5% gap was the dataset** (2026-07-26). Same code gives +58%
  at accept_rate 0.51 (256-out) and +5% at 0.30 (128/128) — draft-friendliness,
  one mechanism, not a second effect. `ARLE_DSV4_SPEC_DECODE` env gate deleted;
  `--spec-type` is the single opt-in.

- **VERDICT — backward re-offload lifts the OPD-writeback device wall
  24576→32768, but 256K needs LA-chunk not more offload** (2026-07-25,
  `e4be96108`; win: [2026-07-25-backward-reoffload-device-wall-24576-to-32768](docs/experience/wins/2026-07-25-backward-reoffload-device-wall-24576-to-32768.md)).
  Matched pod A/B on 27B-FP8: offload ON runs clean through seq 32768 (loss
  10.87) where OFF CUDA-OOMs at 24576 — the backward asymmetry was a real leak
  (replayed hidden fetched per-layer, never re-offloaded, so all N co-resident).
  The device CUDA-OOM wall is now at 40960 (`concat_axis2`, 409 MiB free): one
  GDN layer's O(seq) saved backward context fills 97 GB alone, with checkpoint
  already at one layer resident — a recompute working set, not a retained
  buffer, so offload/bf16 can't touch it. Trainable ceiling 24576→32768 (1.33×),
  not the order of magnitude 256K needs. Keep the fix (net-positive at default);
  redirect to chunking the LA recompute working set. (A first probe mis-attributed
  two foreign-job SIGKILLs to our host memcg; a clean re-measure showed both seqs
  pass — corrected in the win entry.)
- **REJECT — #127 "train a DSv4 draft head"; the trained head is public**
  (2026-07-25; docs/architecture-dsv4.md §7 corrected).
  `deepseek-ai/DeepSeek-V4-Flash-DSpark` (MIT) ships a 3-stage head whose 4705
  tensors match `Dsv4DsparkStage` field-for-field; DSpark on DSv4-Flash is
  already served at TP=4. The wall is trigger
  (`--dspark-max-prompt-tokens 64` routes bench prompts to no-spec — not
  measured, not ineffective) plus the throughput-aware verify scheduler (#124),
  which is inference policy, not weights. Capacity at HEAD is unmeasured: the
  19 GB draft reserve seen on a 2026-07-14 binary no longer exists as a term, and
  the draft now shards through the trunk's own `ExpertSplit`/`TpConfig`.
  Draft width is trunk width (V4 4096, Qwen3.6-DFlash 5120); the paper's
  "~1024" is not an artifact shape. A Rust/autograd draft trainer is killed:
  no FP8 GEMM, no expert parallelism, no MoE backward, and DeepSpec ships no V4
  config — that path builds a distributed MoE trainer to reproduce an existing
  MIT checkpoint.
- **VERDICT — #160 device-fit park gate is a backstop, not a reachable path**
  (2026-07-25, #160 closed; wins:
  [2026-07-24-dsv4-band-exhaustion-park-gate](docs/experience/wins/2026-07-24-dsv4-band-exhaustion-park-gate.md)).
  Four escalating 4×H20 pressure configs (to `kv_free_pages 0`, `queue_depth
  37`) never fire the gate: host admission and the device band pool derive from
  the same solved capacity, so admission binds first by construction. The old
  fatal `band_extend` path never executed — 316 KV-overflow preempts, zero
  errors, serve survived. #156's bench debt cleared in the same session (c4/c8/c16
  +1.9/+11.3/+16.5% vs champion; c1 −8.6% is dataset-attributable, #180).
- **REJECT — "DSv4 cold boot is serialized on rank 0"** (2026-07-25, #181 closed
  not-planned). The 25-min cold boot is a storage ceiling, not a code defect:
  `/host` (ext4 on virtio `/dev/vda2`) reads at 0.19-0.23 GB/s and does **not**
  scale with concurrency (`dd iflag=direct`: 1 stream 0.229, 4 streams 0.20, 16
  streams 0.19 GB/s). `loader.rs`'s single-rank 16-thread prefetch already
  saturates it, and warming the page cache once from rank 0 is what keeps a
  4× read amplification (~100 min) from happening. Warm re-boot: 90 s.
- **DEFAULT FLIP — Qwen KV pool sizing: measured VRAM outranks the page floor**
  (2026-07-25, #178, `5c2931cd3`; wins:
  [2026-07-25-kv-pool-floor-yields-to-measured-vram](docs/experience/wins/2026-07-25-kv-pool-floor-yields-to-measured-vram.md)).
  The non-user-facing `total_pages = 8192` default (8.6 GB BF16) acted as a
  floor over the free-VRAM profile, so a 32 GB V100 booked HBM it did not have
  and OOM'd at first prefill. The profile is now the sizing; the requested count
  is the failed-probe fallback only. Big-box behavior unchanged; DSv4 unaffected.
- **DEFAULT FLIP — `--kv-disk` with a zero derived budget degrades to no-tier**
  (2026-07-25, #158, `59b86ee4c`). Free space below the `max(50 GiB, 10%)`
  reserve used to fail the boot; it now warns and serves without the disk tier.
  An explicit `--kv-disk-limit` still fails loudly.
- **VERDICT — DSpark V100 TP-lockstep stall: FIXED, measured** (2026-07-25,
  #168, `6c5553b45`; errors:
  [2026-07-21-dspark-v100-tp-lockstep-stall-kill](docs/experience/errors/2026-07-21-dspark-v100-tp-lockstep-stall-kill.md)).
  2 h continuous DSpark load at HEAD: zero `lockstep stalled` WARNs, zero
  errors. The earlier −91% KILL is re-grounded to the sm_70 draft path itself
  (#179) and the pool floor (#178), not to lockstep.
- **DEFAULT FLIP — writeback-offload threshold 4096 → 16384** (2026-07-24,
  #172; wins:
  [2026-07-24-writeback-offload-dial-back](docs/experience/wins/2026-07-24-writeback-offload-dial-back.md)).
  Measured seq sweep 5K-28K on 27B: offload buys zero peak headroom, costs
  −29…−38% backward; resident OOM boundary moved 9.6K → 24.6K post fused-CE +
  batched-LA.
- **ACCEPT — FP8 quant loss on 27B: −0.25% PPL vs bf16** (2026-07-24, #174;
  wins:
  [2026-07-24-ppl-harness-fp8-matrix](docs/experience/wins/2026-07-24-ppl-harness-fp8-matrix.md)).
  First GPU run of `arle train ppl` (WikiText-2 test, ctx 2048); also fixed the
  paged-KV panic in `forward_token_logits` (`067849cf3`).
- **REJECT — group-stagger admission for CC preamble prefix reuse**
  (2026-07-24, reverted in `2ab7883f1`; errors:
  [2026-07-24-group-stagger-premise-false](docs/experience/errors/2026-07-24-group-stagger-premise-false.md)).
  Pod A/B: baseline already prefix-hits — hit trajectories identical
  (cumulative hit_tokens 4.243M vs 4.208M), claude-CLI boot spread +
  ~90 s turn-1 publish serialize sample starts naturally; the modeled
  8×21K concurrent cold-prefill waste does not occur.
- **ACCEPT — agent-OPD sandbox staging outside the repo** (2026-07-24,
  `6bd40d663`+`b0a29443e`+`031c8c3f8`+`e21557fbc`). Sandboxes under the ARLE
  checkout made `claude -p` ingest the repo `CLAUDE.md`: ~31K prompt
  tokens/request. Staged at `/tmp/agent-opd` instead; pod-verified
  prompt_tokens → median 21.4K (−9.7K), rollout wall −40% (410 → 204–256 s),
  SAMPLES=8 fits the KV pool again. Measured CC intrinsic floor ~21K/request
  re-ranks hybrid prefix reuse as the top residual lever:
  [wins](docs/experience/wins/2026-07-24-agent-opd-sandbox-staging-verified.md).
- **ACCEPT — batched linear-attention CUDA device path** (2026-07-24,
  `ecc058b20` + `5f68d1f6e`). B>1 LA previously fell to a CPU fallback whose
  scan-assist kernel overflowed i32 on `state_history` (B=4×3150 →
  ILLEGAL_ADDRESS); per-row dispatch over the proven batch==1 kernels fixes
  the crash and the 337 s/micro-batch checkpointed pathology (now 6–23 s/mb,
  loss parity). Checkpoint gate models the 12 LA ctx tensors exactly.
  Pod-verified:
  [wins](docs/experience/wins/2026-07-24-cuda-batched-la-device-path.md).
  Gate boundary calibrated ×3→×4 same day from the measured full-tape ramp
  (#170, `b2a5d6180`); cuda-lane clippy debt cleared (#171, `0e05b052c`).
- **ACCEPT — systematic review-fix sweep (26 findings); one relay regression
  fixed** (2026-07-24, `f0a635e02` + `837b89d39`). 26 confirmed defects fixed
  (2 high, 9 medium, 15 low) across scheduler / serving / OPD / kv-tier /
  autograd / CUDA. Verified on H20 sm_90 (DSv4-Flash TP=4: build + needle 15/15 +
  c1/c4 perf-neutral) and built on Colab G4 sm_120. The sweep introduced one
  regression — an `accept_n` hello-read timeout leaked into the steady-state
  relay reader → TP=4 c8+ serve teardown — found + fixed + pod-confirmed
  (c8 48/48, c16 64/64, no teardown). The c16 deficit first flagged here was
  pod-measured on HEAD `2ffc19736` and does NOT reproduce — c8→c16 1.75× (above
  champion 1.58×), a measurement artifact, not a code regression. Details:
  wins/2026-07-23-systematic-review-fixes,
  errors/2026-07-24-relay-hello-timeout-leak-tp4-teardown.

- **DEFAULT FLIP — self-opd distill path fused → dense** (2026-07-24,
  `38bac08e6`). Self-opd now honors `--fused-distill` (new flag, default
  false) instead of hardcoding fused; dense is the validated direction
  (fused ran lm_head on host, ~205 s/step on 27B). Pod-verified: 8-step smoke
  loss 1.435 ≈ fused 1.440. An 88 GB transient during dense smoke steps is
  flagged for follow-up (wins/errors 2026-07-24).

- **REJECT — checkpoint-gate ×4 tightening reverted** (2026-07-24). The ×4
  estimate fixed the B=4 long-completion writeback OOM (97.5 → 41.1 GB) but
  the newly-engaged batched checkpoint backward crashes in linear_attention
  dqkv, and short B=4 shapes over-fired onto a 3×-slower branch — reverted to
  the prior estimate; long-completion batched writeback runs
  `--writeback-batch 1` (verified, 59 GB) until the LA backward is fixed.
  Details: errors/2026-07-24-batched-checkpoint-la-backward-crash-and-gate-overfire.
  Related cleanups kept: per-lane `--grad-checkpointing` collapsed into the
  shared `--gradient-checkpointing` (now default true, matching the prior
  lane defaults); train control-plane + metrics dead code deleted (−3.2K LOC,
  `9933e21e8`).

- **ACCEPT — agent-OPD rollout is idle-bound; concurrent mega-rollout GO**
  (2026-07-24). First hard-gated `gpu_busy_frac` measurement (new per-group
  timer): 0.30–0.34 on 1×H20 — ~2/3 of the rollout wall the GPU idles on
  CC-side latency, flat under 4→8 concurrency. Unblocked by two serve fixes:
  local-relay ack pump (watchdog-teardown class gone, `e4ac039dc`) and the
  device-pool budget gate (KV exhaustion parks instead of killing the engine,
  `a9d0c5412`). Case-audited passes; ~31K/request prompt traced to CC
  ingesting the repo `CLAUDE.md` from in-repo workdirs. Details:
  [wins](docs/experience/wins/2026-07-24-agent-opd-gpu-busy-frac-measured-go.md).
- **ACCEPT — sm_120 FP8 MoE prefill: CUTLASS grouped GEMM (G2)** (2026-07-22).
  The Blackwell (RTX PRO 6000, sm_120) FP8 MoE prefill now runs the CUTLASS 4.3.5
  sm_120a grouped blockwise-scaling collective instead of the pathological
  hand-grouped GEMV fallback (the Hopper-only DeepGEMM path never compiles for
  sm_120). c=1 cold-prefill TTFT **84,634 → 760 ms (~111×)**, total throughput
  c=8 **26.5×**, c=16 recovered from full collapse. needle exact/DET 115..8000;
  correctness_failed=0. Scale layout matched by construction (SFA custom K-block
  stride, SFB load-time transpose). Opt-in on sm_120 targets only; Hopper path
  byte-unchanged. [wins](docs/experience/wins/2026-07-22-bench-sm120-fp8-moe-cutlass-grouped.md).

### Removed (dead surface — `crates/train`, 2026-07-23)

- **train-crate systematic simplification, −4,134 LOC.** Deleted pivot-orphaned
  dead code and collapsed four single-impl traits, behavior-preserving (238 tests
  green, clippy clean, `arle train {opd,self-opd} --smoke` end-to-end). Removed
  `pub` surfaces: the generic `Trainer<O,C,S>` subsystem (`Trainer`/`TrainerConfig`/
  `StepCtx`/`StepOutcome`/`EvalOutcome`/`GradAccumulator`/`GradClip` trait), the
  stale `MoeWithLora`, and the `SequenceWindowedForward`/`TrajectoryScorer`/
  `TeacherWindowedForward`/`CausalLm` traits (folded into inherent/concrete forms).
  All had zero production callers; live free helpers (`cleanup_after_backward`,
  `clip_grad_norm`, …) retained. Bench-exempt: host-side OPD-training, no serving
  perf surface. [wins](docs/experience/wins/2026-07-23-train-crate-systematic-simplification.md).

## [0.4.0] - 2026-07-22

Headline: **DSv4 production hardening** (prefill chunk 2048 default, FP32
compressor, c32 preemption survival, prefix reuse), **DSpark train sidecar +
batched verify**, and **#167 temp>0 sampling correctness**. ~300 commits since
v0.3.0.

### Added

- **DSpark train sidecar** (`--dspark-train` / serve background trainer) —
  acceptance-weighted Markov-head updates from the live experience buffer with
  hot-swap into the running engine. H20 e2e: 6 steps, loss −4.04→−3.18.
  [win](docs/experience/wins/2026-07-20-dspark-train-sidecar-e2e-verified.md).
- **DSpark batched verify (B>1)** — batch anchor forward + FlashMLA-gated
  verify; TP lockstep proposal/accept extracted. c>1 crash closed; c8/c16 still
  structural regress vs no-spec on some shapes.
- **Qwen3.5/3.6 MTP speculative decode** — paged + contiguous, MoE draft head,
  rejection-sampling acceptance for RL-lossless sampled rollouts (opt-in).
- **Agent-OPD cc-harness path** — in-process serve, pass-rate task selection,
  experience-replay (opt-in), `--staleness 1` IS correction, dense/binary/anchored
  reward shapes, held-out eval concurrency.
- **V100 (sm_70) serving substrate** — BF16→FP16 GEMM cast, paged-attention
  `allow_sm70`, W4A16 MoE grouped GEMV, FA2 hand-written attention.
- **ThinkingCap-27B-FP8** as the canonical CUDA agentic model default.
- **Unified direct L3 storage** (`kv-tier`) + DSv4 L2/L3 hit path measured.
- **DSv4 local-NVMe cold load** — `ARLE_LOADER_PREFETCH=0`; 294 GB → HTTP ready
  in ~81 s on local NVMe (vs ~28 min virtio).
- **Qualified kernel artifact flow** — candidate pack → per-SM fragments →
  aggregate sidecar; release validates exact bundle ID + ancestry.

### Changed

- **DSv4 prefill chunk default 128→2048** (default flip). c1 cold TTFT
  3031→1088 ms; c16/c32 out tok/s +126%/+128%. Override:
  `ARLE_DSV4_PREFILL_CHUNK`.
  [win](docs/experience/wins/2026-07-17-dsv4-prefill-chunk-2048-default.md).
- **DSv4 FP32 main-value compressor default-on** (correctness for #146/#150);
  all compression boundaries; prefill-only probe (unblocks DSpark decode);
  grid-parallel probe; FP32 scratch hoisted off per-slot state (slots 2→59).
- **`--max-running-requests`** frees per-slot VRAM into the compute pool —
  **84k → 1,048,576 tokens (12.5×)**; c32 preemption survival (#164/#162).
- **Agent-OPD curve defaults** — deep-research Phase-2 config (dapo +
  staleness 1 + G=8); grpo temp=1.0 unblocked after #167.

### Fixed

- **#167 Qwen3.6 temp>0 sampled-tail garbage** — dual RMSNorm bugs (hd256 q/k
  OFFSET convention + final-norm `w-1` load). temp=1.0 + greedy coherent;
  on-policy grpo unblocked.
  [errors](docs/experience/errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md).
- **DSv4 extension-prompt prefix reuse** — finish write-through no longer
  clobbers aligned boundary entries; also #165 CSA indexer bf16 pending.
- **DSv4 plan-repair / HostPagedKvPool fatal at c32** — evict-until-freed +
  degrade-to-park on step-path alloc failure.
- **SM-gate Qwen FP8 dense DeepGEMM to Hopper-only** (`major == 9`).
- **DSpark draft latent sliding-window** + oversized prefill chunk rebase.

### Verdicts (selected)

- **2026-07-25 — bf16 tape Stage 1a (frozen prefix K/V) rejected: no VRAM win on
  Qwen3.6-27B.** `--tape-precision bf16` on `PrefixKv.k/v` measured **+288 MiB**
  (not lower) at the writeback and did not move the OOM wall — the frozen K/V is
  0.13 MB/tok (16 full-attn layers of 64), ~400× smaller than the GDN
  linear-attention forward-capture transient that sets the peak (+52.7 GB at
  seq1024). Mechanism correct (loss byte-identical, needle 5/5 DET); it points at
  the wrong buffer. S0 config + S1a quantize/widen substrate kept (no-op at fp32
  default) for Stage 1b, which re-targets the `la_*` forward buffers.
  [errors](docs/experience/errors/2026-07-25-s1a-frozen-prefix-kv-bf16-no-vram-win-kill.md)
- **2026-07-21 — #167 closed: Qwen3.6 temp>0 sampled-tail garbage fixed (accept).**
  `b4b293f0c` carried two independent RMSNorm bugs. Type-A (kernel, `e4d5580ca`):
  hd256 q/k `(1+w)`→`w`. Type-B (load, `d703b5240`): a `w-1` transform on the final
  RMSNorm weight before the correct `(1+w)` kernel, sign-corrupting the STANDARD
  final-norm's negative channels → flattened logits → temp=1.0 garbage; greedy
  survived so it hid behind the greedy gate and persisted to HEAD after the Type-A
  revert. OFFSET-held bisect (sha-verified) → blanket revert to `load_vec`.
  Pod-verified temp=1.0 + greedy COHERENT. **temp=1.0 on-policy grpo unblocked.**
  [errors](docs/experience/errors/2026-07-20-hd256-fp8-temp-sampling-corruption.md) ·
  [#167](https://github.com/cklxx/arle/issues/167).
- **2026-07-20 — DSpark train sidecar Phase 1 shipped (accept, end-to-end verified).**
  `arle serve --spec-type dspark` now spawns a background acceptance-weighted trainer
  that drains the experience buffer the hot path populates and hot-swaps
  updated Markov-head weights back into the running engine. Verified on H20
  (Qwen3.6-27B-FP8 + dspark-aeon draft): 6 training steps, loss −4.04→−3.18,
  zero errors. Fixed: hardcoded `vocab_size` (now lazily inferred from
  experience) and draft-model selection (requires `dspark-sp+markov`, not
  backbone-only DFlash).
  [win](docs/experience/wins/2026-07-20-dspark-train-sidecar-e2e-verified.md).
- **2026-07-17 — DSv4 cold-boot #69 closed (verdict: fixed in code, disk-bound residual).**
  Re-measured on current main: warm boot 33 s, all ranks build concurrent —
  the filed rank-0 serialization and 8× read amplification were fixed by the
  loader rewrite (mmap zero-copy + single-pass rank-0 prefetch). True cold
  boot 26.5 min is 98% one saturated sequential read: 294 GB @ 0.19 GB/s
  virtio device cap (dd-verified at 1/4/16 streams). Remaining lever is
  storage infra, not runtime code.
  [wins](docs/experience/wins/2026-07-17-dsv4-cold-boot-69-attribution.md) · #69.
- **2026-07-17 — DSv4 extension-prompt prefix reuse fixed (accept, wash).**
  The finish write-through's frontier recapture clobbered the prefill
  chunk-end's tail-less boundary entry, so every diverging-suffix prompt (the
  multi-turn shape) licensed 0 reuse blocks and later finishes retroactively
  destroyed hitting shapes. Fix keeps the aligned boundary entry; prefix_reuse
  gates 2000+2003 PASS (reuse_hit 1792t ×3), needle 27/27, bench wash
  (−0.3…−1.2% inside drift band, c32 TTFT −18.6%). Also closed #165 (CSA
  indexer bf16 pending now in the write-through image, pod-verified).
  [wins](docs/experience/wins/2026-07-17-dsv4-extension-prefix-reuse-fix.md) ·
  #165 #166.
- **2026-07-17 — DSv4 prefill chunk default 128→2048 (default flip, accept).**
  The planner's one-unit alignment cap pinned every DSv4 prefill tick at 128
  tokens and made `--chunked-prefill-size` cosmetic. Three-phase unification
  (config honesty + `max_prefill_chunk` capability; SW-ring tail-slice race
  fix + opt-in flag; flip): c1 cold TTFT 3031→1088 ms, c16 out tok/s +126%,
  c32 +128% (209.9 tok/s), ITL p99 better outright; needle zero-miss ×2
  passes. `ARLE_DSV4_PREFILL_CHUNK` overrides. Entry:
  [wins](docs/experience/wins/2026-07-17-dsv4-prefill-chunk-2048-default.md).
- **2026-07-17 — #164/#162 CLOSED (accept): c32 × 300 s oversubscription
  survival with real preemption (192 events, zero teardowns).** Final fix
  chain `f03a54f4a`: the evictor freed live-attached pages (cross-slot KV
  corruption seed) and counted radix-severed orphans as "reclaimed" —
  `page_is_evictable` (retained-once ∧ slot-unattached) now single-sources
  repair capacity and evictor filtering; evict-until-freed; every step-path
  alloc failure degrades to shed/park. Same run accepted `77e0d1d5d`:
  `--max-running-requests 32` frees per-slot VRAM into the comp pool —
  **84k → 1,048,576 tokens (12.5×)**, c32 +42% out tok/s / ITL p99 −84%,
  grid c1–16 unchanged. Entry:
  [wins](docs/experience/wins/2026-07-17-max-running-requests-caps-slot-budget.md).
- **2026-07-16 — polish round: high-effort adversarial review of the day's
  commits fixed 8 confirmed defects pre-deployment.** The first #164 repair
  livelocked warm caches (evictable ≠ free) and could hang TP via rank-local
  `free_pages` — fixed in `459ed5000` (rank-synced free+evictable capacity,
  demand-aware shedding, spec-alloc degrade-to-park;
  [errors](docs/experience/errors/2026-07-16-plan-repair-evictable-not-free.md)).
  `f59dd79af` closes the FP32 carry coherence family: reseed-on-stale at
  restore/reset/decode-advance (cross-request contamination), a mainline
  pending→bf16 mirror gap, and reverts the GLM never-read scratch gate.
  Residual: CSA indexer pending not in the finish write-through (#165).
  Pod needle + c32 repro remain the pending acceptance gates.
- **2026-07-16 — DSv4 FP32 probe scratch hoisted off per-slot state (accept):
  per_slot 9618→338 MB, slot clamp 2→59.** Same-config rolling comparison:
  output tok/s +9% (rate 1) to +48% (rate 16), c32 +72% req/s with TTFT p50
  halved; var-c1 wash (clean null control); needle zero-miss both depths.
  Exposed a previously unreachable fatal path — `HostPagedKvPool out of
  pages` crashes the serve at c32 instead of preempting
  ([errors](docs/experience/errors/2026-07-16-dsv4-c32-hostpagedkvpool-fatal.md));
  now the top high-concurrency blocker. Entry:
  [wins](docs/experience/wins/2026-07-16-dsv4-fp32-scratch-hoist-slots.md).
- **2026-07-16 — DSv4 FP32 prefill compressor grid-parallelized; serial probe
  kernel deleted (accept).** The `<<<1,256>>>` FP32 probe serially swept every
  compressed block per prefill call; now launches the templated grid-parallel
  block/finalize kernels (bit-identical FP32 numerics). Same-shell A/B (4×H20
  TP=4, eager): total tok/s +3.7%..+6.3% at rates 1–16, +18.9% at c=32; needle
  zero-miss at depths 0.0/0.5. Both arms log `256 slots clamped to 2`
  (per-slot FP32 scratch, hoisted in `672b8ac08`, A/B pending) — slots, not
  kernels, are the remaining high-concurrency wall. Entry:
  [wins](docs/experience/wins/2026-07-16-dsv4-fp32-compressor-grid-parallel.md).
- **2026-07-16 — DSv4 FP32 probe limited to prefill; unblocks DSpark (MTP)
  decode.** The all-boundaries FP32 compressor was running the FP32 probe on
  every decode token, including the DSpark draft phase (each draft token
  re-ran the FP32 GEMM+probe). Guarded to prefill only
  (`start_pos_device.is_none()`); simplified the guard from 5 conditions to 3.
  Needle gate: depth 0.0 all-pass, depth 0.5 no-misses (prefill correctness
  preserved). DSpark MTP bench: 48–49 output tok/s, ITL p50 flat at 40.8ms
  (compute-bound). The 48–49 tok/s is MTP speculative decoding now unblocked
  by the probe removal, NOT the isolated probe effect (the no-MTP baseline
  confounds). The all-boundaries bench's −17% to −36% eager total-tok/s is the
  best isolated estimate of the probe-on-decode overhead.
  [bench](docs/experience/wins/2026-07-16-dsv4-fp32-probe-prefill-only-dspark-recovery.md).

- **2026-07-16 — DSv4 FP32 compressor extended to all compression boundaries.**
  The FP32 main-value compressor now runs on every prefill compression boundary
  (any `start_pos`, with prior compressed state), not just the first boundary.
  This fixes the depth=0.5 needle retrieval corruption at len=600 (#146, #150).
  Needle gate: depth 0.0 all-pass (9 lengths), depth 0.5 no-misses. Performance
  cost: −17% to −36% total tok/s vs first-boundary-only (redundant FP32 GEMM per
  boundary). [bench](docs/experience/wins/2026-07-16-dsv4-fp32-compressor-all-boundaries.md).

- **2026-07-16 — DSv4 FP32 compressor promoted to default.** The FP32
  main-value compressor (fixes #146 VIOLET-6529→4929, #150 738291→738292) is
  now always-on, replacing the BF16 compressor for single-prefill. The
  `ARLE_DSV4_COMPRESSOR_FP32` env flags were removed. Throughput impact:
  ±0.7% at c=1/4, −1.1% at c=8, −3.6% at c=16 — acceptable for the
  correctness fix. [bench](docs/experience/wins/2026-07-16-dsv4-fp32-compressor-promotion.md).

- **2026-07-15 — DSv4 long-context correctness blocked.** Full-prefill TP=4
  retrieval became nondeterministic at 5,424 actual prompt tokens and failed
  3/3 at 7,222 tokens for a 50%-depth needle, while the same 7,222-token prompt
  passed at 90% depth. Throughput measurement stops at the correctness gate.
  [error](docs/experience/errors/2026-07-15-dsv4-long-context-needle-failure.md).

- **2026-07-15 — DSv4 local-NVMe cold load shipped.** A zero-residency 294 GB
  checkpoint reached HTTP ready in **80.95 s**; the virtual system-disk run had
  spent **1,675 s** in prefetch alone. Local lazy loading is now explicitly
  selectable with `ARLE_LOADER_PREFETCH=0`.
  [bench](docs/experience/wins/2026-07-15-dsv4-nvme-cold-load.md).

- **2026-07-15 — DSv4 batched greedy lm_head shipped.** Three-trial median
  output throughput improved **+1.53%/+2.25%/+2.52%** at c=4/8/16 with no
  errors or correctness failures. The eligible path is built in with no flag.
  [bench](docs/experience/wins/2026-07-15-dsv4-batched-lm-head.md).

- **2026-07-15 — DSv4 L2/L3 real-hit path measured.** A 1,649-token repeated
  prompt exercised 4 L2 plus 47 L3 state pages and 51 disk reads on the first
  reuse. At c16, L3 cost **4.19%** output throughput and **11.9%** TTFT while
  extending the retained working set.
  [bench](docs/experience/wins/2026-07-15-dsv4-kv-l2-l3-hit-throughput.md).

- **2026-07-15 — DSv4 MegaMoE retained but correctness-blocked.** A no-prefix native A/B
  measured c16 **177.20→256.56 tok/s (+44.8%)**, but one decoded case entered a
  deterministic attractor on MegaMoE and passed on allreduce; direct replay
  reproduced it 5/5 times. The implementation stays for repair, but correctness
  blocks default use. [error](docs/experience/errors/2026-07-15-dsv4-megamoe-decoded-case-failure.md).

- **2026-07-14 — DSv4 DSpark TP=4 concurrency licensed.** Loading the already
  EP4/TP4-sharded draft before KV planning replaced a false 19.9 GB/rank reserve
  with the measured 4,960 MB resident footprint, raising slots **1→33**. The
  TP-unsafe sequential B>1 draft path was deleted; batches use the target decoder
  until a TP-safe batched verify lane exists. GuideLLM c=1/4/8/16 throughput is
  **45.04/80.06/120.66/141.46 tok/s**: -1.0%/+71.6%/+159.1%/+203.9%, zero errors.
  [bench](docs/experience/wins/2026-07-14-dspark-resident-budget-tp4.md).

- **2026-07-14 — DSv4 DSpark prompt router licensed for H20 TP=4.** DSpark
  output throughput moved from +6.3% at 32 prompt tokens to -12.4% at 128 and
  -18.6% at 8K. The opt-in `--dspark-max-prompt-tokens 64` router preserves the
  short-prompt path and restores 128/8K to within 1% of no-spec; defaults remain
  unchanged. [bench](docs/experience/wins/2026-07-14-dspark-prompt-router-tp4.md).

- **2026-07-14 — V100 (sm_70) prefill `cudaErrorNotSupported` fixed.** Two
  fixes, both gated exclusively on compute-major ≤ 7 so the sm_80+ hot path is
  byte-identical: (1) BF16 GEMM on Volta (no BF16 tensor cores — only FP16/FP32)
  now casts BF16→FP16, runs an FP16 tensor-core GEMM, casts back, skipping
  cublasLt's bad-algo heuristic; compute-major cached per device to avoid the
  uncached-per-step −77% decode regression. (2) `allow_sm70 = true` for the
  HD256 q8_kv2 paged-attention prefill/decode kernels (the 0.8B dense config),
  so the sm_70 cubin is compiled instead of the runtime `cudaErrorNotSupported`
  stub. [gemm](docs/experience/wins/2026-07-14-v100-sm70-bf16-gemm-fp16cast.md),
  [paged-attn](docs/experience/wins/2026-07-14-v100-sm70-paged-attention-allow-sm70.md).

- **2026-07-14 — DSv4 DSpark correctness PASS, opt-in unchanged.** Restored the
  official HC-lane mean, native BF16 Markov weights, and accepted-prefix recurrent
  fold: coherent 128-token output with **61/170 accepted (35.9%)** on H20 TP=4.
  Also bounded checkpoint prefetch to rank zero plus page-cache capacity, removing
  the observed 4-rank full-checkpoint read amplification. [correctness](docs/experience/wins/2026-07-14-dspark-dsv4-accept-and-correctness.md),
  [load](docs/experience/wins/2026-07-14-loader-tp-rank0-prefetch.md).


## [0.3.0] - 2026-07-12

Headline: **DSpark speculative decoding** for DSv4/Qwen3.6, a **CUDA kernel
`csrc/` reorg + content-addressed prebuilt-kernel release**, and a
**strategy-driven agent-OPD harness**. ~1 month of runtime + training work since
v0.2.1.

### Added

- **DSpark block-draft speculative decoding** (`--spec-type dspark`) — DSv4
  dual-stream draft + 3-stage backbone orchestrator + draft→verify→accept loop
  (T1–T4.4). P1 LICENSED: **2.39× decode short-ctx / 3.14× at ~3K** vs no-spec
  (Qwen3.6-27B, H20 TP=1). See §Verdicts.
- **Unified kernel set** — one full-build binary serves Qwen AND DSv4; the
  model-family kernel partition was deleted, releases key on SM tier only
  (`89ea8e7c4`). [win](docs/experience/wins/2026-07-11-unified-kernel-set-one-binary-qwen-and-dsv4.md).
- **Content-addressed prebuilt kernel bundle** — immutable source-addressed
  TileLang cubin bundle on the `kernel-artifacts` release; the zero-Python T1
  release lane fetches it instead of regenerating AOT (`6cb2c0054`).
- **Strategy-driven agent-OPD harness** — pluggable update strategy
  (`--update-strategy rejection-ce | sao-dis`) + dense partial-credit reward
  (fraction of fail-to-pass passing), off-policy DIS diagnostics.

### Changed

- **2026-07-12 — CUDA kernel `csrc/` reorg** (`a07a48d90`, `9fc53e7e4`,
  `051edb29b`). Exploded the 19-file `misc/` junk drawer into domain dirs (new
  `sampling/`·`norm/`·`recurrent/`·`elementwise/`; DSv4 MLA/DSA/MHC + FlashMLA/FA3
  shims → `attention/`; `kvcacheio/` → `kv/`) — every family now aligns 1:1 with
  its `src/ffi/*.rs` split. Deleted dead code (−6545 LOC): 3 Marlin W4/W4A8 GEMM
  `.cu` + `marlin_pf8/`, `kv/{paged_kv_append,scatter_kv}.cu`, and 5 `src/ffi/`
  extern decls (all 0-caller). `csrc/` now = 56 `.cu` in 10 kernel dirs.
  Byte-identical, bench-exempt (no runtime path changed).
  [win](docs/experience/wins/2026-07-12-kernel-csrc-reorg.md).

- **2026-07-11 — DSv4 decode-region KV reuse default ON** (`6230d9d3d`).
  Multi-turn concurrent throughput **+25%**; default flip after multi-shape
  verification.

### Verdicts

- **2026-07-11 — DSpark draft-KV: cap full-layer at per-request ceiling** (Qwen3.6-27B,
  CUDA, `1ee72d809`). The DFlash draft full-attention layer sized per-slot KV from the
  128K KV-pool floor (`max_seq_len`), not `max_total_tokens` — 512 MB/slot, clamping
  slots and blocking >4K prompts. Cap at `min(max_seq_len, max_total_tokens)`: lossless
  (scheduler admits nothing longer). Pod-verified: draft/slot **544→64 MB** at
  `--max-total-tokens 8192`, slots **32→256**, dspark tok-s/accept unchanged (2.49×/3.76×,
  above the P1 anchors), 13K prompt now fits one slot. P2.5 prefix-restore partial-ctx
  drafting was ALSO found already-implemented + verified holding (accept 0.18–0.22 on
  prefix-hit turns, 100% partial-ctx chains, no plain-decode fallback).
  [win](docs/experience/wins/2026-07-11-dspark-draft-kv-cap-per-request-ceiling.md).

- **2026-07-11 — DSpark/DFlash block-draft spec-decode: P1 LICENSED** (Qwen3.6-27B,
  CUDA). z-lab DFlash backbone drafter (`--spec-type dspark`) nets **2.39× decode
  tok/s short-ctx / 3.14× at ~3K** vs no-spec on H20 TP=1, B=1 greedy, no-prefix-hit
  — clears the 1.15× kill by a wide margin, above the 1.03× native-MTP ceiling that
  motivated adoption. block-16 verify overhead does NOT eat the win; accept-rate
  *rises* with ctx (0.199→0.228). Correctness PASS (needle + self-consistent).
  NOT a default flip: OPD rollout regime (91% prefix hit, 20–45K ctx) still gated on
  P2.5 (prefix-restore) + the 544 MB/slot draft-KV memory clamp.
  [win](docs/experience/wins/2026-07-11-dspark-p1-license-qwen36-27b.md).

- **2026-07-11 — DSv4 decode-region reuse: DEFAULT FLIPPED ON** (`--dsv4-decode-reuse`,
  was opt-in). Multi-turn concurrent A/B (token-preserving harness, the shape
  guidellm can't express) on H20 TP=4: aggregate throughput **+25.3% at c=16**,
  TTFT p50 halved (−52%), TPOT −18.7% — the win scales monotonically with
  concurrency. No single-shot regression (guidellm independent-prompt A/B is a
  byte-wash; finish-capture D2H ~free when reuse doesn't fire). ON-path
  correctness pod-verified across the campaign (crash-repro 24/24, needle-exact).
  Two binding shapes cleared → flip. The throughput lever was the reuse feature
  itself; the pinned-DRAM (#5) and admission-watermark (#6) knobs were KILLED
  (bad ROI / unsafe-no-cascade → #160).
  [wins](docs/experience/wins/2026-07-11-dsv4-decode-reuse-multiturn-concurrent-throughput.md)

- **2026-07-11 — Agent-OPD round −30.1% wall (H20 GPU1 3-arm A/B), quality-neutral
  (`894be29fa`)**: DSpark serial-B=1 decode LICENSED (rollout −29% / eval −30%,
  1.41×; engagement proven by net speedup + 78 `[dspark-draft]` lines; already
  default-on). Writeback grad-checkpoint offload now **seq-adaptive**
  (`writeback_offload_for_seq` = flag && seq_len≥4096) — short trajectories skip
  the host round-trip (backward −36%, writeback −33% at seq≈1276), long ones
  self-protect from the seq≥~9600 allocator OOM (errors/2026-06-28). Wins:
  [dspark-decode-and-seq-adaptive-offload](docs/experience/wins/2026-07-11-agent-opd-dspark-decode-and-seq-adaptive-offload.md).

- **2026-07-10 — DSv4 finish-write-through decode-region reuse: crash-fix gate
  PASS (opt-in `--dsv4-decode-reuse`), default flip pending perf**: v1
  (`79b5dbb17`) engaged (multi-turn match 640→704, +1 page into the decode
  region) but crashed the TP serve (`pool seq_len 494 != append_pos 485`) —
  the sub-page tail beyond `matched_len` has no radix content identity. v2
  (`28b8cd7bb`) added a continuation guard (reuse the tail only when
  `prompt[matched_len..finish_len] == entry.tail_tokens`). Pod re-verify TP=8:
  OFF 15/15 DET byte-identical; crash-repro 24/24 exact, zero
  `seq_len != append_pos`; multi-turn published 10 pages into the decode region,
  no over-restrict. OFF default byte-identical; flip needs a
  token-id-preserving perf harness.
  [wins](docs/experience/wins/2026-07-10-dsv4-finish-writethrough-decode-reuse.md)
  · [errors](docs/experience/errors/2026-07-10-dsv4-finish-writethrough-tail-content-identity.md)
- **2026-07-10 — DSpark-on-OPD default flip: quality-neutral LICENSED
  (opt-in), concurrency ≥4 DEFERRED**: final gate — pass-rate quality-neutral
  (n=16 dspark 9/16 ≥ plain 7/16, zero systematic per-task loss, CIs overlap;
  lossless-spec expectation confirmed). c=1 aggregate 1.9×; c≥4 unattributable
  under shared-box KV clamp (DFlash draft reserves 2560 MB/slot → co-tenant
  46 GB squeezes slots 256→6, OOM — not a dspark structural failure). No code
  default changed: dspark stays the licensed opt-in (`--dspark-draft-model`)
  until a clean-GPU c-sweep clears the concurrency leg.
  [wins](docs/experience/wins/2026-07-10-dspark-opd-default-flip-gate.md)
- **2026-07-10 — DSv4 Route A prefix reuse "identity formula fix": REVERTED
  (`4ad32362e`)**. The original claim below was wrong: the change never executed
  on DSv4-Flash (demand-paged skips it; its pod numbers licensed the
  copy-restore path) and broke V32/GLM band contiguity. Kept for the record:
  ~~`prepare_kv_batch` and `mirror_full_band` hardcoded `slot*lsp + i` instead of
  using engine-provided `slot_pages[i]`; 89.7% hit rate, 3.3× cold→hot~~.
- **2026-07-10 — Qwen FP8 small-M dense GEMM: DeepGEMM from M=2 LICENSED;
  M=1 GEMV variants KILLED**: measured crossover (DeepGEMM flat 47.5–57.8 µs
  in M vs ~linear GEMV) moves `QWEN_FP8_DEEPGEMM_DENSE_MIN_M` 16→2 — matched
  same-tree A/B +5–9% dspark greedy csv / +2–5% rust, needle ×3 exact DET on
  both lanes. M=1 stays on the GEMV: smem-x and x-in-registers variants both
  measured slower (attributed via the new `fp8_wread_probe`: the per-row x
  tail is the whole 1.78-vs-2.9 TB/s gap; achievable read BW is 3.5 TB/s,
  not the 4.0 spec).
  [wins](docs/experience/wins/2026-07-10-qwen-fp8-smallm-deepgemm-crossover.md)
- **2026-07-10 — DSv4 KV-reuse Phases 2b+3b SHIPPED** (#154): whole-slot
  park deleted (−869 LOC; preemption rides the 2a prefix-state pool;
  `--kv-oversubscription` on DSv4 now fails loud) and FlashMLA bands are
  demand-paged — the 16K slot cliff dissolves (**3 → 117 slots** at
  `--max-total-tokens 16384`, same-day paired A/B). Correctness lanes
  green (E1 15/15 ×2 arms, E2 10/10 @4.16× warm TTFT, restore→batched
  kill-test 25/30→30/30 post codex-R3 fix); E6 c=4 wall **+3.8%** miss
  documented with attribution (slots 0.9pp; zeroing/growth-storm ruled out
  by ablation; residual needs nsys). Wins:
  `docs/experience/wins/2026-07-10-dsv4-park-deletion-phase2b.md`,
  `docs/experience/wins/2026-07-10-dsv4-band-demand-paging-phase3b.md`.

- **2026-07-10 — DSpark on the OPD rollout serve: wall-clock POSITIVE**
  (first e2e A/B, CC-as-harness, 16 real swe_smith tasks): matched-task
  rollout wall **−25.1%**, 4.11 tok/step, partial-ctx engaged on 90% of
  chains, deep-ctx accept 3.46 > cold 2.08; pass-rate movement within
  single-sample noise (9/16 vs 6/16, ~1.1σ). Default flip still gated on:
  multi-sample pass-rate, `/v1/stats` accept export, and wiring
  `dspark_draft_model` into the in-process `train agent-opd` engine
  (train_cli.rs:2434 — serve-only today).
  [wins](docs/experience/wins/2026-07-10-opd-e2e-dspark-rollout-ab.md)
- **2026-07-10 — DSv4 prefix reuse RELICENSED (Phase 2a, content-keyed
  host-resident state pool)**: cross-request reuse relanded on the
  post-Route-A-deletion baseline — entries keyed by radix host-page identity
  (D1 unrepresentable), pool = L2 (zero HBM), L3 mmap spill unbudgeted. Pod
  evidence gate green: warm TTFT **4.19×** (0.768→0.184 s), resend 10/10
  after the derived-state fix (`0b5bd3d55` — the FP8 band is decode-lane
  DERIVED state, never captured/restored; rebuilt from restored bf16
  staging), L3 read-back exact, publish overhead −0.35% (free).
  [wins](docs/experience/wins/2026-07-10-dsv4-prefix-state-pool-phase2a.md).

- **2026-07-10 — DSpark sampled (temp>0) spec decode LICENSED**: device-side
  filter/chain-rejection kernels (`e22a41637`/`9f2dd5b3b`) take sampled spec
  from 34.8 (−7.5% vs plain) to **64–106 tok/s = 1.8–3.0× plain sampling**;
  determinism (cache-off same-seed byte-identical), needle 3/3, greedy lane
  regression-free. OPD 3-turn rollout shape: 62–77 tok/s sampled vs ~36 plain.
  Next walls: 16 per-step draft syncs (~36 ms sampled draft), greedy
  prefix-hit accept drop (3.11→1.92).
  [wins](docs/experience/wins/2026-07-10-dspark-sampled-device-path.md)
- **2026-07-10 — DSpark partial-ctx drafting (P2.5) LICENSED; sampling RNG
  cleared**: prefix-hit requests re-seed speculation (`8edde59c7`); multi-turn
  accept −11/−22% within band, 101–112 tok/s vs 42–44 plain; whole-restore
  −67% accept but 95 tok/s ≥ anchor, greedy byte-identical, needle 3/3 —
  sidecar fallback not needed. Same-seed-twice PASSES with prefix cache
  disabled → the 07-10 "determinism bug" was the lane/ctx confound, not RNG;
  determinism gates must control cache state. Env-sweep smoke Δ≈0%.
  [wins](docs/experience/wins/2026-07-10-dspark-partial-ctx-drafting.md)
- **2026-07-10 — DSpark trained heads NO-LICENSE (z-lab backbone stays);
  P2 sampling verify KILLED as-is**: FR Markov head +0.3–0.9 accept but
  draft 8.1→16.6 ms (per-row host loop) → ≤ z-lab tok/s; confidence
  truncation strictly harmful (conf=0 dominates); AEON block=11 −9% (12-row
  verify misses the B≥16 GEMM lane). Sampling lane: same-seed-twice FAILS
  (spec-path bug, plain lane passes) and host-side sampling lands −7.5% vs
  plain — fix determinism + device-side sampling before OPD rollout use.
  [wins](docs/experience/wins/2026-07-10-qwen36-dspark-dual-head-and-sampling-verdicts.md)
- **2026-07-10 — DSv4 Route A prefix reuse KILLED pending content-keyed
  redesign; warm-cache needle regression FIXED**: the Route A machinery
  (state pools, per-namespace tiers, restore path, host→FlashMLA page
  translation) deleted entirely (`bbaaea93b`, +67/−1553) after #154's
  bisection (origin `0198c3ba7`, amplifier page-sharing series) and a
  9-defect restore-path review; device page tables now refresh via a
  dirty-bit contract on every host-band change. Pod acceptance 6/6 (solo
  15/15 both cache states, concurrent 120/120, +193 MB pool budget
  reclaimed, park intact):
  [wins](docs/experience/wins/2026-07-10-dsv4-route-a-deletion-regression-fix-acceptance.md).

- **2026-07-10 — Qwen3.6 DSpark block draft LICENSED (short-ctx greedy)**:
  36.2 ms/step, 104–108 tok/s = 2.4× plain decode on H20 after quant-lane
  routing (row-serial GEMV → DeepGEMM/cuBLASLt at B≥16); needle ×3 +
  self-consistency PASS, plain-decode control unregressed. OPD-rollout claim
  still gated on long-ctx A/B + the prefix-restore draft-ctx gap.
  [wins](docs/experience/wins/2026-07-10-qwen36-dspark-block-draft-licensed-2p4x.md)

- **DSv4 decode-kernel levers #141/#142/#143 LICENSED (2026-07-04).** uint4-vectorized
  FP8 GEMVs + TILE-templated batch accumulator + warp-parallel mhc_params tail with a
  fused params|pre_rms_norm decode-graph pair. Matched binary-pair A/B (TP=4/EP=4,
  8×H20, same shell): decode TPOT 39.57→24.90 ms (−37.1%, MTP-off c=1) and
  31.27→20.94 ms/committed-tok (−33.0%, MTP-on 2015-in); needle 3/3 + count gates
  clean, MTP drift shown pre-existing via paired baseline control.
  [wins/gemv](docs/experience/wins/2026-07-03-dsv4-gemv-uint4-tile-template.md) ·
  [wins/mhc](docs/experience/wins/2026-07-03-dsv4-mhc-tail-parallel-fused.md)

- **Agent-OPD toy-corpus capability lane KILLED; harness + 12-round loop SHIPPED (2026-07-03).**
  Five measured escalations (surface cues, gold-module scenery, turn budgets)
  all left the untrained 27B at ceiling on synthetic small-repo bug-fix tasks
  (8/8 → 24/24 → 22/24→0/24 cliff) — classic single-line bugs are
  pattern-matched, and read→edit completes in 2 turns. What shipped: the full
  curve harness (corpus gen + self-check, `scripts/agent_opd_curve.sh`,
  plotter, held-out eval channel), the tape-footprint 3× margin fix (OOM at
  seq≈1350), sandbox `__pycache__` staleness fix, and a 12-round 27B run
  (loss 0.376→0.155, pass-rate ≥ baseline, zero OOM). Phase 2 =
  teacher-rescue on real SWE-Pro.
  ([kill](docs/experience/errors/2026-07-03-agent-opd-toy-corpus-saturation-kill.md) ·
  [run](docs/experience/wins/2026-07-03-agent-opd-27b-loop-stability-12rounds.md))

- **Phase 2 re-scoped; whole-step decode CUDA graph RE-KILLED (2026-06-21).**
  The B=1 chain-map/roofline shows the wall is foundation-bound (per-step
  `ctx.sync` + cross-process barrier; HBM ~2.8% util, 36× below roofline) —
  the graph lever measured −41%. MTP stays acceptance-gated opt-in
  (break-even ~57% accept; typical 50–53% is a wash); no universal spec-decode
  default. #70 closed.

### Train / OPD

- **OPD stack review-driven hardening (2026-07-06).** **Landed:** KL scale
  centralized behind `kl_batchmean_scale` + gradient regression test (guards the
  2026-06-16 LR-collapse); rollout arm → `--rollout-engine {infer,train}`
  (`ARLE_OPD_INFER_ROLLOUT` deleted); 490-line `gkd_anchor` split into phase
  helpers (490→251, zero behavior change); dead `rubric_writeback_ce_step`
  deleted; OPD-vs-RFT naming de-drifted (`agent-opd`/`rubric-opd` are RFT, not
  distillation). **Planned (pod-gated):** Metal OPD backend, real-SWE
  teacher-in-loop curve, overload-chain collapse.
  ([kl-guard](docs/experience/wins/2026-07-06-opd-kl-batchmean-scale-guard.md) ·
  [flags](docs/experience/wins/2026-07-06-opd-engine-knobs-cli-flags-pending-remote.md) ·
  [split](docs/experience/wins/2026-07-06-opd-gkd-anchor-phase-helpers-pending-remote.md) ·
  [dedrift](docs/experience/wins/2026-07-06-opd-rft-naming-dedrift-dead-code.md))

### CUDA

- **Qwen3.6 serves on CUDA (2026-06-29):** FP8 MoE via DeepGEMM; batched paged
  decode scales c=1→8 (Qwen3.6-27B-FP8, 1×H20: 21 → 26 tok/s aggregate).
  ([wins](docs/experience/wins/2026-06-29-cuda-qwen36-paged-batched-decode.md))
- **Qwen3.5-122B-A10B serves at TP4** via GQA KV-head replication;
  numerical-completion gate pending a clean re-run.
  ([wins](docs/experience/wins/2026-06-29-cuda-gqa-replication-122b-tp4.md))
- **GLM-5.2 (`glm_moe_dsa`, DSv4-DSA family) wired on the DSv4 path** —
  forward tranches landed, verification pending-remote; not
  production-verified. (wins `2026-06-19-glm52-*`)

### Metal

- **Qwen3.6 NextN/MTP spec decode shipped (2026-06-21)** on the canonical
  Metal model.
  ([wins](docs/experience/wins/2026-06-21-metal-qwen36-mtp-spec-decode.md))
- **VLM bring-up:** Gemma4 forward + image smoke landed (2026-06-15);
  DeepSeek-OCR wired (2026-06-24/25, vision numerics not yet faithful).
  Quality/throughput validation pending for both.

### Server

- **`/v1/chat/completions` now supports `stream=true`** (SSE
  `chat.completion.chunk` frames with `reasoning_content`/`content` deltas;
  closes the R5 tranche-2 deferral, #79). Multimodal chat streaming still
  fails closed with 400.
  ([wins](docs/experience/wins/2026-07-02-http-chat-sse-streaming.md))

### Repo

- **Renamed `agent-infer` → `arle`** across source, config, and docs
  (2026-06-29).

## [0.2.1] — 2026-06-15

> Consolidated section: tags `v0.1.5` (2026-05-02), `v0.2.0` and `v0.2.1`
> (both 2026-06-15) were cut without changelog sections. Everything below
> spans v0.1.4 → v0.2.1; per-tag artifacts live on GitHub Releases.

### Runtime rewrite — `infer-*` stack becomes the serving truth (2026-06-04)

- **Breaking:** the monolithic `infer` crate is deleted (`e81b98fb`,
  ~167k LOC). Serving stack: `infer-plan` → `infer-seam` → `infer-core` →
  `infer-cuda`/`infer-metal` → `infer-server`/`infer-api`; `infer-api`
  (`LoadedInferenceEngine`) is the single programmatic front door. Any command
  referencing `-p infer` is stale. Consolidated verification + performance
  verdict:
  final report.

### Training surface — OPD-only (2026-05-18)

- **Breaking:** scratch pretrain / SFT / GRPO / multi-turn RL surfaces are
  deleted; OPD is the only training axis.

### DSv4 perf campaign — adopt official kernels (2026-06-06 → 06-15)

- Official DSA indexer default-on: decode 124 ms → 26 ms flat @4096.
  ([wins](docs/experience/wins/2026-06-07-dsv4-official-dsa-default-on.md))
- FlashMLA `sparse_fwd` + FP8 DeepGEMM prefill default-on: 7.2 s → 3.48 s.
  ([wins](docs/experience/wins/2026-06-07-dsv4-prefill-official-kernels-default-on.md))
- Phase 0 debt closed 2026-06-10 (#56–#59). KV precision parity gate re-ported
  as correct-inference (needle ladder, not byte-identity); FlashMLA decode +
  fused-wqkv correctness LICENSED; pooled/contig-MoE default flip KILLED
  (−24%).
  ([lever verdicts](docs/experience/wins/2026-06-10-dsv4-lever-gate-license-or-kill.md))
- Seam-level KV-dtype dispatch `--kv-cache-dtype` (default bf16 unchanged);
  INT8/FP8 correctness LICENSED, opt-in pending a perf license (2026-06-12).
  ([wins](docs/experience/wins/2026-06-12-cuda-quant-kv-dispatch-int8-fp8.md))
- Phase 1 batched-lane keystone closed (#61 2026-06-11, #60 2026-06-15): DSv4
  B>1 decode takes the batched serving lane by default; residual c>1
  throughput lever is DP-attn (#89).

### OPD train (CUDA) — new beta surface

- **OPD mainline queue moved from experiment-only to operator-facing workflow.**
  `arle train opd --student-model <dir>` now runs the real HF-dir OPD path
  instead of the old pending stub, using the Qwen3.5 loader and `opd_step`
  directly. The 2026-05-24/25 queue also landed code-only chunked-logits KL
  parity, KV-tier observability counters, the default-off T2 coordinator
  wireframe, SFT-anchor corpus attribution, and a CPU-only capability-eval
  preflight for the P5 pure-OPD 5k adapter.
- **End-to-end OPD CUDA training stack landed on Qwen3-0.6B.** Single-session
  32-commit arc through kill-or-license-gated wins brings the OPD step at the
  moderate Qwen3.5-like shape to **48.5 ms** on RTX 4070 Ti SUPER —
  **1.71× faster than the like-for-like PyTorch CUDA reference (83 ms)** —
  and the real Qwen3-0.6B checkpoint OPD step to **0.164 s/step** (~170×
  over a naive scratch CPU baseline). CPU/CUDA loss bit-equivalent to
  relerr 1.276e-6. Convergence verified at lr=1e-7 with held-out
  exact-overlap **50 → 82.8 %** by step 5000 (KL/NLL still monotonically
  falling). Five parallel axes killed cleanly via SOLID gates with
  recorded errors entries (forward_last_logits, merge_grad sharing, SDPA
  mask-softmax fusion, high-level CUDA Graph rollout capture, SwiGLU
  silu+multiply fusion). New CUDA op surfaces: `matmul_bt` forward +
  backward, in-place AdamW, KV cache for OPD rollout, device-resident
  RoPE / argmax, fused causal-SDPA decode, fused attention-prepare
  layout, fused grad clip.

### Observability

- Added low-overhead HTTP `request_trace` JSON summaries for streaming and
  buffered requests, including TTFT, total latency, token throughput,
  KV/prefix-cache state, scheduler phase EMA, pipeline, and preprocess
  snapshots. Added `scripts/bench_dsv4_trace_http.py` to run DSv4 HTTP smoke
  cases and collect matching `request_trace` entries from server logs without
  enabling CUDA-synchronizing per-layer tracing.
- Fixed DSv4 distributed HTTP submissions so concurrent client requests keep
  the same logical queue order on every rank. `DistributedSchedulerGroup` now
  serializes cross-rank fanout submission, preventing rank 0 and follower ranks
  from entering different per-request token coordinators under concurrent
  traffic.
- Allowed DSv4 decode to run scheduler batches larger than one via the existing
  per-slot decode path. This keeps multi-slot distributed HTTP fanout alive
  while the vectorized DSv4 B>1 decode kernel work remains pending.
- Added DSv4 HTTP TP/EP axis overrides through the existing `INFER_TP_SIZE`
  / `ARLE_TP_SIZE` and `INFER_EP_SIZE` / `ARLE_EP_SIZE` env vars. The default
  remains the legacy overlapping TP=world, EP=world layout. The first 8xH20
  profiling pass confirms the current runnable DSv4 layout is decode
  communication-bound: default TP=8/EP=8 performs 86 all-reduces per generated
  token per rank, and nsys observed 22016 NCCL all-reduce kernels for a
  32-token decode window. Evidence and industry comparison are recorded in
  `docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`.
- Added committed DSv4 trace artifacts under
  `docs/trace-artifacts/2026-05-14-dsv4-decode/`,
  including the compressed raw nsys report/database, `nsys stats`, client JSON,
  server log, and SHA256 manifest. The trace record no longer depends on remote
  `/tmp` files.
- Added DSv4 DeepEP MoE trace artifacts under
  `docs/trace-artifacts/2026-05-14-dsv4-deepep/`,
  including compressed BF16 and FP8 combine trace logs, parsed summaries, remote
  build evidence, default trace-off post-checks, and the current bottleneck
  callout for return-side combine exchange plus local expert GEMMs.
- Added a current 8xH20 DSv4 single-token Nsight trace under
  `docs/trace-artifacts/2026-05-14-dsv4-deepep/nsys-one-token-current/`.
  The `max_tokens=2` streaming request returned `霓灯` and produced exactly one
  `step_decode_kernel_launch` wave across 8 ranks. The isolated token takes
  266.020 ms wall; decode-only nsys shows `cuStreamSynchronize`,
  async allocation/free, launch/memset churn, and NCCL send/recv ahead of the
  actual attention and GEMV kernels.
- Added a refreshed 2026-05-15 DSv4 single-token Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-one-token-current/`.
  With send/recv route and route-logits scratch reuse in place, the same
  one-token decode shape is now 158.439 ms wall. The remaining ranked costs are
  async allocation/free, launch/memset churn, D2H route readbacks, NCCL
  SendRecv/AllReduce, and local expert FP8/FP4 GEMV.
- Added 2026-05-15 DSv4 padded-dispatch Nsight records under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/`.
  The negative first trace (`nsys-single-token-padded-dispatch`) shows that
  padding without removing the dead send-count kernel regresses to 136.908 ms;
  the fixed trace (`nsys-single-token-padded-dispatch-skip-count`) validates the
  shipped B=1 decode path at 123.955 ms and records the remaining ranked costs.
- Added the DSv4 padded peer-combine Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-token-padded-peer-combine/`.
  The real 8xH20 run keeps the `霓彩` output and shows the single-token decode
  wave at 112.133 ms after pre-summing padded return rows per origin peer.
- Added the DSv4 fused dispatch payload Nsight trace and matching HTTP smoke
  under `docs/trace-artifacts/2026-05-15-dsv4-deepep/`.
  The real 8xH20 run keeps the `霓彩` output, cuts decode-window SendRecv
  launches from 1,032 to 688 by exchanging hidden rows and route metadata in
  one BF16 payload, and records a fresh isolated single-token decode wave at
  118.985 ms. The trace-off `decode64` smoke returns normal English content at
  12.22 post-first tok/s and the arithmetic case returns `410`; the nsys run
  makes clear that NCCL exchange/reduction, launch overhead, allocator churn,
  D2H, and local expert GEMV still dominate.
- Added the DSv4 route-grouped pair GEMV Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-route-pair-gemv/`.
  The opt-in `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1` run keeps the `霓彩` output
  and measures a 117.894 ms single-token decode wave. The decode-window top
  costs are now explicit: `ncclDevKernel_SendRecv` at 50.338 ms per rank
  range, FP4 route pair GEMV at 19.616 ms, FP4 route `w2` GEMV at 10.487 ms,
  FP8 GEMV at 9.408 ms, plus allocator/free and launch overhead.
- Added a fresh user-requested single-token `nsys` rerun under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-current-user/`.
  The real 8xH20 `/root/DeepSeek-V4-Flash` run returns exact arithmetic `406`
  and measures a 94.841 ms decode wave. The slow stack is reduce-scatter
  combine, local FP8/FP4 expert GEMV, residual all-reduce/send-recv,
  attention/MHC/route kernels, and high per-token launch/alloc/free/D2H
  runtime overhead, not sampler time.
- Added the matching DSv4 single-token `NCCL_PROTO=LL128` negative trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-nccl-ll128/`.
  The arithmetic request still returns `406`, but the isolated decode wave is
  94.936 ms versus 94.841 ms on the current default reference, and
  reduce-scatter combine is slightly worse at 21.371 ms per rank-range.
  Protocol selection alone is therefore not the next default decode fix.
- Added an opt-in DSv4 return-combine overlap experiment behind
  `ARLE_DSV4_COMBINE_OVERLAP=1`. The path creates a second EP NCCL
  communicator on a dedicated communication stream and delays routed-output
  consumption with an explicit CUDA fence so shared expert compute can overlap
  the reduce-scatter. The real 8xH20 run returns exact arithmetic `406`, but
  the trace regresses from 94.841 ms to 104.359 ms because all-reduce timing
  and cross-stream event overhead outweigh the reduce-scatter improvement.
  The matching default-off HTTP smoke still reaches 12.05 post-first tok/s,
  so the overlap experiment remains disabled by default.
- Added a fused DSv4 B=1 padded DeepEP local expert prepare kernel and matching
  trace records under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-small-local-pack-prepare/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-small-local-pack-prepare-smoke/`.
  The real 8xH20 run returns exact arithmetic `406`, cuts H2D runtime calls
  from 1,040 to 696, cuts `cuMemsetD8Async` calls from 1,232 to 544, and keeps
  trace-off `decode64` at 12.05 post-first tok/s. The single captured nsys wave
  is 92.602 ms due to noisier D2H/AllReduce timing, so this is recorded as
  small-call cleanup rather than a wall-time win.
- Reused per-layer DSv4 incremental attention projection buffers for `c_q`,
  `c_q_normed`, `q_raw`, `kv_raw`, and `kv_normed`. The real 8xH20
  single-token `nsys` run returns exact arithmetic `406` and moves the decode
  wave from 94.841 ms to 90.946 ms, while `cuMemAllocAsync` calls drop from
  6,760 to 5,040 and `cuMemFreeAsync` calls drop from 3,048 to 1,328 inside
  the decode range. The matching HTTP smoke keeps normal Chinese/English
  streaming output and exact math, with `decode64` at 11.89 post-first tok/s.
- Added a direct current-path single decode-token Nsight breakdown under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-current-breakdown/`.
  The real 8xH20 `/root/DeepSeek-V4-Flash` run returns exact arithmetic `406`
  and measures a 105.205 ms isolated second-token decode wave. The top stack is
  now explicit: 16,177 CUDA launches, reduce-scatter combine, local FP8/FP4
  expert GEMV, all-reduce, attention/MHC/route kernels, and 347 D2H calls for
  per-layer synchronization. The actual D2H activity payload is only 44,044
  bytes, confirming the current issue is MoE communication/compute plus
  launch/runtime synchronization granularity, not sampler time or copy bandwidth.
- Switched additional full-write DSv4 runtime scratch buffers from zeroed
  allocation to uninitialized allocation: expert/shared/grouped hidden scratch,
  route logits, per-layer hidden scratch, and MHC parameter scratch. The real
  8xH20 single-token `nsys` trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-expanded-uninit/`
  returns exact arithmetic `406`, moves the isolated decode wave from
  105.205 ms to 88.554 ms, and cuts `cuMemsetD8Async` from 3,640 calls /
  6.932 ms per rank range to 1,920 calls / 2.839 ms. The trace still points at
  reduce-scatter combine and local FP8/FP4 expert GEMV as the main bottlenecks.
  The matching HTTP smoke under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-expanded-uninit-smoke/`
  keeps normal Chinese/English multi-token output, exact math `410`, and
  `decode64` at 11.94 post-first tok/s.
- Extended the DSv4 uninitialized scratch cleanup to MoE dispatch, payload,
  recv/local-route, active grouped, and combine buffers. The real 8xH20
  single-token `nsys` artifact under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-moe-scratch-uninit-rerun/`
  returns exact arithmetic `406`, moves the isolated decode wave from
  88.554 ms to 87.667 ms after a rerun, and cuts `cuMemsetD8Async` from
  1,920 calls / 2.839 ms per rank range to 1,232 calls / 1.558 ms. The
  matching HTTP smoke under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-moe-scratch-uninit-smoke/`
  keeps normal Chinese/English multi-token output, exact math `410`, and
  `decode64` at 12.06 post-first tok/s.
- Moved DSv4 grouped expert weight/scale pointer tables into
  `DeepseekV4MoeBlock` load-time caches for the opt-in grouped/route-grouped
  expert paths and future raw-pointer DeepGEMM integration. On the real 8xH20
  `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1` trace, exact arithmetic remains `406`,
  H2D activity drops from 1,918 calls / 374,752 bytes to 440 calls / 7,808
  bytes, H2D runtime drops from 5.490 ms to 1.380 ms, and the route-grouped
  single-token wave moves from 105.808 ms to 94.828 ms. The path remains
  default-off because reduce-scatter combine and route-wise FP4/FP8 GEMV still
  dominate; the default DeepEP smoke still returns math `410`, normal Chinese
  writing, and normal English decode text.
- Added the DSv4 default-path warm decode Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-default-warm-decode/`.
  The run warms a real decode first, then profiles a second single decode token
  on 8xH20. The output remains `霓彩`, the decode wave is 128.130 ms, and the
  trace confirms allocator/free overhead is steady-state rather than only
  first-decode initialization: the decode window still records 8,453
  `cuMemAllocAsync` calls and 6,048 `cuMemFreeAsync` calls. The slow stack is
  NCCL SendRecv/AllReduce, local FP8/FP4 expert GEMV, launch/runtime overhead,
  allocator/free churn, and route-count D2H synchronization.
- Added the DSv4 expert-wise grouped GEMV negative Nsight trace under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-expert-grouped/`.
  With `ARLE_DSV4_GROUPED_EXPERTS=1`, the real 8xH20 run keeps the `霓彩`
  output but regresses the warmed single-token decode wave to 145.693 ms.
  The trace shows `ncclDevKernel_SendRecv` at 58.049 ms per rank range, FP4
  grouped gate/up GEMV at 23.162 ms, FP4 grouped `w2` GEMV at 11.428 ms, and
  elevated route-count D2H synchronization. This confirms the opt-in grouped
  GEMV path remains default-off and that the target remains true grouped
  GEMM/DeepGEMM with DeepEP overlap.
- Added the DSv4 route-grouped pair trace-off HTTP comparison under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-route-grouped-pair-vs-default/`.
  Default fused-dispatch decode keeps `decode64` at 11.47 completion tok/s and
  arithmetic at `410`; `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1` returns normal text
  and the same arithmetic answer but regresses `decode64` to 6.54 completion
  tok/s. Route-wise grouped GEMV remains default-off.
- Added DSv4 incremental stream scratch recycling and captured both the HTTP
  smoke and Nsight follow-up under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-stream-recycle/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-stream-recycle/`.
  The real 8xH20 run keeps normal text and arithmetic `410`; the isolated
  warmed decode wave improves from 128.130 ms to 111.798 ms, with
  `cuMemAllocAsync` dropping from 8,453 calls / 16.802 ms to 7,757 calls /
  12.574 ms and `cuMemFreeAsync` from 6,048 calls / 13.801 ms to 5,352 calls /
  11.096 ms. HTTP `decode64` stays effectively flat at 11.48 tok/s, so the
  main target remains NCCL plus local expert GEMV.
- Added DSv4 GPU compressor projection scratch reuse for `kv_raw` and
  `score_raw`, with trace artifacts under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-compressor-projection-scratch/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-compressor-projection-scratch/`.
  The real-output checks still pass (`decode64` normal text, arithmetic `410`)
  and alloc/free calls fall again (`cuMemAllocAsync` 7,757 -> 6,765,
  `cuMemFreeAsync` 5,352 -> 4,360), but HTTP `decode64` remains flat at
  11.47 tok/s and the single nsys wave is not a wall-time win because D2H/NCCL
  timing dominates this capture.
- Added DSv4 incremental attention scratch Nsight and HTTP artifacts under
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/bench-attention-scratch/`
  and
  `docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-attention-scratch/`.
  The real 8xH20 run returns normal multi-token output and arithmetic `410`;
  the isolated single-token decode wave is 97.042 ms after B=1 attention
  scratch cuts decode-window free calls from 4,360 to 3,048 without retaining
  prompt-sized prefill buffers. The trace directly answers the current
  bottleneck question: sampler is not in the top stack; NCCL SendRecv/AllReduce,
  D2H route-count synchronization, launch/runtime overhead, local expert
  FP8/FP4 GEMV, and attention/MHC kernels dominate.

### CUDA

- Added a default DSv4 B=1 padded BF16 combine reduce-scatter path behind
  `ARLE_DSV4_COMBINE_REDUCE_SCATTER` (default `1`). Expert ranks now sum padded
  route outputs into one row per origin peer and call NCCL `ReduceScatter`
  directly into the owner-rank output hidden row, with `0` preserving the prior
  grouped SendRecv combine. Real 8xH20 DSv4 validation against
  `/root/DeepSeek-V4-Flash` keeps normal Chinese/English streaming output and
  exact arithmetic `410`; `decode64` measures 12.05 post-first tok/s. The
  matching single-token nsys wave moves from 97.071 ms to 94.923 ms, replacing
  the old 23.163 ms SendRecv combine bucket with a 20.443 ms ReduceScatter
  bucket plus 3.259 ms residual SendRecv. This is a modest communication-shape
  cleanup; local expert grouped GEMM/DeepGEMM, DeepEP overlap, launch reduction,
  scratch reuse, and D2H readback removal remain the main performance targets.
- Reused per-layer DSv4 DeepEP dispatch scratch for route setup, rank count
  exchange buffers, packed send hidden rows/metadata, and local expert
  count/offset/cursor buffers. On the 8xH20 default path, trace-off math smoke
  reached 7.7-7.8 tok/s for 12 generated tokens, traced
  `ffn_deepep_dispatch_combine` p50 dropped to 1.552 ms, and the profiled
  `cuMemAllocAsync`/`cuMemFreeAsync` call count fell from 136,825 to 111,531 in
  the 8-token Nsight window. Remaining bottlenecks are still stream sync,
  return-side NCCL send/recv, and local expert GEMV/GEMM.
- Reused DSv4 DeepEP send-route token/slot buffers across decode steps and
  removed the unused `expert_token` output from `dsv4_pack_received_experts`.
  The 8xH20 trace-off math/writing smoke remained normal at 7.94-8.09
  completion tok/s, while the single-token nsys window reduced decode-only
  `cuMemAllocAsync` calls from 11,980 to 11,097 and `cuMemFreeAsync` calls from
  11,988 to 11,105. Remaining allocator pressure now sits in recv/local route
  buffers plus combine scratch and still needs a broader lifetime/graph pass.
- Reused DSv4 DeepEP B=1 decode recv/local route scratch for received hidden
  rows and metadata, local expert packed rows/weights/route slots, and
  route-output rows. Prefill preallocates only a small `ep_world * topk` decode
  capacity so long prompts do not retain prompt-sized route buffers. The real
  8xH20 DSv4 smoke stayed correct at 8.24-8.79 completion tok/s, and the
  single-token nsys window improved from 191.152 ms to 148.253 ms while
  reducing decode-only `cuMemAllocAsync`/`cuMemFreeAsync` calls to
  9,480/9,488 and `cuMemsetD8Async` calls to 10,554.
- Reused the DSv4 B=1 decode MoE route-logits buffer and preallocated its
  one-token scratch during prefill. This is an allocator-count cleanup rather
  than a confirmed wall-time win: the single-token nsys window reduced
  decode-only `cuMemAllocAsync`/`cuMemFreeAsync` calls again to 9,136/9,144 and
  `cuMemsetD8Async` calls to 10,210, while the captured wall time was noisy
  at 162.062 ms versus the prior 148.253 ms.
- Reran the post-scratch DSv4 single-token Nsight capture on 2026-05-15. The
  fresh one-token decode wave measured 158.439 ms wall and confirms the
  remaining cost center is not sampler or KV-cache lookup: runtime allocation,
  launch, memset, D2H routing readbacks, NCCL exchange/reduction, and per-expert
  GEMV still dominate before attention.
- Reused per-layer DSv4 shared expert scratch during DeepEP decode and added an
  in-place BF16 add kernel for accumulating shared expert output into the routed
  MoE output. Real 8xH20 smoke stayed correct at 9.07-9.50 completion tok/s,
  while the single-token nsys wave improved from 158.439 ms to 140.111 ms and
  decode-only `cuMemAllocAsync`/`cuMemFreeAsync` calls fell from 9,136/9,144 to
  7,416/7,424. The same step restored the CUDA `argmax_batch_readback_into`
  re-export required by Qwen3.5 CUDA builds. The scratch is gated to B=1 decode
  so long prefill does not retain prompt-sized shared expert buffers.
- Optimized the gated DSv4 grouped expert prototype behind
  `ARLE_DSV4_GROUPED_EXPERTS=1` by caching per-layer local expert weight
  pointer arrays and launching indexed active experts instead of rebuilding
  active pointer tables every step. The route remains opt-in: 8xH20 trace-off
  smoke improved grouped math latency to 2.37-2.40 s and short writing latency
  to 2.69 s, but traced `ffn_deepep_local_experts` p50 is still 1.196 ms versus
  roughly 0.46 ms on the default scratch-reuse path. The harness is ready for
  the next replacement with real grouped GEMM/DeepGEMM.
- Added a gated DSv4 grouped gate/up pair GEMV launch for the same
  `ARLE_DSV4_GROUPED_EXPERTS=1` harness. The FP8/FP4 pair kernels compute
  `w1` and `w3` in one grouped launch when format, shape, and block-scale
  layout match, otherwise the path falls back to separate grouped GEMV
  launches. 8xH20 nsys with `ARLE_DSV4_MOE_BACKEND=deepep` confirms
  `dsv4_fp4_grouped_gemv_pair_batch_kernel` runs in decode, but the grouped
  harness remains default-off: the decode window is still dominated by NCCL
  send/recv plus allocation/free and launch churn, not by the missing gate/up
  fusion alone.
- Added a gated DSv4 MoE combine exchange experiment via
  `ARLE_DSV4_COMBINE_DTYPE=fp8`. The path quantizes return-route BF16 rows to
  FP8 E4M3 with per-row FP32 scales, exchanges the FP8 payload through NCCL
  `Uint8` send/recv plus scale exchange, and dequantizes back to BF16 before
  the existing route-slot combine kernel. It is validated on 8xH20 but remains
  opt-in because the measured 1,039-token prefill trace is not faster than the
  BF16 combine default.
- Reused per-layer DSv4 HyperConnection/MHC temporary buffers in the
  incremental attention and FFN paths. The 8xH20 trace-off smoke set improved
  from roughly 5.5/5.6/6.0 tok/s to 6.3/6.2/7.3 tok/s for two math cases and
  one short writing case, while traced decode `attn_mhc` and `ffn_mhc` p50
  dropped to 0.088 ms and 0.085 ms respectively.
- Reused DSv4 incremental attention scratch for prepared Q/K, local attention
  output, and the `wo_a` latent projection, gated to B=1 decode so prefill does
  not retain prompt-sized buffers. The real 8xH20 HTTP smoke remains correct
  (`decode64` normal text, writing normal Chinese, arithmetic `410`), while the
  paired single-token Nsight capture reduces warmed decode `cuMemFreeAsync`
  calls from 4,360 to 3,048.
- Added a default DSv4 local expert segment-input path for DeepEP decode. When
  `w1` and `w3` are DSv4 block-scaled FP8/FP4 matrices, the per-expert fallback
  now runs their GEMV directly from the packed `expert_hidden` segment and
  skips the old D2D copy into `scratch.input`; unsupported formats still use
  the original copy fallback. Real 8xH20 nsys against `/root/DeepSeek-V4-Flash`
  kept the `霓虹` streaming output, reduced decode-only `cuMemcpyDtoDAsync_v2`
  from 871 calls / 1.795 ms per rank range to 613 calls / 1.240 ms, and moved
  the isolated single-token wave from 146.448 ms to 145.104 ms. The trace
  confirms this is a small cleanup: allocator/runtime churn, D2H route
  readback, NCCL SendRecv/AllReduce, and per-expert FP8/FP4 GEMV remain the
  dominant costs.
- Reused per-layer DSv4 incremental hidden scratch for attention/FFN
  HyperConnection pre-projection and RMSNorm temporaries. Real 8xH20 nsys
  against `/root/DeepSeek-V4-Flash` kept the streaming `霓虹` output and moved
  the isolated decode wave from 145.104 ms to 135.390 ms. Decode-only
  `cuMemAllocAsync`, `cuMemFreeAsync`, and `cuMemsetD8Async` calls each dropped
  by 1,376, matching four one-token temporary buffers across 43 layers and 8
  ranks. The remaining ranked costs are launch/runtime overhead, D2H route
  readback, NCCL SendRecv/AllReduce, and local expert FP8/FP4 GEMV.
- Removed the default DSv4 DeepEP AllGather route's redundant 32-byte
  `send_rank_counts` host readback. The AllGather count matrix is now
  collected before route packing and reused to derive both send and receive
  counts; the `ARLE_DSV4_COUNT_EXCHANGE=sendrecv` fallback keeps the previous
  readback. Real 8xH20 nsys kept the `霓虹` output, moved the single-token
  decode wave from 135.390 ms to 129.768 ms, and reduced decode-only D2H calls
  from 887 to 543. The remaining D2H cost is the 256-byte all-rank count matrix
  readback, ahead of deeper device-side count-prefix or countless dispatch
  work.
- Added the default DSv4 B=1 padded dispatch fast path for DeepEP decode. When
  the count exchange mode is the default AllGather route, decode now uses fixed
  `ep_world * topk` route slots, initializes unused slots as invalid, skips the
  unused send-rank zero/count kernel, and avoids the count AllGather plus its
  256-byte all-rank D2H readback. Set `ARLE_DSV4_PADDED_DISPATCH=0` to force
  exact-count dispatch. Real 8xH20 nsys kept the `霓彩` streaming output, moved
  the single-token decode wave from 129.768 ms to 123.955 ms, removed
  `ncclDevKernel_AllGather` from the decode window, and reduced decode-only D2H
  calls from 543 to 344. The remaining slow stack is NCCL SendRecv/AllReduce,
  launch/runtime and allocator/memset/free churn, local-count D2H, and local
  expert FP8/FP4 GEMV.
- Optimized the B=1 padded return-side combine exchange by summing valid padded
  route outputs into one BF16 row per origin peer on the expert rank before the
  return send/recv. This keeps the same `霓彩` streaming output, reduces
  returned combine rows by 8x, moves the real 8xH20 single-token decode wave
  from 123.955 ms to 112.133 ms, and drops `ncclDevKernel_SendRecv` time from
  25.211 ms to 23.329 ms per rank range. The local expert FP8/FP4 GEMV timings
  are unchanged, so true grouped GEMM/DeepGEMM remains the next compute target.
- Added and gated the default-path DSv4 single-expert `w1`/`w3` pair GEMV
  experiment behind `ARLE_DSV4_PAIR_EXPERT_GEMV=1`. The real 8xH20 trace kept
  the `霓彩` output and proved the new `dsv4_fp4_gemv_pair_batch_kernel` runs,
  but it regressed the local expert work on the current B=1 decode shape
  (`23.207 ms` per rank range for the pair kernel, 127.412 ms decode wave), so
  the shipped default remains the split GEMV path while the next compute target
  stays true grouped GEMM/DeepGEMM.
- Added and gated a DSv4 route-wise grouped expert experiment behind
  `ARLE_DSV4_ROUTE_GROUPED_EXPERTS=1`. It runs local experts directly from
  padded received route slots and removes the local-count D2H readback from the
  top decode runtime list, but the real 8xH20 nsys trace regressed to a
  145.669 ms single-token wave because `dsv4_fp4_route_gemv_batch_kernel`
  costs 35.895 ms per rank range. The path remains default-off and documents
  why the next compute step needs DeepGEMM-style grouped GEMM rather than
  route-wise GEMV.
- Added a clean 8xH20 decode-only HTTP comparison for the gated
  `ARLE_DSV4_PAIR_EXPERT_GEMV=1` path. The default split expert GEMV path
  reaches 11.79 post-first tok/s on `decode64`, while pair GEMV reaches
  7.70 tok/s; both return normal sequence text and the arithmetic check returns
  `410`. This keeps pair GEMV default-off and confirms the next compute target
  is real grouped GEMM/DeepGEMM rather than single-expert gate/up fusion.
- Added `HiddenStates::uninit` for CUDA call sites that immediately overwrite
  every element and switched DSv4 decode temporaries plus generic GEMM/add/SwiGLU
  outputs to use it where safe. Real 8xH20 DSv4 HTTP smoke remains correct
  (`decode64` reaches 11.99 post-first tok/s and the arithmetic check returns
  `410`), and single-token nsys shows `cuMemsetD8Async` dropping from 8,789
  calls / 11.855 ms per rank range to 2,957 calls / 4.180 ms. The isolated
  decode wave moves from 125.497 ms to 112.724 ms; NCCL exchange, launch
  overhead, async allocation/free, and local expert FP8/FP4 GEMV remain the top
  targets.
- Added the DSv4 B=1 fused dispatch payload experiment. Padded DeepEP decode
  appended route metadata as raw BF16 words behind each hidden row and exchanged
  hidden+metadata through one BF16 grouped send/recv instead of separate BF16
  hidden and I32 metadata exchanges. Real 8xH20 nsys kept the output correct,
  reduced SendRecv launches from 1,032 to 688, and recorded the isolated decode
  wave at 118.985 ms; NCCL SendRecv/AllReduce, launch/runtime overhead,
  allocator churn, D2H, and local expert FP8/FP4 GEMV remained the next targets.
- Optimized the gated route-wise grouped expert experiment by pairing its
  route-local `w1` and `w3` GEMV launches for matching DSv4 block-scaled FP8 or
  FP4 weights, falling back to split route GEMV when format or shape differs.
  The real 8xH20 nsys run lowers the prior route-grouped regression from
  145.669 ms to 117.894 ms, but it remains default-off because single-token
  decode is still dominated by NCCL SendRecv, route GEMV work, launch overhead,
  and async allocation/free. The main target remains true grouped
  GEMM/DeepGEMM with DeepEP overlap.
- **🎉 W4-hybrid prefill graph capture closes 4k/c=4 gap — Tier 1 STRONG
  PROCEED** (`a56b7a9`/`c44788f` 2026-05-10). Path B.2 bucketed prefill
  graph allocation key reduces capture key churn from 388 unique → **7
  unique** (98% reduction) with **98.5% LRU dominant key reuse rate**.
  Engine-side TTFT p50 **2000ms → 150ms = -92.5%** improvement on
  4k/c=4 prefill-dominant workload (server-side ground truth via
  `/v1/stats engine_ttft_us`; client-side guidellm 0.6.0 TTFT
  measurement separately broken per `e8d82b0` — bench tool bug, not
  substrate). Throughput **+632%** in matched-control 60s window
  (53 → 388 requests). Codex's "second-order bucketing" insight
  (captured scalar launch parameters use bucket capacity, not exact
  dim from first capture) was load-bearing for the win and added to
  skill v1.7.0 anti-pattern catalog. Followup: n=3 σ-tight re-bench +
  guidellm streaming fix. Evidence:
  `docs/experience/wins/2026-05-10-bench-40-pathB2-tier1-strong-proceed.md`.
- W4-hybrid Qwen3 paged-prefill **CUDA Graph capture** lands as opt-in
  via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1` (`35fc3cf`).
  Phase 1 functional gate: prefill-lifetime `MarlinPrefillScratch`
  lifecycle + multi-key 8-d graph cache (token / page layout / start_pos)
  + W4 graphsafe weight gating for dense BF16, W4A16 Marlin, W4A8 Marlin,
  and W4-hybrid. Default behavior unchanged when env vars unset.
  Throughput license deferred: scout bench A vs B (graph OFF baseline
  TTFT p50 1628.9 ms vs graph ON 1627.8 ms = Δ -0.07%) detected
  capture-key churn — Path A multi-key direction KILLED, Path B
  device-memory `start_pos` re-licensed P0 (`e462c53`). Evidence:
  `docs/experience/wins/2026-05-10-bench-p24-w4a8-prefill-graph-hoist.md`,
  `docs/experience/errors/2026-05-10-37-throughput-bench-killed-pathA-multikey-churn.md`.

### Long-context (cross-backend)

- **RoPE scaling support** (YARN / Linear / NtkAware) wired through
  `Qwen3Config::rope_scaling` and `Qwen35Config::rope_scaling` (Phase
  1+2 closed via 7 atomic commits + 51 unit tests). Helpers
  `compute_scaled_inv_freq` and `compute_attention_factor` ship in both
  spec crates. CUDA backend integration via
  `weight_loader::precompute_rope_with_scaling` (qwen3 path) +
  `precompute_rope_with_qwen35_scaling` thin shim. Vanilla path
  (`rope_scaling = None`) is bit-equivalent to the legacy
  `precompute_rope` formula (verified by
  `vanilla_inv_freq_matches_legacy_formula` test). Long-ctx bench
  validation (Qwen3-4B 64k YARN×2 / 128k YARN×4 + FP8 KV) deferred to
  Phase 3; CUDA-side viable on RTX 4070 Ti SUPER 16 GB.
  Apply to a model dir via [`scripts/setup_qwen3_yarn_config.py`](scripts/setup_qwen3_yarn_config.py).
  Consolidation:
  `docs/experience/wins/2026-05-10-m-rope-yarn-scaling-phase1-phase2-landed.md`.

### Structured-output (xgrammar)

- `crates/xgrammar-sys` Rust safe wrapper over upstream
  `mlc-ai/xgrammar` v0.1.34 lands as Phase 1 FFI scaffold (codex's #26).
  Default build is a stub that compiles without native sources or
  network; `--features real` builds a C++ shim against a pinned
  upstream checkout via `cc`. Wrapper surface:
  `GrammarCompiler` / `CompiledGrammar` / `GrammarMatcher` /
  `bitmask_size` / per-step bitmask fill APIs. No HTTP, scheduler,
  sampler, or GPU sampling integration yet — that is follow-up
  tranche work.

### Metal

- Qwen3.5-0.8B MLX 4bit single-request step-driver reaches 305.5 tok/s mean
  / 304.7 p50 on M4 Pro 20c for `1024/256`. The matched GGUF Q4_K_M
  exact default remains 202.1 tok/s direct for correctness, while the
  opt-in native-q4 load path reaches 236.7 tok/s direct / 239.8 tok/s
  step-driver, so current status surfaces no longer present the historical
  211.7 tok/s GGUF-only profile as the Metal SOTA headline. Evidence:
  `docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-mlx4bit-qknorm-default.md`.
  Native-q4 GGUF evidence:
  `docs/experience/wins/2026-04-28-bench-metal-qwen35-0p8b-gguf-native-q4.md`.


> Older releases (0.1.x — pre-rewrite): see [CHANGELOG-history.md](CHANGELOG-history.md)

### 2026-08-04 — default flip: DSpark train sidecar `learning_rate` 1e-4 → 1e-3

The online DSpark Markov head trained but could not be observed: the bias reached
~1e-3 while the serve adds it into bf16 draft logits whose half-ulp is ~0.03, so
`base + bias` returned `base` bit-for-bit and drafting never changed. Compounding
it, the cold-start `w1` used `0.02·sin(0.1·(i mod 1000))`, which aliases with
period `gcd(1000, rank)` — 125 distinct rows for a 248320 vocab, all in a ~4-dim
subspace, and `∂bias/∂w2 = w1[c]` made that the whole head's ceiling.

`w1` now uses a per-element hash; `update_markov_weights` logs
`rms|w1| rms|w2| est|bias|` against the bf16 floor on every publish.
See [errors/2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md](docs/experience/errors/2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md).
Serve-side measurement `pending-remote`.

### 2026-08-04 — removed: the DSpark online train sidecar

`--dspark-train`, `--dspark-train-out`, `--dspark-train-lr`,
`--dspark-train-batch`, `--dspark-train-iso` and `--dspark-prob-match-alpha` are
gone, with the trainer, the hot-path experience capture and the weight-publish
channel. `--dspark-markov-init` stays: installing a trained head is now the only
way a head reaches a serve.

The sidecar saw 120 training rows per optimizer step against DeepSpec's
1,835,008. That gap is architectural, not a data-rate knob — DSpark's 512x
amplification comes from a training-time attention mask over sampled anchors,
which a path that only observes what the serve actually drafted cannot have.
See [errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md](docs/experience/errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md).

New `spec-train` crate holds the artifact layer (Markov head I/O, ISO frames).
Kept out of `train`, which stays OPD-only: the two share only autograd.
