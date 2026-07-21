# sm_120 (Blackwell RTX PRO 6000) — ThinkingCap-Qwen3.6-27B-FP8 kernel compatibility map

> Status: Active — (A) correctness floor shipped; **(B) G2 SHIPPED 2026-07-22**
> (CUTLASS sm_120a grouped FP8 MoE GEMM: c=1 prefill TTFT 84.6 s → 760 ms ~111×,
> needle exact/DET; [wins](../experience/wins/2026-07-22-bench-sm120-fp8-moe-cutlass-grouped.md)).

Two deliverables, both planned:
- **(A) correctness floor** — route sm_120 to the portable fallback tier, pass the
  needle gate. Done: 1 code edit (§2). This is the interim/safety net.
- **(B) peak performance** — the goal. Add a native FP8 tensor-core GEMM route for
  Blackwell via **cuBLASLt FP8** (§6). The dequant→BF16 fallback (A) leaves ~2× FP8
  throughput on the table; (B) recovers it.

**Verdict.** The TC 27B FP8 path has **0** ops without a portable fallback, so (A)
is 1 edit. Peak (B) is **not** a per-kernel Blackwell rewrite and **not** hand-written
tcgen05 — sm_120 (GB202 workstation Blackwell) has no tcgen05; the peak path is the
vendored library (cuBLASLt FP8), wired as one more dispatch route. The existing
policy-driven dispatch (`quant_linear.rs:185`) and the cuBLASLt scaffold already in
`gemm/gemv.cu:539` make it an additive route, not a restructure.

**Critical path = the sm_120 bench loop, not the code.** There is no local sm_120;
the `.cu` cannot even compile here (no nvcc). Any peak claim is a hypothesis until
benched on the Colab RTX PRO 6000. Stand up Colab build+bench FIRST, then write,
then bench+gate needle, then decide the scale-format tier.

## 0. Model resolution — TC-27B is DENSE, not MoE

Ground truth: `crates/infer-vulkan/src/config.rs:9-10` — the on-box `Qwen3.6-27B`
checkpoint is `arch = qwen35` (**dense**); the MoE arch `qwen35moe` is the *35B-A3B*
variant. The loader hard-splits on `general.architecture` (`config.rs:74-77`):
`qwen35` ⇒ `num_experts = 0`. Per-layer type is by tensor presence (`config.rs:277`:
`ssm_conv1d.weight` ⇒ LinearAttention, `attn_q.weight` ⇒ FullAttention), so the 27B
is a **dense hybrid**: GDN linear-attention layers + periodic full-attention layers,
all FFNs dense SwiGLU.

Consequence: the MoE grouped-FP8-GEMM + host-router fallback paths (`qwen35.rs:~4184`,
`moe.rs:539`, `warm_fp8_deepgemm_grouped_prefill` at `qwen35.rs:2719`) are **never
entered** for TC — out of scope for (A).

**Pod-only unknown:** the FP8 *safetensors* `config.json` is not on this box (only
`models/Qwen3.5-0.8B`, `models/Qwen3-0.6B`). Confirm on the pod that the 27B FP8
checkpoint has `num_experts = 0` / no router tensors. The dense conclusion is from
the GGUF sibling + code, not the exact FP8 config.

## 1. Op inventory (TC dense FP8 hybrid: prefill + decode + rollout)

Method: every `csrc` file grepped for `wgmma / tcgen05 / mma.sync / cp.async.bulk /
__nv_fp8 / asm volatile / sm_90a / __CUDA_ARCH__`. Global result — **wgmma**: only a
comment (`gemm/quantized_gemv.cu:98`); **tcgen05**: none in-tree; **TMA**: only
`gemm/deepgemm_native.cu`; **mma.sync**: none; **sm_90a**: only
`attention/arle_fa3_shim.cu`; **__CUDA_ARCH__** guards: `kv/transfer.cu:30`,
`comm/custom_all_reduce.cuh:117/175`, `gemm/quantized_gemv.cu:1281` (`==700` Volta).

| # | op | file:line | Hopper-fast | portable fallback | HW asm? | sm_120 status |
|---|----|----|----|----|----|----|
| 1 | dense FP8 proj GEMM | dispatch `ops/quant_linear.rs:535` | DeepGEMM `deepgemm_native.cu:1641` (sm_90a) | dequant→cuBLAS `quant_linear.rs:441` + GEMV `quantized_gemv.cu:1349` | DeepGEMM=TMA/wgmma; fallback=`__nv_fp8` | **FIXED (was G1)** |
| 3 | full-attn prefill | `attention/nonpaged_prefill_attention.cu` (`qwen35.rs:6020`) | FA3 shim (`qwen35.rs:5960`) | this row | prefill: none; FA3: sm_90a | portable (FA3 auto-off) |
| 4 | full-attn decode | `nonpaged_prefill_attention_devpos` (`qwen35.rs:5945`) | FA3 split / FA2-sm70 | this row | none | portable |
| 8 | FA3 hopper fwd | `attention/arle_fa3_shim.cu` | FA3 (opt-in, sm_90a) | falls to #3/#4 | **sm_90a wgmma** | build-pinned sm_90a, runtime-off |
| 9 | FA2 sm70 | `attention/arle_fa2_sm70.cu` | — | falls to #3/#4 | none | gated off (major<8) |
| 11 | GDR prefill | `recurrent/gated_delta_rule.cu` `_prefill_recurrent` | FlashQLA (sm_90a) | this row | none | portable |
| 13 | FlashQLA chunked GDR | `recurrent/gdr_prefill_*.cu` + TileLang AOT | this (opt-in) | falls to #11 | TileLang sm_90a cubin | runtime-off (flag default false) |
| 20 | dense bf16 GEMM (MLP/o-proj) | `ops.rs:187/340` `gemm_cuda` | cuBLASLt | cuBLASLt | none | portable (cuBLAS Blackwell) |
| 24 | LoRA merge (rollout) | `qwen35.rs:8834` `lora_device_gemm`; `dequantize_fp8_block_scaled_to_bf16` (`:5206`) | cuBLAS + dequant | same | `__nv_fp8` | portable |

Plain portable (in-tree, no Hopper path, no HW asm, sm_120-native): KV prep (#2), batched decode (#5), paged resolve (#6), attn gate (#7), conv1d (#10), GDR decode (#12), RMSNorm/l2norm/input norm (#14/#15/#16), embedding (#17), SwiGLU (#18), residual (#19), RoPE (#21), sampling (#22), spec-decode (#23), KV transfer (#25) — see `csrc/` file:lines above.

**~25 kernel families (~30 FFI entry points).** Off-path but present (DSv4-only, not
TC): `decode_attention_{quantized,turboquant,varlen_fp8}.cu` (0 hits for
wgmma/tcgen05/sm_90a — portable anyway), `dsv4_*`, `moe/*` grouped, `marlin_*` (int4).

## 2. Gap list

**G1 (the only blocker — FIXED 2026-07-21). Dense FP8 GEMM SM-gate mis-selects
Blackwell.** `quant_linear.rs:171` was `major >= 9`; sm_120 has `major = 12` → gate
returned true → `dsv4_deepgemm_fp8_gemm_nt` launched → `deepgemm_native.cu:1641`
`if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED` → `.result()?` hard-aborts the
run. Same defect flows through the warm path (`qwen35.rs:2662`). This is **not** a
missing fallback — the portable dequant→cuBLAS + scalar block-scaled GEMV path
already exists; the gate simply failed to pick it on `major != 9`.
Fix: `major >= 9` → `major == 9` (DeepGEMM is Hopper-exclusive; `deepgemm_native.cu`
checks `== 9` at every entry: 1155/1505/1574/1641/1703). Zero-code interim:
`--qwen35-deepgemm false` (`args.rs:818`).

Every other Hopper-only op is build-pinned to sm_90a (dormant) or runtime-gated
off, each with a wired portable fallback: FA3 `qwen35.rs:782` `== (9,0)`;
FlashQLA `runtime_flags.rs:126` default false; FA2-sm70 `qwen35.rs:831` `major < 8`.

## 3. Build story for sm_120 (T2 opt-in)

- Enable: `TORCH_CUDA_ARCH_LIST="12.0"` (or `"9.0;12.0"` for a fat binary).
  `build.rs:12` lists `120` in `T2_SMS`; `validate_sm`/`parse_sm_token` accept `12.0`→`120`.
- Compiles for sm_120: every `.cu` gets `arch=compute_120,code=sm_120`
  (`build.rs:2825` else-arm) — all TC ops in §1.
- Stays OFF sm_120 (no assemble failure): `build.rs:2816-2823` pins FA3/FlashMLA/shims
  (`is_sm90a_only`) to `compute_90a,sm_90a` regardless of the arch list, so the wgmma
  asm never targets sm_120. FA3/FlashMLA also opt-in (`ARLE_CUDA_ENABLE_FA3`); default
  build swaps stubs.
- `deepgemm_native.cu`: gets the sm_120 gencode but is a host-side JIT harness
  (device kernels are CUTLASS strings JIT'd at runtime for sm_90a); the TU builds for
  sm_120 as it already does for T1 80/86/89. Opt-in via `enable_deepgemm_native`.
- FP8 intrinsics: the dequant fallback uses `__nv_fp8_e4m3`/`__nv_fp8x4_e4m3`
  (`quantized_gemv.cu:50,150,354,619`), native on sm_89+/Blackwell. The only
  arch-conditional (`__CUDA_ARCH__ == 700`, `:1281`) is Volta-only; sm_120 takes the
  portable `#else` (`:1349`).
- No sm_90a-only asm reaches the sm_120 gencode: only `arle_fa3_shim.cu` is sm_90a
  (pinned away); `kv/transfer.cu` asm is generic `ld/st.global` PTX;
  `custom_all_reduce.cuh` is guarded `>= 800`/`>= 700` and retained-out on single-GPU.

## 4. Phased (A) plan

1. **Build** `crates/cuda-kernels` with `TORCH_CUDA_ARCH_LIST="12.0"`; confirm
   FA3/FlashMLA stayed sm_90a-pinned (`build.rs:2816`).
2. **Fix the dispatch defect** — `quant_linear.rs:171` `>= 9` → `== 9` (**DONE
   2026-07-21**). Interim alternative: `--qwen35-deepgemm false`.
3. **Existing runtime gates already exclude sm_120 — no edits**: FA3 `== (9,0)`
   (`qwen35.rs:782`), FlashQLA default false (`runtime_flags.rs:126`), FA2-sm70
   `major < 8` (`qwen35.rs:831`).
4. **Gate**: serve the 27B FP8 checkpoint on the sm_120 box and run
   `scripts/needle_gate.py` across `115..8000` (spanning the 241 boundary); end state
   = exact needle recall + `deterministic? true` per length.

**4 steps, 1 code edit (landed); the rest are build + verify-existing-gate.**

## 4b. sm_120 first-light — VERIFIED 2026-07-22 (Colab RTX PRO 6000)

Real-hardware results on NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB, CUDA 12.8):
- **Build ✅** — `TORCH_CUDA_ARCH_LIST=12.0 cargo build --release --features cuda`
  succeeds (6m19s, Rust cached). **TileLang emitted sm_120 codegen with the DEFAULT
  arch — no JIT hang (#2328 did NOT manifest at build), no error**, ~45 kernels. The
  §5 load-bearing assumption is confirmed positive; `ARLE_TILELANG_CUDA_ARCH=90` not
  needed for the build.
- **G1 fix ✅ real-hardware** — serve.log: `Qwen FP8 dense DeepGEMM SM-gated OFF on
  sm_120 … using dequant→BF16 GEMM / scalar GEMV fallback`. **No CUDA_ERROR_NOT_SUPPORTED
  abort.** The `major == 9` gate routes sm_120 to the portable path correctly.
- Model loads (49.9 GB, BF16 KV pool 21104 pages, L2 DRAM tier). `/v1/models` → 200.
- Model is `Qwen/Qwen3.6-27B-FP8` = **MoE-VLM** (`Qwen3_5ForConditionalGeneration`,
  64 layers, per-layer `shared_expert_gate`, MTP, `out_hidden_size`) — corrects §0's
  dense inference.

**The "serve hang" DISSOLVED — direct measurement falsified it (DP2, 2026-07-22, clean
main 9dbcb54).** The prior report (GPU 0%, 1077/1080 threads in `futex_wait`, serve on
`coordinator_local_router` with 1024 `arle-relay-worker` threads) **does NOT reproduce
on current main.** Single-GPU serve routes through `serve_multiproc.rs:112 world_size
<= 1 → serving single-process (no workers)` — the engine runs **in-process, bypassing
the coordinator / `lockstep_loop` / relay-driver chain entirely**; that chain is the
TP>1 multiproc path, structurally unreachable on 1 GPU. The prior observation came from
a non-`world=1` code/config state, not main. **The full FP8 sm_120 loop is end-to-end
healthy on current main** — no blocker gates the peak-perf work.

**Loop VERIFIED end-to-end (DP1 + DP2, two Colab RTX PRO 6000 sessions):**
- **DP1** — dense **bf16** Qwen3-0.6B, single-process serve + `bench_throughput.py`
  clean: c=1 **315 tok/s output, ITL p50 3.16 ms, 12/12 complete**. Runtime + serve +
  canonical bench harness all work.
- **DP2a offline** — **`Qwen3.6-35B-A3B-FP8` MoE decodes coherently** on sm_120,
  **~11 tok/s** (`arle run`, in-process). G1 line confirmed verbatim
  (`quant_linear.rs:176`). FP8 MoE kernels are **correct** on sm_120.
- **DP2b serve** — same MoE, `arle serve` healthy: non-stream 0.3 s, streaming 26
  chunks, chat 26 chunks, **4 concurrent all `finish_reason=length`**, zero stall.
- **Current FP8 path runs on FALLBACKS = the peak headroom.** Dense FP8 → dequant→BF16
  (G1). **MoE grouped FP8 → hand-grouped kernels** (DeepGEMM native bridge
  `CUDA_ERROR_NOT_SUPPORTED`, `loader.rs:1391` — Hopper-only, never sm_120). The 11
  tok/s offline baseline is on these fallbacks; Cut-1/Cut-2 (dense) + G2 (MoE grouped
  CUTLASS block-scaled) are exactly what replaces them for peak.

**Model-support facts (source-verified) that pin the vehicle to MoE:**
- **No dense FP8 CUDA vehicle exists.** `Qwen3ForCausalLM` FP8 is REJECTED at load — the
  `Qwen3Dense` executor is `from_qwen3_bf16_safetensors` (`loaded.rs:2130`), bf16-only.
  The FP8 dense `gemm_batch`/G1/Cut path is reachable ONLY via the Qwen35 family.
- **Vanilla Qwen3-MoE also rejected** (`Qwen3MoeForCausalLM` → `Qwen3MoeUnsupported`,
  `loaded.rs:2055`). The peak-work vehicle is **`Qwen3.6-35B-A3B-FP8`** (or TC-27B) —
  MoE, and (now proven) it serves + benches fine on a single GPU.

**Next: baseline bench then peak kernels.** Run the canonical `bench_throughput.py`
1/4/8/16 grid on `Qwen3.6-35B-A3B-FP8` (the champion row Cut-1/Cut-2/G2 must beat), then
implement per §6. Warm Colab VM `moehang` (binary + model on disk) can serve the
baseline before it idle-reclaims.

## 5. Colab RTX PRO 6000 build+bench loop (critical-path prerequisite)

Peak-perf work is unverifiable without an sm_120 box. Stand this up FIRST; it gates
every (B) claim.

- Target: RTX PRO 6000 Blackwell (sm_120, GB202, 96 GB) via Colab CLI. 96 GB fits the
  27B FP8 checkpoint (~29 GB) with room for KV.
- Build: sync the tree, `TORCH_CUDA_ARCH_LIST="12.0" cargo build --release --features
  cuda` — confirm the sm_120 gencode compiles the portable tier (§3) and FA3/FlashMLA
  stayed sm_90a-pinned (stubbed off by default).
- Serve+bench: `arle serve --backend cuda --model-path <27B-FP8>` +
  `scripts/needle_gate.py` (correctness) + `scripts/bench_throughput.py` (perf, vs the
  (A) dequant-BF16 arm as the baseline the (B) route must beat).
- Loop: write → sync → build-on-Colab → needle+bench → iterate. Every (B) row in
  wins/ cites a Colab sm_120 bench, never a local number.
- Open: confirm the FP8 safetensors `config.json` has `num_experts=0` (§0 pod caveat)
  on this box while it's up.

## 6. (B) peak-perf plan — CUTLASS sm_120 block-scaled GEMM

**Priority — MEASURED, three corrections converged (baseline bench + nsys, 2026-07-22,
[wins](../experience/wins/2026-07-22-bench-sm120-fp8-moe-baseline.md)). The ONE target is
G2 = the MoE grouped GEMM. Cut-2 (dense) is ~0.04% — skip it.**
- The bottleneck is **PREFILL** (cold ~85 s / 3013 tok = ~35 tok/s; decode ITL ~11 ms is
  already healthy; c=16 collapses on prefill starvation).
- nsys prefill breakdown: **~99.5% is the MoE grouped GEMM** —
  `fp8_f32_block_grouped_gemv_batch` + `_pair_batch` = 91.66 s of 92 s. **Dense projections
  (Cut-2 target) = ~0.04%** (already bf16 CUTLASS GEMM). Linear-attn ~0.4%, attn <0.001%.
- **Why so slow:** the sm_120 MoE path fell to the **hand-grouped fallback** because
  DeepGEMM native won't compile for sm_120 (`loader.rs:1393`, Hopper-only). That fallback is
  **GEMV-shaped** (per-token batched GEMV, no tensor cores, no M-batching across a group's
  tokens) — pathological at prefill M=3013.
- **G2 = replace the hand-grouped GEMV with the CUTLASS sm_120 grouped blockwise-scaling
  collective** (real FP8 tensor-core grouped GEMM). Est. 10–50× on the MoE prefill (GEMV →
  tensor-core grouped GEMM) → prefill ~35 tok/s could reach the 100s–1000s tok/s range.
- **Cut-2 (dense) is dropped as a shippable kernel** — 0.04% headroom. Any collective
  build/FFI/dispatch de-risk it would have provided is folded into G2's first compile.

**Gates RESOLVED on the VM (no blockers to G2):**
- **CUTLASS 4.3.5** ships in-repo at `crates/cuda-kernels/vendor/flashmla/csrc/cutlass`
  (build.rs:2576) — **has the sm_120 grouped collectives**: `sm120_blockscaled_mma_array_tma.hpp`
  + `sm120_mma_array_tma_blockwise_scaling.hpp` + `sm120_blockscaled_mma_builder.inl`.
  **No CUTLASS bump.** Point the new `.cu`'s include at that tree.
- **Scale format = 128×128 block, BF16 `weight_scale_inv`** (`weight_block_size=[128,128]`,
  `fmt=e4m3`, `activation_scheme=dynamic`) — this is the DeepSeek-style **blockwise-scaling**
  (float scale) collective family, NOT MXFP8/e8m0. Load-time work = **BF16→FP32 scale
  widen only, no UE8M0 repack**.

### G2 — CUTLASS sm_120 grouped blockwise-scaling MoE GEMM (the one deliverable)

Replace the hand-grouped GEMV fallback (`quantized_gemv.cu:2865/2898`
`fp8_f32_block_grouped_gemv_batch` / `_pair_batch`) with a real FP8 tensor-core grouped
GEMM built on the vendored **CUTLASS 4.3.5** sm_120 array/grouped blockwise-scaling
collective. This is the sm_120 replacement for the Hopper-only DeepGEMM
`m_grouped_fp8_gemm_nt_contiguous` (which never compiles for sm_120).

Adopt-vendored-first: the collective already exists in-tree —
`vendor/flashmla/csrc/cutlass/include/cutlass/gemm/collective/sm120_mma_array_tma_blockwise_scaling.hpp`
(+ `sm120_blockscaled_mma_array_tma.hpp`, builder `sm120_blockscaled_mma_builder.inl`).
Instantiate it; do not hand-write MMA.

**file:line**
1. `crates/cuda-kernels/csrc/gemm/fp8_moe_grouped_cutlass_sm120.cu` (new): instantiate the
   sm_120a grouped blockwise-scaling collective. Signature mirrors
   `cuda_moe::dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous` (called at `moe.rs:1386/1434`;
   reference impl `deepgemm_native.cu`) so it's a drop-in on the SAME grouped buffers:
   E4M3 grouped weights (w13/down), E4M3 activations + **dynamic per-token** act scales
   (`activation_scheme=dynamic` — reuse the existing FP8-activation quant the DeepGEMM path
   already feeds), **128×128 weight scales widened BF16→FP32**, per-expert group offsets /
   problem sizes (`m_grouped` contiguous layout).
2. `crates/cuda-kernels/src/*.rs` (FFI) + `crates/infer-cuda/src/moe.rs` `mod cuda_moe`:
   extern + safe wrapper, mirroring the `dsv4_deepgemm_m_grouped_*` binding.
3. **Loader** (`loader.rs:~1391`): on sm_120, the DeepGEMM disable currently CLEARS the
   grouped-B caches (`w13_fp8_grouped`/`down_fp8_grouped`) → `has_deepgemm_grouped=false`
   (`moe.rs:537`) → hand-grouped GEMV. **Keep the grouped caches built on sm_120** (the
   contiguous grouped memory layout is identical; only the GEMM callee changes) so the
   grouped-contiguous route is taken.
4. **Dispatch** (`moe.rs:539` `use_deepgemm`): add an sm_120 arm — when `major==12` &&
   grouped caches present, route the expert GEMM to the CUTLASS grouped call instead of
   `dsv4_deepgemm_m_grouped_*` (Hopper) and instead of the hand GEMV. Keep the existing
   `QWEN35_DEEPGEMM_MIN_ROUTES` hybrid crossover (hand kernels still win tiny decode bands;
   the CUTLASS grouped path is the prefill / large-R win).
5. `build.rs`: collect the new `.cu` with **sm_120a** gencode (`compute_120a`), include path
   = the flashmla cutlass tree (build.rs:2576). Guard so non-sm_120 builds skip it.

**Scale format (resolved):** 128×128, BF16 `weight_scale_inv`, `fmt=e4m3` — the
DeepSeek-style **blockwise-scaling** (float-scale) collective family, NOT MXFP8/e8m0. Only
a BF16→FP32 widen at load; no UE8M0 repack.

### G2 de-risk — collective VERIFIED 2026-07-22 (RTX PRO 6000, standalone)
The sm_120a grouped blockwise-scaling FP8 GEMM **compiles and is bit-exact** vs an FP32
reference (`max_rel_err=0`, ragged M∈[100,500] + heavy M=3000 groups). Source =
`scratchpad/fp8_moe_grouped_cutlass_sm120.cu`, copied from CUTLASS 4.3.5 example
**`87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu`** (the only in-tree sm_120a
grouped blockwise instantiation). **The hard part (does the collective instantiate on
sm_120a) is done.** Load-bearing facts for the production wiring:
- **Type stack:** `arch::Sm120` `OpClassTensorOp`; `MmaTileShape=Shape<_128,_128,_128>`,
  `ClusterShape=Shape<_1,_1,_1>` (RTX Blackwell = no multicast); `Sm120BlockwiseScaleConfig<
  M=1,N=128,K=128>` (= DeepSeek per-token-A / 128×128-B); `KernelScheduleSm120Blockwise` +
  `EpilogueScheduleAuto`; `GroupProblemShape` + `GemmUniversalMode::kGrouped`, TileScheduler
  `void`. A RowMajor `[M,K]`, **B ColumnMajor `[N,K]` (NT)**, C/D RowMajor bf16. SF element
  = f32. **N%128==0 && K%128==0** required (Qwen3.6 N=768,K=2048 comply).
- **Build flags (add for this `.cu`):** `--expt-relaxed-constexpr` (collective's device
  `std::min`) + **link `-lcuda`** (device TMA setup calls `cuDriverGetVersion`). Both are
  build-flag, not template constraints.
- **API is array-of-pointers, not DeepGEMM single-contiguous.** Map `ptr_A[g]=base+
  row_offset[g]*K` (B/SFA/SFB likewise) — trivial. Build per-group `LayoutSFA/SFB` via
  `ScaleConfig::tile_atom_to_shape_SFA/SFB(make_shape(M,N,K,1))` host-side, upload arrays.
- **REMAINING integration risk = the ONE thing to validate on the real model:** the
  checkpoint's `weight_scale_inv` memory layout vs CUTLASS's expected **SFB layout**
  (`tile_atom_to_shape_SFB`), and the DeepGEMM activation-quant's **SFA layout** vs
  CUTLASS's per-token expectation. Fake-data standalone can't prove this — the
  `needle_gate` on the real checkpoint does. If layouts differ, add a host/load-time
  repack of the scale tensors into CUTLASS's SF layout (cheap, load-time only).

### verify
Colab sm_120. **Correct-inference gate** (`needle_gate.py` + self-consistency: the CUTLASS
grouped output's own autoregressive generation is the reference, vs the hand-grouped
fallback envelope — MoE non-determinism forbids byte-identity). THEN `bench_throughput.py`
1/4/8/16 vs the [baseline](../experience/wins/2026-07-22-bench-sm120-fp8-moe-baseline.md)
(prefill ~35 tok/s → target multiples; the GEMV→tensor-core-grouped change is the win).
A stable positive prefill Δ ships; wins/ entry cites the Colab bench. Floor: the hand
grouped GEMV stays the always-correct fallback for shapes/SM the CUTLASS route rejects.

### Blackwell attention (deferred, second hotspot)
GEMM dominates FLOPs; land the GEMM route first. Research verdict: **no Blackwell-native
FMHA for hd256** — FA3 refuses Blackwell, FA4 is hd≤128, CUTLASS-ex77/trtllm-gen are
sm_100. The peak is **FA2-class** (our TileLang `batch_prefill_paged_hd256`, GemmMMASm70
MMA, forward-compiled to sm_120 via `ARLE_TILELANG_CUDA_ARCH=90`); `nonpaged_prefill_attention`
(§1 #3/#4) is the SIMT floor. TileLang sm_120 has open bugs (#2328 hang / #2703 non-det)
— gate on needle determinism.

### Strategic (long-term) — NVFP4
NVFP4 (E2M1 + 16-element UE4M3 scale) is the ecosystem-consensus **Blackwell-native**
format, the only un-throttled ~4× path that also fits 96 GB comfortably. For a
workstation-Blackwell *deployment* target, a 4-bit NVFP4 requant may beat FP8 —
evaluate as a follow-up after the FP8 route is proven, not the first cut.
