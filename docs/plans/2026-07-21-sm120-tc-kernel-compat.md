# sm_120 (Blackwell RTX PRO 6000) — ThinkingCap-Qwen3.6-27B-FP8 kernel compatibility map

> Status: Active

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

Grounded by the 4-stream deep research ([2026-07-22-sm120-fp8-peak-landscape.md](../research/2026-07-22-sm120-fp8-peak-landscape.md)).
Correction to the first draft: the peak path is **NOT cuBLASLt** — cuBLASLt on sm_120
is per-tensor only and runs at the *throttled* legacy-MMA rate (2×). The framework
consensus (vLLM PR #22131, CUTLASS example-79) is the **CUTLASS sm_120 block-scaled
collective** (`mma.sync…block_scale`), the only route to un-throttled ~4× over BF16.

### Two cuts, evidence-ordered
- **Cut 1 (prove the route, mature, ~2×)**: cuBLASLt **per-tensor** scaled FP8
  (`CUDA_R_8F_E4M3` + `SCALAR_32F`), reusing the `gemv.cu:539` cuBLASLt scaffold.
  Validates FFI + `major==12` dispatch + sm_120 build end-to-end. Numerics change
  (per-tensor vs 128-block) — a prove-route, not the endpoint. cuBLASLt sm_120 needs
  the **TN layout** (no non-TN MXFP8 today).
- **Cut 2 (true peak, ~4×)**: adopt the **CUTLASS sm_120 block-scaled collective**
  (integrate CUTLASS example-79c MXFP8 / the vLLM PR #22131 kernel as a new `.cu`),
  consuming 128-block E4M3 with **UE8M0** scales. This is the adopt-vendored-first
  path — not hand-written.
- Floor: (A) dequant→BF16 stays the always-correct fallback for any shape/SM the FP8
  route rejects.

### Scale-format action item (blocks Cut 2)
Confirm the checkpoint's 128-block scales are **UE8M0** (power-of-2) — then CUTLASS
sm_120 takes them directly (vLLM `block_shape=[128,128]`). If arbitrary f32, a
load-time requant to UE8M0 is required (a checkpoint with `scale_fmt ≠ ue8m0` crashes
the block-scaled loader — [vLLM #47436](https://github.com/vllm-project/vllm/issues/47436)).
DeepGEMM's own SM100 path already repacks to UE8M0, so the format likely exists.

### file:line
1. `crates/cuda-kernels/csrc/gemm/fp8_cutlass_sm120_gemm.cu` (new, Cut 2): adopt the
   CUTLASS example-79c sm_120a block-scaled collective; inputs E4M3 + UE8M0 128-block
   scales. Cut 1 interim: `fp8_cublaslt_gemm.cu` per-tensor on the `gemv.cu:539` scaffold.
2. `crates/cuda-kernels/src/gemm.rs`: extern + safe wrapper.
3. `crates/infer-cuda/src/ops/generated/qwen_fp8_dense_projection.rs`: add
   `Route::CutlassSm120Fp8` (and the Cut-1 cuBLASLt variant). `@generated` by
   `scripts/reduce_operator_evidence.py` — add via the generator or a hand-written
   override over `select_exact`, NOT by editing the generated file.
4. `crates/infer-cuda/src/ops/quant_linear.rs:185 qwen_fp8_dense_route`: `major==12` →
   the sm_120 route; `major==9` stays `PackDeepGemm`; other SMs unchanged. Dispatch at
   `~:535`.
5. **MoE grouped (G2)**: the sm_120 MoE path is the CUTLASS block-scaled **grouped**
   GEMM (CUTLASS ex79d / vLLM), NOT DeepGEMM grouped. `moe.rs:537` gates on cache
   presence, not SM — confirm sm_120 routes away from `dsv4_deepgemm_m_grouped_*`
   (empirically via Pass B, then wire the CUTLASS grouped route).
6. `build.rs`: new `.cu` auto-collected; needs CUTLASS sm_120a includes for Cut 2.

### verify
Colab sm_120: `needle_gate.py` per cut, THEN `bench_throughput.py` vs the (A)
dequant-BF16 arm AND vs Cut 1 — Cut 2 must show a stable positive Δ (target ~2× over
Cut 1, ~4× over BF16) or it is reverted. wins/ entry cites the Colab bench.

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
