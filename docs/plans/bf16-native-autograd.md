# bf16-native autograd tensor pipeline (CUDA OPD train lane)

**Status:** design — not started. **Owner:** ckl. **Scope:** `crates/autograd`
CUDA backend + `crates/train` estimators. CPU reference, Metal backend, and the
host `Vec<f32>` canonical contract are explicitly untouched.
**Baseline (measured, 2026-07-02, H20 GPU-pinned toy agent-OPD round, 27B
Qwen3.6-FP8 `--share-frozen-base`, LoRA attention-qv r16, seq≈1010):**
forward 2.2–2.8 s, backward 5.06 s (post-adaptive-checkpointing `f6d11206`,
of which LinearAttention backward 4.2 s = 83%), full-attn layer wall 96–97 ms,
linear-attn 32.5 ms (`ARLE_OPD_PROFILE_SYNC=1`). Sources:
[wins/2026-07-02-agent-opd-full10-e2e](../experience/wins/2026-07-02-agent-opd-full10-e2e.md),
[wins/2026-07-02-opd-sdpa-fused-prefill-kernel](../experience/wins/2026-07-02-opd-sdpa-fused-prefill-kernel.md),
[wins/2026-07-02-opd-partial-rotary-rope-device](../experience/wins/2026-07-02-opd-partial-rotary-rope-device.md).

**One-sentence version:** the autograd store is f32-canonical while every heavy
kernel already computes in bf16, so each GEMM/SDPA call pays 2–4 conversion
kernels + allocs and every activation/tape byte is 2× — make
`DeviceHandle::CudaBf16` the default activation storage with f32 master
weights / f32 param-grad accumulation / f32 reductions (Megatron + torch-amp
semantics), in four flag-gated tranches, licensed or killed by a tranche-0
conversion-share measurement.

**Honesty upfront (framing, per §0):** the pure-traffic arithmetic in §5 says
conversion kernels + halvable f32 traffic explain only ~4–10% of today's
*measured* forward wall at seq≈1010 — the measured per-layer walls sit ~5–10×
above the bandwidth+FLOP roofline, so most of the 96 ms/layer is **unattributed
overhead** (launch/alloc/dispatch), and how much of that the conversion-op
removal recovers is a *hypothesis* until T0's nsys attribution. The
arithmetic-solid wins are (a) tape VRAM ≈ −45% → the no-recompute /
adaptive-checkpoint ceilings move ~2× in seq_len, which at production agentic
trajectory lengths (≥8K) is worth up to a full forward-recompute per backward,
and (b) train-forward numerics equal to the serving forward (both bf16), which
is a correctness feature for OPD, not just perf.

---

## 1 · Inventory — where f32 is load-bearing vs incidental

### 1.1 DeviceHandle variants (`crates/autograd/src/backend.rs:266-277`)

| Variant | Storage | Today's role |
|---|---|---|
| `Cpu(Vec<f32>)` | host | CPU backend + tests |
| `Metal(MlxHandle)` | MLX array | Metal lane — out of scope |
| `Cuda(CudaStorage)` | `Arc<CudaSlice<f32>>` (backend.rs:109-128) | **the only activation format on CUDA** — every op output |
| `CudaBf16(CudaBf16Storage)` | `Arc<CudaSlice<u16>>` (backend.rs:130-149) | frozen weights (`upload_bf16_bits`, backend_cuda.rs:1029; loader `crates/train/src/qwen35_loader.rs:1324`), embed table, and — precedent — LinearAttention's saved forward intermediates `qkv_conv/q/k/v/a_inv/raw_output` (backend_cuda.rs:4152-4161) |
| `CudaFp8BlockScaled(…)` | u8 weight + f32 scales (backend.rs:151-264) | frozen base under `--share-frozen-base` (borrowed import backend_cuda.rs:1101-1175) |

No new variant is needed: `CudaBf16` already exists, already round-trips
through `readback` (backend_cuda.rs:1220-1233 converts to `Vec<f32>` on host),
and is already produced/consumed inside the LinearAttention op family. The
design promotes it from "weights + LA internals" to "default activation".

### 1.2 Incidental f32 — the conversion churn (delete)

Every conversion site below is one `alloc_zeros` (alloc **+ memset**) plus one
elementwise kernel:

- **Bridge helpers**: `local_f32_as_bf16` (backend_cuda.rs:466-491),
  `import_local_bf16_as_f32` (:434-463); kernels `f32_to_bf16_bits` /
  `bf16_bits_to_f32` (`backend_cuda/kernels/bridge.cu`).
- **Every frozen-weight GEMM, forward**: `matmul_bt_device_f32_bf16`
  (:692-793) converts the activation in at :716 and imports the bf16 GEMM
  output back to f32 at :791; `matmul_device_f32_bf16` (:796-889) ditto at
  :820/:887. Dispatched from `matmul_bt` (:1495-1521) for `CudaBf16` and
  `CudaFp8BlockScaled` weights — i.e. **all 7 base projections per layer**.
- **Every frozen-weight GEMM, backward**: grad-input rides the same pair via
  `cuda_matmul_bt_input_grad_device` (:5116-5138) and
  `cuda_matmul_bt_backward_device` grad_a (:5209-5232).
- **FP8 weights additionally re-dequant per GEMM call**:
  `fp8_block_scaled_as_bf16` (:896-941) materializes the full bf16 weight on
  every forward *and* every backward GEMM touching it (:944-963). ~448 forward
  + ~450 backward dequants/step of ~0.4 GB matrices ≈ 1–2 GB/layer extra
  traffic. (A dequant-once bf16 weight cache is an adjacent orthogonal fix —
  §3 T1 note — with a +27 GB VRAM tradeoff vs per-call dequant.)
- **Fused SDPA**: q/k/v converted f32→bf16 at :3658-3660, output imported back
  at :3701 — around a kernel (`nonpaged_prefill_attention_cuda`) that is
  **already bf16-native**.
- **LinearAttention forward**: b/a/dt converted at :3764-3766; the big
  `qkv`/`z` inputs are consumed as f32 (:3733-3734) by
  `linear_attention_conv1d_silu_forward_f32_to_bf16` — the kernel converts to
  bf16 *itself* one op later.
- **All elementwise / norm / layout / softmax / rope / gather / embedding
  kernels are `*_f32`** (`backend_cuda/kernels.rs:70-138` FUNCTION_NAMES), and
  every CUDA op output allocates f32 (102 `alloc_zeros::<f32>` sites in
  backend_cuda.rs). Activations moving through them pay 2× the bytes bf16
  would.
- **Tape storage**: every saved forward tensor a backward op needs is f32 —
  the ~24 GB tape estimate at seq≈1010
  ([full10-e2e addendum 2](../experience/wins/2026-07-02-agent-opd-full10-e2e.md)).

Conversion-launch count per step (from per-layer op composition,
`crates/train/src/qwen35.rs` `forward_full_attention` :1542 /
`forward_linear_attention` :2313, 16 full + 48 linear layers):
full-attn layer fwd = 7 GEMMs×2 + SDPA 4 = 18; linear layer fwd = 5 GEMMs×2 +
LA 3 = 13; + lm_head 2 → **≈ 915 conversion kernels + ≈ 915 allocs per
forward**; backward re-runs grad-input conversions per GEMM plus FP8
re-dequants → **≈ 1 000–1 500 more**. Roughly a third of all kernel launches
in a step are dtype plumbing.

### 1.3 Load-bearing f32 — keep (with the kernels that already do it)

| # | Site | Evidence it must stay f32 (or already accumulates f32) |
|---|---|---|
| 1 | CE / softmax / log_softmax reductions | `softmax.cu` `softmax_last_axis_f32` / `log_softmax_last_axis_f32`: row max + Σexp in f32 smem. Loss-lane outputs (gathered log-probs, `mean`, scalar loss) stay f32 — torch-amp autocasts CE to f32. |
| 2 | RMSNorm sum-of-squares | `rms_norm.cu:24-38` f32 `local_sq` tree-reduction; saved `rms_norm_inv_rms_f32` stats for backward stay f32. Only the x-in/out dtype changes. |
| 3 | SDPA accumulation | `nonpaged_prefill_attention_cuda` is bf16-I/O with f32 online-softmax accumulators internally (inference kernel, adopted b92ec601). Nothing to change. |
| 4 | LinearAttention recurrence states | `preact/g/g_cumsum/beta/chunk_state/state_history` are f32 buffers (backend_cuda.rs:3768-3810) and the serial scan backward is all-f32 (`linear_attention.cu:108-128`). Precision-critical recurrence — bf16 I/O only at the qkv/z/upstream edges. |
| 5 | AdamW master weights + moments | `adamw.cu` `adamw_step_f32` mutates f32 param/m/v in place; `optim.rs:205-375 step_device` keeps f32 device handles. **Trainable LoRA params stay f32 = master weights.** |
| 6 | Grad-clip norm | `Backend::sum_squares → f64` (backend.rs:912; kernels `sum_squares_partial_f32`/`grad_clip_sumsq_f32`, backend_cuda.rs:6428/6537), consumed by `crates/train/src/grad_clip.rs`. |
| 7 | Param-grad accumulation | `TensorStore::accumulate_grad` (tensor.rs:576-651) + `add_into_device`/`add_into_f32` (backend_cuda.rs:5565-5598). Megatron accumulates grads in f32; param-grads stay f32 end-to-end (§2.2). |
| 8 | RoPE cos/sin caches | angle tables stay f32 (`rope_f32` reads f32 cache); only x/grad dtype changes. |
| 9 | Host canonical `Tensor.data: Vec<f32>` | tensor.rs:21. `readback` already converts CudaBf16 → f32 (backend_cuda.rs:1220-1233) — `ensure_host`/`to_host` contracts hold with zero changes. |

---

## 2 · Target design

### 2.1 Dtype contract (Megatron `--bf16` / torch-autocast envelope, adapted to LoRA)

| Lane | Dtype | Where enforced |
|---|---|---|
| Forward activations + tape-saved activations | **bf16** (`DeviceHandle::CudaBf16`) | every CUDA op's output alloc |
| Activation gradients (backward flow) | **bf16** | backward ops, dtype-follows-input |
| Frozen weights | bf16 / FP8 (unchanged) | loader |
| Trainable (LoRA) master params | **f32** (unchanged; host mirror + `Cuda` f32 handle) | `lora.rs:186-187` params |
| Param-grads (`*.lora_a`/`*.lora_b` grad_b) | **f32**, produced directly by `gemm_ex` bf16×bf16→**f32 C** (`CUDA_R_32F` output, `CUBLAS_COMPUTE_32F`) — no conversion pass | `cuda_matmul_bt_backward_device` grad_b branch |
| GEMM accumulation | f32 (`CUBLAS_COMPUTE_32F`, unchanged) | :782/:880 |
| softmax/rmsnorm/SDPA/LA internal accumulation | f32 (unchanged, §1.3) | kernels |
| log_softmax/CE loss lane output | f32 | `softmax_last_axis`/`log_softmax_last_axis` bf16-in→f32-out |
| Optimizer state (m/v) | f32 (unchanged) | `adamw_step_f32` |
| Grad-clip norm | f64 (unchanged) | `sum_squares` |
| Host mirror | `Vec<f32>` (unchanged) | tensor.rs |

**Rules:**
- **dtype-follows-input** for unary/binary/layout ops: bf16 in → bf16 out.
- **Named exceptions** produce f32: loss-lane reductions (`sum_all`,
  `mean`, log_softmax output), param-grad GEMMs, `sum_squares`.
- **Mixed binary op (bf16 ⊕ f32)** → f32 path via an on-device upgrade shim
  (§2.3); post-migration the shim must measure ~0 on the hot path (T4 gate).
- LoRA GEMMs: activation is bf16, LoRA weight is f32 master → cast the weight
  operand f32→bf16 per call (r16×hidden ≈ 82K elems, trivial). A cached bf16
  shadow invalidated by `replace_device_handle` after `adamw` is a later
  optimization, not tranche scope.

### 2.2 Why this decomposition is industry-standard

Megatron `--bf16`: bf16 model params + activations, f32 main params inside the
optimizer, f32 grad accumulation. torch autocast(bf16): matmul-class ops in
bf16, softmax/norm/CE upcast to f32. Our variant: the "model params" are
already bf16/FP8 (frozen base), the f32 "main params" are exactly the LoRA
tensors, and grads split into bf16 activation-grads (flow) vs f32 param-grads
(accumulate) — the same split, with the mixed-output `gemm_ex` replacing
Megatron's grad-accum cast.

### 2.3 Mechanics

- **No ops-layer changes for dispatch.** `ops/*.rs` pass `DeviceHandle`
  opaquely (e.g. `ops/matmul.rs:37-40`, `ops/elementwise.rs:35/:109`,
  `ops/norm.rs:37`); dtype dispatch lives entirely inside `CudaBackend`
  methods. Device-residency gates (`dirty != Host && handle.is_some()`) are
  dtype-agnostic and unchanged.
- **`ActSlice` helper** in backend_cuda.rs:
  `enum ActSlice<'a> { F32(&'a CudaSlice<f32>), Bf16(&'a CudaSlice<u16>) }` +
  `fn act_slice(&self, h: &DeviceHandle, op: &str) -> Result<ActSlice>`
  replacing `cuda_slice` at activation inputs. Plus the migration shim
  `fn act_f32_view(&self, h) -> Result<Cow-like CudaSlice<f32>>` that upgrades
  a bf16 handle on-device (one `import_local_bf16_as_f32`) for **not-yet-ported
  ops** — this is what makes every tranche a complete, correct state instead of
  a half-state: an unported op costs exactly today's conversion, never an
  error, never a silent host demotion (the rope-fix lesson: one host fallback
  demotes the whole downstream chain —
  [wins/2026-07-02-opd-partial-rotary-rope-device](../experience/wins/2026-07-02-opd-partial-rotary-rope-device.md)).
- **Shim + conversion counters** (`AtomicU64` count + bytes, mirroring Metal's
  `eval_count`) so "did the hot path stop converting" is a measured number,
  not code-reading (the flat-A/B lesson from the SDPA win: silent fallbacks
  need a probe switch).
- **Kernels**: NVRTC-concatenated C sources with `unsigned short` bf16 bit
  helpers already in-tree (`la_bf16_to_float`/`la_float_to_bf16`,
  `linear_attention.cu:13-22`; same trick in `bridge.cu`). New `*_bf16`
  variants follow that pattern — f32 math inside, bf16 loads/stores — plus new
  entries in `FUNCTION_NAMES` (kernels.rs:70-138). No `cuda_bf16.h` dependency.
- **Rollout flag**: `--bf16-activations` CLI flag on the train binaries
  (runtime config = CLI flags, not env), plumbed to a
  `CudaBackend { act_dtype }` field. Default **off** until the T3 gate; the
  flag is deleted in T4 after the default flip (no long-lived dual default).
- **Host/`tensor.rs` contract: zero changes.** `data: Vec<f32>` stays
  canonical; `ensure_host`/`to_host`/`flush_to_host_batch` work today for
  `CudaBf16` via `readback`'s conversion; `ensure_device` on a host tensor
  keeps producing an f32 handle (legal mixed graph — first device op consuming
  it either takes the f32 lane or the shim). `clone_tensor`'s Arc-sharing of
  handles (tensor.rs:674-714) is dtype-agnostic. The checkpoint offload pool
  stays `Vec<f32>` (tensor.rs:484-519); a u16 host pool that halves offload
  PCIe is a named follow-up, not in scope.
- **VRAM estimators become dtype-aware** (T3): `should_checkpoint`'s `* 4`
  (`crates/train/src/qwen35.rs:2606-2611`) and `ckpt_group_size`'s
  `.saturating_mul(4)` (:301-312) read the activation byte width from the
  backend. This is where the −45% tape directly becomes a 2× seq ceiling.

---

## 3 · Migration path — shippable tranches

Every tranche exits with: compiles under
`cargo check -p autograd --no-default-features --features cuda,no-cuda`
(Mac gate) + `cargo test -p autograd --release` green + pod toy round
(`run-*-toy1r` config, GPU-pinned) `RUN_EXIT=0` with **loss in the 0.24–0.33
band** + its own commit + a dated wins/errors entry (bench mandate). Tranches
land flag-off by default until T3.

### T0 — attribution + license gate (~0.5 day) — BLOCKS everything

1. Add conversion counters (count + bytes) to `local_f32_as_bf16`,
   `import_local_bf16_as_f32`, `fp8_block_scaled_as_bf16`,
   printed in the existing `ARLE_OPD_PROFILE=1` round summary.
2. One nsys capture of a toy round (fwd+bwd window): bucket GPU time into
   {GEMM, conversion kernels, memset/alloc, elementwise/layout f32 ops,
   SDPA/LA kernels, idle/launch gaps}. This attributes the ~80 ms/layer gap
   between the 96 ms measured wall and the ~10–15 ms roofline (§5.1).
3. One long-seq probe (seq ≥ 8K writeback) to record where
   `should_checkpoint` flips and what backward costs across the cliff.

**License:** proceed iff (conversion + memset + halvable-f32-traffic +
attributable launch overhead) ≥ 10% of round wall-clock **or** the target
workload's seq crosses the f32 recompute/OOM cliff that bf16 would defer.
Otherwise **KILL** (§5.3) — write the errors entry and stop.

### T1 — GEMM outputs stay bf16 + elementwise bf16 lane (~1–2 days)

- `CudaBackend` gains `act_dtype` + `--bf16-activations` plumbing
  (`train_cli`).
- `matmul_bt` (backend_cuda.rs:1478-1558) / `matmul` (:1456-1476): accept
  bf16 lhs (skip the :716/:820 convert), and when `act_dtype=bf16` return
  `DeviceHandle::CudaBf16` directly (skip the :791/:887 import). Keep the
  skinny-N SIGFPE padding (:493-529) — only the C dtype changes.
- Replace `cuda_slice` with `act_slice`/`act_f32_view` at activation inputs of
  **all** CUDA methods (mechanical; the shim keeps unported ops correct).
- bf16 kernel twins + dispatch for the ops adjacent to GEMMs:
  `add`/`mul`/`mul_scalar`/`silu`/`sigmoid` (+ their `*_backward_device`) and
  `add_into` — files `kernels/elementwise.cu`, `silu.cu`,
  `activation_backward.cu`, `mul_backward.cu`, `add_into.cu`,
  `kernels.rs` FUNCTION_NAMES; trait impls at backend_cuda.rs :1560 (add),
  :2097 (silu), :2109 (sigmoid), :2121 (mul), :1864 (add_into_device), plus
  the backward-device methods.
- *Adjacent orthogonal fix (separate commit, own A/B):* cache the FP8→bf16
  dequant per matrix per step instead of per GEMM call (:944-963), or promote
  once like the LoRA-sync fix
  ([wins/2026-07-02-cuda-lora-fp8-promote-bf16](../experience/wins/2026-07-02-cuda-lora-fp8-promote-bf16.md));
  measure VRAM (+~27 GB dense-bf16 27B) vs the ~1–2 GB/layer re-dequant
  traffic before choosing.
- **Gate:** matched same-binary A/B (flag on/off, same GPU, same task):
  forward Δ, VRAM Δ, loss in band, shim-counter reported.

### T2 — norms / rope / layout / SDPA / embedding / LA edges (~2–3 days)

- `rms_norm` (:2247) bf16-I/O kernel (f32 sumsq unchanged) + backward
  (`rms_norm_backward.cu`; grad_w stays f32-out — norm weights are potential
  trainables).
- `rope` (:2390) bf16 x, f32 cos/sin cache, `rot_half` param preserved
  (`rope.cu`/`rope_backward.cu`).
- Layout: `transpose_axes_swap` (:2525), `slice` (:2543), `concat_axis2`,
  `slice_backward`, `reshape` (:2504) — pure u16 moves in `layout.cu`; these
  were the 10.5 s host-cascade cost in the rope-fix entry, so they must ride
  bf16 natively, not the shim.
- SDPA prefill: delete the :3658-3660/:3701 conversions when inputs are bf16
  (kernel is bf16-native). `causal_sdpa_recompute_backward_f32` (:5300+): keep
  the f32 kernel, convert q/k/v/upstream in at entry via the shim first; port
  the kernel to bf16-I/O only if T0/T2 profiling shows it matters.
- `embedding` (:2283): `embedding_bf16` kernel (table already bf16) → bf16
  output.
- LinearAttention: forward accepts bf16 `qkv`/`z` (bf16-in variant of
  `linear_attention_conv1d_silu_forward*`; delete the b/a/dt converts
  :3764-3766 when bf16); backward (:4166) converts bf16 `upstream` in at the
  edge — **scan internals stay f32** (§1.3 #4).
- `log_softmax_last_axis` (:2040): bf16-in → f32-out; its backward emits bf16
  grad (feeds the lm_head GEMM backward). `gather`/`add_broadcast`/`mean`/
  `sum_last_axis` + their backwards as tape coverage requires.
- **Gate:** toy round flag-on with **shim counter ≈ 0 on the hot path**
  (allowed: loss lane, first-step uploads), loss in band, matched A/B wall +
  VRAM vs T1.

### T3 — grad flow, estimators, default flip (~1 day)

- Param-grad lane: `cuda_matmul_bt_backward_device` grad_b (:5257+) and
  `cuda_matmul_backward_device` grad_b switch to `gemm_ex` bf16×bf16→f32-C so
  `accumulate_grad`/`add_into_f32`/AdamW/grad-clip see f32 with zero new
  conversions. `merge_grad` (tape.rs:919) needs no change beyond
  `add_into_device` dispatch: bf16+bf16 → `add_into_bf16`; mixed → f32
  promote.
- Backward seed: `fill_like(loss, 1.0)` + `ensure_device` (tape.rs:600-608)
  stays a scalar f32 handle — mixed rule covers it.
- Dtype-aware estimators: qwen35.rs:2606-2611 and :301-312 (§2.3).
- Long-seq probe rerun: confirm the no-recompute ceiling moved ~2× and record
  backward Δ across the old cliff.
- **Flip `--bf16-activations` default on** after: toy round in band, full10
  e2e (10 rounds) loss trajectory descending and terminal loss comparable to
  the 0.1740-class runs, grad-norm trace within band of the f32 run (same
  seed/config), VRAM ledger matches the bit-exact prediction.

### T4 — cleanup deletion pass (~0.5 day)

- Delete shim call-sites whose counters measure 0 across toy + full10 + one
  long-seq round; delete the flag (bf16 becomes the CUDA lane's only
  activation default; CPU/Metal untouched); delete now-unreachable conversion
  plumbing.
- Update `crates/autograd/AGENTS.md` (DeviceHandle contract §, tolerance
  note §4) — it currently documents f32-only handles.
- Final wins entry with the cumulative A/B table.

Dependency DAG: T0 → T1 → T2 → T3 → T4, strictly serial; the FP8-dequant-cache
side fix is parallel to T1+ but must be its own commit/A/B (single-variable
rule).

---

## 4 · Numerics gates

- **f32-accumulation invariants** (§1.3 table) are non-negotiable; every new
  bf16 kernel keeps f32 math internally (load-convert / store-convert only) —
  the established pattern of `linear_attention.cu` and the inference SDPA
  kernel.
- **Tolerance:** the AGENTS.md CPU-parity bar (≤1e-3 rel) is unachievable for
  bf16 *storage* (8 mantissa bits, per-element rel step ~3.9e-3). New test
  file `tests/test_cuda_bf16_act_ops.rs` (mirroring
  `test_cuda_bf16_frozen_ops.rs`) checks each ported op against the CPU f32
  reference at **2e-2 rel / 1e-2 abs**, documented in AGENTS.md at T4. CPU
  reference itself stays f32 — it remains the semantic contract.
- **Correct-inference-style gate, not byte-identity:** the training-loss
  analogue of the needle gate — toy-round loss in the 0.24–0.33 band
  (baseline runs: 0.2795 / 0.2793 / 0.2824 / 0.2827), full10 trajectory
  descending, grad-norm trace in band. MoE/FP8 non-determinism plus bf16
  rounding makes token/loss byte-identity a non-goal
  (`feedback_correct_inference_not_baseline_identity`).
- **Train–infer alignment is a numerics argument FOR this change:** the
  serving forward (infer-cuda) runs the whole hidden stream in bf16; today's
  train forward runs a f32/bf16 hybrid that matches neither pure-f32 nor the
  serving numerics. Post-migration the student's train forward ≈ its rollout
  forward, shrinking the OPD train/infer mismatch regime
  (`project_opd_train_infer_unified_mismatch_regime`).
- **Loss-band caveat:** the band is a coarse gate; the T3 default flip
  additionally requires the full10 *trajectory* comparison (10 points, same
  seed/config) because a single toy round can sit in band while drifting.

---

## 5 · Risk + expected gain + kill conditions

### 5.1 Gain arithmetic (grounded, with the uncertainty named)

Op counts per step (§1.2): ≈ 915 conversion kernels + allocs in forward,
≈ 1 000–1 500 in backward. Traffic formula per bf16-weight GEMM: input convert
= 4B read + 2B write + 2B memset ≈ 8 B/elem; output import = 4B memset + 2B
read + 4B write ≈ 10 B/elem. Summed over the per-token GEMM I/O widths of all
64 layers (~8M elems/token incl. lm_head) × 1010 tokens →
**≈ 60–80 GB of pure conversion traffic per forward ≈ 25–30 ms at H20's
~2.5 TB/s achievable — 1–2% of the 2.2–2.8 s measured forward.** Halving the
f32 traffic of non-GEMM activation ops (norm/rope/layout/elementwise, ~30–50
GB/forward) adds ~3–8%. Backward similar. **Pure-bandwidth roofline gain:
4–10% — a wash-risk number by itself**
(`feedback_b1_decode_gpu_bound_overhead_removal_wash`).

The open variable: measured full-attn layer wall is 96 ms vs a ~10–15 ms
GEMM+traffic roofline. If T0's nsys shows the ~80 ms gap is launch/alloc/
dispatch overhead, removing ~⅓ of launches+allocs tracks to **10–30%**; if
the gap is elsewhere (cuBLASLt heuristics, hidden syncs, kernel inefficiency),
conversion removal recovers little. **This is exactly what T0 exists to
decide before the ~5-day surface investment** — launch-count surveys alone are
hypothesis, not license
(`errors/2026-05-21-arle-cuda-opd-swiglu-fused-kill.md`).

Arithmetic-solid regardless of T0:
- **Tape VRAM ≈ −45%** (activations halve; LA f32 states, saved norm stats,
  f32 loss lane don't): est ~24 GB → ~13 GB at seq≈1010. Via
  `should_checkpoint` (qwen35.rs:2606-2611) the no-recompute ceiling moves
  ~2× in seq; past the old cliff, backward currently pays a full forward
  recompute (the measured 10.0 s → 5.06 s adaptive-checkpointing delta *is*
  that recompute at seq≈1010) — **up to ~−45% backward at seqs between the
  old and new cliffs**, plus `ckpt_group_size` doubling → half the host
  offloads beyond it. Production agentic trajectories (8–32K) live in exactly
  this regime.
- **Backward's current #1 wall is untouched by design:** LinearAttention
  backward 4.2 s (83% of 5.06 s) is an f32 serial scan; this plan only trims
  its I/O edges. bf16 does not compete with — and must not be confused with —
  a future LA-scan optimization (separate axis).

### 5.2 Risks

| Risk | Mitigation |
|---|---|
| Silent fallback cascade (one unported op demotes a chain) | shim converts **on-device**, never host; counters + `ARLE_OPD_PROFILE` visibility; T2 gate requires shim ≈ 0 |
| cuBLASLt bf16 SIGFPE heuristic on skinny-N | padding logic (:493-529) preserved verbatim; only C dtype changes |
| Loss drift beyond band | tranche granularity is the bisect unit; revert the tranche commit (small-tranche rule), re-attribute at op level via the parity test before re-landing |
| MoE ops (35B-A3B student) not ported | out of scope for T1–T4 (27B dense is the toy target); shim keeps MoE correct at today's cost; named follow-up |
| VRAM transient regressions during migration (dual copies) | flag-off default until T3; VRAM ledger check per tranche gate |
| `--share-frozen-base` FP8 borrow lifetimes | untouched — FP8 handles and their `Drop` semantics (backend.rs:171-191) are not modified by this plan |
| Mac/CI breakage | every tranche runs the `cuda,no-cuda` typecheck; kernels are runtime-NVRTC so no build-graph change |

### 5.3 Kill conditions (explicit)

1. **T0 kill:** conversion + memset + halvable-traffic + attributable launch
   overhead < 10% of round wall-clock at seq≈1010 **and** the long-seq probe
   shows target workloads stay under the f32 recompute cliff → KILL the
   project, errors entry, keep the f32 pipeline. (VRAM-only motivation does
   not license a ~5-day surface change if the cliff isn't binding.)
2. **T1 kill:** matched A/B shows < 3% forward improvement **and** no VRAM
   delta at flag-on → stop after T1, revert the default (flag stays available
   only if it's a strict wash *and* deletes net code; otherwise revert fully —
   no half-states).
3. **Numerics kill:** any tranche that cannot hold the loss band / full10
   trajectory after one focused fix attempt → revert that tranche, errors
   entry with the decoded failing op (case-as-fact: attribute the op via the
   parity test before generalizing to "bf16 can't work here").

### 5.4 Verification checklist for the executing session

- [ ] T0 numbers in a wins/errors entry **before** any T1 code.
- [ ] Every A/B: same binary, same GPU, same task, flag as the only variable.
- [ ] Per-tranche: Mac typecheck, `cargo test -p autograd --release`, pod toy
      round, VRAM ledger, commit + entry.
- [ ] T3 flip only with full10 + long-seq + grad-norm evidence attached.
- [ ] T4 deletes the flag and updates `crates/autograd/AGENTS.md`.
