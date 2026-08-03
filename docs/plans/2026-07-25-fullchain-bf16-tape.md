# Full-chain bf16 mixed-precision training — design + implementation spec

> **STATUS 2026-08-03 — SUPERSEDED by the shipped `--tape-dtype` design**
> (commits 23e03e590 / 42dcd537f / f2f6e5aea / 8f6deb43f): kernel templating via
> dtype-keyed NVRTC injection, flag surface on agent-opd + opd + self-opd. The pod
> A/B ladder is pending and MUST include the checkpoint-path no-op probe from
> `docs/experience/errors/2026-07-27-tape-bf16-noop-on-checkpoint-path.md`.

> **STATUS 2026-07-25.** S0 (config) + S1a (frozen prefix K/V) SHIPPED & correct
> (loss byte-identical, needle 5/5 DET). **S1a REJECTED as a VRAM lever** — on
> Qwen3.6-27B the frozen K/V is 0.13 MB/tok (16 full-attn layers of 64), ~400×
> smaller than the GDN linear-attention forward-capture transient that actually
> sets the peak (+52.7 GB at seq1024); bf16 there measured **+288 MiB** (quantize
> double-buffer) and did not move the OOM wall. §7 Stage-1a below is superseded.
> **The peak is the forward `la_*` capture transient, not retained K/V** — S1b
> re-targets the GDN linear-attention forward buffers, and any store-time downcast
> must quantize in place / free-then-alloc (the +320 MiB double-buffer sank S1a).
> KILL entry: `docs/experience/errors/2026-07-25-s1a-frozen-prefix-kv-bf16-no-vram-win-kill.md`

Goal: halve the two buffers that set the LoRA-backward peak (retained activations +
transient emitted grads) so the agent-OPD token wall moves from ~30K to ~60K, via a
single ergonomic knob `--tape-precision {fp32,bf16}` (default `fp32`, byte-identical).
Store-bf16 / compute-fp32: every GEMM accumulate, every fp32 island, and every
persistent grad accumulator stay f32.

All load-bearing facts verified against source (workflow-designed, adversarially
critiqued). Key reconciliations: (1) `runtime_flags.rs` field+static+apply+accessor
pattern; (2) CLI value-enum via `OpdEngineOffloadArg`→`to_flags()` map; (3) `cuda_slice`
(backend_cuda.rs:546) hard-errors on `CudaBf16`, and `cuda_bf16_slice` (:585) already
exists; (4) reshape (:2532) calls `cuda_slice(x,"reshape")?` which errors on bf16 *before*
the `x.clone()` view at :2541 — reshape is an f32-only blocker, not dtype-transparent;
(5) `local_f32_as_bf16` (:480) returns bare `CudaSlice<u16>`, `import_local_bf16_as_f32`
(:448) returns `CudaSlice<f32>`.

## 1. VERDICT

**Feasible, but scoped — the "half the plumbing is built" premise is false for the
backward chain.** bf16 is wired only for the *forward* `matmul_bt`/`matmul` B-operand
(`matmul_device_f32_bf16`, backend_cuda.rs:812). The backward/grad chain is essentially
unbuilt: the single f32 gate `cuda_slice` (backend_cuda.rs:546) hard-errors on any
`CudaBf16` handle and is hit by ~20 `*_backward_device` consumers plus `adamw_step`
(:3198) and `clip_grad_norm` (:6942). "Store as bf16" is therefore not a free lever — it
requires a bf16 read-branch in each consumer.

The correct product is **one enum knob `--tape-precision {fp32,bf16}`** (default `fp32`,
byte-identical), where `bf16` means: *forward retained activations and transient per-op
emitted gradients are stored bf16 (u16), all compute stays fp32 (cuBLAS accumulate +
all fp32-islands untouched), and every persistent grad accumulator + red-line island
stays f32.* This halves the two buffers that set the backward peak — the
O(layers·seq·hidden) retained-activation chain (checkpoint.rs:40 keep-set, qwen35.rs:3869
inter-layer carry, 1298 MoE activations) and the transient per-expert LoRA emitted grads
(moe.rs:941/955).

**Wall it moves:** 27B LoRA writeback currently OOMs at ~30K tokens; halving retained
activations + emitted grads targets **~2× the token wall (~60K)**. This is a modeled
ceiling, not a measured number — Stage 1 alone (activations) captures the retained-chain
half; the full ~2× needs Stage 2 (emitted grads) landed. The number is confirmed only by
the pod peak-VRAM A/B in §7.

## 2. CONFIG SURFACE

**Enum, not bool.** A bare `--bf16-tape` bool conflates the safe axis (activations,
write-once grads) with the unsafe one (multi-touch grad accumulator → swamping). The enum
matches the shipped precedent (`OpdEngineOffloadArg`, `ServeKvCacheDtypeArg`,
`SaveDtypeArg`) and leaves headroom for a future `fp8` without a new flag. We do **not**
expose "activations-only vs grads-too" as separate user flags (YAGNI); that split is an
internal staging detail (§6). The accumulator-stays-f32 rule is an implementation
invariant, not a knob.

**autograd — `crates/autograd/src/runtime_flags.rs`** (enum stored as `AtomicU8`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapePrecision { Fp32 = 0, Bf16 = 1 }
impl TapePrecision { fn from_u8(v: u8) -> Self { if v == 1 { Self::Bf16 } else { Self::Fp32 } } }

pub tape_precision: TapePrecision,            // field on AutogradRuntimeFlags
// Default: TapePrecision::Fp32
static TAPE_PRECISION: AtomicU8 = AtomicU8::new(0);
TAPE_PRECISION.store(f.tape_precision as u8, Relaxed);   // in apply_runtime_flags

#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn tape_precision() -> TapePrecision { TapePrecision::from_u8(TAPE_PRECISION.load(Relaxed)) }
#[cfg_attr(any(not(feature = "cuda"), feature = "no-cuda"), allow(dead_code))]
pub(crate) fn tape_bf16() -> bool { matches!(tape_precision(), TapePrecision::Bf16) }
```

**CLI — `crates/cli/src/args.rs`** (value-enum + map in `to_flags()`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum TapePrecisionArg { Fp32, Bf16 }

#[arg(long, value_enum, default_value_t = TapePrecisionArg::Fp32)]
pub(crate) tape_precision: TapePrecisionArg,

tape_precision: match self.tape_precision {
    TapePrecisionArg::Fp32 => autograd::TapePrecision::Fp32,
    TapePrecisionArg::Bf16 => autograd::TapePrecision::Bf16,
},
```

`TapePrecision` re-exported from `crates/autograd/src/lib.rs`. Default `Fp32` → the store
decision (§4) is skipped entirely → byte-identical.

**Status: Stage 0 SHIPPED** (config plumbing, zero behavior change). The above is live in
the three files; the remaining stages add the store/read branches behind `tape_bf16()`.

## 3. PER-BUFFER DISPOSITION TABLE

Two lenses reconciled: **(A)** enumerated `on_retained_chain`/`on_grad_chain` tags, **(B)**
adversarial critic. Where they disagree the decision + reason is stated. Dispositions:
`RED-LINE` (stays f32 always), `BF16-S1` (activation, Stage 1), `BF16-S2` (transient
emitted grad, Stage 2), `ACCUM-f32` (persistent accumulator, stays f32 — §4), `NO-OP`
(host Vec/view/rollout-only — flag never applies).

### Red-line islands (stay f32 regardless of knob)

| file:line | role | reason |
|---|---|---|
| backend_cuda.rs:3678/3682 `q/k/v/out_bf16` | flash-kernel operands | already bf16-internal, fp32 online-softmax accumulate — do not touch |
| ops/attention.rs:176 `CausalSdpaRecomputeCtx{q,k,v}` + bcuda:5823-5826 `d_q/d_k/d_v` | SDPA recompute-read q/k/v | **B over A**: recompute-read via f32 gate; needs a bf16 recompute-read branch, not just a store change. Island: keep f32 in S1/S2. |
| bcuda:9056 `d_inv` (inv_rms) | RMSNorm 1/√(mean x²+ε) | fp32-island |
| tape.rs:82 `RMSNormCtx.inv_rms: Vec<f32>` | saved inv_rms | fp32-island; host metadata |
| qwen35.rs:3736 `logits` (lm_head) | [b,seq,vocab] | logits+CE island; GEMM accumulate + storage f32 |
| bcuda:3462/3590 log_softmax/gather-bwd [B,S,V] grad | CE-path grad | **B over A**: biggest single buffer but CE-sensitive (catastrophic cancellation over vocab). Gets its **own** A/B (§8), NOT folded into the S1/S2 flip. Default f32. |
| bcuda:8662 `d_grad` embedding-bwd [V,H] + :7409 host scatter | atomicAdd accumulator | ACCUM red-line: bf16 atomicAdd lossy across duplicate tokens; no fast bf16 atomicAdd on all SM |
| optim.rs / adamw_state.rs master weights + moments | — | out of scope, explicit red line |
| qwen35.rs:3645 cos/sin cache, ops/rope.rs:81 cos_data/sin_data | RoPE caches | frozen read-only, tiny, RoPE-precision adjacent |
| bcuda:1921 `all_reduce_sum` out | NCCL reduce | hardcodes `DType::F32` @1936 — dtype must move in lockstep with NCCL; defer (S2 non-goal) |

### Forward retained activations → BF16-S1

| file:line | role | disposition | note |
|---|---|---|---|
| checkpoint.rs:40 keep-set retained device activations | O(layers·seq·hidden) retained | **BF16-S1** | dtype-transparent (holds TensorId/handle); the dominant lever |
| qwen35.rs:3762 embedding out; 3869 inter-layer hidden; 1298 MoE activations; 3725 post-final-norm hidden | layer-stack activations | **BF16-S1** | consumers need bf16 reads (§4): add/rmsnorm/mul/matmul-`a` |
| qwen35.rs:84 `PrefixKv.k/v`; 92 `PrefixState.state/conv_window` | frozen cross-pass prompt K/V + LA boundary | **BF16-S1 (first slice)** | frozen (requires_grad=false), no grad chain, biggest single retained buffer; **the ideal Stage-1a A/B** — consumer is the gen-segment flash path that already converts to bf16 |
| bcuda:632/905/1602 matmul/add fwd outputs; 6180/6253/6216 silu/mul/mul_scalar out; 6480 embedding out; 3410 gather out | fwd op outputs | **BF16-S1** | 6253 (`silu(gate)*up`) feeds down_proj matmul which already accepts bf16 operand |
| bcuda:3674/3725 q_t / transposed out; 3723 `out_f32` | attention layout bridges + prefill out | **BF16-S1** | 3723 re-imports kernel's bf16 out to f32 — bf16 store skips the widen |
| bcuda:3950/4002/3970/3974/3978/4018 LA fwd (preact/chunk_state/g/g_cumsum/beta/output) | LA retained ctx | **BF16-S1** | consumers = `cuda_slice` f32 gate → need bf16 read; `chunk_state` largest |
| softmax.rs:85 fwd `y` | softmax/log_softmax probs | **RED-LINE (defer)** | **B over A**: probs in bf16 is a numeric-precision flag; couples fwd-save + bwd-read (bcuda:3525/3463). Keep f32 until its own A/B. |

### Transient per-op emitted grads → BF16-S2

| file:line | role | disposition |
|---|---|---|
| bcuda:5371/5402/5446/5491/5618/5712/5729 matmul & matmul_bt grad_a/grad_b (2×2, batched, bt) | GEMM backward emitted grads | **BF16-S2** (round c→bf16 before wrapping; cuBLAS accumulate stays f32) |
| moe.rs:941/955 `grad_lora_a`/`grad_lora_b` per-expert | **primary OOM writeback target** | **BF16-S2** (write-once per step → safe) |
| moe.rs:916 grad_weights, 902 grad_packed_input, 333/438/566/578/1217/1229 MoE grads | MoE emitted grads (host Vec) | **NO-OP** (host path; S2 device-only) |
| bcuda:5836/5840/5844 grad_q/k/v (SDPA); 5926/8854/8972/8989 elementwise-bwd grads | emitted grads | **BF16-S2** |
| bcuda:9081 `d_grad` grad_x (RMSNorm) | main RMSNorm bwd grad | **BF16-S2**; 9114 grad_w [hidden] tiny → leave f32 |
| bcuda:4830-4525 LA bwd grads (dqkv/dconv/dz/db/da/ddt/da_log/dnorm) | LA emitted grads | **BF16-S2** |
| bcuda:6006/5975/3544/3482/9209/8745/7671/8551 reduce/softmax/rope/broadcast/layout bwd grads | rest-ops emitted grads | **BF16-S2** |
| bcuda:6047 `cuda_add_into_device` out (grad fuse); 7858 concat_axis2 | grad-merge output | **ACCUM-f32** (§4 — output stays f32; widen-on-read incoming bf16) |

### Persistent accumulators → ACCUM-f32 (never bf16)

| file:line | role | reason |
|---|---|---|
| tensor.rs:855/866/872 accumulate_grad target; tape.rs:974/982 merge_grad | persistent param .grad | swamping (§4): bf16 mantissa ~8-bit, running sum >> increment loses the tail |
| tensor.rs:945 fill_like seed grad | ones_like(loss) | chain origin; keep f32 |
| **fused_linear_distill.rs:586** grad_hidden_2d_accum + grad_weight_accum | **critic finding — uncited by mapper** | OPD production CE accumulators across chunks; ACCUM-f32 (grad_weight is tied-LM-head, ≥2× touched) |

### NO-OP (flag never applies)

Host `Vec<f32>` fallbacks (activation.rs / elementwise.rs / norm.rs:278/298 / moe.rs host
paths / reduce/softmax/broadcast/layout host grads / linear_attention.rs:1323/1771 /
backend.rs:314) — already dtype-safe via `readback` decode (bcuda:1243) but **demote to
host f32**, forfeiting the win; documented, not targeted. Views (broadcast.rs:175 alias,
reshape metadata) inherit upstream dtype. Rollout-only decode buffers
(bcuda:8034/8125/8284/8429) — gated behind `!tape.enabled` (attention.rs:246), never in
backward. Tiny buffers (bcuda:7116 reduce out [rows], 9114 grad_w [hidden], 6372 gamma
upload) — not worth the cast.

## 4. GRAD_OUT CONSUMER BRANCHES

Every consumer below reads its grad_out/activation through `cuda_slice` (:546) and
**hard-errors on `CudaBf16`**. Each needs a bf16-aware read: match `CudaBf16` →
`import_local_bf16_as_f32` (:448) widen to a scratch f32 slice → feed the existing f32
kernel. (Reuse `cuda_bf16_slice` :585 to grab the u16 slice, then widen.) A shared helper
is the low-entropy move (§5).

**Backward-device consumers needing a `CudaBf16` widen-on-read arm** (grad_out side):

| file:line | fn | arm |
|---|---|---|
| bcuda:5350 | `cuda_matmul_backward_device` | widen `d_g` (grad_out). **Also `a`/`b` @5348/5349** — `a` is the retained *activation*, equally f32-gated, no bf16 branch (unlike matmul_bt @5576). Both need widen for S1 activations to flow here (MoE uploads into this, moe.rs:2227). |
| bcuda:5559 | `cuda_matmul_bt_input_grad_device` | widen `d_g`; B-operand already bf16-capable (@5583) |
| bcuda:5647 | `cuda_matmul_bt_backward_device` | widen `d_g`; B already branches (@5676) |
| bcuda:5826 | `cuda_causal_sdpa_recompute_backward_device` | widen `upstream` (grad); q/k/v stay f32 island |
| bcuda:9045 | `cuda_rms_norm_backward_device` | widen `upstream`; x is retained bf16 (S1) → also widen here @9046 |
| bcuda:8844 | `cuda_elementwise_backward_with_saved` (silu/gelu/sigmoid/exp) | widen `upstream` + saved input/output |
| bcuda:8957/5916 | `cuda_mul_backward_device` / `cuda_mul_scalar_backward_device` | widen `upstream` (+ saved a/b for mul) |
| bcuda:8736 | `cuda_add_broadcast_backward_device` (grad_b path) | widen `upstream` |
| bcuda:5998/5954/3524/3462 | sum/mean/softmax/log_softmax bwd | widen `upstream` (+ retained `y` for softmax @3525/3463) |
| bcuda:9184 | `cuda_rope_backward_device` | widen `upstream` |
| bcuda:8527 | `cuda_slice_backward_device` | widen `upstream` |
| bcuda:6942 | `cuda_clip_grad_norm_device` | widen each grad; **@7037**: also allocates a *second* full-size f32 copy per grad (`out_slices`) — with bf16 grads, write the clipped output bf16 too, else clip re-inflates peak |
| bcuda:3198 | `cuda_adamw_step_device` | widen `grad`; param/m/v stay f32 (red line) |

**Forward-op gates that fire on grad-derived buffers (they fire BEFORE any `*_backward_device`):**

| file:line | fn | why it fires first |
|---|---|---|
| bcuda:2532 | `Backend::reshape` | `cuda_slice(x,"reshape")?` errors on bf16 before `x.clone()` @2541. sdpa-chunked bwd reshapes grad_out first (attention.rs:509). Needs a bf16 pass-through (return `x.clone()` for CudaBf16 after a len-check on the u16 slice). |
| bcuda:7751 | `Backend::slice` (fwd) | applied directly to grad_out in LoRA-tiled (matmul.rs:438) + sdpa-chunked (attention.rs:544). **27B LoRA writeback takes the tiled path** (matmul.rs:420) → on the primary chain. Needs bf16 slice kernel or widen-on-read. |
| bcuda:1494/1592 | fwd `matmul`/`add` | sdpa-chunked composes ~7 forward-op gates over grad buffers (attention.rs:563/566/572). Each needs the widen arm. |
| fused_linear_distill.rs:1140 | `scale_saved_grad` → `cuda_scalar_1d_device` (:6207) | scales saved bf16 grad_hidden/grad_weight; OPD hot path |

**add_into_device / merge_grad — the accumulation decision (critic's core concern):**

`add_into_device` (bcuda:6036/6037) reads *both* operands f32-only and cannot sum any bf16
combination. Decision:

- **Per-op emitted grads may be bf16 (write-once, safe).**
- **The persistent accumulation target stays f32.** Rule: **first write into a persistent
  grad may land bf16; the second touch (any `add_into`) promotes the accumulator to f32
  for the rest of the step.** Self-adjusting — write-once params (LoRA A/B) stay bf16 (the
  win); multi-touch params (tied [V,H], accumulated ≥2× per tensor.rs:828-832) promote to
  f32 (kills swamping). Implementation: `add_into_device` widens any bf16 incoming operand
  on read, **always outputs f32** `CudaStorage`; `accumulate_grad` None-branch
  (tensor.rs:872) keeps the clone's dtype, merge-branch (tensor.rs:855) produces f32.
- **No separate fp32 master-grad copy is required** — the persistent `.grad` *is* the f32
  master. The optimizer path (adamw @3198) reads f32 as today. `clip_grad_norm` @6942 runs
  first, touches every grad, and widens to f32 on its scaled output, so adamw always sees
  f32 → its bf16 arm is optional. Prefer widening at the clip boundary.

## 5. BUILD SURFACE

**Already built (reuse):**
- `local_f32_as_bf16` (bcuda:480) — f32→bf16 RNE store cast, returns bare `CudaSlice<u16>`.
- `import_local_bf16_as_f32` (bcuda:448) — bf16→f32 widen, the read-side primitive.
- `matmul_device_f32_bf16` (bcuda:812) — f32-lhs × bf16-rhs, fp32 accumulate.
- `cuda_bf16_slice` (bcuda:585) — grabs the `&CudaSlice<u16>`.
- `CudaBf16Storage` (backend.rs:133) + `DeviceHandle::CudaBf16` (:274) — no enum change needed.
- `cuda_row_slice`/`cuda_concat_rows` (bcuda:3786/3808) — **exact dual-dtype match-on-both-variants pattern to copy** for add/slice/matmul-output.
- `readback` (bcuda:1243) decodes CudaBf16→f32 → all host fallbacks are dtype-safe (but demote to host f32).

**Must be written (minimal set):**
1. **`f32_handle_to_bf16(handle) -> DeviceHandle::CudaBf16`** — wrap `local_f32_as_bf16`'s
   bare `CudaSlice<u16>` as `CudaBf16Storage::new` → `DeviceHandle::CudaBf16`. Every S1/S2
   store site calls this behind `tape_bf16()`.
2. **`cuda_slice_or_widen(handle, op) -> Cow<CudaSlice<f32>>`** — read-side helper:
   `Cuda`→borrow (as today); `CudaBf16`→`import_local_bf16_as_f32` into owned scratch f32.
   All ~20 consumers in §4 route through this instead of `cuda_slice`. One line per consumer.
3. **bf16-aware `add_into_device`** — widen bf16 `src`/`dest` on read, accumulate f32,
   output f32 (§4). No new kernel (reuse `add_into_f32` after widen); scratch f32 transient.
4. **bf16 pass-through in `reshape`** (bcuda:2532) — len-check the u16 slice, `Ok(x.clone())` for CudaBf16.
5. **bf16 `slice` fwd via widen-on-read** (bcuda:7751) — tiled LoRA path needs it on the primary chain. Prefer widen-on-read (no new kernel).

No new struct, no enum variant, no new `.cu` kernel required if widen-on-read is used
everywhere (the extra widen launch is the cost; a native bf16 elementwise kernel is a
later optimization, not needed for correctness).

## 6. STAGED LANDING ORDER

Each stage independently revertible (the store site is behind `tape_bf16()`; default fp32 =
untouched). No half-state: a stage lands its store sites *and* all consumer read-branches
together, or not at all.

- **Stage 0 — config plumbing (no behavior change). SHIPPED.** §2 additions. Default `fp32`
  byte-identical. Ships alone.
- **Stage 1a — frozen prompt-prefix K/V bf16 (the minimum first A/B).** One buffer:
  `PrefixKv.k/v` (qwen35.rs:84), frozen, no grad chain, consumer is the gen-segment flash
  path that already converts to bf16. Store via helper #1; gen-segment reads via helper #2
  (or feeds bf16 straight to the kernel). Gate: gradcheck N/A (frozen), needle gate +
  peak-VRAM A/B. **Smallest slice that proves the store cast + a real consumer + a VRAM delta.**
- **Stage 1b — forward retained activations bf16.** checkpoint.rs:40 keep-set +
  qwen35.rs:3869/1298/3725 + fwd op outputs. Lands helper #2 in: add (1592), rmsnorm
  fwd/bwd read of x (6355/9046), mul/silu, matmul `a`/`b` (5348), reshape (2532), slice
  (7751). **Also the checkpoint offload/reload self-heal** (bcuda:1044 reload always
  produces f32) — reload must re-quantize via `upload_bf16_bits` (:1050) when `tape_bf16()`,
  else dtype flips mid-tape. Gate: full gradcheck @ §7 tolerance + peak-VRAM.
- **Stage 2 — transient per-op emitted grads bf16.** GEMM/elementwise/LA/rest-ops emitted
  grads (§3) + the accumulate promotion rule (§4) + bf16-aware `add_into_device` + clip
  widen (7037). Excludes CE/logits grad + softmax probs (their own A/B, §8). This is where
  the ~20 backward read-branches land. Gate: gradcheck + needle + peak-VRAM at the 30K→target wall.
- **Stage 3 (deferred) — CE/logits grad + softmax probs bf16.** Highest VRAM win, highest
  numeric risk; own A/B against the correct-inference envelope. Only if Stage 2's ~2× is insufficient.

## 7. GATE

**Gradcheck tolerance — NOT the existing 1e-4.** bf16 rel-eps is ~4e-3 (7-8 bit mantissa).
The chunked-attention gradchecks assert 1e-4 (attention.rs:1189/1259/1263/1267/1308) — a
bf16 tape **false-fails** there. safetensors_io.rs:391 already uses 1e-2 rel-tol *because
bf16 is lossy*. Per CLAUDE.md, the gate for a dtype swap is **correct-inference, not
grad-exact-vs-f32**:
- Gradcheck rel-tol **1e-2** (matching the bf16 precedent), against the f32 reference — as a smoke, not the acceptance gate.
- **Acceptance = correct-inference:** `scripts/needle_gate.py` needle ladder ×3 same-config
  vs the f32 baseline envelope + self-consistency (the bf16 run's own autoregressive output
  as reference), NOT token/grad exactness (MoE non-determinism confounds it).
- **Per-step backward peak-VRAM A/B:** same-binary, same-shell, same-prompt, two
  `--tape-precision` values side-by-side; measure device peak at the writeback step;
  confirm the retained-activation + emitted-grad halving and the new max-token wall (30K→target).
- **All pod (CUDA can't build on Mac).** Mac pre-push typecheck only:
  `cargo check -p infer-api --release --no-default-features --features cuda,no-cuda --lib`.
  Bench entry per the bench spec, `pending-remote` stub if the pod A/B lags the commit.

## 8. OPEN RISKS

1. **add_into_device swamping (blocker → resolved by design).** Accumulator promotes to f32
   on second touch (§4); no fp32 master copy needed. **Resolution shipped in Stage 2.**
   Mitigated by the tied-[V,H] case being explicitly ≥2×-touched (tensor.rs:828-832).
2. **`a`-operand + reshape + fwd-slice gates (blocker → resolved).** The store-half breaks
   at matmul `a` (5348), reshape (2532), fwd slice (7751) — not just grad_out. **Resolution:**
   helper #2 routes all of them; Stage 1b lands them together. Do not trust a
   "reshape dtype-transparent" claim.
3. **CE/logits grad + softmax probs (deferred).** Biggest buffer, most CE-sensitive.
   **Explicit defer to Stage 3** with its own correct-inference A/B; never folded into S1/S2.
4. **Checkpoint offload/reload dtype flip (important → resolved).** Reload (bcuda:1044)
   always produces f32; `Tensor.data` is `Vec<f32>` (tensor.rs:31) with no dtype memory.
   **Resolution:** Stage 1b reload re-quantizes via `upload_bf16_bits` (:1050) under
   `tape_bf16()`. The offload→reload round-trip is lossy-then-requantized — acceptable (it
   was already an f32 activation), but must be A/B'd on the long-seq offload path (the exact OOM path).
5. **NCCL all_reduce dtype lockstep (deferred).** bcuda:1936 hardcodes `DType::F32`.
   **Defer** — TP grad all-reduce stays f32 in S1/S2. Revisit only if TP training needs the
   bf16 win on the reduce buffer.
6. **Host-fallback silent demotion (minor, accept).** Every host fallback (readback decode,
   bcuda:1243) accepts bf16 by demoting to host f32 — dtype-safe but forfeits the
   device-resident win. **Accept:** host fallbacks are a correctness floor, not a savings path.
7. **Metal no-op (minor, resolved).** backend_metal.rs:126 errors on `CudaBf16`.
   `tape_precision()` accessor is `#[cfg]`-gated (§2); the store decision is behind
   `tape_bf16()`, only consulted on CUDA. Flag doc states CUDA-only; on Metal it's an inert
   field. **Resolution shipped in Stage 0.**
8. **Host/device matmul-backward divergence (important → track).** The host-eager
   `cuda_matmul_backward` (bcuda:5148/5179/5225/5267) is a distinct path from the device
   sibling. **Resolution:** host path is NO-OP (already f32, demotes) — explicitly documented;
   not a divergence because the host path never stored bf16.
