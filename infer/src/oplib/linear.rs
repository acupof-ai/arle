//! Backend-neutral linear / GEMM dispatch **selection** (FlashInfer-style
//! `plan()`), relocated out of the CUDA launch path.
//!
//! This module owns the *selection* half of the linear operator: given a fully
//! host-side description of the weight matrix (its format + the precomputed
//! alignment predicates), the batch size, the dispatch phase, and the resolved
//! [`DispatchPolicy`](crate::dispatch_policy::DispatchPolicy), [`plan`] returns
//! the named [`LinearKernel`] that should run.
//!
//! ## The pure-`plan()` property
//!
//! [`plan`] is a pure function. It names **no** CUDA/cudarc type, touches no
//! device memory, launches no kernel, and reads only host-side metadata. The
//! consequence — and the headline of
//! [`docs/plans/backend-operator-library.md`](../../docs/plans/backend-operator-library.md) —
//! is that "is my kernel even selected for shape X?" becomes a GPU-free unit
//! test (`assert_eq!(plan(inputs, policy), Expected)`), runnable under the
//! crate's default feature set on a machine with no nvcc and no GPU.
//!
//! The CUDA launch path (`infer/src/ops/linear.rs`) extracts a
//! [`LinearPlanInputs`] off its `&DeviceMatrix` (computing the alignment bools
//! with the existing CUDA-side helpers), calls [`plan`], and dispatches the
//! returned [`LinearKernel`] to the matching kernel launch. The selection logic
//! lives here exactly once; the launch logic stays on the CUDA side.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::dispatch_policy::DispatchPolicy;

/// Backend-neutral mirror of the weight storage format the kernel ABI selects
/// on. This is the host-side selector consumed by [`plan`]; it carries no
/// device buffers and is available under the crate's default feature set
/// (unlike `cuda_kernels::tensor::WeightFormat`, which is CUDA-gated). The CUDA
/// dispatch site maps `cuda_kernels::tensor::WeightFormat` onto this 1:1.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightFormat {
    /// Dense row-major BF16 weights.
    #[default]
    DenseBf16,
    /// Uniform per-group signed INT8 weights with BF16 scales.
    W8A16,
    /// Uniform per-group packed INT4 weights with BF16 scales.
    W4A16,
    /// Marlin W4 weights with dynamic INT8 activations.
    MarlinW4A8,
    /// Uniform per-group packed INT2 weights with BF16 scales.
    W2A16,
    /// GGUF Q3_K packed superblocks.
    GgufQ3K,
    /// GGUF Q4_K packed superblocks.
    GgufQ4K,
    /// GGUF Q5_K packed superblocks.
    GgufQ5K,
    /// GGUF Q6_K packed superblocks.
    GgufQ6K,
    /// TurboQuant packed indices + FP16 group norms + Hadamard signs.
    TurboQuant,
    /// DeepSeek V4 row-major FP8 E4M3 weights with FP8 E8M0 block scales.
    Dsv4Fp8BlockScaled,
    /// DeepSeek V4 row-major packed FP4 E2M1 weights with FP8 E8M0 block scales.
    Dsv4Fp4BlockScaled,
}

/// Whether the linear is being dispatched on the decode or prefill path.
///
/// Backend-neutral mirror of the CUDA-side `ops::LinearDispatchPhase`; the CUDA
/// dispatch site maps the two variants 1:1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearPhase {
    Decode,
    Prefill,
}

/// The named dispatch artifact — which logical linear kernel `plan` selected.
///
/// Backend-neutral: it names the *logical* kernel, not a device function
/// pointer. The CUDA launch path maps each variant to an FFI symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinearKernel {
    Bf16Gemv,
    Bf16GraphsafeGemm,
    Bf16CublasGemm,
    W2A16Gemv,
    W4A16Gemv,
    W8A16Gemv,
    Dsv4Fp8Gemv,
    Dsv4Fp4Gemv,
    W2A16BatchGemv,
    W4A16BatchGemv,
    W8A16BatchGemv,
    Dsv4Fp8BatchGemv,
    Dsv4Fp4BatchGemv,
    Q3KGemv,
    Q4KGemv,
    Q5KGemv,
    Q6KGemv,
    Q3KBatchGemv,
    Q4KBatchGemv,
    Q5KBatchGemv,
    Q6KBatchGemv,
    Q3KDequantCublasGemm,
    Q4KDequantCublasGemm,
    Q5KDequantCublasGemm,
    Q6KDequantCublasGemm,
    MarlinW4Gemm,
    MarlinW4A8Gemm,
    MarlinW4Hybrid,
    /// PF8.4 — Prefill-only W4+FP8 Marlin GEMM dispatch.
    ///
    /// Opt-in via `INFER_MARLIN_W4_FP8_PREFILL=1`. Decode path keeps existing
    /// W4A8 (FP8 mma is the wrong lever for HBM-bound decode per
    /// `docs/research/2026-05-10-phase0a-decode-kill-architectural-implication.md`).
    MarlinW4FP8Prefill,
    TurboQuantGemv,
    TurboQuantDequantCublasGemm,
}

impl LinearKernel {
    /// Stable string name of the selected kernel — the Observe counter label
    /// and the value asserted by the legacy `linear_kernel_plan_for_test`
    /// harness. One source of names: the launch site no longer maintains a
    /// parallel `&str` map.
    #[must_use]
    pub fn kernel_label(self) -> &'static str {
        match self {
            Self::Bf16Gemv => "Bf16Gemv",
            Self::Bf16GraphsafeGemm => "Bf16GraphsafeGemm",
            Self::Bf16CublasGemm => "Bf16CublasGemm",
            Self::W2A16Gemv => "W2A16Gemv",
            Self::W4A16Gemv => "W4A16Gemv",
            Self::W8A16Gemv => "W8A16Gemv",
            Self::Dsv4Fp8Gemv => "Dsv4Fp8Gemv",
            Self::Dsv4Fp4Gemv => "Dsv4Fp4Gemv",
            Self::W2A16BatchGemv => "W2A16BatchGemv",
            Self::W4A16BatchGemv => "W4A16BatchGemv",
            Self::W8A16BatchGemv => "W8A16BatchGemv",
            Self::Dsv4Fp8BatchGemv => "Dsv4Fp8BatchGemv",
            Self::Dsv4Fp4BatchGemv => "Dsv4Fp4BatchGemv",
            Self::Q3KGemv => "Q3KGemv",
            Self::Q4KGemv => "Q4KGemv",
            Self::Q5KGemv => "Q5KGemv",
            Self::Q6KGemv => "Q6KGemv",
            Self::Q3KBatchGemv => "Q3KBatchGemv",
            Self::Q4KBatchGemv => "Q4KBatchGemv",
            Self::Q5KBatchGemv => "Q5KBatchGemv",
            Self::Q6KBatchGemv => "Q6KBatchGemv",
            Self::Q3KDequantCublasGemm => "Q3KDequantCublasGemm",
            Self::Q4KDequantCublasGemm => "Q4KDequantCublasGemm",
            Self::Q5KDequantCublasGemm => "Q5KDequantCublasGemm",
            Self::Q6KDequantCublasGemm => "Q6KDequantCublasGemm",
            Self::MarlinW4Gemm => "MarlinW4Gemm",
            Self::MarlinW4A8Gemm => "MarlinW4A8Gemm",
            Self::MarlinW4Hybrid => "MarlinW4Hybrid",
            Self::MarlinW4FP8Prefill => "MarlinW4FP8Prefill",
            Self::TurboQuantGemv => "TurboQuantGemv",
            Self::TurboQuantDequantCublasGemm => "TurboQuantDequantCublasGemm",
        }
    }

    /// The number of distinct `LinearKernel` variants — the size of the
    /// process-global [`LINEAR_KERNEL_FIRED`] counter array. Kept in lockstep
    /// with [`metric_index`](Self::metric_index): every variant maps to a
    /// distinct index in `0..VARIANT_COUNT`.
    pub const VARIANT_COUNT: usize = 31;

    /// O(1), branch-table mapping from variant to a distinct counter slot in
    /// `0..VARIANT_COUNT`. This is the index the lock-free dispatch counter uses
    /// — no string compare on the hot path. Aligned 1:1 with
    /// [`kernel_label`](Self::kernel_label) (same variant order); the unit test
    /// `metric_index_is_distinct_per_variant` asserts no two variants collide
    /// and that every index is in range.
    #[must_use]
    pub fn metric_index(self) -> usize {
        match self {
            Self::Bf16Gemv => 0,
            Self::Bf16GraphsafeGemm => 1,
            Self::Bf16CublasGemm => 2,
            Self::W2A16Gemv => 3,
            Self::W4A16Gemv => 4,
            Self::W8A16Gemv => 5,
            Self::Dsv4Fp8Gemv => 6,
            Self::Dsv4Fp4Gemv => 7,
            Self::W2A16BatchGemv => 8,
            Self::W4A16BatchGemv => 9,
            Self::W8A16BatchGemv => 10,
            Self::Dsv4Fp8BatchGemv => 11,
            Self::Dsv4Fp4BatchGemv => 12,
            Self::Q3KGemv => 13,
            Self::Q4KGemv => 14,
            Self::Q5KGemv => 15,
            Self::Q6KGemv => 16,
            Self::Q3KBatchGemv => 17,
            Self::Q4KBatchGemv => 18,
            Self::Q5KBatchGemv => 19,
            Self::Q6KBatchGemv => 20,
            Self::Q3KDequantCublasGemm => 21,
            Self::Q4KDequantCublasGemm => 22,
            Self::Q5KDequantCublasGemm => 23,
            Self::Q6KDequantCublasGemm => 24,
            Self::MarlinW4Gemm => 25,
            Self::MarlinW4A8Gemm => 26,
            Self::MarlinW4Hybrid => 27,
            Self::MarlinW4FP8Prefill => 28,
            Self::TurboQuantGemv => 29,
            Self::TurboQuantDequantCublasGemm => 30,
        }
    }

    /// Reconstruct the variant from its [`metric_index`](Self::metric_index).
    /// Used only by the cold render path ([`linear_kernel_fired_counts`]) to
    /// recover the `kernel_label` for each populated counter slot. The match is
    /// the inverse of `metric_index`; the unit test exercises the round-trip.
    #[must_use]
    fn from_metric_index(index: usize) -> Self {
        match index {
            0 => Self::Bf16Gemv,
            1 => Self::Bf16GraphsafeGemm,
            2 => Self::Bf16CublasGemm,
            3 => Self::W2A16Gemv,
            4 => Self::W4A16Gemv,
            5 => Self::W8A16Gemv,
            6 => Self::Dsv4Fp8Gemv,
            7 => Self::Dsv4Fp4Gemv,
            8 => Self::W2A16BatchGemv,
            9 => Self::W4A16BatchGemv,
            10 => Self::W8A16BatchGemv,
            11 => Self::Dsv4Fp8BatchGemv,
            12 => Self::Dsv4Fp4BatchGemv,
            13 => Self::Q3KGemv,
            14 => Self::Q4KGemv,
            15 => Self::Q5KGemv,
            16 => Self::Q6KGemv,
            17 => Self::Q3KBatchGemv,
            18 => Self::Q4KBatchGemv,
            19 => Self::Q5KBatchGemv,
            20 => Self::Q6KBatchGemv,
            21 => Self::Q3KDequantCublasGemm,
            22 => Self::Q4KDequantCublasGemm,
            23 => Self::Q5KDequantCublasGemm,
            24 => Self::Q6KDequantCublasGemm,
            25 => Self::MarlinW4Gemm,
            26 => Self::MarlinW4A8Gemm,
            27 => Self::MarlinW4Hybrid,
            28 => Self::MarlinW4FP8Prefill,
            29 => Self::TurboQuantGemv,
            30 => Self::TurboQuantDequantCublasGemm,
            other => unreachable!(
                "metric_index {other} out of range 0..{}",
                Self::VARIANT_COUNT
            ),
        }
    }
}

/// Process-global, lock-free per-variant "this GEMM kernel actually FIRED"
/// counter — the Observe gate of GPU-dispatch governance Phase 1
/// (`docs/plans/gpu-dispatch-governance.md`). One `AtomicU64` slot per
/// [`LinearKernel`] variant, indexed by [`LinearKernel::metric_index`].
///
/// Backend-neutral: no CUDA / MLX / metrics type is named here. The CUDA launch
/// path calls [`record_linear_kernel`] at the dispatch site (the *fired* fact,
/// not the *selected* fact — `plan()` stays uncounted); the cold `/v1/stats`
/// render path reads [`linear_kernel_fired_counts`].
static LINEAR_KERNEL_FIRED: [AtomicU64; LinearKernel::VARIANT_COUNT] =
    [const { AtomicU64::new(0) }; LinearKernel::VARIANT_COUNT];

/// Record that `k` was the kernel actually dispatched (launched). Hot path:
/// exactly one `Relaxed` `fetch_add` indexed by [`LinearKernel::metric_index`]
/// — no Mutex / RwLock / HashMap, no allocation, no string compare. Called once
/// per real dispatch from the CUDA launch site.
#[inline]
pub fn record_linear_kernel(k: LinearKernel) {
    LINEAR_KERNEL_FIRED[k.metric_index()].fetch_add(1, Ordering::Relaxed);
}

/// Snapshot the fired-counts as `(kernel_label, count)` pairs, **only for
/// variants with count > 0** (a `/v1/stats` scrape stays small — quiet kernels
/// emit no line). Cold path: allocates a `Vec`, reads each slot with `Relaxed`.
/// The render block (`metrics/render.rs`) iterates this to emit
/// `infer_dispatch_kernel_total{op="linear",variant="<label>"}` lines.
#[must_use]
pub fn linear_kernel_fired_counts() -> Vec<(&'static str, u64)> {
    LINEAR_KERNEL_FIRED
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| {
            let count = slot.load(Ordering::Relaxed);
            (count > 0).then(|| (LinearKernel::from_metric_index(index).kernel_label(), count))
        })
        .collect()
}

/// Host-side description of a linear dispatch, holding every predicate the
/// CUDA resolver used to read directly off the `&DeviceMatrix` / env.
///
/// The alignment booleans (`*_aligned`) are the `is_ok()` results of the
/// CUDA-side `marlin_prefill_aligned` / `hybrid_w4a8_aligned` /
/// `marlin_w4a8_aligned` / `hybrid_w4_fp8_aligned` helpers, precomputed on the
/// CUDA dispatch side and passed in so [`plan`] stays free of device types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearPlanInputs {
    /// The weight storage format (`weight.weight_format()`).
    pub weight_format: WeightFormat,
    /// Number of rows / tokens in the batch (decode `gemv` uses `1`; batched
    /// GEMM uses `x.seq_len`).
    pub batch: usize,
    /// Decode vs prefill phase.
    pub phase: LinearPhase,
    /// `weight.is_hybrid_w4_marlin()`.
    pub is_hybrid_w4_marlin: bool,
    /// `weight.has_marlin()`.
    pub has_marlin: bool,
    /// `marlin_prefill_aligned(weight).is_ok()`.
    pub marlin_prefill_aligned: bool,
    /// `hybrid_w4a8_aligned(weight).is_ok()`.
    pub hybrid_w4a8_aligned: bool,
    /// `marlin_w4a8_aligned(weight).is_ok()`.
    pub marlin_w4a8_aligned: bool,
    /// `hybrid_w4_fp8_aligned(weight).is_ok()`.
    pub hybrid_w4_fp8_aligned: bool,
}

/// PURE. Select the linear kernel for `inputs` under `policy`.
///
/// This is the relocated body of the former `LinearKernelPlan::{batched,decode}`
/// resolvers — behavior-preserving and bit-identical, proven by the
/// plan-equivalence test below. No device memory is touched and no CUDA type is
/// named, so this runs on CPU under the default feature set.
///
/// The decode path (single-token `gemv`) is `batch == 1` with
/// [`LinearPhase::Decode`]; the batched GEMM path passes `x.seq_len` and the
/// phase the caller is on. The `batch == 1` GEMM case falls through to the
/// decode resolver exactly as the legacy `batched` match did.
#[must_use]
pub fn plan(inputs: &LinearPlanInputs, policy: &DispatchPolicy) -> LinearKernel {
    let LinearPlanInputs {
        weight_format,
        batch,
        phase,
        is_hybrid_w4_marlin,
        // `has_marlin` is part of the documented input contract (computed on the
        // CUDA side and used there for the loud Marlin-fallback trace), but the
        // selection itself does not branch on it.
        has_marlin: _,
        marlin_prefill_aligned,
        hybrid_w4a8_aligned,
        marlin_w4a8_aligned,
        hybrid_w4_fp8_aligned,
    } = *inputs;

    // PF8.4 — opt-in W4+FP8 prefill dispatch (decode keeps W4+INT8).
    if phase == LinearPhase::Prefill
        && batch > 1
        && policy.marlin_w4_fp8_prefill
        && hybrid_w4_fp8_aligned
    {
        return LinearKernel::MarlinW4FP8Prefill;
    }
    if is_hybrid_w4_marlin {
        if phase == LinearPhase::Prefill
            && batch > 1
            && policy.hybrid_w4a8_prefill
            && hybrid_w4a8_aligned
        {
            return LinearKernel::MarlinW4Hybrid;
        }
        if marlin_prefill_aligned {
            return LinearKernel::MarlinW4Gemm;
        }
    }
    if marlin_w4a8_aligned {
        return LinearKernel::MarlinW4A8Gemm;
    }
    // M_quant Round 4 #6: env-gated override to prefer W4A16BatchGemv (BF16-native,
    // 1 launch) over MarlinW4Gemm (3 launches) ONLY for decode-batched (batch ∈ 2..=8).
    // Prefill (batch > 8 = seq_len > 8) always uses Marlin per Round 1 baseline.
    // See docs/research/2026-05-09-eod106-r4-6-bench-preliminary-solid-gap.md.
    if batch > 1 && marlin_prefill_aligned && !(batch <= 8 && policy.r4_w4a16_gemv_override) {
        return LinearKernel::MarlinW4Gemm;
    }

    match (batch, weight_format) {
        (1, WeightFormat::DenseBf16) => LinearKernel::Bf16GraphsafeGemm,
        (_, WeightFormat::DenseBf16) => LinearKernel::Bf16CublasGemm,
        (1, _) => plan_decode(weight_format, is_hybrid_w4_marlin),
        (_, WeightFormat::W2A16) => LinearKernel::W2A16BatchGemv,
        (_, WeightFormat::W4A16) => LinearKernel::W4A16BatchGemv,
        (_, WeightFormat::W8A16) => LinearKernel::W8A16BatchGemv,
        (_, WeightFormat::Dsv4Fp8BlockScaled) => LinearKernel::Dsv4Fp8BatchGemv,
        (_, WeightFormat::Dsv4Fp4BlockScaled) => LinearKernel::Dsv4Fp4BatchGemv,
        (2..=8, WeightFormat::GgufQ3K) => LinearKernel::Q3KBatchGemv,
        (2..=8, WeightFormat::GgufQ4K) => LinearKernel::Q4KBatchGemv,
        (2..=8, WeightFormat::GgufQ5K) => LinearKernel::Q5KBatchGemv,
        (2..=8, WeightFormat::GgufQ6K) => LinearKernel::Q6KBatchGemv,
        (_, WeightFormat::GgufQ3K) => LinearKernel::Q3KDequantCublasGemm,
        (_, WeightFormat::GgufQ4K) => LinearKernel::Q4KDequantCublasGemm,
        (_, WeightFormat::GgufQ5K) => LinearKernel::Q5KDequantCublasGemm,
        (_, WeightFormat::GgufQ6K) => LinearKernel::Q6KDequantCublasGemm,
        (_, WeightFormat::MarlinW4A8) => LinearKernel::MarlinW4A8Gemm,
        (_, WeightFormat::TurboQuant) => LinearKernel::TurboQuantDequantCublasGemm,
    }
}

/// PURE. Single-token (decode) selection — the relocated body of the former
/// `LinearKernelPlan::decode`. Exposed via the `batch == 1` fall-through in
/// [`plan`]; the CUDA `gemv` path constructs `LinearPlanInputs` with
/// `batch == 1` + `LinearPhase::Decode` and reads back this same result.
#[must_use]
fn plan_decode(weight_format: WeightFormat, is_hybrid_w4_marlin: bool) -> LinearKernel {
    if is_hybrid_w4_marlin {
        return LinearKernel::MarlinW4Gemm;
    }
    match weight_format {
        WeightFormat::DenseBf16 => LinearKernel::Bf16Gemv,
        WeightFormat::W2A16 => LinearKernel::W2A16Gemv,
        WeightFormat::W4A16 => LinearKernel::W4A16Gemv,
        WeightFormat::W8A16 => LinearKernel::W8A16Gemv,
        WeightFormat::Dsv4Fp8BlockScaled => LinearKernel::Dsv4Fp8Gemv,
        WeightFormat::Dsv4Fp4BlockScaled => LinearKernel::Dsv4Fp4Gemv,
        WeightFormat::GgufQ3K => LinearKernel::Q3KGemv,
        WeightFormat::GgufQ4K => LinearKernel::Q4KGemv,
        WeightFormat::GgufQ5K => LinearKernel::Q5KGemv,
        WeightFormat::GgufQ6K => LinearKernel::Q6KGemv,
        WeightFormat::MarlinW4A8 => LinearKernel::MarlinW4A8Gemm,
        WeightFormat::TurboQuant => LinearKernel::TurboQuantGemv,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_policy::DispatchPolicy;

    /// Reference reimplementation of the *pre-relocation* resolver, expressed
    /// directly in terms of the host-side predicates. Mirrors the legacy
    /// `LinearKernelPlan::{batched,decode}` bodies line-for-line so the
    /// equivalence sweep proves [`plan`] selects bit-identically. If [`plan`]
    /// ever grows logic the old `match` did not have, this oracle diverges and
    /// the sweep goes red — the deletion-style behavior gate.
    fn legacy_oracle(inputs: &LinearPlanInputs, policy: &DispatchPolicy) -> LinearKernel {
        let batch = inputs.batch;
        let phase = inputs.phase;
        let wf = inputs.weight_format;

        // --- legacy LinearKernelPlan::batched ---
        if phase == LinearPhase::Prefill
            && batch > 1
            && policy.marlin_w4_fp8_prefill
            && inputs.hybrid_w4_fp8_aligned
        {
            return LinearKernel::MarlinW4FP8Prefill;
        }
        if inputs.is_hybrid_w4_marlin {
            if phase == LinearPhase::Prefill
                && batch > 1
                && policy.hybrid_w4a8_prefill
                && inputs.hybrid_w4a8_aligned
            {
                return LinearKernel::MarlinW4Hybrid;
            }
            if inputs.marlin_prefill_aligned {
                return LinearKernel::MarlinW4Gemm;
            }
        }
        if inputs.marlin_w4a8_aligned {
            return LinearKernel::MarlinW4A8Gemm;
        }
        if batch > 1
            && inputs.marlin_prefill_aligned
            && !(batch <= 8 && policy.r4_w4a16_gemv_override)
        {
            return LinearKernel::MarlinW4Gemm;
        }

        // --- legacy match (batch, weight_format), with batch==1 → decode ---
        match (batch, wf) {
            (1, WeightFormat::DenseBf16) => LinearKernel::Bf16GraphsafeGemm,
            (_, WeightFormat::DenseBf16) => LinearKernel::Bf16CublasGemm,
            (1, _) => legacy_decode_oracle(wf, inputs.is_hybrid_w4_marlin),
            (_, WeightFormat::W2A16) => LinearKernel::W2A16BatchGemv,
            (_, WeightFormat::W4A16) => LinearKernel::W4A16BatchGemv,
            (_, WeightFormat::W8A16) => LinearKernel::W8A16BatchGemv,
            (_, WeightFormat::Dsv4Fp8BlockScaled) => LinearKernel::Dsv4Fp8BatchGemv,
            (_, WeightFormat::Dsv4Fp4BlockScaled) => LinearKernel::Dsv4Fp4BatchGemv,
            (2..=8, WeightFormat::GgufQ3K) => LinearKernel::Q3KBatchGemv,
            (2..=8, WeightFormat::GgufQ4K) => LinearKernel::Q4KBatchGemv,
            (2..=8, WeightFormat::GgufQ5K) => LinearKernel::Q5KBatchGemv,
            (2..=8, WeightFormat::GgufQ6K) => LinearKernel::Q6KBatchGemv,
            (_, WeightFormat::GgufQ3K) => LinearKernel::Q3KDequantCublasGemm,
            (_, WeightFormat::GgufQ4K) => LinearKernel::Q4KDequantCublasGemm,
            (_, WeightFormat::GgufQ5K) => LinearKernel::Q5KDequantCublasGemm,
            (_, WeightFormat::GgufQ6K) => LinearKernel::Q6KDequantCublasGemm,
            (_, WeightFormat::MarlinW4A8) => LinearKernel::MarlinW4A8Gemm,
            (_, WeightFormat::TurboQuant) => LinearKernel::TurboQuantDequantCublasGemm,
        }
    }

    fn legacy_decode_oracle(wf: WeightFormat, is_hybrid_w4_marlin: bool) -> LinearKernel {
        if is_hybrid_w4_marlin {
            return LinearKernel::MarlinW4Gemm;
        }
        match wf {
            WeightFormat::DenseBf16 => LinearKernel::Bf16Gemv,
            WeightFormat::W2A16 => LinearKernel::W2A16Gemv,
            WeightFormat::W4A16 => LinearKernel::W4A16Gemv,
            WeightFormat::W8A16 => LinearKernel::W8A16Gemv,
            WeightFormat::Dsv4Fp8BlockScaled => LinearKernel::Dsv4Fp8Gemv,
            WeightFormat::Dsv4Fp4BlockScaled => LinearKernel::Dsv4Fp4Gemv,
            WeightFormat::GgufQ3K => LinearKernel::Q3KGemv,
            WeightFormat::GgufQ4K => LinearKernel::Q4KGemv,
            WeightFormat::GgufQ5K => LinearKernel::Q5KGemv,
            WeightFormat::GgufQ6K => LinearKernel::Q6KGemv,
            WeightFormat::MarlinW4A8 => LinearKernel::MarlinW4A8Gemm,
            WeightFormat::TurboQuant => LinearKernel::TurboQuantGemv,
        }
    }

    const ALL_FORMATS: [WeightFormat; 12] = [
        WeightFormat::DenseBf16,
        WeightFormat::W8A16,
        WeightFormat::W4A16,
        WeightFormat::MarlinW4A8,
        WeightFormat::W2A16,
        WeightFormat::GgufQ3K,
        WeightFormat::GgufQ4K,
        WeightFormat::GgufQ5K,
        WeightFormat::GgufQ6K,
        WeightFormat::TurboQuant,
        WeightFormat::Dsv4Fp8BlockScaled,
        WeightFormat::Dsv4Fp4BlockScaled,
    ];

    /// Every `DispatchPolicy` combination [`plan`] reads: the three Marlin/W4
    /// prefill knobs plus the decode-batched override. The remaining policy
    /// fields are not consulted by linear selection, so they are held at their
    /// defaults.
    fn policy_permutations() -> Vec<DispatchPolicy> {
        let base = DispatchPolicy {
            marlin_w4_fp8_prefill: false,
            hybrid_w4a8_prefill: false,
            marlin_w4a8_autoconfig: false,
            r4_w4a16_gemv_override: false,
            deterministic_gemm: false,
            tilelang_bf16_split_kv: false,
            prefill_graph: false,
            bypass_tilelang_prefill: false,
            dsv4_grouped_gemm_m_threshold: 4,
        };
        let mut out = Vec::new();
        for fp8 in [false, true] {
            for hybrid in [false, true] {
                for r4 in [false, true] {
                    out.push(DispatchPolicy {
                        marlin_w4_fp8_prefill: fp8,
                        hybrid_w4a8_prefill: hybrid,
                        r4_w4a16_gemv_override: r4,
                        ..base
                    });
                }
            }
        }
        out
    }

    /// The headline property: over a representative sweep of
    /// `weight_format × batch × phase × alignment-bools × policy-flags`,
    /// [`plan`] returns exactly what the pre-relocation resolver returned.
    /// Pure, CPU-only, runs under the default (no-CUDA) feature set.
    #[test]
    fn plan_matches_legacy_resolver_over_full_sweep() {
        for &weight_format in &ALL_FORMATS {
            for &batch in &[1usize, 2, 4, 8, 16] {
                for &phase in &[LinearPhase::Decode, LinearPhase::Prefill] {
                    for &is_hybrid_w4_marlin in &[false, true] {
                        for &has_marlin in &[false, true] {
                            for &marlin_prefill_aligned in &[false, true] {
                                for &hybrid_w4a8_aligned in &[false, true] {
                                    for &marlin_w4a8_aligned in &[false, true] {
                                        for &hybrid_w4_fp8_aligned in &[false, true] {
                                            let inputs = LinearPlanInputs {
                                                weight_format,
                                                batch,
                                                phase,
                                                is_hybrid_w4_marlin,
                                                has_marlin,
                                                marlin_prefill_aligned,
                                                hybrid_w4a8_aligned,
                                                marlin_w4a8_aligned,
                                                hybrid_w4_fp8_aligned,
                                            };
                                            for policy in policy_permutations() {
                                                assert_eq!(
                                                    plan(&inputs, &policy),
                                                    legacy_oracle(&inputs, &policy),
                                                    "plan diverged from legacy resolver for {inputs:?} under {policy:?}"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Spot-check the documented-expected kernel for the canonical shapes, so a
    /// regression in the oracle itself can't silently mask a `plan` regression.
    #[test]
    fn plan_documented_expected_for_canonical_shapes() {
        let default_policy = DispatchPolicy {
            marlin_w4_fp8_prefill: false,
            hybrid_w4a8_prefill: false,
            marlin_w4a8_autoconfig: false,
            r4_w4a16_gemv_override: false,
            deterministic_gemm: false,
            tilelang_bf16_split_kv: false,
            prefill_graph: false,
            bypass_tilelang_prefill: false,
            dsv4_grouped_gemm_m_threshold: 4,
        };
        let unaligned = |weight_format, batch, phase| LinearPlanInputs {
            weight_format,
            batch,
            phase,
            is_hybrid_w4_marlin: false,
            has_marlin: false,
            marlin_prefill_aligned: false,
            hybrid_w4a8_aligned: false,
            marlin_w4a8_aligned: false,
            hybrid_w4_fp8_aligned: false,
        };

        // Dense BF16: decode GEMV, batch-1 graph-safe GEMM, batch-N cuBLAS GEMM.
        assert_eq!(
            plan(
                &unaligned(WeightFormat::DenseBf16, 1, LinearPhase::Decode),
                &default_policy
            ),
            LinearKernel::Bf16GraphsafeGemm
        );
        assert_eq!(
            plan(
                &unaligned(WeightFormat::DenseBf16, 8, LinearPhase::Prefill),
                &default_policy
            ),
            LinearKernel::Bf16CublasGemm
        );

        // W4A16 batched → W4A16BatchGemv when not Marlin-aligned.
        assert_eq!(
            plan(
                &unaligned(WeightFormat::W4A16, 4, LinearPhase::Decode),
                &default_policy
            ),
            LinearKernel::W4A16BatchGemv
        );

        // GGUF Q4_K: small batch → batch GEMV; large batch → dequant+cuBLAS.
        assert_eq!(
            plan(
                &unaligned(WeightFormat::GgufQ4K, 4, LinearPhase::Decode),
                &default_policy
            ),
            LinearKernel::Q4KBatchGemv
        );
        assert_eq!(
            plan(
                &unaligned(WeightFormat::GgufQ4K, 16, LinearPhase::Prefill),
                &default_policy
            ),
            LinearKernel::Q4KDequantCublasGemm
        );

        // Marlin-aligned W4A16: batched Marlin GEMM; r4 override flips the
        // decode-batched window (2..=8) to W4A16BatchGemv.
        let marlin_aligned = LinearPlanInputs {
            weight_format: WeightFormat::W4A16,
            batch: 4,
            phase: LinearPhase::Decode,
            marlin_prefill_aligned: true,
            has_marlin: true,
            ..unaligned(WeightFormat::W4A16, 4, LinearPhase::Decode)
        };
        assert_eq!(
            plan(&marlin_aligned, &default_policy),
            LinearKernel::MarlinW4Gemm
        );
        let r4_policy = DispatchPolicy {
            r4_w4a16_gemv_override: true,
            ..default_policy
        };
        assert_eq!(
            plan(&marlin_aligned, &r4_policy),
            LinearKernel::W4A16BatchGemv
        );
        // ...but prefill (batch > 8) ignores the override.
        let marlin_aligned_prefill = LinearPlanInputs {
            batch: 16,
            phase: LinearPhase::Prefill,
            ..marlin_aligned
        };
        assert_eq!(
            plan(&marlin_aligned_prefill, &r4_policy),
            LinearKernel::MarlinW4Gemm
        );
    }

    #[test]
    fn kernel_label_round_trips_through_every_variant() {
        // Every variant has a non-empty stable label.
        for &weight_format in &ALL_FORMATS {
            for &batch in &[1usize, 4, 16] {
                let inputs = LinearPlanInputs {
                    weight_format,
                    batch,
                    phase: LinearPhase::Prefill,
                    is_hybrid_w4_marlin: false,
                    has_marlin: false,
                    marlin_prefill_aligned: false,
                    hybrid_w4a8_aligned: false,
                    marlin_w4a8_aligned: false,
                    hybrid_w4_fp8_aligned: false,
                };
                let policy = DispatchPolicy {
                    marlin_w4_fp8_prefill: false,
                    hybrid_w4a8_prefill: false,
                    marlin_w4a8_autoconfig: false,
                    r4_w4a16_gemv_override: false,
                    deterministic_gemm: false,
                    tilelang_bf16_split_kv: false,
                    prefill_graph: false,
                    bypass_tilelang_prefill: false,
                    dsv4_grouped_gemm_m_threshold: 4,
                };
                assert!(!plan(&inputs, &policy).kernel_label().is_empty());
            }
        }
    }

    /// Every variant that `plan` can return — used to exercise `metric_index`
    /// distinctness and the round-trip through `from_metric_index`. Listed by
    /// hand (not derived) so adding an enum variant without extending the
    /// counter contract makes this list, and the asserts below, go stale loudly.
    const ALL_KERNELS: [LinearKernel; LinearKernel::VARIANT_COUNT] = [
        LinearKernel::Bf16Gemv,
        LinearKernel::Bf16GraphsafeGemm,
        LinearKernel::Bf16CublasGemm,
        LinearKernel::W2A16Gemv,
        LinearKernel::W4A16Gemv,
        LinearKernel::W8A16Gemv,
        LinearKernel::Dsv4Fp8Gemv,
        LinearKernel::Dsv4Fp4Gemv,
        LinearKernel::W2A16BatchGemv,
        LinearKernel::W4A16BatchGemv,
        LinearKernel::W8A16BatchGemv,
        LinearKernel::Dsv4Fp8BatchGemv,
        LinearKernel::Dsv4Fp4BatchGemv,
        LinearKernel::Q3KGemv,
        LinearKernel::Q4KGemv,
        LinearKernel::Q5KGemv,
        LinearKernel::Q6KGemv,
        LinearKernel::Q3KBatchGemv,
        LinearKernel::Q4KBatchGemv,
        LinearKernel::Q5KBatchGemv,
        LinearKernel::Q6KBatchGemv,
        LinearKernel::Q3KDequantCublasGemm,
        LinearKernel::Q4KDequantCublasGemm,
        LinearKernel::Q5KDequantCublasGemm,
        LinearKernel::Q6KDequantCublasGemm,
        LinearKernel::MarlinW4Gemm,
        LinearKernel::MarlinW4A8Gemm,
        LinearKernel::MarlinW4Hybrid,
        LinearKernel::MarlinW4FP8Prefill,
        LinearKernel::TurboQuantGemv,
        LinearKernel::TurboQuantDequantCublasGemm,
    ];

    /// `metric_index` is a bijection onto `0..VARIANT_COUNT`: every variant maps
    /// to a distinct in-range slot, `VARIANT_COUNT` is exactly the number of
    /// variants, and `from_metric_index` round-trips. A new variant that forgot
    /// to extend `metric_index` / `from_metric_index` / `VARIANT_COUNT` trips
    /// one of these asserts (or fails to compile against `ALL_KERNELS`).
    #[test]
    fn metric_index_is_distinct_per_variant() {
        let mut seen = [false; LinearKernel::VARIANT_COUNT];
        for &k in &ALL_KERNELS {
            let idx = k.metric_index();
            assert!(
                idx < LinearKernel::VARIANT_COUNT,
                "metric_index {idx} for {k:?} out of range 0..{}",
                LinearKernel::VARIANT_COUNT
            );
            assert!(
                !seen[idx],
                "metric_index {idx} collides — {k:?} shares a counter slot with another variant"
            );
            seen[idx] = true;
            // Round-trips back to the same label (the cold render path relies on this).
            assert_eq!(
                LinearKernel::from_metric_index(idx).kernel_label(),
                k.kernel_label()
            );
        }
        assert!(
            seen.iter().all(|&hit| hit),
            "VARIANT_COUNT={} exceeds the number of distinct metric_index values produced",
            LinearKernel::VARIANT_COUNT
        );
    }

    /// Recording a kernel bumps exactly its own counter by one. `LINEAR_KERNEL_FIRED`
    /// is process-global, so this asserts on before→after deltas (other tests in
    /// the binary may also record) and confirms only the chosen slots move.
    #[test]
    fn record_linear_kernel_increments_only_its_slot() {
        // Three distinct variants; record Bf16Gemv twice so the delta is 2.
        let recorded = [
            (LinearKernel::Bf16Gemv, 2u64),
            (LinearKernel::W4A16BatchGemv, 1),
            (LinearKernel::MarlinW4FP8Prefill, 3),
        ];

        let before: Vec<u64> = LINEAR_KERNEL_FIRED
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .collect();

        for &(k, times) in &recorded {
            for _ in 0..times {
                record_linear_kernel(k);
            }
        }

        // Per-slot deltas equal exactly what we recorded; untouched slots unchanged.
        let mut expected_delta = [0u64; LinearKernel::VARIANT_COUNT];
        for &(k, times) in &recorded {
            expected_delta[k.metric_index()] += times;
        }
        for (idx, slot) in LINEAR_KERNEL_FIRED.iter().enumerate() {
            let delta = slot.load(Ordering::Relaxed) - before[idx];
            assert_eq!(
                delta,
                expected_delta[idx],
                "slot {idx} ({}) moved by {delta}, expected {}",
                LinearKernel::from_metric_index(idx).kernel_label(),
                expected_delta[idx]
            );
        }

        // The cold reader surfaces our recorded variants with count > 0 and the
        // correct labels. (It returns only nonzero slots, so each must appear.)
        let counts = linear_kernel_fired_counts();
        for &(k, _) in &recorded {
            let label = k.kernel_label();
            let (_, count) = counts
                .iter()
                .find(|(name, _)| *name == label)
                .unwrap_or_else(|| panic!("{label} missing from linear_kernel_fired_counts()"));
            assert!(*count > 0, "{label} count should be > 0 after recording");
        }
    }
}
