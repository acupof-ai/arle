//! Model architectures (`ModelArch` impls) — R3 track.
//!
//! Scope of this crate at R2:
//! - [`NoopCommunicator`] is a no-op [`infer_seam::Communicator`] over a host
//!   `Vec<f32>` tensor. It exists so [`SkeletonModel`] has a concrete `Comm`
//!   associated type and the [`infer_seam::ModelArch`] seam is provably
//!   implementable without any backend collective library wired up.
//! - [`SkeletonModel`] is a **documented placeholder** [`infer_seam::ModelArch`]
//!   whose `forward` returns zeroed logits sized to a small vocab. It proves the
//!   seam type relationships hold (`Comm: Communicator<Tensor = Self::Tensor>`)
//!   and sets the R3 template.
//!
//! R3 fills this crate with the real `infer_seam::ModelArch` implementations for
//! the Qwen3 / Qwen3.5 / DeepSeek-V4 families, written against `Communicator`
//! (TP/EP collectives) and `KvPool`. The numerical forward math is **PORTED**
//! from the existing `infer/src/model/*` (not re-derived) and gated by the
//! parity suite (`kv_precision_parity` / `greedy_consistency` / `e2e`) per
//! [`docs/projects/2026-06-03-rewrite-verification-targets.md`].
//!
//! Depends only on the stable `infer-plan` + `infer-seam` contracts (+ the
//! `*-spec` config crates when the real ports land).

use infer_plan::ForwardPlan;
use infer_seam::{Communicator, KvPool, ModelArch};

/// Vocab size used by the [`SkeletonModel`] placeholder forward.
///
/// Deliberately tiny — this is not a real model vocabulary, only enough to give
/// the zeroed-logits return a concrete, non-empty shape so callers can exercise
/// the seam. R3 replaces this with the per-model config vocab size.
const SKELETON_VOCAB: usize = 8;

/// No-op communicator over a host `Vec<f32>` tensor.
///
/// Every collective is a deliberate no-op: there is no tensor-parallel,
/// expert-parallel, or pipeline group in the R2 skeleton. It exists solely to
/// satisfy [`SkeletonModel`]'s `Comm` associated type and to prove the
/// `Communicator<Tensor = Self::Tensor>` bound on [`ModelArch`] is satisfiable
/// with a plain host tensor.
///
/// R3 replaces this with the backend collective implementations (NCCL on CUDA,
/// MLX/Metal collectives, DeepEP all-to-all for DSv4 EP).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCommunicator;

impl Communicator for NoopCommunicator {
    type Tensor = Vec<f32>;

    fn all_reduce(&self, _tensor: &mut Self::Tensor) {
        // No-op: single-rank skeleton has no TP group to reduce across.
    }

    fn all_to_all(&self, _send: &Self::Tensor, _recv: &mut Self::Tensor) {
        // No-op: single-rank skeleton has no EP group to dispatch/combine across.
    }

    fn send_recv(&self, _stage: u32, _tensor: &mut Self::Tensor) {
        // No-op: single-stage skeleton has no pipeline peer to exchange with.
    }
}

/// Documented placeholder [`ModelArch`] for the R3 model-port track.
///
/// This is **not** a numerically correct model. Its [`forward`](SkeletonModel::forward)
/// returns a zero-filled logits vector of length [`SKELETON_VOCAB`], regardless
/// of the plan contents. Its only job at R2 is to prove that the
/// [`infer_seam::ModelArch`] seam — including the associated-type web
/// (`Tensor` / `Logits` / `Comm: Communicator<Tensor = Tensor>`) and the
/// `forward(&plan, &mut dyn KvPool, &comm)` signature — is implementable, and to
/// fix the call shape R3 ports build against.
///
/// R3 replaces the body with the real Qwen3 / Qwen3.5 / DeepSeek-V4 forward math
/// PORTED from `infer/src/model/*`, reading KV via `kv.page_indices(slot)` and
/// driving real collectives through `comm`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkeletonModel;

impl SkeletonModel {
    /// Build a skeleton model.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ModelArch for SkeletonModel {
    type Tensor = Vec<f32>;
    type Logits = Vec<f32>;
    type Comm = NoopCommunicator;

    /// PLACEHOLDER forward — returns zeroed logits sized to [`SKELETON_VOCAB`].
    ///
    /// TODO(R3): replace with the ported Qwen3 / Qwen3.5 / DeepSeek-V4 forward
    /// (see crate-level docs). The real path reads KV pages from `kv`, runs the
    /// attention/MoE stack, and drives TP/EP/PP collectives through `comm`. This
    /// stub intentionally ignores `plan`, `kv`, and `comm` so the seam is
    /// exercisable on any host with no backend wired up.
    fn forward(
        &mut self,
        _plan: &ForwardPlan,
        _kv: &mut dyn KvPool,
        _comm: &Self::Comm,
    ) -> anyhow::Result<Self::Logits> {
        Ok(vec![0.0_f32; SKELETON_VOCAB])
    }
}

#[cfg(test)]
mod tests {
    use infer_seam::{KvAllocator, KvPrefixStore, KvQuery};

    use super::*;

    /// Minimal inline [`KvPool`] mock: every method returns a default / inert
    /// value. The skeleton forward never reads KV, so this only has to satisfy
    /// the `&mut dyn KvPool` argument type and the dyn-safe trait surface.
    #[derive(Default)]
    struct MockKvPool;

    impl KvQuery for MockKvPool {
        fn is_active(&self) -> bool {
            false
        }

        fn page_size(&self) -> usize {
            1
        }

        fn free_pages(&self) -> usize {
            0
        }

        fn free_tokens(&self) -> usize {
            0
        }

        fn seq_len(&self, _slot: usize) -> usize {
            0
        }

        fn slot_epoch(&self, _slot: usize) -> u64 {
            0
        }

        fn append_pages_needed(&self, _slot: usize, _tokens: usize) -> usize {
            0
        }

        fn page_indices(&self, _slot: usize) -> &[u32] {
            &[]
        }

        fn page_indices_for_token_range(&self, _slot: usize, _start: usize, _len: usize) -> &[u32] {
            &[]
        }
    }

    impl KvAllocator for MockKvPool {
        fn alloc(&mut self, _slot: usize, _tokens: usize) -> anyhow::Result<()> {
            Ok(())
        }

        fn alloc_detached_pages(&mut self, _pages: usize) -> anyhow::Result<Vec<u32>> {
            Ok(Vec::new())
        }

        fn free_slot(&mut self, _slot: usize) {}

        fn truncate_slot(&mut self, _slot: usize, _new_len: usize) -> anyhow::Result<()> {
            Ok(())
        }

        fn migrate(&mut self, _slot: usize, _start: usize, _len: usize) -> anyhow::Result<()> {
            Ok(())
        }
    }

    impl KvPrefixStore for MockKvPool {
        fn retain_pages(&mut self, _pages: &[u32]) {}

        fn release_pages(&mut self, _pages: &[u32]) {}

        fn retained_count(&self) -> usize {
            0
        }

        fn attach_pages(
            &mut self,
            _slot: usize,
            _pages: &[u32],
            _token_count: usize,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn skeleton_forward_idle_returns_ok_zeroed_logits() {
        let mut model = SkeletonModel::new();
        let comm = NoopCommunicator;
        let mut kv = MockKvPool;
        let plan = ForwardPlan::idle();

        let logits = model
            .forward(&plan, &mut kv, &comm)
            .expect("skeleton forward must succeed for an idle plan");

        assert_eq!(logits.len(), SKELETON_VOCAB);
        assert!(logits.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn noop_communicator_collectives_are_inert() {
        let comm = NoopCommunicator;
        let mut tensor = vec![1.0_f32, 2.0, 3.0];
        let original = tensor.clone();

        comm.all_reduce(&mut tensor);
        let send = vec![4.0_f32, 5.0];
        let mut recv = vec![0.0_f32, 0.0];
        comm.all_to_all(&send, &mut recv);
        comm.send_recv(0, &mut tensor);

        // No-op collectives must leave the in-place tensor untouched.
        assert_eq!(tensor, original);
        // and must not write into the recv buffer.
        assert_eq!(recv, vec![0.0_f32, 0.0]);
    }

    #[test]
    fn skeleton_model_is_dyn_modelarch_shaped() {
        // Compile-time proof that the associated-type web resolves: a
        // `SkeletonModel` value used purely through the `ModelArch` bound, with
        // its `Comm` constrained to `Communicator<Tensor = Self::Tensor>`.
        fn drive<M: ModelArch>(model: &mut M, comm: &M::Comm, kv: &mut dyn KvPool) -> M::Logits {
            model
                .forward(&ForwardPlan::idle(), kv, comm)
                .expect("forward ok")
        }

        let mut model = SkeletonModel::new();
        let comm = NoopCommunicator;
        let mut kv = MockKvPool;
        let logits = drive(&mut model, &comm, &mut kv);
        assert_eq!(logits.len(), SKELETON_VOCAB);
    }
}
