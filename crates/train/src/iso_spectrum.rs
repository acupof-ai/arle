//! ISO fixed-spectrum retraction for RLVR-shaped heads (arXiv:2607.19331).
//!
//! The paper's load-bearing observation: an RLVR update barely moves a weight's
//! singular spectrum (κ_spec 1.02–1.35, i.e. indistinguishable from isotropic
//! noise) while dense SFT supervision moves it hard (κ_spec 89–1364). The
//! behaviour change is a rotation of the singular *frames*, not a rescaling of
//! the singular *values* — so a base optimizer that spends updates on Σ is
//! spending them on the one thing RLVR does not need. Constraining the
//! iterate to the fixed-spectrum family ℱ(W₀) = {U Σ₀ Vᵀ} bought the paper
//! ~2.2× fewer steps to the same accuracy.
//!
//! **Where this applies here, and where it must not.** The DSpark acceptance
//! head is our one RLVR-shaped trainable: outcome-level sparse reward
//! (`accepted / block_size`), seeded from a base checkpoint, small proximal
//! steps. The OPD path is dense per-token distillation — the regime the paper's
//! own control condition says moves the spectrum by two orders of magnitude —
//! so this must not be wired there.
//!
//! Retraction, for a factor `W [n, k]` with `n >= k`:
//!
//! ```text
//! S = Wᵀ W            M = S^{-1/2} S₀^{1/2}            W ← W M
//! ```
//!
//! `W M = polar(W) · S₀^{1/2}`, and multiplying by an isometry cannot change
//! singular values, so `σ(W M) = σ(S₀^{1/2}) = σ(W₀)` — exactly ℱ(W₀), the
//! projection ISO's step 4 asks for. Cost is two `n·k²` GEMMs per parameter per
//! step; the matrix roots are `k × k` and computed in f64, as the paper does.

use anyhow::{Result, ensure};
use autograd::{Tape, TensorId, TensorStore, ops};
use std::collections::HashSet;

/// Newton–Schulz iterations for the matrix square root. Convergence is
/// quadratic once the iterate is near the fixed point; 16 is comfortable for a
/// ridge-regularized Gram and keeps the `k³` cost off the critical path.
const NS_ITERS: usize = 16;

/// Fixed-spectrum retraction state: one `S₀^{1/2}` per parameter, captured from
/// the base weights before the first step.
pub struct FixedSpectrum {
    /// `(param, sqrt(W₀ᵀ W₀) [k, k] row-major f64, k)`, parallel to the param list.
    cores: Vec<(TensorId, Vec<f64>, usize)>,
    /// Relative drift `‖W − W M‖_F / ‖W‖_F` of the most recent retraction, per
    /// parameter — how far the base optimizer's step left ℱ(W₀). This is the
    /// paper's premise, measured on our head rather than assumed: near-zero
    /// means the update was already a frame rotation and the constraint is free.
    pub last_drift: Vec<f32>,
}

impl FixedSpectrum {
    /// Capture the base spectrum of each parameter. Parameters must be rank-2
    /// with `rows >= cols`; anything else is a caller bug, not a fallback case.
    pub fn capture(params: &[TensorId], store: &mut TensorStore) -> Result<Self> {
        let mut cores = Vec::with_capacity(params.len());
        for &id in params {
            let shape = store
                .get(id)
                .map(|t| t.shape.clone())
                .ok_or_else(|| anyhow::anyhow!("iso: parameter {id:?} not in store"))?;
            ensure!(
                shape.len() == 2 && shape[0] >= shape[1],
                "iso: expected a rank-2 [n, k] parameter with n >= k, got {shape:?}"
            );
            let k = shape[1];
            let gram = gram(id, store)?;
            let (sqrt, _) = matrix_roots(&gram, k)?;
            cores.push((id, sqrt, k));
        }
        let last_drift = vec![0.0; cores.len()];
        Ok(Self { cores, last_drift })
    }

    /// Project every captured parameter back onto its fixed-spectrum family.
    /// Call after the base optimizer's step — ISO composes with any of them.
    pub fn retract(&mut self, store: &mut TensorStore) -> Result<()> {
        for (slot, (id, sqrt0, k)) in self.cores.iter().enumerate() {
            let (id, k) = (*id, *k);
            let gram = gram(id, store)?;
            let (_, inv_sqrt) = matrix_roots(&gram, k)?;
            // M = S^{-1/2} S₀^{1/2}, folded on the host so the big GEMM runs once.
            let m = matmul_f64(&inv_sqrt, sqrt0, k);
            let m32: Vec<f32> = m.iter().map(|&x| x as f32).collect();

            let before = store.to_host(id)?;
            let live: HashSet<TensorId> = store.live_ids().into_iter().collect();
            let mut tape = Tape::new();
            let m_id = store.from_slice(&m32, &[k, k])?;
            let out_id = ops::matmul(id, m_id, store, &mut tape)?;
            let after = store.to_host(out_id)?;
            let _ = store.free_new_except(&live, &HashSet::new());

            self.last_drift[slot] = relative_drift(&before, &after);
            let param = store
                .get_mut(id)
                .ok_or_else(|| anyhow::anyhow!("iso: parameter {id:?} vanished mid-retraction"))?;
            param.data.copy_from_slice(&after);
        }
        Ok(())
    }
}

/// `Wᵀ W` for a `[n, k]` parameter, as f64 row-major `[k, k]`. Uses the store's
/// tuned GEMM — this is the only `n·k²` work in the capture path.
fn gram(id: TensorId, store: &mut TensorStore) -> Result<Vec<f64>> {
    let live: HashSet<TensorId> = store.live_ids().into_iter().collect();
    let mut tape = Tape::new();
    let wt = ops::transpose(id, 0, 1, store, &mut tape)?;
    let s = ops::matmul(wt, id, store, &mut tape)?;
    let host = store.to_host(s)?;
    let _ = store.free_new_except(&live, &HashSet::new());
    Ok(host.iter().map(|&x| f64::from(x)).collect())
}

/// `(S^{1/2}, S^{-1/2})` for a symmetric positive-definite `[k, k]` matrix, by
/// coupled Newton–Schulz. Scaling by `‖S‖_F` puts the spectrum in `(0, 1]`,
/// which is what makes the iteration contract; a trace-relative ridge keeps a
/// near-singular Gram (a collapsed singular direction in the head) from
/// stalling it.
fn matrix_roots(s: &[f64], k: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(s.len() == k * k, "iso: gram is not [k, k]");
    let trace: f64 = (0..k).map(|i| s[i * k + i]).sum();
    ensure!(
        trace > 0.0 && trace.is_finite(),
        "iso: non-positive gram trace {trace} — parameter is zero or diverged"
    );
    let ridge = 1e-6 * trace / k as f64;
    let mut a: Vec<f64> = s.to_vec();
    for i in 0..k {
        a[i * k + i] += ridge;
    }
    let norm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let scale = norm.sqrt();
    for x in &mut a {
        *x /= norm;
    }

    let mut y = a;
    let mut z = identity(k);
    for _ in 0..NS_ITERS {
        // T = (3I − Z Y) / 2
        let mut t = matmul_f64(&z, &y, k);
        for x in &mut t {
            *x *= -0.5;
        }
        for i in 0..k {
            t[i * k + i] += 1.5;
        }
        y = matmul_f64(&y, &t, k);
        z = matmul_f64(&t, &z, k);
    }
    // Undo the 1/norm scaling: (S/c)^{1/2} = S^{1/2}/√c.
    for x in &mut y {
        *x *= scale;
    }
    for x in &mut z {
        *x /= scale;
    }
    Ok((y, z))
}

fn identity(k: usize) -> Vec<f64> {
    let mut m = vec![0.0; k * k];
    for i in 0..k {
        m[i * k + i] = 1.0;
    }
    m
}

/// Row-major `[k, k] × [k, k]`.
// ponytail: ikj triple loop — k is the head rank (256), so this is ~33 MFLOP a
// call against the retraction's ~10 GFLOP GEMM. Block or hand off to the
// backend's sgemm if a head ever pushes k into the thousands.
fn matmul_f64(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; k * k];
    for i in 0..k {
        for inner in 0..k {
            let a_ik = a[i * k + inner];
            if a_ik == 0.0 {
                continue;
            }
            let b_row = &b[inner * k..(inner + 1) * k];
            let out_row = &mut out[i * k..(i + 1) * k];
            for j in 0..k {
                out_row[j] += a_ik * b_row[j];
            }
        }
    }
    out
}

fn relative_drift(before: &[f32], after: &[f32]) -> f32 {
    let (mut diff, mut base) = (0.0f64, 0.0f64);
    for (&a, &b) in before.iter().zip(after) {
        diff += f64::from(a - b).powi(2);
        base += f64::from(a).powi(2);
    }
    if base == 0.0 {
        return 0.0;
    }
    (diff.sqrt() / base.sqrt()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use autograd::{CpuBackend, TensorStore};
    use std::sync::Arc;

    /// The invariant the whole method rests on: after retraction the singular
    /// values are the base's, whatever the optimizer did in between. Checked
    /// through `WᵀW`, whose eigenvalues are σ² — comparing the Gram spectra is
    /// the same statement without an eigensolver.
    #[test]
    fn retraction_restores_the_base_spectrum() {
        let backend: Arc<dyn autograd::Backend> = Arc::new(CpuBackend);
        let mut store = TensorStore::with_backend(backend);
        let (n, k) = (64usize, 8usize);
        // Deterministic non-orthogonal base with a spread-out spectrum.
        let base: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * (1.0 + (i % k) as f32))
            .collect();
        let id = store.from_slice(&base, &[n, k]).unwrap();
        let mut iso = FixedSpectrum::capture(&[id], &mut store).unwrap();
        let gram_before = gram(id, &mut store).unwrap();

        // Simulate an optimizer step that scales the spectrum hard — the exact
        // thing ISO says an RLVR update should not be doing.
        {
            let p = store.get_mut(id).unwrap();
            for (i, x) in p.data.iter_mut().enumerate() {
                *x = *x * 1.5 + 0.01 * (i % 7) as f32;
            }
        }
        let gram_kicked = gram(id, &mut store).unwrap();
        assert!(
            frob(&gram_kicked, &gram_before) > 0.2,
            "the kick must actually move the spectrum, else the test proves nothing"
        );

        iso.retract(&mut store).unwrap();
        let gram_after = gram(id, &mut store).unwrap();
        assert!(
            frob(&gram_after, &gram_before) < 1e-4,
            "retraction must restore σ(W₀): relative Gram error {}",
            frob(&gram_after, &gram_before)
        );
        assert!(iso.last_drift[0] > 0.0, "drift must report the kick");
    }

    /// Relative Frobenius distance between two Grams.
    fn frob(a: &[f64], b: &[f64]) -> f64 {
        let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
        let base: f64 = b.iter().map(|x| x * x).sum();
        (diff / base).sqrt()
    }
}
