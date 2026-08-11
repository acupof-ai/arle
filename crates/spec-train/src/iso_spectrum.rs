//! Paper-faithful ISO optimizer for RLVR-shaped heads (arXiv:2607.19331).
//!
//! The paper's load-bearing observation: an RLVR update barely moves a weight's
//! singular spectrum (κ_spec 1.02–1.35, i.e. indistinguishable from isotropic
//! noise) while dense SFT supervision moves it hard (κ_spec 89–1364). The
//! behaviour change is a rotation of the singular *frames*, not a rescaling of
//! the singular *values* — so an optimizer that spends Adam moments on Σ is
//! spending them on the one thing RLVR does not need. ISO removes Σ from the
//! coordinates entirely: it factors `W₀ = U₀ diag(Σ₀) V₀ᵀ`, freezes Σ₀, and
//! optimizes the orthonormal frames `U,V` — every step is a frame rotation by
//! construction. That bought the paper ~2.2× fewer steps to equal accuracy.
//!
//! **Where this applies, and where it must not.** The DSpark acceptance head is
//! our one RLVR-shaped trainable: outcome-level sparse reward
//! (`accepted / block_size`), seeded from a base checkpoint, small proximal
//! steps. The OPD path is dense per-token distillation — the regime the paper's
//! own control condition says moves the spectrum by two orders of magnitude — so
//! this is not wired there.
//!
//! Mechanics, for a head weight `W [n, k]` with `n ≥ k`:
//!
//! - **Capture** (once): thin SVD of the seeded head. `S = WᵀW [k,k]`,
//!   symmetric-eig `S = V diag(λ) Vᵀ`, `Σ₀ = √λ`, `U = W V diag(1/Σ₀)`. `U,V`
//!   become trainable leaves; Σ₀ is frozen.
//! - **Reconstruct** (per forward): `W = U diag(Σ₀) Vᵀ`, on the tape — so a
//!   backward through the head yields `G_U, G_V` by the chain rule, no manual
//!   gradient. The base optimizer then steps `U,V` with independent state.
//! - **Retract** (per cadence): `U ← polar(U)`, `V ← polar(V)`, where
//!   `polar(X) = X (XᵀX)^{-1/2}` restores orthonormal columns. With `UᵀU = VᵀV
//!   = I`, `σ(U diag(Σ₀) Vᵀ) = Σ₀` exactly — the iterate stays on ℱ(W₀).
//!
//! Matrix roots are `k × k` in f64 (as the paper does); the only `n·k²` work is
//! the one-time capture and the per-cadence `U`-polar GEMM.

use anyhow::{Result, ensure};
use autograd::{Tape, Tensor, TensorId, TensorStore, ops};
use std::collections::HashSet;

/// Newton–Schulz iterations for the matrix square root. Quadratic convergence
/// near the fixed point; 16 is comfortable for a ridge-regularized Gram.
const NS_ITERS: usize = 16;
/// Cyclic-Jacobi sweeps for the one-time capture eig. 30 diagonalizes a `k×k`
/// Gram to f64 round-off well past `k = 256`.
const JACOBI_SWEEPS: usize = 30;

/// One factored head weight: `W = U diag(Σ₀) Vᵀ`.
struct Factor {
    u: TensorId,      // [n, k] left frame, trainable
    v: TensorId,      // [k, k] right frame, trainable
    sigma0: Vec<f64>, // [k] frozen singular values
    k: usize,
}

/// ISO frame state: one factor per head weight, captured from the seeded head
/// before the first step.
pub struct IsoFrames {
    factors: Vec<Factor>,
    /// Relative drift `‖X − polar(X)‖_F / ‖X‖_F` of the last retraction, per
    /// factor — the paper's premise measured on our head: near-zero means the
    /// base optimizer's step was already (near) a frame rotation.
    pub last_drift: Vec<f32>,
}

impl IsoFrames {
    /// Capture the thin SVD of each parameter. Parameters must be rank-2 with
    /// `rows ≥ cols`. Rejects a zero factor — ISO adapts an existing head, it
    /// cannot grow one from a cold start.
    pub fn capture(params: &[TensorId], store: &mut TensorStore) -> Result<Self> {
        let mut factors = Vec::with_capacity(params.len());
        for &id in params {
            let shape = store
                .get(id)
                .map(|t| t.shape.clone())
                .ok_or_else(|| anyhow::anyhow!("iso: parameter {id:?} not in store"))?;
            ensure!(
                shape.len() == 2 && shape[0] >= shape[1],
                "iso: expected a rank-2 [n, k] parameter with n >= k, got {shape:?}"
            );
            let (n, k) = (shape[0], shape[1]);
            let w = store.to_host(id)?;
            let gram = gram(id, store)?;
            let (v, lambda) = jacobi_eig(&gram, k);
            let trace: f64 = lambda.iter().sum();
            ensure!(
                trace > 0.0 && trace.is_finite(),
                "iso: non-positive gram trace {trace} — a zero factor has no base \
                 spectrum to preserve; ISO cannot grow a head from a cold start"
            );
            let ridge = 1e-12 * trace / k as f64;
            let sigma0: Vec<f64> = lambda.iter().map(|&l| (l.max(ridge)).sqrt()).collect();
            // U[i,r] = (Σ_c W[i,c]·V[c,r]) / σ[r]; V column r is eigenvector r.
            let mut u = vec![0.0f32; n * k];
            for i in 0..n {
                for r in 0..k {
                    let mut acc = 0.0f64;
                    for c in 0..k {
                        acc += f64::from(w[i * k + c]) * v[c * k + r];
                    }
                    u[i * k + r] = (acc / sigma0[r]) as f32;
                }
            }
            let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            let u = store.alloc(Tensor::new(u, vec![n, k], true)?);
            let v = store.alloc(Tensor::new(v32, vec![k, k], true)?);
            factors.push(Factor { u, v, sigma0, k });
        }
        let last_drift = vec![0.0; factors.len()];
        Ok(Self {
            factors,
            last_drift,
        })
    }

    /// Trainable frame ids `[U₀, V₀, U₁, V₁, …]` — the optimizer's parameter set.
    pub fn frame_ids(&self) -> Vec<TensorId> {
        self.factors.iter().flat_map(|f| [f.u, f.v]).collect()
    }

    /// Reconstruct `W_i = U diag(Σ₀) Vᵀ` on the tape. Backward through the result
    /// populates `U.grad, V.grad` = `G_W V Σ₀, G_Wᵀ U Σ₀`.
    pub fn reconstruct(
        &self,
        i: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        let f = &self.factors[i];
        let mut diag = vec![0.0f32; f.k * f.k];
        for r in 0..f.k {
            diag[r * f.k + r] = f.sigma0[r] as f32;
        }
        let sigma = store.from_slice(&diag, &[f.k, f.k])?;
        let vs = ops::matmul_bt(f.v, sigma, store, tape)?; // V·diag(σ)  [k,k]
        Ok(ops::matmul_bt(f.u, vs, store, tape)?) // U·diag(σ)·Vᵀ  [n,k]
    }

    /// Materialize `W_i` off-tape (for publish/checkpoint). Retract first, so the
    /// materialized head is always on ℱ(W₀).
    pub fn materialize(&self, i: usize, store: &mut TensorStore) -> Result<Vec<f32>> {
        let live: HashSet<TensorId> = store.live_ids().into_iter().collect();
        let mut tape = Tape::new();
        let w = self.reconstruct(i, store, &mut tape)?;
        let host = store.to_host(w)?;
        let _ = store.free_new_except(&live, &HashSet::new());
        Ok(host)
    }

    /// Project every frame back onto the orthonormal manifold: `U ← polar(U)`,
    /// `V ← polar(V)`. Call after the base optimizer's step.
    pub fn retract(&mut self, store: &mut TensorStore) -> Result<()> {
        for (slot, f) in self.factors.iter().enumerate() {
            let du = polar_in_place(f.u, f.k, store)?;
            polar_in_place(f.v, f.k, store)?;
            self.last_drift[slot] = du;
        }
        Ok(())
    }
}

/// Non-mutating spectrum measurement, independent of retraction. Captures the
/// base head's singular values (as sorted Gram eigenvalues σ²) once, then reports
/// each parameter's relative spectral drift `‖σ²(W) − σ²(W₀)‖ / ‖σ²(W₀)‖` on
/// demand — so an ISO-**off** run genuinely measures whether unconstrained
/// updates leave the spectrum near-fixed (the paper's premise), not just the
/// ISO-on retraction residual. This is the drift the α-sweep reads.
pub struct SpectrumProbe {
    /// `(param, sorted σ²(W₀), k)`, parallel to the captured param list.
    bases: Vec<(TensorId, Vec<f64>, usize)>,
}

impl SpectrumProbe {
    /// Capture the base spectrum of each parameter. Rank-2 `[n, k]`, `n >= k`.
    pub fn capture(params: &[TensorId], store: &mut TensorStore) -> Result<Self> {
        let mut bases = Vec::with_capacity(params.len());
        for &id in params {
            let k = store
                .get(id)
                .map(|t| t.shape.clone())
                .ok_or_else(|| anyhow::anyhow!("iso probe: parameter {id:?} not in store"))?
                .get(1)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("iso probe: parameter {id:?} is not rank-2"))?;
            bases.push((id, sorted_eigvals(&gram(id, store)?, k), k));
        }
        Ok(Self { bases })
    }

    /// Relative spectral drift per captured parameter, measured on the current
    /// weights without mutating them.
    pub fn drift(&self, store: &mut TensorStore) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.bases.len());
        for (id, base, k) in &self.bases {
            let now = sorted_eigvals(&gram(*id, store)?, *k);
            let num: f64 = base.iter().zip(&now).map(|(a, b)| (a - b).powi(2)).sum();
            let den: f64 = base.iter().map(|a| a * a).sum();
            // A cold param (w2=0 at a seedless start) has no base spectrum to be
            // isospectral against; relative drift is undefined, so report 0 rather
            // than dividing by ~0 and emitting an astronomical fake drift.
            // The test must be relative to the CURRENT spectrum: `den` sums σ⁴ of
            // a near-zero head, so an absolute floor lets a base 1e14× smaller
            // than `now` through and reports the ratio as real drift (the
            // warm-head run's 2.82e14).
            let now_den: f64 = now.iter().map(|a| a * a).sum();
            out.push(if den <= 1e-30 || den <= 1e-12 * now_den {
                0.0
            } else {
                (num / den).sqrt() as f32
            });
        }
        Ok(out)
    }
}

/// Sorted eigenvalues of a symmetric `[k, k]` — the σ² multiset.
fn sorted_eigvals(gram: &[f64], k: usize) -> Vec<f64> {
    let (_, mut lambda) = jacobi_eig(gram, k);
    lambda.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lambda
}

/// `X ← X (XᵀX)^{-1/2}` for a `[·, k]` tensor; returns the relative drift
/// `‖X − X'‖_F / ‖X‖_F`.
fn polar_in_place(id: TensorId, k: usize, store: &mut TensorStore) -> Result<f32> {
    let gram = gram(id, store)?;
    let (_, inv_sqrt) = matrix_roots(&gram, k)?;
    let inv32: Vec<f32> = inv_sqrt.iter().map(|&x| x as f32).collect();

    let before = store.to_host(id)?;
    let live: HashSet<TensorId> = store.live_ids().into_iter().collect();
    let mut tape = Tape::new();
    let m_id = store.from_slice(&inv32, &[k, k])?;
    let out_id = ops::matmul(id, m_id, store, &mut tape)?;
    let after = store.to_host(out_id)?;
    let _ = store.free_new_except(&live, &HashSet::new());

    let drift = relative_drift(&before, &after);
    store
        .get_mut(id)
        .ok_or_else(|| anyhow::anyhow!("iso: frame {id:?} vanished mid-retraction"))?
        .data
        .copy_from_slice(&after);
    Ok(drift)
}

/// `Wᵀ W` for a `[n, k]` parameter, as f64 row-major `[k, k]`.
fn gram(id: TensorId, store: &mut TensorStore) -> Result<Vec<f64>> {
    let live: HashSet<TensorId> = store.live_ids().into_iter().collect();
    let mut tape = Tape::new();
    let wt = ops::transpose(id, 0, 1, store, &mut tape)?;
    let s = ops::matmul(wt, id, store, &mut tape)?;
    let host = store.to_host(s)?;
    let _ = store.free_new_except(&live, &HashSet::new());
    Ok(host.iter().map(|&x| f64::from(x)).collect())
}

/// Symmetric eigendecomposition of an SPD `[k, k]` by cyclic Jacobi. Returns
/// `(V, λ)`: `V` row-major `[k, k]` with eigenvector `r` in column `r`, and
/// eigenvalues `λ[r]`. Exposed for the DSpark spectrum-invariant test.
pub fn jacobi_eig(s: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = s.to_vec();
    let mut v = identity(k);
    for _ in 0..JACOBI_SWEEPS {
        let mut off = 0.0;
        for p in 0..k {
            for q in (p + 1)..k {
                off += a[p * k + q] * a[p * k + q];
            }
        }
        if off.sqrt() <= 1e-15 {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                let apq = a[p * k + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                // Rotation (c, s) that zeros a[p,q].
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for i in 0..k {
                    let (aip, aiq) = (a[i * k + p], a[i * k + q]);
                    a[i * k + p] = c * aip - sn * aiq;
                    a[i * k + q] = sn * aip + c * aiq;
                }
                for i in 0..k {
                    let (api, aqi) = (a[p * k + i], a[q * k + i]);
                    a[p * k + i] = c * api - sn * aqi;
                    a[q * k + i] = sn * api + c * aqi;
                }
                for i in 0..k {
                    let (vip, viq) = (v[i * k + p], v[i * k + q]);
                    v[i * k + p] = c * vip - sn * viq;
                    v[i * k + q] = sn * vip + c * viq;
                }
            }
        }
    }
    let lambda = (0..k).map(|r| a[r * k + r]).collect();
    (v, lambda)
}

/// `(S^{1/2}, S^{-1/2})` for a symmetric PD `[k, k]` by coupled Newton–Schulz.
/// Scaling by `‖S‖_F` puts the spectrum in `(0, 1]` so the iteration contracts;
/// a trace-relative ridge keeps a near-singular Gram from stalling it.
fn matrix_roots(s: &[f64], k: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(s.len() == k * k, "iso: gram is not [k, k]");
    let trace: f64 = (0..k).map(|i| s[i * k + i]).sum();
    ensure!(
        trace > 0.0 && trace.is_finite(),
        "iso: non-positive gram trace {trace} in retraction"
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
// ponytail: ikj triple loop — k is the head rank (256), one call per retraction,
// ~33 MFLOP against the ~10 GFLOP polar GEMM. Hand to the backend sgemm only if
// a head pushes k into the thousands.
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
