// Fused AdamW per-element update. Matches `cpu_adamw_step_in_place`
// in `crates/autograd/src/backend.rs` to within float-rounding:
//
//   param *= (1 - lr*wd)                  // decoupled weight decay (skip if wd == 0)
//   m      = beta1*m + (1-beta1)*g
//   v      = beta2*v + (1-beta2)*g*g
//   m_hat  = m / bc1
//   v_hat  = v / bc2
//   param -= lr * m_hat / (sqrt(v_hat) + eps)
//
// Each thread owns one element of `param`, `m`, `v`. All three are
// mutated in place; `grad` is read-only. One launch fuses upload of
// `grad` (done by the host caller via `clone_htod`) + the entire AdamW
// formula, replacing the 3× readback + 3× upload + host loop the
// default `Backend::adamw_step` would do.
extern "C" __global__ void adamw_step_f32(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bc1,
    float bc2
) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }
    float p = param[i];
    if (wd > 0.0f) {
        p *= (1.0f - (lr * wd));
    }
    float g = grad[i];
    float m_new = (beta1 * m[i]) + ((1.0f - beta1) * g);
    float v_new = (beta2 * v[i]) + ((1.0f - beta2) * g * g);
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    p -= lr * m_hat / (sqrtf(v_hat) + eps);

    param[i] = p;
    m[i] = m_new;
    v[i] = v_new;
}
