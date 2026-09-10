//! Host-side model geometry and routing for the inference backends: the
//! `model → operator sequence → kernel selection` stages of the request axis
//! (architecture refactor §3.2, step 3b). Everything here is pure arithmetic
//! on config + per-rank values; device probes enter as plain host values and
//! every upload/launch stays in `infer-cuda`.

pub mod dspark;
pub mod qwen35;
