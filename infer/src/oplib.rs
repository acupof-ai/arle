//! Backend operator library — FlashInfer-style `plan()` / `run()` split.
//!
//! See [`docs/plans/backend-operator-library.md`](../../docs/plans/backend-operator-library.md).
//! `plan()` is the **pure** selection half of an operator family: given a
//! host-side shape/quant/policy description it returns a named, inspectable
//! kernel plan, touching no device memory and naming no CUDA/MLX type. `run()`
//! (which stays on the backend side) launches exactly the kernel `plan` named.
//!
//! The pure-`plan()` property makes kernel reachability a GPU-free unit test:
//! `assert_eq!(plan(inputs, policy), Expected)` runs under the crate's default
//! feature set with no nvcc and no GPU.
//!
//! **Phase 1** lands the `linear` family: the relocated linear / GEMM dispatch
//! selection. **Phase 2** lands the `attention` family: the relocated HD256
//! TileLang `(num_qo_heads, num_kv_heads)` head-config resolver. Further
//! families (KV quant, grouped/MoE GEMM) are added one-per-migrated-family in
//! later tranches — never stubbed ahead of a real consumer.

#[path = "oplib/attention.rs"]
pub mod attention;
#[path = "oplib/linear.rs"]
pub mod linear;
