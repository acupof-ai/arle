//! GPU operations on device tensors.

#[path = "ops/attention.rs"]
mod attention;
#[path = "ops/elementwise.rs"]
mod elementwise;
#[path = "ops/embedding.rs"]
mod embedding;
#[path = "ops/kv_ops.rs"]
mod kv_ops;
#[path = "ops/linear.rs"]
mod linear;
#[path = "ops/norm.rs"]
mod norm;
#[path = "ops/recurrent.rs"]
mod recurrent;
#[path = "ops/sampling.rs"]
mod sampling;

#[cfg(test)]
#[path = "ops/tests.rs"]
mod tests;

// pub re-exports
pub(crate) use attention::{
    decode_prep_paged, decode_prep_paged_fused_qkv, prefill_attention_batch,
    prefill_attention_hd256_batch, prefill_attention_hd256_batch_with_scratch,
};
pub use attention::{
    flashinfer_run_layer, flashinfer_tc_run_layer, fused_attention_decode_batched_into,
    fused_attention_decode_into,
};
pub use elementwise::{add_batch, silu_mul_batch};
pub use embedding::{embedding_batch, embedding_decode_into};
pub use kv_ops::scatter_write_kv;
pub use linear::{fused_mlp_into, gemm, gemv};
pub use norm::{
    fused_add_rms_norm_into, fused_add_rms_norm_offset_into, rms_norm_batch_offset_into,
    rms_norm_gated_into, rms_norm_into, rms_norm_offset_into,
};
pub use recurrent::gated_delta_rule_prefill_chunkwise_into;
pub use sampling::{
    argmax, argmax_with_logprob, gpu_sample, gpu_sample_into, gpu_sample_launch,
    gpu_sample_readback,
};

// pub(crate) re-exports
#[cfg(test)]
pub(crate) use attention::flash_attention_prefill_hd256_into;
pub(crate) use attention::{
    HeadConfig, NormRopeParams, PagedKVMeta, attention_gate_paged_hd256, decode_prep_paged_hd256,
    flashinfer_run_layer_hd256,
};
pub(crate) use elementwise::{
    add_batch_into, add_bias_batch_into, extract_vec, extract_vec_into, silu_mul_batch_into,
    silu_mul_fused_batch_into, split_qkv_batch, vec_add_inplace,
};
pub(crate) use linear::{gemm_into, linear};
pub(crate) use norm::{
    add_bf16_into_f32, cast_bf16_to_f32, cast_f32_to_bf16, fused_add_rms_norm_batch_into, rms_norm,
    rms_norm_batch_f32_in_into, rms_norm_batch_into, rms_norm_gated_batch_into,
};
pub(crate) use recurrent::{
    conv1d_decode_batch_into, conv1d_prefill_batch_into, gated_delta_rule_decode_into,
    gdr_decode_batch_into,
};
pub(crate) use sampling::{
    argmax_batch_launch, argmax_batch_logprob_launch, argmax_batch_readback_into,
    gpu_sample_launch_raw,
};
