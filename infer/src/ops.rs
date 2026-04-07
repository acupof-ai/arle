//! GPU operations on device tensors.

mod attention;
mod elementwise;
mod embedding;
mod kv_ops;
mod linear;
mod norm;
mod recurrent;
mod sampling;

#[cfg(test)]
mod tests;

// pub re-exports
pub use attention::{
    FlashInferWorkspace, decode_prep_paged, flashinfer_batch_decode, flashinfer_plan,
    flashinfer_run_layer, flashinfer_tc_plan, flashinfer_tc_run_layer,
    fused_attention_decode_batched_into, fused_attention_decode_into, prefill_attention_batch,
    prefill_attention_hd256_batch, prefill_attention_hd256_batch_with_scratch,
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
pub use sampling::{argmax, gpu_sample, gpu_sample_into, gpu_sample_launch, gpu_sample_readback};

// pub(crate) re-exports
#[cfg(test)]
pub(crate) use attention::flash_attention_prefill_hd256_into;
pub(crate) use attention::{
    HeadConfig, NormRopeParams, PagedKVMeta, attention_gate_paged_hd256, decode_prep_paged_hd256,
    flashinfer_plan_hd256, flashinfer_run_layer_hd256,
};
pub(crate) use elementwise::{
    add_batch_into, add_bias_batch_into, extract_vec, extract_vec_into, silu_mul_batch_into,
    silu_mul_fused_batch_into, split_qkv_batch, vec_add_inplace,
};
pub(crate) use linear::{gemm_into, linear};
pub(crate) use norm::{
    fused_add_rms_norm_batch_into, rms_norm, rms_norm_batch_into, rms_norm_gated_batch_into,
};
pub(crate) use recurrent::{
    conv1d_decode_batch_into, conv1d_prefill_batch_into, gated_delta_rule_decode_into,
    gdr_decode_batch_into,
};
pub(crate) use sampling::{argmax_batch_launch, argmax_batch_readback_into, gpu_sample_launch_raw};
