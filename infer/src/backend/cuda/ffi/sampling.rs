use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub(crate) fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream) -> CUresult;

    pub(crate) fn argmax_logprob_cuda(
        x: *const Half,
        out_idx: *mut i32,
        out_logprob: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_batch_logprob_cuda(
        logits: *const Half,
        token_ids: *mut i32,
        logprobs: *mut f32,
        batch_size: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn argmax_batch_cuda(
        logits: *const Half,
        token_ids: *mut i32,
        batch_size: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub(crate) fn gpu_sample_cuda(
        logits: *const Half,
        probs_scratch: *mut f32,
        output: *mut i32,
        vocab_size: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        random_val: f32,
        stream: CUstream,
    ) -> CUresult;
}
