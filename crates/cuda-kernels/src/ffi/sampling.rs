use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn argmax_cuda(x: *const Half, out: *mut i32, n: i32, stream: CUstream) -> CUresult;

    pub fn argmax_logprob_cuda(
        x: *const Half,
        out_idx: *mut i32,
        out_logprob: *mut f32,
        n: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn argmax_batch_logprob_cuda(
        logits: *const Half,
        token_ids: *mut i32,
        logprobs: *mut f32,
        batch_size: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn argmax_batch_cuda(
        logits: *const Half,
        token_ids: *mut i32,
        batch_size: i32,
        vocab_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn gpu_sample_cuda(
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

    /// Batched engine-sampler filter: `probs[row] :=` filtered, renormalized
    /// dist of `logits[row]` (temperature softmax → top-k → top-p → min-p).
    pub fn dspark_filter_probs_cuda(
        logits: *const Half,
        probs: *mut f32,
        rows: i32,
        vocab: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Filter one logits row into `q_row` AND multinomial-sample from it;
    /// only the token id returns to the host (draft markov step).
    pub fn dspark_draft_sample_cuda(
        logits: *const Half,
        q_row: *mut f32,
        token_out: *mut i32,
        vocab: i32,
        inv_temperature: f32,
        top_k: i32,
        top_p: f32,
        min_p: f32,
        random_val: f32,
        stream: CUstream,
    ) -> CUresult;

    /// Chain rejection sampling over pre-filtered `q [depth, vocab]` /
    /// `p [depth+1, vocab]` dists with host-supplied uniforms;
    /// `out = [accepted_len, residual-or-bonus token]`.
    pub fn dspark_chain_accept_cuda(
        q: *const f32,
        p: *const f32,
        draft_tokens: *const i32,
        u_accept: *const f32,
        u_residual: *const f32,
        out: *mut i32,
        depth: i32,
        vocab: i32,
        stream: CUstream,
    ) -> CUresult;
}
