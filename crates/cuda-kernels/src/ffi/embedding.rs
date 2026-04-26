use super::{CUresult, CUstream, Half};

#[allow(dead_code)]
unsafe extern "C" {
    pub fn embedding_batched_cuda(
        embed: *const Half,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn embedding_decode_cuda(
        embed: *const Half,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q8_embedding_batched_cuda(
        weight: *const i8,
        scales: *const Half,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q8_embedding_decode_cuda(
        weight: *const i8,
        scales: *const Half,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        group_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q3k_embedding_batched_cuda(
        weight: *const u8,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_embedding_batched_cuda(
        weight: *const u8,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_embedding_batched_cuda(
        weight: *const u8,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_embedding_batched_cuda(
        weight: *const u8,
        token_ids: *const i32,
        out: *mut Half,
        hidden_size: i32,
        seq_len: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q3k_embedding_decode_cuda(
        weight: *const u8,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q4k_embedding_decode_cuda(
        weight: *const u8,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q5k_embedding_decode_cuda(
        weight: *const u8,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;

    pub fn q6k_embedding_decode_cuda(
        weight: *const u8,
        decode_meta: *const i32,
        out: *mut Half,
        hidden_size: i32,
        stream: CUstream,
    ) -> CUresult;
}
