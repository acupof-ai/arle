use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};

use super::KV_CACHE_CHUNK;
use super::config::{MetalModelArch, MetalModelConfig};
use super::dflash::{self, MetalDflashRuntime};
use super::forward::build_forward_graph;
use super::gdr::MetalRecurrentState;
use super::kv_pool::MetalKVPool;
use super::mlx::{MlxArray, async_eval, concatenate_axis, eval, slice, take_axis, zeros};
use super::ops::{clear_metal_cache, extend_kv_cache};
use super::qwen35::{
    CppQwen35Model, Qwen35MetalWeights, qwen35_dflash_supported, qwen35_forward_step,
    qwen35_forward_with_hidden_states,
};
use super::sampling::{gpu_sample_token, gpu_sample_token_batched, validate_metal_sampling_params};
use super::weights::{MetalWeights, StandardMetalWeights};
use crate::sampler::SamplingParams;

const METAL_REQUEST_STATE_ID: usize = 0;

#[path = "request_state/helpers.rs"]
mod helpers;

#[path = "request_state/tests.rs"]
mod tests_mod;

use helpers::{
    left_pad_kv_cache_row, metal_qwen35_trace_enabled, round_up_kv_capacity, slice_row,
    strip_left_padding_from_packed_row,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalRequestPhase {
    Prefill,
    Decode,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DflashRequestMetrics {
    pub block_count: usize,
    pub block_size: usize,
    pub avg_accepted_inputs: f64,
    pub acceptance_rate: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefillChunkResult {
    pub processed_tokens: usize,
    pub emitted_token: Option<u32>,
    pub phase: MetalRequestPhase,
    pub finish_reason: Option<&'static str>,
}

/// Result of `MetalRequestState::try_mixed_batch`.
///
/// The contract is intentionally backend-generic: the caller gets the
/// sampled decode tokens for the rows that participated in the mixed batch
/// plus the prefill chunk outcome for the single prefill request that shared
/// the step. The current fused implementation is still Qwen3-only, but the
/// entrypoint and result type are Metal-wide so the runtime can stay uniform.
pub(crate) struct MetalMixedBatchResult {
    pub(crate) decode_tokens: Vec<u32>,
    pub(crate) prefill: PrefillChunkResult,
}

/// Result of `MetalRequestState::try_decode_qwen35_dflash_speculative_batch`:
/// identifies which rows of the caller's input slice were dispatched through
/// the batched speculative kernel, and the first sampled token for each.
///
/// Contract:
///   - `ready_indices` is sorted ascending, `len() >= 2`.
///   - `tokens.len() == ready_indices.len()`; `tokens[i]` is the sampled
///     token for the input row at `ready_indices[i]`.
///   - Every input-slice index NOT in `ready_indices` must be run scalar by
///     the caller — those rows were left untouched.
#[derive(Clone, Debug)]
pub(crate) struct DflashBatchOutcome {
    pub(crate) ready_indices: Vec<usize>,
    pub(crate) tokens: Vec<u32>,
}

enum Qwen3PackedBatchRowKind {
    Decode,
    Prefill { terminal_prompt: bool },
}

struct Qwen3PackedBatchRow<'a> {
    state: *mut ResumableRequestState<Qwen3StepDriver<'a>>,
    query_tokens: Vec<u32>,
    kind: Qwen3PackedBatchRowKind,
}

trait StepDriver {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>>;
    fn prefill_tokens(&mut self, tokens: &[u32], terminal_prompt: bool) -> Result<Option<u32>> {
        let mut emitted = None;
        for (idx, &token) in tokens.iter().enumerate() {
            let is_terminal = terminal_prompt && idx + 1 == tokens.len();
            let sampled = self.prefill_token(token, is_terminal)?;
            if is_terminal {
                emitted = sampled;
            } else if sampled.is_some() {
                bail!("non-terminal prefill step unexpectedly emitted a sampled token");
            }
        }
        Ok(emitted)
    }
    fn decode_token(&mut self, token: u32) -> Result<u32>;
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

struct ResumableRequestState<D: StepDriver> {
    driver: D,
    prompt_tokens: Vec<u32>,
    prompt_cursor: usize,
    max_new_tokens: usize,
    generated_tokens: usize,
    last_token: Option<u32>,
    stop_token_ids: Vec<u32>,
    eos_token_id: u32,
    ignore_eos: bool,
    phase: MetalRequestPhase,
    finish_reason: Option<&'static str>,
    cleaned_up: bool,
}

impl<D: StepDriver> ResumableRequestState<D> {
    fn new(
        driver: D,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
        stop_token_ids: Vec<u32>,
        eos_token_id: u32,
        ignore_eos: bool,
    ) -> Result<Self> {
        ensure!(
            !prompt_tokens.is_empty(),
            "Metal request state requires at least one prompt token"
        );
        ensure!(max_new_tokens > 0, "max_new_tokens must be >= 1");
        Ok(Self {
            driver,
            prompt_tokens,
            prompt_cursor: 0,
            max_new_tokens,
            generated_tokens: 0,
            last_token: None,
            stop_token_ids,
            eos_token_id,
            ignore_eos,
            phase: MetalRequestPhase::Prefill,
            finish_reason: None,
            cleaned_up: false,
        })
    }

    fn phase(&self) -> MetalRequestPhase {
        self.phase
    }

    fn prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }

    fn prompt_progress(&self) -> usize {
        self.prompt_cursor
    }

    fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    fn finish_reason(&self) -> Option<&'static str> {
        self.finish_reason
    }

    fn prefill_chunk(&mut self, budget: usize) -> Result<PrefillChunkResult> {
        ensure!(budget > 0, "prefill budget must be >= 1");
        ensure!(
            self.phase == MetalRequestPhase::Prefill,
            "prefill_chunk requires Prefill phase, got {:?}",
            self.phase
        );

        let remaining = self.prompt_tokens.len() - self.prompt_cursor;
        let processed = budget.min(remaining);
        let prompt_end = self.prompt_cursor + processed;
        let terminal_prompt = prompt_end == self.prompt_tokens.len();
        let sampled = self.driver.prefill_tokens(
            &self.prompt_tokens[self.prompt_cursor..prompt_end],
            terminal_prompt,
        )?;
        self.prompt_cursor = prompt_end;

        if terminal_prompt {
            let sampled_token =
                sampled.context("terminal prefill step did not emit a sampled token")?;
            self.record_sampled_token(sampled_token)?;
            return Ok(PrefillChunkResult {
                processed_tokens: processed,
                emitted_token: Some(sampled_token),
                phase: self.phase,
                finish_reason: self.finish_reason,
            });
        }

        Ok(PrefillChunkResult {
            processed_tokens: processed,
            emitted_token: None,
            phase: self.phase,
            finish_reason: self.finish_reason,
        })
    }

    fn decode_step(&mut self) -> Result<Option<u32>> {
        ensure!(
            self.phase == MetalRequestPhase::Decode,
            "decode_step requires Decode phase, got {:?}",
            self.phase
        );
        let input_token = self
            .last_token
            .context("decode_step requires a committed prefill token")?;
        let sampled_token = self.driver.decode_token(input_token)?;
        self.record_sampled_token(sampled_token)?;
        Ok(Some(sampled_token))
    }

    fn cancel(&mut self) -> Result<()> {
        if self.phase != MetalRequestPhase::Finished {
            self.phase = MetalRequestPhase::Finished;
            self.finish_reason = Some("cancelled");
            self.cleanup_once()?;
        }
        Ok(())
    }

    fn record_sampled_token(&mut self, sampled_token: u32) -> Result<()> {
        self.last_token = Some(sampled_token);
        self.generated_tokens += 1;

        if self.should_stop(sampled_token) {
            self.phase = MetalRequestPhase::Finished;
            self.finish_reason = Some("stop");
            self.cleanup_once()?;
        } else if self.generated_tokens >= self.max_new_tokens {
            self.phase = MetalRequestPhase::Finished;
            self.finish_reason = Some("length");
            self.cleanup_once()?;
        } else {
            self.phase = MetalRequestPhase::Decode;
        }

        Ok(())
    }

    fn should_stop(&self, token: u32) -> bool {
        (!self.ignore_eos && token == self.eos_token_id) || self.stop_token_ids.contains(&token)
    }

    fn cleanup_once(&mut self) -> Result<()> {
        if self.cleaned_up {
            return Ok(());
        }
        self.driver.cleanup()?;
        self.cleaned_up = true;
        Ok(())
    }
}

impl<D: StepDriver> Drop for ResumableRequestState<D> {
    fn drop(&mut self) {
        if self.cleaned_up {
            return;
        }
        if let Err(err) = self.driver.cleanup() {
            log::warn!("Metal request state cleanup failed during drop: {err:#}");
        }
        self.cleaned_up = true;
    }
}

pub struct MetalRequestState<'a> {
    inner: MetalRequestStateInner<'a>,
}

enum MetalRequestStateInner<'a> {
    Qwen3(Box<ResumableRequestState<Qwen3StepDriver<'a>>>),
    Qwen35(Box<ResumableRequestState<Qwen35StepDriver<'a>>>),
}

pub(crate) struct Qwen35PrefixSnapshot {
    pub token_ids: Vec<u32>,
    pub kv_flat: Vec<MlxArray>,
    pub gdr_flat: Vec<MlxArray>,
    pub cache_len: i32,
    pub kv_capacity: i32,
}

const QWEN35_PREFIX_SNAPSHOT_MAGIC: [u8; 8] = *b"Q35PFX01";
const QWEN35_PREFIX_SNAPSHOT_VERSION: u16 = 2;
const QWEN35_PREFIX_SNAPSHOT_FIXED_HEADER_LEN: usize = 14;

#[derive(serde::Serialize, serde::Deserialize)]
struct Qwen35PrefixSnapshotHeader {
    model_fingerprint: Vec<u8>,
    token_ids: Vec<u32>,
    cache_len: i32,
    kv_capacity: i32,
    metadata_checksum: Vec<u8>,
    body_checksum: Vec<u8>,
    kv_flat: Vec<MlxArrayBytesHeader>,
    gdr_flat: Vec<MlxArrayBytesHeader>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct MlxArrayBytesHeader {
    name: String,
    dtype: i32,
    shape: Vec<i32>,
    byte_len: u64,
}

impl Qwen35PrefixSnapshot {
    pub(crate) fn encode_for_disk(&self, model_fingerprint: &[u8]) -> Result<Vec<u8>> {
        ensure!(
            !model_fingerprint.is_empty(),
            "Qwen3.5 prefix snapshot encode requires a model fingerprint"
        );
        ensure!(
            self.cache_len > 0 && self.kv_capacity >= self.cache_len,
            "Qwen3.5 prefix snapshot has invalid cache_len/capacity: {}/{}",
            self.cache_len,
            self.kv_capacity
        );
        ensure!(
            self.token_ids.len() == self.cache_len as usize,
            "Qwen3.5 prefix snapshot token count {} does not match cache_len {}",
            self.token_ids.len(),
            self.cache_len
        );

        let mut body = Vec::new();
        let kv_flat = encode_mlx_array_headers("kv", &self.kv_flat, &mut body)?;
        let gdr_flat = encode_mlx_array_headers("gdr", &self.gdr_flat, &mut body)?;
        let metadata_checksum = qwen35_prefix_snapshot_metadata_checksum(
            model_fingerprint,
            &self.token_ids,
            self.cache_len,
            self.kv_capacity,
            &kv_flat,
            &gdr_flat,
        );
        let body_checksum = blake3::hash(&body).as_bytes().to_vec();
        let header = Qwen35PrefixSnapshotHeader {
            model_fingerprint: model_fingerprint.to_vec(),
            token_ids: self.token_ids.clone(),
            cache_len: self.cache_len,
            kv_capacity: self.kv_capacity,
            metadata_checksum,
            body_checksum,
            kv_flat,
            gdr_flat,
        };
        let header_bytes =
            postcard::to_allocvec(&header).context("encode Qwen3.5 prefix snapshot header")?;
        let header_len = u32::try_from(header_bytes.len())
            .context("Qwen3.5 prefix snapshot header is too large")?;

        let mut payload = Vec::with_capacity(
            QWEN35_PREFIX_SNAPSHOT_FIXED_HEADER_LEN + header_bytes.len() + body.len(),
        );
        payload.extend_from_slice(&QWEN35_PREFIX_SNAPSHOT_MAGIC);
        payload.extend_from_slice(&QWEN35_PREFIX_SNAPSHOT_VERSION.to_le_bytes());
        payload.extend_from_slice(&header_len.to_le_bytes());
        payload.extend_from_slice(&header_bytes);
        payload.extend_from_slice(&body);
        Ok(payload)
    }

    pub(crate) fn estimated_disk_payload_len(&self, model_fingerprint: &[u8]) -> Result<u64> {
        ensure!(
            !model_fingerprint.is_empty(),
            "Qwen3.5 prefix snapshot estimate requires a model fingerprint"
        );
        ensure!(
            self.cache_len > 0 && self.kv_capacity >= self.cache_len,
            "Qwen3.5 prefix snapshot has invalid cache_len/capacity: {}/{}",
            self.cache_len,
            self.kv_capacity
        );
        ensure!(
            self.token_ids.len() == self.cache_len as usize,
            "Qwen3.5 prefix snapshot token count {} does not match cache_len {}",
            self.token_ids.len(),
            self.cache_len
        );

        let (kv_flat, kv_bytes) = describe_mlx_array_headers("kv", &self.kv_flat)?;
        let (gdr_flat, gdr_bytes) = describe_mlx_array_headers("gdr", &self.gdr_flat)?;
        let metadata_checksum = qwen35_prefix_snapshot_metadata_checksum(
            model_fingerprint,
            &self.token_ids,
            self.cache_len,
            self.kv_capacity,
            &kv_flat,
            &gdr_flat,
        );
        let header = Qwen35PrefixSnapshotHeader {
            model_fingerprint: model_fingerprint.to_vec(),
            token_ids: self.token_ids.clone(),
            cache_len: self.cache_len,
            kv_capacity: self.kv_capacity,
            metadata_checksum,
            body_checksum: vec![0; blake3::OUT_LEN],
            kv_flat,
            gdr_flat,
        };
        let header_bytes = postcard::to_allocvec(&header)
            .context("encode Qwen3.5 prefix snapshot header estimate")?;
        let fixed_len = u64::try_from(QWEN35_PREFIX_SNAPSHOT_FIXED_HEADER_LEN)
            .context("Qwen3.5 prefix fixed header length exceeds u64")?;
        let header_len = u64::try_from(header_bytes.len())
            .context("Qwen3.5 prefix snapshot header length exceeds u64")?;
        fixed_len
            .checked_add(header_len)
            .and_then(|len| len.checked_add(kv_bytes))
            .and_then(|len| len.checked_add(gdr_bytes))
            .context("Qwen3.5 prefix snapshot estimated payload length overflow")
    }

    pub(crate) fn decode_from_disk(
        bytes: &[u8],
        expected_model_fingerprint: &[u8],
    ) -> Result<Self> {
        let (header, body) =
            decode_qwen35_prefix_snapshot_header(bytes, expected_model_fingerprint, true)?;
        let mut cursor = 0usize;
        let kv_flat = decode_mlx_array_headers("kv", &header.kv_flat, body, &mut cursor)?;
        let gdr_flat = decode_mlx_array_headers("gdr", &header.gdr_flat, body, &mut cursor)?;
        ensure!(
            cursor == body.len(),
            "Qwen3.5 prefix snapshot body has {} trailing bytes",
            body.len() - cursor
        );

        Ok(Self {
            token_ids: header.token_ids,
            kv_flat,
            gdr_flat,
            cache_len: header.cache_len,
            kv_capacity: header.kv_capacity,
        })
    }

    pub(crate) fn peek_disk_token_ids(
        bytes: &[u8],
        expected_model_fingerprint: &[u8],
    ) -> Result<Vec<u32>> {
        let (header, _body) =
            decode_qwen35_prefix_snapshot_header(bytes, expected_model_fingerprint, false)?;
        Ok(header.token_ids)
    }

    pub(crate) fn looks_like_disk_payload(bytes: &[u8]) -> bool {
        bytes.starts_with(&QWEN35_PREFIX_SNAPSHOT_MAGIC)
    }
}

fn decode_qwen35_prefix_snapshot_header<'a>(
    bytes: &'a [u8],
    expected_model_fingerprint: &[u8],
    validate_body: bool,
) -> Result<(Qwen35PrefixSnapshotHeader, &'a [u8])> {
    ensure!(
        !expected_model_fingerprint.is_empty(),
        "Qwen3.5 prefix snapshot decode requires a model fingerprint"
    );
    ensure!(
        bytes.len() >= QWEN35_PREFIX_SNAPSHOT_FIXED_HEADER_LEN,
        "Qwen3.5 prefix snapshot payload is shorter than fixed header"
    );

    let magic: [u8; 8] = bytes[..8]
        .try_into()
        .context("read Qwen3.5 prefix snapshot magic")?;
    ensure!(
        magic == QWEN35_PREFIX_SNAPSHOT_MAGIC,
        "Qwen3.5 prefix snapshot has invalid magic"
    );

    let version = u16::from_le_bytes(
        bytes[8..10]
            .try_into()
            .context("read Qwen3.5 prefix snapshot version")?,
    );
    ensure!(
        version == QWEN35_PREFIX_SNAPSHOT_VERSION,
        "Qwen3.5 prefix snapshot version {} is unsupported",
        version
    );

    let header_len = u32::from_le_bytes(
        bytes[10..14]
            .try_into()
            .context("read Qwen3.5 prefix snapshot header length")?,
    ) as usize;
    let header_end = QWEN35_PREFIX_SNAPSHOT_FIXED_HEADER_LEN
        .checked_add(header_len)
        .context("Qwen3.5 prefix snapshot header length overflow")?;
    ensure!(
        bytes.len() >= header_end,
        "Qwen3.5 prefix snapshot payload is shorter than declared header"
    );

    let header: Qwen35PrefixSnapshotHeader =
        postcard::from_bytes(&bytes[QWEN35_PREFIX_SNAPSHOT_FIXED_HEADER_LEN..header_end])
            .context("decode Qwen3.5 prefix snapshot header")?;
    ensure!(
        header.model_fingerprint == expected_model_fingerprint,
        "Qwen3.5 prefix snapshot model fingerprint mismatch"
    );
    ensure!(
        header.cache_len > 0 && header.kv_capacity >= header.cache_len,
        "Qwen3.5 prefix snapshot has invalid cache_len/capacity: {}/{}",
        header.cache_len,
        header.kv_capacity
    );
    ensure!(
        header.token_ids.len() == header.cache_len as usize,
        "Qwen3.5 prefix snapshot token count {} does not match cache_len {}",
        header.token_ids.len(),
        header.cache_len
    );
    validate_qwen35_prefix_snapshot_metadata_checksum(&header)?;

    let body = &bytes[header_end..];
    if validate_body {
        validate_qwen35_prefix_snapshot_body(&header, body)?;
    }
    Ok((header, body))
}

fn qwen35_prefix_snapshot_metadata_checksum(
    model_fingerprint: &[u8],
    token_ids: &[u32],
    cache_len: i32,
    kv_capacity: i32,
    kv_flat: &[MlxArrayBytesHeader],
    gdr_flat: &[MlxArrayBytesHeader],
) -> Vec<u8> {
    fn update_bytes(hasher: &mut blake3::Hasher, label: &[u8], bytes: &[u8]) {
        hasher.update(label);
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }

    fn update_array_records(
        hasher: &mut blake3::Hasher,
        label: &[u8],
        records: &[MlxArrayBytesHeader],
    ) {
        hasher.update(label);
        hasher.update(&(records.len() as u64).to_le_bytes());
        for record in records {
            update_bytes(hasher, b"name\0", record.name.as_bytes());
            hasher.update(b"dtype\0");
            hasher.update(&record.dtype.to_le_bytes());
            hasher.update(b"shape\0");
            hasher.update(&(record.shape.len() as u64).to_le_bytes());
            for dim in &record.shape {
                hasher.update(&dim.to_le_bytes());
            }
            hasher.update(b"byte_len\0");
            hasher.update(&record.byte_len.to_le_bytes());
        }
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(b"arle-qwen35-prefix-snapshot-metadata-v1\0");
    update_bytes(&mut hasher, b"model\0", model_fingerprint);
    hasher.update(b"tokens\0");
    hasher.update(&(token_ids.len() as u64).to_le_bytes());
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    hasher.update(b"cache_len\0");
    hasher.update(&cache_len.to_le_bytes());
    hasher.update(b"kv_capacity\0");
    hasher.update(&kv_capacity.to_le_bytes());
    update_array_records(&mut hasher, b"kv_flat\0", kv_flat);
    update_array_records(&mut hasher, b"gdr_flat\0", gdr_flat);
    hasher.finalize().as_bytes().to_vec()
}

fn validate_qwen35_prefix_snapshot_metadata_checksum(
    header: &Qwen35PrefixSnapshotHeader,
) -> Result<()> {
    ensure!(
        header.metadata_checksum.len() == blake3::OUT_LEN,
        "Qwen3.5 prefix snapshot metadata checksum has invalid length {}",
        header.metadata_checksum.len()
    );
    let actual = qwen35_prefix_snapshot_metadata_checksum(
        &header.model_fingerprint,
        &header.token_ids,
        header.cache_len,
        header.kv_capacity,
        &header.kv_flat,
        &header.gdr_flat,
    );
    ensure!(
        header.metadata_checksum == actual,
        "Qwen3.5 prefix snapshot metadata checksum mismatch"
    );
    Ok(())
}

fn validate_qwen35_prefix_snapshot_body(
    header: &Qwen35PrefixSnapshotHeader,
    body: &[u8],
) -> Result<()> {
    let mut cursor = 0usize;
    validate_mlx_array_header_layout("kv", &header.kv_flat, body.len(), &mut cursor)?;
    validate_mlx_array_header_layout("gdr", &header.gdr_flat, body.len(), &mut cursor)?;
    ensure!(
        header.body_checksum.len() == blake3::OUT_LEN,
        "Qwen3.5 prefix snapshot body checksum has invalid length {}",
        header.body_checksum.len()
    );
    let actual_body_checksum = blake3::hash(body);
    ensure!(
        header.body_checksum.as_slice() == actual_body_checksum.as_bytes(),
        "Qwen3.5 prefix snapshot body checksum mismatch"
    );
    ensure!(
        cursor == body.len(),
        "Qwen3.5 prefix snapshot body has {} trailing bytes",
        body.len() - cursor
    );
    Ok(())
}

fn validate_mlx_array_header_layout(
    prefix: &str,
    records: &[MlxArrayBytesHeader],
    body_len: usize,
    cursor: &mut usize,
) -> Result<()> {
    for (idx, record) in records.iter().enumerate() {
        let expected_name = format!("{prefix}.{idx}");
        ensure!(
            record.name == expected_name,
            "Qwen3.5 prefix snapshot array name mismatch: got {}, expected {}",
            record.name,
            expected_name
        );
        ensure!(
            super::mlx::Dtype::from_raw(record.dtype).is_some(),
            "Qwen3.5 prefix snapshot array {} has unknown dtype {}",
            record.name,
            record.dtype
        );
        let byte_len = usize::try_from(record.byte_len)
            .context("Qwen3.5 prefix snapshot array byte length exceeds usize")?;
        let end = cursor
            .checked_add(byte_len)
            .context("Qwen3.5 prefix snapshot array byte range overflow")?;
        ensure!(
            end <= body_len,
            "Qwen3.5 prefix snapshot body is truncated while reading {}",
            record.name
        );
        *cursor = end;
    }
    Ok(())
}

fn encode_mlx_array_headers(
    prefix: &str,
    arrays: &[MlxArray],
    body: &mut Vec<u8>,
) -> Result<Vec<MlxArrayBytesHeader>> {
    arrays
        .iter()
        .enumerate()
        .map(|(idx, array)| {
            let bytes = array
                .to_bytes()
                .with_context(|| format!("export Qwen3.5 prefix array {prefix}.{idx}"))?;
            body.extend_from_slice(&bytes);
            Ok(MlxArrayBytesHeader {
                name: format!("{prefix}.{idx}"),
                dtype: array.dtype().to_raw(),
                shape: array.shape().to_vec(),
                byte_len: u64::try_from(bytes.len())
                    .context("Qwen3.5 prefix array byte length exceeds u64")?,
            })
        })
        .collect()
}

fn describe_mlx_array_headers(
    prefix: &str,
    arrays: &[MlxArray],
) -> Result<(Vec<MlxArrayBytesHeader>, u64)> {
    let mut total_bytes = 0u64;
    let mut headers = Vec::with_capacity(arrays.len());
    for (idx, array) in arrays.iter().enumerate() {
        let byte_len = u64::try_from(array.nbytes())
            .context("Qwen3.5 prefix array byte length exceeds u64")?;
        total_bytes = total_bytes
            .checked_add(byte_len)
            .context("Qwen3.5 prefix array byte length sum overflow")?;
        headers.push(MlxArrayBytesHeader {
            name: format!("{prefix}.{idx}"),
            dtype: array.dtype().to_raw(),
            shape: array.shape().to_vec(),
            byte_len,
        });
    }
    Ok((headers, total_bytes))
}

fn decode_mlx_array_headers(
    prefix: &str,
    records: &[MlxArrayBytesHeader],
    body: &[u8],
    cursor: &mut usize,
) -> Result<Vec<MlxArray>> {
    records
        .iter()
        .enumerate()
        .map(|(idx, record)| {
            let expected_name = format!("{prefix}.{idx}");
            ensure!(
                record.name == expected_name,
                "Qwen3.5 prefix snapshot array name mismatch: got {}, expected {}",
                record.name,
                expected_name
            );
            let dtype = super::mlx::Dtype::from_raw(record.dtype).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 prefix snapshot array {} has unknown dtype {}",
                    record.name,
                    record.dtype
                )
            })?;
            let byte_len = usize::try_from(record.byte_len)
                .context("Qwen3.5 prefix snapshot array byte length exceeds usize")?;
            let end = cursor
                .checked_add(byte_len)
                .context("Qwen3.5 prefix snapshot array byte range overflow")?;
            ensure!(
                end <= body.len(),
                "Qwen3.5 prefix snapshot body is truncated while reading {}",
                record.name
            );
            let array = MlxArray::from_bytes(&body[*cursor..end], &record.shape, dtype)
                .with_context(|| format!("import Qwen3.5 prefix array {}", record.name))?;
            *cursor = end;
            Ok(array)
        })
        .collect()
}

pub(crate) struct Qwen35PackedDecodeBatch<'a> {
    weights: &'a Qwen35MetalWeights,
    config: &'a MetalModelConfig,
    arch: &'a super::config::MetalQwen35ArchConfig,
    batch_cache_len: i32,
    kv_capacity: i32,
    left_padding: Vec<i32>,
    n_kv_per_request: i32,
    n_gdr_per_request: i32,
    packed_kv_flat: Vec<MlxArray>,
    packed_gdr_flat: Vec<MlxArray>,
}

impl<'a> Qwen35PackedDecodeBatch<'a> {
    pub(crate) fn batch_size(&self) -> usize {
        self.packed_kv_flat
            .first()
            .or_else(|| self.packed_gdr_flat.first())
            .map_or(0, |array| {
                array.shape().first().copied().unwrap_or(0).max(0) as usize
            })
    }

    /// Shared column cursor for this packed batch. All rows write their next
    /// decode token at this column; rows that joined late are left-padded so
    /// their valid KV data sits in `[left_padding[i], batch_cache_len)`.
    pub(crate) fn batch_cache_len(&self) -> i32 {
        self.batch_cache_len
    }

    fn matches_driver(&self, driver: &Qwen35StepDriver<'a>) -> bool {
        std::ptr::eq(self.weights, driver.weights)
            && std::ptr::eq(self.config, driver.config)
            && std::ptr::eq(self.arch, driver.arch)
            && matches!(driver.mode, Qwen35StepMode::Cpp(_))
    }

    fn ensure_capacity_for_states(
        &mut self,
        states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
        needed_tokens: i32,
    ) {
        for state in states.iter_mut() {
            state.driver.kv_capacity = state.driver.kv_capacity.max(self.kv_capacity);
        }
        while needed_tokens > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            for cache in &mut self.packed_kv_flat {
                extend_kv_cache(
                    cache,
                    self.config.num_key_value_heads as i32,
                    self.config.head_dim as i32,
                    new_cap,
                );
            }
            self.kv_capacity = new_cap;
            for state in states.iter_mut() {
                state.driver.kv_capacity = new_cap;
            }
        }
    }

    pub(crate) fn retain_rows(&mut self, row_indices: &[usize]) -> Result<()> {
        if row_indices.len() == self.batch_size()
            && row_indices.iter().enumerate().all(|(idx, row)| idx == *row)
        {
            return Ok(());
        }

        let indices: Vec<i32> = row_indices
            .iter()
            .map(|&row| i32::try_from(row).context("Qwen3.5 packed batch row index overflow"))
            .collect::<Result<_>>()?;
        let index_arr = MlxArray::from_slice_i32(
            &indices,
            &[i32::try_from(indices.len()).context("Qwen3.5 packed batch row count overflow")?],
        );

        for tensor in &mut self.packed_kv_flat {
            let old = std::mem::replace(tensor, take_axis(tensor, &index_arr, 0));
            drop(old);
        }
        for tensor in &mut self.packed_gdr_flat {
            let old = std::mem::replace(tensor, take_axis(tensor, &index_arr, 0));
            drop(old);
        }
        self.left_padding = row_indices
            .iter()
            .map(|&row| self.left_padding[row])
            .collect();

        let mut eval_refs =
            Vec::with_capacity(self.packed_kv_flat.len() + self.packed_gdr_flat.len());
        eval_refs.extend(self.packed_kv_flat.iter());
        eval_refs.extend(self.packed_gdr_flat.iter());
        let eval_refs: Vec<&MlxArray> = eval_refs.into_iter().collect();
        eval(&eval_refs);
        Ok(())
    }

    pub(crate) fn admit_rows(
        &mut self,
        states: &mut [&mut MetalRequestState<'a>],
        new_indices: &[usize],
    ) -> Result<()> {
        if new_indices.is_empty() {
            return Ok(());
        }

        let mut target_kv_capacity = self.kv_capacity;
        for &idx in new_indices {
            let Some(state) = states.get(idx) else {
                bail!("Qwen3.5 packed batch admit_rows index {idx} out of range");
            };
            let state_ref: &MetalRequestState<'a> = state;
            let MetalRequestStateInner::Qwen35(qwen35) = &state_ref.inner else {
                bail!("Qwen3.5 packed batch admit_rows received mixed model batch");
            };
            if qwen35.phase() != MetalRequestPhase::Decode {
                bail!("Qwen3.5 packed batch admit_rows requires Decode phase");
            }
            target_kv_capacity = target_kv_capacity.max(qwen35.driver.kv_capacity);
        }
        target_kv_capacity = round_up_kv_capacity(target_kv_capacity);
        while target_kv_capacity > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            for cache in &mut self.packed_kv_flat {
                extend_kv_cache(
                    cache,
                    self.config.num_key_value_heads as i32,
                    self.config.head_dim as i32,
                    new_cap,
                );
            }
            self.kv_capacity = new_cap;
        }

        let mut new_kv_rows: Vec<Vec<MlxArray>> = (0..self.n_kv_per_request)
            .map(|_| Vec::with_capacity(new_indices.len()))
            .collect();
        let mut new_gdr_rows: Vec<Vec<MlxArray>> = (0..self.n_gdr_per_request)
            .map(|_| Vec::with_capacity(new_indices.len()))
            .collect();
        let mut new_left_padding = Vec::with_capacity(new_indices.len());

        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            if let MetalRequestStateInner::Qwen35(qwen35) = &mut state_ref.inner {
                if self.matches_driver(&qwen35.driver) {
                    qwen35.driver.kv_capacity = qwen35.driver.kv_capacity.max(self.kv_capacity);
                }
            }
        }

        for &idx in new_indices {
            let state_ref: &mut MetalRequestState<'a> = states.get_mut(idx).ok_or_else(|| {
                anyhow::anyhow!("Qwen3.5 packed batch admit_rows index {idx} out of range")
            })?;
            let MetalRequestStateInner::Qwen35(qwen35) = &mut state_ref.inner else {
                bail!("Qwen3.5 packed batch admit_rows received mixed model batch");
            };
            ensure!(
                qwen35.phase() == MetalRequestPhase::Decode,
                "Qwen3.5 packed batch admit_rows requires Decode phase"
            );
            ensure!(
                std::ptr::eq(qwen35.driver.weights, self.weights)
                    && std::ptr::eq(qwen35.driver.config, self.config)
                    && std::ptr::eq(qwen35.driver.arch, self.arch),
                "Qwen3.5 packed batch admit_rows requires matching model handles"
            );

            qwen35.driver.ensure_cpp_session_drained()?;
            qwen35.driver.ensure_capacity(self.kv_capacity)?;
            qwen35.driver.kv_capacity = self.kv_capacity;
            let left_pad = self.batch_cache_len - qwen35.driver.cache_len;
            ensure!(
                left_pad >= 0,
                "Qwen3.5 packed batch cannot admit cache_len {} into batch_cache_len {}",
                qwen35.driver.cache_len,
                self.batch_cache_len
            );

            match &mut qwen35.driver.mode {
                Qwen35StepMode::Cpp(cpp) => {
                    ensure!(
                        i32::try_from(cpp.kv_flat.len())
                            .context("Qwen3.5 packed batch admit_rows kv count overflow")?
                            == self.n_kv_per_request
                            && i32::try_from(cpp.gdr_flat.len())
                                .context("Qwen3.5 packed batch admit_rows gdr count overflow")?
                                == self.n_gdr_per_request,
                        "Qwen3.5 packed batch admit_rows requires matching state vector counts"
                    );
                    for (slot_idx, slot) in cpp.kv_flat.iter().enumerate() {
                        new_kv_rows[slot_idx].push(left_pad_kv_cache_row(
                            slot,
                            left_pad,
                            qwen35.driver.cache_len,
                            self.kv_capacity,
                        ));
                    }
                    for (slot_idx, slot) in cpp.gdr_flat.iter().enumerate() {
                        new_gdr_rows[slot_idx].push(slot.clone());
                    }
                }
                Qwen35StepMode::Rust(_) => {
                    bail!("Qwen3.5 packed batch admit_rows requires compiled Qwen3.5 state")
                }
            }

            new_left_padding.push(left_pad);
        }

        for (slot_idx, appended_rows) in new_kv_rows.iter_mut().enumerate() {
            let mut concatenated = Vec::with_capacity(1 + appended_rows.len());
            concatenated.push(self.packed_kv_flat[slot_idx].clone());
            concatenated.append(appended_rows);
            let old = std::mem::replace(
                &mut self.packed_kv_flat[slot_idx],
                concatenate_axis(&concatenated, 0),
            );
            drop(old);
        }
        for (slot_idx, appended_rows) in new_gdr_rows.iter_mut().enumerate() {
            let mut concatenated = Vec::with_capacity(1 + appended_rows.len());
            concatenated.push(self.packed_gdr_flat[slot_idx].clone());
            concatenated.append(appended_rows);
            let old = std::mem::replace(
                &mut self.packed_gdr_flat[slot_idx],
                concatenate_axis(&concatenated, 0),
            );
            drop(old);
        }
        self.left_padding.extend(new_left_padding);

        let mut eval_refs =
            Vec::with_capacity(self.packed_kv_flat.len() + self.packed_gdr_flat.len());
        eval_refs.extend(self.packed_kv_flat.iter());
        eval_refs.extend(self.packed_gdr_flat.iter());
        let eval_refs: Vec<&MlxArray> = eval_refs.into_iter().collect();
        eval(&eval_refs);
        Ok(())
    }
}

/// Union of per-request `stop_token_ids` and the model's resolved
/// `stop_token_ids` (HF eos array). Preserves first-seen order, dedups.
/// Skips the model stops when `ignore_eos` is set, matching the C++
/// generate paths so benchmarks can still generate past EOS.
fn merge_stop_ids(request_stops: &[u32], config: &MetalModelConfig, ignore_eos: bool) -> Vec<u32> {
    let model_stops: &[u32] = if ignore_eos {
        &[]
    } else {
        &config.stop_token_ids
    };
    let mut out = Vec::with_capacity(request_stops.len() + model_stops.len());
    for id in request_stops.iter().chain(model_stops.iter()) {
        if !out.contains(id) {
            out.push(*id);
        }
    }
    out
}

impl<'a> MetalRequestState<'a> {
    pub(super) fn new(
        weights: &'a MetalWeights,
        config: &'a MetalModelConfig,
        prompt_tokens: Vec<u32>,
        params: &SamplingParams,
        use_kv_pool: bool,
        max_new_tokens: usize,
        dflash_runtime: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)>,
    ) -> Result<Self> {
        validate_metal_sampling_params(params)?;

        let inner = match weights {
            MetalWeights::Qwen3(weights) => {
                // DFlash needs direct KV cache access — disable pool when DFlash is active.
                let effective_kv_pool = use_kv_pool && dflash_runtime.is_none();
                let driver = Qwen3StepDriver::new(
                    weights,
                    config,
                    params,
                    effective_kv_pool,
                    &prompt_tokens,
                    max_new_tokens,
                    dflash_runtime,
                )?;
                let state = ResumableRequestState::new(
                    driver,
                    prompt_tokens,
                    max_new_tokens,
                    merge_stop_ids(&params.stop_token_ids, config, params.ignore_eos),
                    config.eos_token_id,
                    params.ignore_eos,
                )?;
                MetalRequestStateInner::Qwen3(Box::new(state))
            }
            MetalWeights::Qwen35(weights) => {
                let driver = Qwen35StepDriver::new(
                    weights,
                    config,
                    params,
                    &prompt_tokens,
                    max_new_tokens,
                    dflash_runtime,
                )?;
                let state = ResumableRequestState::new(
                    driver,
                    prompt_tokens,
                    max_new_tokens,
                    merge_stop_ids(&params.stop_token_ids, config, params.ignore_eos),
                    config.eos_token_id,
                    params.ignore_eos,
                )?;
                MetalRequestStateInner::Qwen35(Box::new(state))
            }
        };

        Ok(Self { inner })
    }

    pub fn phase(&self) -> MetalRequestPhase {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.phase(),
            MetalRequestStateInner::Qwen35(state) => state.phase(),
        }
    }

    pub fn prompt_len(&self) -> usize {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.prompt_len(),
            MetalRequestStateInner::Qwen35(state) => state.prompt_len(),
        }
    }

    pub fn prompt_progress(&self) -> usize {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.prompt_progress(),
            MetalRequestStateInner::Qwen35(state) => state.prompt_progress(),
        }
    }

    pub fn generated_tokens(&self) -> usize {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.generated_tokens(),
            MetalRequestStateInner::Qwen35(state) => state.generated_tokens(),
        }
    }

    pub fn last_token(&self) -> Option<u32> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.last_token,
            MetalRequestStateInner::Qwen35(state) => state.last_token,
        }
    }

    pub fn is_qwen3(&self) -> bool {
        matches!(self.inner, MetalRequestStateInner::Qwen3(_))
    }

    pub fn is_qwen35(&self) -> bool {
        matches!(self.inner, MetalRequestStateInner::Qwen35(_))
    }

    /// Drain a live Qwen3.5 C++ session back into this request's owned state.
    ///
    /// The compiled MLX model supports exactly one active session at a time.
    /// Runtime code calls this before starting another Qwen3.5 request's
    /// prefill so scalar decode and prefill cannot overlap through nested
    /// `qwen35_session_begin` calls.
    pub(crate) fn drain_qwen35_cpp_session(&mut self) -> Result<bool> {
        let MetalRequestStateInner::Qwen35(state) = &mut self.inner else {
            return Ok(false);
        };
        let was_active = state.driver.cpp_session_active();
        state.driver.ensure_cpp_session_drained()?;
        Ok(was_active)
    }

    pub(crate) fn can_import_qwen35_prefix_snapshot(&self) -> bool {
        match &self.inner {
            MetalRequestStateInner::Qwen35(state) => state.driver.can_import_prefix_snapshot(),
            MetalRequestStateInner::Qwen3(_) => false,
        }
    }

    pub fn finish_reason(&self) -> Option<&'static str> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.finish_reason(),
            MetalRequestStateInner::Qwen35(state) => state.finish_reason(),
        }
    }

    pub fn kv_pool_usage(&self) -> Option<(usize, usize)> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state
                .driver
                .kv_pool
                .as_ref()
                .map(|pool| (pool.total_tokens_used(), pool.max_total_tokens())),
            MetalRequestStateInner::Qwen35(_) => None,
        }
    }

    pub fn prefill_chunk(&mut self, budget: usize) -> Result<PrefillChunkResult> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.prefill_chunk(budget),
            MetalRequestStateInner::Qwen35(state) => {
                let trace = metal_qwen35_trace_enabled();
                let started = trace.then(Instant::now);
                let prompt_before = state.prompt_progress();
                let phase_before = state.phase();
                let result = state.prefill_chunk(budget);
                if let (true, Some(started), Ok(chunk)) = (trace, started, result.as_ref()) {
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "metal_trace[qwen35_prefill_chunk]: phase={:?}->{:?} budget={} prompt={}..{} processed={} emitted={} elapsed_ms={:.1}",
                        phase_before,
                        chunk.phase,
                        budget,
                        prompt_before,
                        prompt_before + chunk.processed_tokens,
                        chunk.processed_tokens,
                        chunk.emitted_token.is_some(),
                        elapsed_ms,
                    );
                }
                result
            }
        }
    }

    pub fn decode_step(&mut self) -> Result<Option<u32>> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.decode_step(),
            MetalRequestStateInner::Qwen35(state) => state.decode_step(),
        }
    }

    /// Try to execute one cross-request Qwen3 decode batch.
    ///
    /// Returns `Ok(None)` when the batch is not eligible for the Qwen3 batched
    /// path (for example Qwen3.5 requests or non-decode phases), so the caller
    /// can fall back to per-request decode. Returns sampled tokens in the same
    /// order as the input slice on success.
    pub fn decode_batch(states: &mut [&mut MetalRequestState<'a>]) -> Result<Option<Vec<u32>>> {
        if states.is_empty() {
            return Ok(None);
        }

        match &mut states[0].inner {
            MetalRequestStateInner::Qwen3(_) => {
                let mut qwen3_states = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    match &mut state.inner {
                        MetalRequestStateInner::Qwen3(qwen3) => {
                            if qwen3.phase() != MetalRequestPhase::Decode {
                                return Ok(None);
                            }
                            qwen3_states.push(&mut **qwen3);
                        }
                        MetalRequestStateInner::Qwen35(_) => return Ok(None),
                    }
                }

                if qwen3_states
                    .iter()
                    .any(|state| state.driver.kv_pool.is_some())
                {
                    let first_cache_len = qwen3_states[0].driver.cache_len;
                    if qwen3_states
                        .iter()
                        .any(|state| state.driver.cache_len != first_cache_len)
                    {
                        return Ok(None);
                    }
                }

                let sampled = decode_qwen3_batch(&mut qwen3_states)?;
                Ok(Some(sampled))
            }
            MetalRequestStateInner::Qwen35(_) => {
                let mut qwen35_states = Vec::with_capacity(states.len());
                for state in states.iter_mut() {
                    match &mut state.inner {
                        MetalRequestStateInner::Qwen35(qwen35) => {
                            if qwen35.phase() != MetalRequestPhase::Decode {
                                return Ok(None);
                            }
                            qwen35_states.push(&mut **qwen35);
                        }
                        MetalRequestStateInner::Qwen3(_) => return Ok(None),
                    }
                }

                let first_cache_len = qwen35_states[0].driver.cache_len;
                let first_kv_capacity = qwen35_states[0].driver.kv_capacity;
                let cpp_mode = matches!(qwen35_states[0].driver.mode, Qwen35StepMode::Cpp(_));
                if !cpp_mode
                    || qwen35_states.iter().any(|state| {
                        state.driver.cache_len != first_cache_len
                            || state.driver.kv_capacity != first_kv_capacity
                            || !matches!(state.driver.mode, Qwen35StepMode::Cpp(_))
                    })
                {
                    return Ok(None);
                }

                let sampled = decode_qwen35_batch(&mut qwen35_states)?;
                Ok(Some(sampled))
            }
        }
    }

    /// Try to execute one mixed Metal batch.
    ///
    /// This is the generic runtime-facing entrypoint. The current fused
    /// implementation still only accepts Qwen3 decode rows paired with a
    /// Qwen3 prefill row; other model mixes return `Ok(None)` so the caller
    /// can fall back to the scalar path without changing behavior.
    pub(crate) fn try_mixed_batch(
        decode_states: &mut [&mut MetalRequestState<'a>],
        prefill_state: &mut MetalRequestState<'a>,
        budget: usize,
    ) -> Result<Option<MetalMixedBatchResult>> {
        ensure!(budget > 0, "Metal mixed batch requires budget > 0");

        if decode_states.is_empty() {
            return Ok(None);
        }

        let mut qwen3_decode_states = Vec::with_capacity(decode_states.len());
        for state in decode_states.iter_mut() {
            match &mut state.inner {
                MetalRequestStateInner::Qwen3(qwen3) => {
                    if qwen3.phase() != MetalRequestPhase::Decode
                        || qwen3.driver.kv_pool.is_some()
                        || qwen3.driver.dflash.is_some()
                    {
                        return Ok(None);
                    }
                    qwen3_decode_states.push(&mut **qwen3);
                }
                MetalRequestStateInner::Qwen35(_) => return Ok(None),
            }
        }

        let MetalRequestStateInner::Qwen3(prefill_state) = &mut prefill_state.inner else {
            return Ok(None);
        };
        if prefill_state.phase() != MetalRequestPhase::Prefill
            || prefill_state.driver.kv_pool.is_some()
            || prefill_state.driver.dflash.is_some()
        {
            return Ok(None);
        }

        let remaining = prefill_state.prompt_tokens.len() - prefill_state.prompt_cursor;
        let processed = budget.min(remaining);
        let prompt_end = prefill_state.prompt_cursor + processed;
        let terminal_prompt = prompt_end == prefill_state.prompt_tokens.len();
        let query_tokens =
            prefill_state.prompt_tokens[prefill_state.prompt_cursor..prompt_end].to_vec();
        let (decode_tokens, emitted_token) = {
            let mut rows = Vec::with_capacity(qwen3_decode_states.len() + 1);
            for state in &mut qwen3_decode_states {
                let token = state
                    .last_token
                    .context("Qwen3 packed batch requires committed decode tokens")?;
                rows.push(Qwen3PackedBatchRow {
                    state: std::ptr::from_mut(&mut **state),
                    query_tokens: vec![token],
                    kind: Qwen3PackedBatchRowKind::Decode,
                });
            }
            rows.push(Qwen3PackedBatchRow {
                state: std::ptr::from_mut(&mut **prefill_state),
                query_tokens,
                kind: Qwen3PackedBatchRowKind::Prefill { terminal_prompt },
            });
            let sampled = execute_qwen3_packed_batch(&mut rows)?;
            let emitted_token = sampled.last().copied().flatten();
            let decode_tokens = sampled[..qwen3_decode_states.len()]
                .iter()
                .map(|token| token.context("Metal mixed batch missing sampled decode token"))
                .collect::<Result<Vec<_>>>()?;
            (decode_tokens, emitted_token)
        };
        let prefill = PrefillChunkResult {
            processed_tokens: processed,
            emitted_token,
            phase: prefill_state.phase(),
            finish_reason: prefill_state.finish_reason(),
        };
        Ok(Some(MetalMixedBatchResult {
            decode_tokens,
            prefill,
        }))
    }

    /// Return this request's Qwen3.5 decode cursor (`cache_len`) if it is a
    /// Qwen3.5 request currently in the `Decode` phase. Used by the scheduler
    /// runtime to decide whether a freshly prefilled row can join an existing
    /// packed batch without forcing a full cache rebuild.
    pub(crate) fn qwen35_decode_cursor(&self) -> Option<i32> {
        match &self.inner {
            MetalRequestStateInner::Qwen35(state) if state.phase() == MetalRequestPhase::Decode => {
                Some(state.driver.cache_len)
            }
            MetalRequestStateInner::Qwen3(_) | MetalRequestStateInner::Qwen35(_) => None,
        }
    }

    /// Whether this request has DFlash speculative decode enabled.
    pub(crate) fn is_dflash_enabled(&self) -> bool {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => state.driver.dflash.is_some(),
            MetalRequestStateInner::Qwen35(state) => state.driver.dflash.is_some(),
        }
    }

    /// DFlash aggregate acceptance metrics for this request.
    /// Returns `None` if DFlash is disabled or no speculative block executed.
    pub fn dflash_metrics(&self) -> Option<DflashRequestMetrics> {
        let (block_count, acceptance_lengths, block_size) = self.dflash_block_stats()?;
        let total_accepted: usize = acceptance_lengths.iter().copied().sum();
        let avg_accepted_inputs = total_accepted as f64 / block_count as f64;
        let acceptance_rate = if total_accepted > 0 {
            total_accepted.saturating_sub(block_count) as f64 / total_accepted as f64
        } else {
            0.0
        };
        Some(DflashRequestMetrics {
            block_count,
            block_size,
            avg_accepted_inputs,
            acceptance_rate,
        })
    }

    /// DFlash per-block acceptance lengths + block size for runtime metrics
    /// flush. Returns `(block_count, acceptance_lengths, block_size)` or
    /// `None` if not a DFlash request.
    pub(crate) fn dflash_block_stats(&self) -> Option<(usize, &[usize], usize)> {
        match &self.inner {
            MetalRequestStateInner::Qwen3(state) => {
                let d = state.driver.dflash.as_ref()?;
                Some((
                    d.acceptance_lengths.len(),
                    &d.acceptance_lengths,
                    d.runtime.block_size(),
                ))
            }
            MetalRequestStateInner::Qwen35(state) => {
                let d = state.driver.dflash.as_ref()?;
                Some((
                    d.acceptance_lengths.len(),
                    &d.acceptance_lengths,
                    d.runtime.block_size(),
                ))
            }
        }
    }

    pub(crate) fn try_build_qwen35_packed_decode_batch(
        states: &mut [&mut MetalRequestState<'a>],
    ) -> Result<Option<Qwen35PackedDecodeBatch<'a>>> {
        if states.is_empty() {
            return Ok(None);
        }

        let mut qwen35_states = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => {
                    if qwen35.phase() != MetalRequestPhase::Decode {
                        return Ok(None);
                    }
                    qwen35_states.push(&mut **qwen35);
                }
                MetalRequestStateInner::Qwen3(_) => return Ok(None),
            }
        }

        try_build_qwen35_packed_decode_batch(&mut qwen35_states)
    }

    pub(crate) fn try_decode_qwen35_packed_batch(
        states: &mut [&mut MetalRequestState<'a>],
        batch: &mut Qwen35PackedDecodeBatch<'a>,
    ) -> Result<Option<Vec<u32>>> {
        if states.is_empty() {
            return Ok(None);
        }

        let mut qwen35_states = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => {
                    if qwen35.phase() != MetalRequestPhase::Decode
                        || !batch.matches_driver(&qwen35.driver)
                    {
                        return Ok(None);
                    }
                    qwen35_states.push(&mut **qwen35);
                }
                MetalRequestStateInner::Qwen3(_) => return Ok(None),
            }
        }

        if qwen35_states.len() != batch.batch_size() {
            return Ok(None);
        }

        let sampled = decode_qwen35_packed_batch(&mut qwen35_states, batch)?;
        Ok(Some(sampled))
    }

    pub(crate) fn sync_qwen35_packed_decode_batch(
        states: &mut [&mut MetalRequestState<'a>],
        batch: &Qwen35PackedDecodeBatch<'a>,
    ) -> Result<()> {
        let mut qwen35_states = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => qwen35_states.push(&mut **qwen35),
                MetalRequestStateInner::Qwen3(_) => {
                    bail!("sync_qwen35_packed_decode_batch received mixed model batch")
                }
            }
        }

        sync_qwen35_packed_decode_batch(&mut qwen35_states, batch)
    }

    /// Phase-2B batched DFlash speculative decode with per-row filter.
    ///
    /// Partitions the input slice into a READY subset (every eligibility
    /// predicate holds, and all ready rows agree on cross-row shape/handle
    /// invariants anchored on the first ready row) and a STALE subset
    /// (everyone else). If `ready.len() >= 2`, dispatches one
    /// `qwen35_dflash_speculative_block_batched` call across the ready rows
    /// only, fanning per-row outputs (sampled tokens, cache_pos, KV/GDR,
    /// updated `target_hidden`) back into each `MetalRequestState`. Remaining
    /// accepted tokens land in each row's `token_buffer` for scalar drain on
    /// the next tick.
    ///
    /// Returns:
    ///   - `Ok(None)` when fewer than two rows are ready — caller runs every
    ///     row through scalar `execute_decode_single`.
    ///   - `Ok(Some(outcome))` when at least two rows were batched; callers
    ///     MUST run every input index NOT in `outcome.ready_indices` through
    ///     `execute_decode_single` themselves. Stale rows are left untouched
    ///     by this function.
    ///
    /// Per-row eligibility:
    ///   - Qwen3.5 Decode phase, `Qwen35StepMode::Cpp`, DFlash enabled.
    ///   - Captured `target_hidden` (post-prefill).
    ///   - Empty `token_buffer` (still draining a prior block → scalar path).
    ///   - Committed `last_token` (post-prefill invariant).
    ///   - `runtime.batched_draft_path_eligible()` — scalar draft routing
    ///     gates hold.
    ///
    /// Cross-row eligibility (checked vs the first per-row-ready anchor):
    ///   - Identical `cache_len`, `target_hidden.shape()[0]`,
    ///     `draft_state.active_len()`.
    ///   - Shared `weights`/`config`/`arch`/DFlash `runtime`/`config` ptrs.
    pub(crate) fn try_decode_qwen35_dflash_speculative_batch(
        states: &mut [&mut MetalRequestState<'a>],
    ) -> Result<Option<DflashBatchOutcome>> {
        if states.len() < 2 {
            return Ok(None);
        }

        // Current caller (`execute_qwen35_dflash_packed_batch`) only passes
        // DFlash-enabled rows, which are always Qwen3.5. Defensive bail if a
        // Qwen3 row ever leaks through: the inner function requires a uniform
        // `ResumableRequestState<Qwen35StepDriver>` slice, so there's no
        // middle ground — demote the whole bucket to scalar.
        let mut qwen35_states: Vec<&mut ResumableRequestState<Qwen35StepDriver<'a>>> =
            Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let state_ref: &mut MetalRequestState<'a> = state;
            match &mut state_ref.inner {
                MetalRequestStateInner::Qwen35(qwen35) => qwen35_states.push(qwen35),
                MetalRequestStateInner::Qwen3(_) => return Ok(None),
            }
        }

        try_decode_qwen35_dflash_speculative_batch(&mut qwen35_states)
    }

    pub fn cancel(&mut self) -> Result<()> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen3(state) => state.cancel(),
            MetalRequestStateInner::Qwen35(state) => state.cancel(),
        }
    }

    pub(crate) fn import_qwen35_prefix_snapshot(
        &mut self,
        snapshot: &Qwen35PrefixSnapshot,
        matched_len: usize,
    ) -> Result<bool> {
        match &mut self.inner {
            MetalRequestStateInner::Qwen35(state) => {
                ensure!(
                    state.phase == MetalRequestPhase::Prefill,
                    "Qwen3.5 prefix import requires Prefill phase, got {:?}",
                    state.phase
                );
                ensure!(
                    state.prompt_cursor == 0 && state.generated_tokens == 0,
                    "Qwen3.5 prefix import requires a fresh request state"
                );
                ensure!(
                    matched_len == snapshot.token_ids.len(),
                    "Qwen3.5 prefix import length {} does not match snapshot {}",
                    matched_len,
                    snapshot.token_ids.len()
                );
                ensure!(
                    snapshot.cache_len == matched_len as i32,
                    "Qwen3.5 prefix snapshot cache_len {} does not match {} tokens",
                    snapshot.cache_len,
                    matched_len
                );
                if matched_len == 0 || matched_len >= state.prompt_tokens.len() {
                    return Ok(false);
                }

                state
                    .driver
                    .import_prefix_snapshot(snapshot)
                    .context("import Qwen3.5 cached prefix snapshot")?;
                state.prompt_cursor = matched_len;
                Ok(true)
            }
            MetalRequestStateInner::Qwen3(_) => Ok(false),
        }
    }

    /// Snapshot the live Qwen3.5 C++ session's KV+GDR state at this request's
    /// fully-prefilled prompt cursor. Returns `None` when the cursor is
    /// shorter than one in-memory block. Snapshots at exactly `prompt_cursor`
    /// (= `cache_len` post-terminal-prefill) so the recurrent GDR state in the
    /// snapshot matches the KV state — Qwen3.5's linear-attention recurrent
    /// state advances in stream and cannot be rewound to a shorter prefix
    /// without replay, so any truncated snapshot would import stale state on a
    /// future hit. Drains the C++ session as a side effect; the next
    /// decode/prefill tick will re-attach via `begin_session`. In-memory tier
    /// publish uses this to reuse the work the live request already paid for,
    /// without paying a second replay-prefill.
    pub(crate) fn export_qwen35_live_prefix_snapshot(
        &mut self,
        block_size: usize,
    ) -> Result<Option<Qwen35PrefixSnapshot>> {
        let MetalRequestStateInner::Qwen35(state) = &mut self.inner else {
            return Ok(None);
        };
        let live_len = state.prompt_cursor;
        if live_len < block_size {
            return Ok(None);
        }
        let token_ids = state.prompt_tokens[..live_len].to_vec();
        let snapshot = state
            .driver
            .export_drained_prefix_snapshot(token_ids, live_len)?;
        Ok(Some(snapshot))
    }

    pub(crate) fn export_qwen35_disk_prompt_prefixes(
        &self,
        block_size: usize,
    ) -> Result<Vec<Qwen35PrefixSnapshot>> {
        match &self.inner {
            MetalRequestStateInner::Qwen35(state) => {
                let target_lens = qwen35_disk_publish_prefix_lens(
                    state.prompt_cursor,
                    state.prompt_tokens.len(),
                    block_size,
                );
                let Some(&aligned_len) = target_lens.last() else {
                    return Ok(Vec::new());
                };
                let mut snapshots = Vec::with_capacity(target_lens.len());
                state.driver.stream_prefix_snapshots_at_lengths(
                    &state.prompt_tokens[..aligned_len],
                    block_size,
                    &target_lens,
                    |snapshot| {
                        snapshots.push(snapshot);
                        Ok(())
                    },
                )?;
                Ok(snapshots)
            }
            MetalRequestStateInner::Qwen3(_) => Ok(Vec::new()),
        }
    }
}

fn qwen35_disk_publish_prefix_lens(
    prompt_cursor: usize,
    prompt_len: usize,
    block_size: usize,
) -> Vec<usize> {
    let aligned_len = longest_reusable_aligned_prefix_len(prompt_cursor, prompt_len, block_size);
    if aligned_len == 0 {
        return Vec::new();
    }

    let mut target_lens = Vec::with_capacity(2);
    if aligned_len >= prompt_len {
        let importable_len = aligned_len.saturating_sub(block_size);
        if importable_len > 0 {
            target_lens.push(importable_len);
        }
    }
    if target_lens.last().copied() != Some(aligned_len) {
        target_lens.push(aligned_len);
    }
    target_lens
}

fn longest_reusable_aligned_prefix_len(
    prompt_cursor: usize,
    prompt_len: usize,
    block_size: usize,
) -> usize {
    if block_size == 0 {
        return 0;
    }
    (prompt_cursor.min(prompt_len) / block_size) * block_size
}

fn decode_qwen3_batch(
    states: &mut [&mut ResumableRequestState<Qwen3StepDriver<'_>>],
) -> Result<Vec<u32>> {
    use super::mlx::{concatenate_axis, eval, rms_norm, slice, take_axis, transpose_axes};
    use super::ops::linear;

    ensure!(
        !states.is_empty(),
        "decode_qwen3_batch requires at least one request state"
    );

    if states.iter().all(|state| state.driver.kv_pool.is_none()) {
        let mut rows = Vec::with_capacity(states.len());
        for state in states.iter_mut() {
            let token = state
                .last_token
                .context("Qwen3 packed batch requires committed decode tokens")?;
            rows.push(Qwen3PackedBatchRow {
                state: std::ptr::from_mut(&mut **state),
                query_tokens: vec![token],
                kind: Qwen3PackedBatchRowKind::Decode,
            });
        }
        let sampled = execute_qwen3_packed_batch(&mut rows)?;
        return sampled
            .into_iter()
            .map(|token| token.context("Qwen3 packed decode missing sampled token"))
            .collect();
    }

    let batch = i32::try_from(states.len()).context("decode_qwen3_batch batch size overflow")?;
    let first = &states[0].driver;
    let weights = first.weights;
    let n_heads = first.n_heads;
    let n_kv_heads = first.n_kv_heads;
    let head_dim = first.head_dim;
    let attn_scale = first.attn_scale;
    let rope_base = first.rope_base;
    let eps = first.eps;
    let kv_dim = n_kv_heads * head_dim;
    let cache_len = first.cache_len;
    let end_pos = cache_len + 1;

    for state in states.iter() {
        ensure!(
            std::ptr::eq(state.driver.weights, weights),
            "decode_qwen3_batch requires identical Qwen3 weight handles"
        );
        ensure!(
            state.driver.n_heads == n_heads
                && state.driver.n_kv_heads == n_kv_heads
                && state.driver.head_dim == head_dim,
            "decode_qwen3_batch requires identical Qwen3 geometry"
        );
    }

    let input_tokens: Vec<u32> = states
        .iter()
        .map(|state| {
            state
                .last_token
                .context("decode_qwen3_batch requires a committed prefill token")
        })
        .collect::<Result<_>>()?;

    if states.iter().any(|state| {
        let cache_len = state.driver.cache_len;
        cache_len > 0 && cache_len % KV_CACHE_CHUNK == 0
    }) {
        clear_metal_cache();
    }

    for state in states.iter_mut() {
        let driver = &mut state.driver;
        if let Some(pool) = driver.kv_pool.as_mut() {
            pool.alloc_tokens(METAL_REQUEST_STATE_ID, 1)
                .context("alloc MetalKVPool slot for batched decode")?;
        } else {
            driver.ensure_capacity(driver.cache_len + 1)?;
        }
    }

    let token_values: Vec<i32> = input_tokens.iter().map(|&token| token as i32).collect();
    let token_arr = MlxArray::from_slice_i32(&token_values, &[batch]);
    let mut x = take_axis(&weights.embed_tokens, &token_arr, 0);

    // MLX 0.31.1 scalar-rope `[B>1, H, S=1, D]` workaround: always feed an
    // int32[B] offsets array so the `fast::rope(..., const array&)` overload
    // is used. Same-length batch here, so every entry is `cache_len`; when
    // this path grows to varlen the values diverge.
    // See docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md.
    let rope_offsets_data: Vec<i32> = vec![cache_len; states.len()];
    let rope_offsets = MlxArray::from_slice_i32(&rope_offsets_data, &[batch]);

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let residual = x.clone();
        let x_norm = rms_norm(&x, &layer.input_layernorm, eps);
        let (q_raw, k_raw, v_raw) = layer.attention_inputs.project(&x_norm);

        let q = super::mlx::reshape(&q_raw, &[batch, 1, n_heads, head_dim]);
        let q = rms_norm(&q, &layer.q_norm, eps);
        let q = transpose_axes(&q, &[0, 2, 1, 3]);
        let q = super::mlx::rope_dynamic(&q, head_dim, false, rope_base, 1.0f32, &rope_offsets);

        let k = super::mlx::reshape(&k_raw, &[batch, 1, n_kv_heads, head_dim]);
        let k = rms_norm(&k, &layer.k_norm, eps);
        let k = transpose_axes(&k, &[0, 2, 1, 3]);
        let k = super::mlx::rope_dynamic(&k, head_dim, false, rope_base, 1.0f32, &rope_offsets);

        let v = super::mlx::reshape(&v_raw, &[batch, 1, n_kv_heads, head_dim]);
        let v = transpose_axes(&v, &[0, 2, 1, 3]);

        let k_rows = transpose_axes(&k, &[0, 2, 1, 3]);
        let k_rows = super::mlx::reshape(&k_rows, &[batch, kv_dim]);
        let v_rows = transpose_axes(&v, &[0, 2, 1, 3]);
        let v_rows = super::mlx::reshape(&v_rows, &[batch, kv_dim]);

        let mut batch_k = Vec::with_capacity(states.len());
        let mut batch_v = Vec::with_capacity(states.len());
        for (row_idx, state) in states.iter_mut().enumerate() {
            let row = i32::try_from(row_idx).context("decode_qwen3_batch row index overflow")?;
            let row_k = slice(&k_rows, &[row, 0], &[row + 1, kv_dim], &[1, 1]);
            let row_v = slice(&v_rows, &[row, 0], &[row + 1, kv_dim], &[1, 1]);

            let (k_full, v_full) = if let Some(pool) = state.driver.kv_pool.as_mut() {
                pool.write_kv(layer_idx, METAL_REQUEST_STATE_ID, &row_k, &row_v)
                    .context("write MetalKVPool during batched decode")?;
                pool.gather_kv(layer_idx, METAL_REQUEST_STATE_ID)
                    .context("gather MetalKVPool during batched decode")?
            } else {
                let k_token = slice(
                    &k,
                    &[row, 0, 0, 0],
                    &[row + 1, n_kv_heads, 1, head_dim],
                    &[1, 1, 1, 1],
                );
                let v_token = slice(
                    &v,
                    &[row, 0, 0, 0],
                    &[row + 1, n_kv_heads, 1, head_dim],
                    &[1, 1, 1, 1],
                );
                state.driver.k_caches[layer_idx] = super::mlx::slice_update(
                    &mut state.driver.k_caches[layer_idx],
                    &k_token,
                    &[0, 0, state.driver.cache_len, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                );
                state.driver.v_caches[layer_idx] = super::mlx::slice_update(
                    &mut state.driver.v_caches[layer_idx],
                    &v_token,
                    &[0, 0, state.driver.cache_len, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                );
                let k_full = slice(
                    &state.driver.k_caches[layer_idx],
                    &[0, 0, 0, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                    &[1, 1, 1, 1],
                );
                let v_full = slice(
                    &state.driver.v_caches[layer_idx],
                    &[0, 0, 0, 0],
                    &[1, n_kv_heads, end_pos, head_dim],
                    &[1, 1, 1, 1],
                );
                (k_full, v_full)
            };

            batch_k.push(k_full);
            batch_v.push(v_full);
        }

        let k_full = concatenate_axis(&batch_k, 0);
        let v_full = concatenate_axis(&batch_v, 0);
        let attn_out =
            super::mlx::scaled_dot_product_attention(&q, &k_full, &v_full, attn_scale, None);
        let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = super::mlx::reshape(&attn_out, &[batch, n_heads * head_dim]);
        let attn_out = linear(&attn_out, &layer.o_proj);
        x = super::mlx::add(&residual, &attn_out);

        let residual2 = x.clone();
        let xn = rms_norm(&x, &layer.post_attention_layernorm, eps);
        let (gate_raw, up) = layer.mlp_inputs.project(&xn);
        let gate = super::mlx::silu(&gate_raw);
        let mlp = linear(&super::mlx::multiply(&gate, &up), &layer.down_proj);
        x = super::mlx::add(&residual2, &mlp);
    }

    let logits = linear(&rms_norm(&x, &weights.norm, eps), &weights.lm_head);
    let mut sampled_tokens = Vec::with_capacity(states.len());
    let mut sampled_arrays = Vec::with_capacity(states.len());
    for (row_idx, state) in states.iter().enumerate() {
        let row = i32::try_from(row_idx).context("decode_qwen3_batch sample row overflow")?;
        let row_logits = slice(&logits, &[row, 0], &[row + 1, logits.shape()[1]], &[1, 1]);
        sampled_arrays.push(gpu_sample_token(&row_logits, &state.driver.sample_params));
    }
    let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
    eval(&sample_refs);

    for (state, sampled) in states.iter_mut().zip(sampled_arrays.iter()) {
        let token = sampled.item_i32() as u32;
        state.driver.cache_len += 1;
        state.record_sampled_token(token)?;
        sampled_tokens.push(token);
    }

    Ok(sampled_tokens)
}

fn execute_qwen3_packed_batch(rows: &mut [Qwen3PackedBatchRow<'_>]) -> Result<Vec<Option<u32>>> {
    use super::mlx::{
        build_varlen_verify_mask, eval, reshape, rms_norm, rope_dynamic,
        scaled_dot_product_attention_masked, take_axis, transpose_axes,
    };
    use super::ops::linear;

    let row_count = rows.len();
    ensure!(
        row_count > 0,
        "Qwen3 packed batch requires at least one row"
    );

    let batch = i32::try_from(rows.len()).context("Qwen3 packed batch size overflow")?;
    let first = unsafe { &*rows[0].state };
    let first = &first.driver;
    let weights = first.weights;
    let n_heads = first.n_heads;
    let n_kv_heads = first.n_kv_heads;
    let head_dim = first.head_dim;
    let attn_scale = first.attn_scale;
    let rope_base = first.rope_base;
    let eps = first.eps;
    let mut batch_cache_len = 0;
    let mut target_kv_capacity = 0;
    let mut max_query_len = 0;
    let mut query_lens = Vec::with_capacity(rows.len());
    let mut left_padding = Vec::with_capacity(rows.len());
    for row in rows.iter_mut() {
        let state = unsafe { &mut *row.state };
        ensure!(
            std::ptr::eq(state.driver.weights, weights),
            "Qwen3 packed batch requires identical weight handles"
        );
        ensure!(
            state.driver.n_heads == n_heads
                && state.driver.n_kv_heads == n_kv_heads
                && state.driver.head_dim == head_dim,
            "Qwen3 packed batch requires identical geometry"
        );
        ensure!(
            state.driver.kv_pool.is_none() && state.driver.dflash.is_none(),
            "Qwen3 packed batch requires plain non-DFlash request states without MetalKVPool"
        );
        state.driver.ensure_cpp_prefill_drained()?;
        batch_cache_len = batch_cache_len.max(state.driver.cache_len);
        target_kv_capacity = target_kv_capacity.max(state.driver.kv_capacity);
        max_query_len = max_query_len.max(
            i32::try_from(row.query_tokens.len())
                .context("Qwen3 packed batch query len overflow")?,
        );
    }
    target_kv_capacity =
        target_kv_capacity.max(round_up_kv_capacity(batch_cache_len + max_query_len));

    for row in rows.iter_mut() {
        let state = unsafe { &mut *row.state };
        state.driver.ensure_capacity(target_kv_capacity)?;
        state.driver.kv_capacity = target_kv_capacity;
        let query_len = i32::try_from(row.query_tokens.len())
            .context("Qwen3 packed batch query len overflow")?;
        query_lens.push(query_len);
        left_padding.push(batch_cache_len - state.driver.cache_len);
    }

    if rows.iter().any(|row| {
        let state = unsafe { &*row.state };
        state.driver.cache_len > 0 && state.driver.cache_len % KV_CACHE_CHUNK == 0
    }) {
        clear_metal_cache();
    }

    let mut packed_tokens = Vec::with_capacity(rows.len() * max_query_len as usize);
    for row in rows.iter() {
        packed_tokens.extend(row.query_tokens.iter().map(|&token| token as i32));
        packed_tokens.resize(
            packed_tokens.len() + (max_query_len as usize - row.query_tokens.len()),
            0,
        );
    }
    let token_arr = MlxArray::from_slice_i32(&packed_tokens, &[batch, max_query_len]);
    let token_flat = reshape(&token_arr, &[batch * max_query_len]);
    let x_flat = take_axis(&weights.embed_tokens, &token_flat, 0);
    let hidden_size = weights.embed_tokens.shape()[1];
    let mut x = reshape(&x_flat, &[batch, max_query_len, hidden_size]);

    let rope_offsets_data: Vec<i32> = rows
        .iter()
        .map(|row| unsafe { (&*row.state).driver.cache_len })
        .collect();
    let rope_offsets = MlxArray::from_slice_i32(&rope_offsets_data, &[batch]);
    let attn_mask = build_varlen_verify_mask(&left_padding, max_query_len, batch_cache_len);
    let key_len = batch_cache_len + max_query_len;

    let mut packed_k_caches = Vec::with_capacity(weights.layers.len());
    let mut packed_v_caches = Vec::with_capacity(weights.layers.len());
    for layer_idx in 0..weights.layers.len() {
        let mut k_rows = Vec::with_capacity(rows.len());
        let mut v_rows = Vec::with_capacity(rows.len());
        for (row_idx, row) in rows.iter().enumerate() {
            let state = unsafe { &mut *row.state };
            let left_pad = left_padding[row_idx];
            k_rows.push(left_pad_kv_cache_row(
                &state.driver.k_caches[layer_idx],
                left_pad,
                state.driver.cache_len,
                target_kv_capacity,
            ));
            v_rows.push(left_pad_kv_cache_row(
                &state.driver.v_caches[layer_idx],
                left_pad,
                state.driver.cache_len,
                target_kv_capacity,
            ));
        }
        packed_k_caches.push(concatenate_axis(&k_rows, 0));
        packed_v_caches.push(concatenate_axis(&v_rows, 0));
    }

    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        let residual = x.clone();
        let x_norm = rms_norm(&x, &layer.input_layernorm, eps);
        let (q_raw, k_raw, v_raw) = layer.attention_inputs.project(&x_norm);

        let q = reshape(&q_raw, &[batch, max_query_len, n_heads, head_dim]);
        let q = rms_norm(&q, &layer.q_norm, eps);
        let q = transpose_axes(&q, &[0, 2, 1, 3]);
        let q = rope_dynamic(&q, head_dim, false, rope_base, 1.0, &rope_offsets);

        let k = reshape(&k_raw, &[batch, max_query_len, n_kv_heads, head_dim]);
        let k = rms_norm(&k, &layer.k_norm, eps);
        let k = transpose_axes(&k, &[0, 2, 1, 3]);
        let k = rope_dynamic(&k, head_dim, false, rope_base, 1.0, &rope_offsets);

        let v = reshape(&v_raw, &[batch, max_query_len, n_kv_heads, head_dim]);
        let v = transpose_axes(&v, &[0, 2, 1, 3]);

        for (row_idx, query_len) in query_lens.iter().copied().enumerate() {
            let row = i32::try_from(row_idx).context("Qwen3 packed batch row overflow")?;
            let k_row = slice(
                &k,
                &[row, 0, 0, 0],
                &[row + 1, n_kv_heads, query_len, head_dim],
                &[1, 1, 1, 1],
            );
            let v_row = slice(
                &v,
                &[row, 0, 0, 0],
                &[row + 1, n_kv_heads, query_len, head_dim],
                &[1, 1, 1, 1],
            );
            packed_k_caches[layer_idx] = super::mlx::slice_update(
                &mut packed_k_caches[layer_idx],
                &k_row,
                &[row, 0, batch_cache_len, 0],
                &[row + 1, n_kv_heads, batch_cache_len + query_len, head_dim],
            );
            packed_v_caches[layer_idx] = super::mlx::slice_update(
                &mut packed_v_caches[layer_idx],
                &v_row,
                &[row, 0, batch_cache_len, 0],
                &[row + 1, n_kv_heads, batch_cache_len + query_len, head_dim],
            );
        }

        let k_full = slice(
            &packed_k_caches[layer_idx],
            &[0, 0, 0, 0],
            &[batch, n_kv_heads, key_len, head_dim],
            &[1, 1, 1, 1],
        );
        let v_full = slice(
            &packed_v_caches[layer_idx],
            &[0, 0, 0, 0],
            &[batch, n_kv_heads, key_len, head_dim],
            &[1, 1, 1, 1],
        );
        let attn_out =
            scaled_dot_product_attention_masked(&q, &k_full, &v_full, attn_scale, &attn_mask);
        let attn_out = transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = reshape(&attn_out, &[batch, max_query_len, n_heads * head_dim]);
        let attn_out = linear(&attn_out, &layer.o_proj);
        x = super::mlx::add(&residual, &attn_out);

        let residual2 = x.clone();
        let xn = rms_norm(&x, &layer.post_attention_layernorm, eps);
        let (gate_raw, up) = layer.mlp_inputs.project(&xn);
        let gate = super::mlx::silu(&gate_raw);
        let mlp = linear(&super::mlx::multiply(&gate, &up), &layer.down_proj);
        x = super::mlx::add(&residual2, &mlp);
    }

    let logits = linear(&rms_norm(&x, &weights.norm, eps), &weights.lm_head);
    let logits_shape = logits.shape().to_vec();
    let vocab = *logits_shape
        .last()
        .context("Qwen3 packed batch logits missing vocab dim")?;
    let mut sampled_arrays = Vec::new();
    let mut sampled_row_indices = Vec::new();
    for (row_idx, row) in rows.iter().enumerate() {
        let state = unsafe { &*row.state };
        let sample_row = match row.kind {
            Qwen3PackedBatchRowKind::Decode => true,
            Qwen3PackedBatchRowKind::Prefill { terminal_prompt } => terminal_prompt,
        };
        if !sample_row {
            continue;
        }
        let row_i32 = i32::try_from(row_idx).context("Qwen3 packed batch sample row overflow")?;
        let query_len = query_lens[row_idx];
        let row_logits = if logits_shape.len() == 2 {
            slice(&logits, &[row_i32, 0], &[row_i32 + 1, vocab], &[1, 1])
        } else {
            slice(
                &logits,
                &[row_i32, query_len - 1, 0],
                &[row_i32 + 1, query_len, vocab],
                &[1, 1, 1],
            )
        };
        let row_logits = reshape(&row_logits, &[1, vocab]);
        sampled_arrays.push(gpu_sample_token(&row_logits, &state.driver.sample_params));
        sampled_row_indices.push(row_idx);
    }
    let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
    let mut outputs: Vec<&MlxArray> =
        Vec::with_capacity(1 + sample_refs.len() + packed_k_caches.len() + packed_v_caches.len());
    outputs.push(&logits);
    outputs.extend(sample_refs.iter().copied());
    outputs.extend(packed_k_caches.iter());
    outputs.extend(packed_v_caches.iter());
    eval(&outputs);

    let mut sampled_tokens = vec![None; rows.len()];
    for (row_idx, sampled) in sampled_row_indices.into_iter().zip(sampled_arrays.iter()) {
        sampled_tokens[row_idx] = Some(sampled.item_i32() as u32);
    }

    for (row_idx, row) in rows.iter_mut().enumerate() {
        let state = unsafe { &mut *row.state };
        let row_i32 = i32::try_from(row_idx).context("Qwen3 packed batch row overflow")?;
        let left_pad = left_padding[row_idx];
        for layer_idx in 0..weights.layers.len() {
            let old_k = std::mem::replace(
                &mut state.driver.k_caches[layer_idx],
                strip_left_padding_from_packed_row(
                    &packed_k_caches[layer_idx],
                    row_i32,
                    left_pad,
                    key_len,
                    target_kv_capacity,
                ),
            );
            drop(old_k);
            let old_v = std::mem::replace(
                &mut state.driver.v_caches[layer_idx],
                strip_left_padding_from_packed_row(
                    &packed_v_caches[layer_idx],
                    row_i32,
                    left_pad,
                    key_len,
                    target_kv_capacity,
                ),
            );
            drop(old_v);
        }
        state.driver.kv_capacity = target_kv_capacity;
        state.driver.cache_len += query_lens[row_idx];
        match row.kind {
            Qwen3PackedBatchRowKind::Decode => {
                let token = sampled_tokens[row_idx]
                    .context("Metal packed batch missing sampled decode token")?;
                state.record_sampled_token(token)?;
            }
            Qwen3PackedBatchRowKind::Prefill { terminal_prompt } => {
                state.prompt_cursor += row.query_tokens.len();
                if terminal_prompt {
                    let token = sampled_tokens[row_idx]
                        .context("Metal packed batch missing terminal prefill token")?;
                    state.record_sampled_token(token)?;
                }
            }
        }
    }

    Ok(sampled_tokens)
}

fn try_build_qwen35_packed_decode_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
) -> Result<Option<Qwen35PackedDecodeBatch<'a>>> {
    ensure!(
        !states.is_empty(),
        "try_build_qwen35_packed_decode_batch requires at least one request state"
    );

    for state in states.iter_mut() {
        state.driver.ensure_cpp_session_drained()?;
    }

    let first = &states[0].driver;
    let weights = first.weights;
    let config = first.config;
    let arch = first.arch;

    let (n_kv_per_request, n_gdr_per_request) = match &first.mode {
        Qwen35StepMode::Cpp(state) => (
            i32::try_from(state.kv_flat.len())
                .context("Qwen3.5 packed decode batch kv count overflow")?,
            i32::try_from(state.gdr_flat.len())
                .context("Qwen3.5 packed decode batch gdr count overflow")?,
        ),
        Qwen35StepMode::Rust(_) => return Ok(None),
    };

    // Shape / model identity check only. `cache_len` and `kv_capacity` are
    // allowed to differ across rows — we unify them below via a shared
    // `batch_cache_len` cursor + per-row `left_padding` (mlx-lm BatchKVCache
    // pattern). Correctness of variable-length batching depends on both the
    // attention mask (columns [0, left_pad) zeroed) AND per-row RoPE offsets
    // (each row's Q/K rotated at its own logical position). The rope offsets
    // ride through the bridge via `current_rope_offsets` on
    // `Qwen35CompiledModel`; see `decode_qwen35_packed_batch` below.
    for state in states.iter() {
        if !std::ptr::eq(state.driver.weights, weights)
            || !std::ptr::eq(state.driver.config, config)
            || !std::ptr::eq(state.driver.arch, arch)
        {
            return Ok(None);
        }
        match &state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                if i32::try_from(cpp.kv_flat.len())
                    .context("Qwen3.5 packed decode batch kv count overflow")?
                    != n_kv_per_request
                    || i32::try_from(cpp.gdr_flat.len())
                        .context("Qwen3.5 packed decode batch gdr count overflow")?
                        != n_gdr_per_request
                {
                    return Ok(None);
                }
            }
            Qwen35StepMode::Rust(_) => return Ok(None),
        }
    }

    // Shared batch cursor = max of all per-row cache_lens. Rows with shorter
    // caches get left-padded up to this cursor so every row writes its next
    // decode token at the same column (`batch_cache_len`).
    let mut batch_cache_len: i32 = 0;
    let mut target_kv_capacity: i32 = 0;
    for state in states.iter() {
        batch_cache_len = batch_cache_len.max(state.driver.cache_len);
        target_kv_capacity = target_kv_capacity.max(state.driver.kv_capacity);
    }
    // Capacity must fit the next decode write (batch_cache_len + 1) rounded up
    // to KV_CACHE_CHUNK so future grow steps stay aligned.
    target_kv_capacity = target_kv_capacity.max(round_up_kv_capacity(batch_cache_len + 1));

    // Normalize every state's own storage up to target_kv_capacity before we
    // read its KV arrays — concatenate_axis requires identical trailing shapes.
    for state in states.iter_mut() {
        state.driver.ensure_capacity(target_kv_capacity)?;
        state.driver.kv_capacity = target_kv_capacity;
    }

    let mut left_padding = Vec::with_capacity(states.len());
    for state in states.iter() {
        let pad = batch_cache_len - state.driver.cache_len;
        debug_assert!(pad >= 0);
        left_padding.push(pad);
    }

    let mut packed_kv_flat = Vec::with_capacity(n_kv_per_request as usize);
    for kv_idx in 0..n_kv_per_request as usize {
        let mut per_request = Vec::with_capacity(states.len());
        for (row_idx, state) in states.iter().enumerate() {
            let Qwen35StepMode::Cpp(cpp) = &state.driver.mode else {
                unreachable!("checked above");
            };
            let pad = left_padding[row_idx];
            if pad == 0 {
                per_request.push(cpp.kv_flat[kv_idx].clone());
            } else {
                per_request.push(left_pad_kv_cache_row(
                    &cpp.kv_flat[kv_idx],
                    pad,
                    state.driver.cache_len,
                    target_kv_capacity,
                ));
            }
        }
        packed_kv_flat.push(concatenate_axis(&per_request, 0));
    }

    // GDR state is per-request recurrent state (not a time-series cache), so
    // it does NOT get left-padded — just stacked along the batch axis.
    let mut packed_gdr_flat = Vec::with_capacity(n_gdr_per_request as usize);
    for gdr_idx in 0..n_gdr_per_request as usize {
        let mut per_request = Vec::with_capacity(states.len());
        for state in states.iter() {
            let Qwen35StepMode::Cpp(cpp) = &state.driver.mode else {
                unreachable!("checked above");
            };
            per_request.push(cpp.gdr_flat[gdr_idx].clone());
        }
        packed_gdr_flat.push(concatenate_axis(&per_request, 0));
    }

    let mut eval_refs = Vec::with_capacity(packed_kv_flat.len() + packed_gdr_flat.len());
    eval_refs.extend(packed_kv_flat.iter());
    eval_refs.extend(packed_gdr_flat.iter());
    let eval_refs: Vec<&MlxArray> = eval_refs.into_iter().collect();
    eval(&eval_refs);

    Ok(Some(Qwen35PackedDecodeBatch {
        weights,
        config,
        arch,
        batch_cache_len,
        kv_capacity: target_kv_capacity,
        left_padding,
        n_kv_per_request,
        n_gdr_per_request,
        packed_kv_flat,
        packed_gdr_flat,
    }))
}

fn sync_qwen35_packed_decode_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
    batch: &Qwen35PackedDecodeBatch<'a>,
) -> Result<()> {
    ensure!(
        states.len() == batch.batch_size(),
        "sync_qwen35_packed_decode_batch expected {} states, got {}",
        batch.batch_size(),
        states.len()
    );

    for (row_idx, state) in states.iter_mut().enumerate() {
        ensure!(
            batch.matches_driver(&state.driver),
            "sync_qwen35_packed_decode_batch state mismatch at row {row_idx}"
        );
        state.driver.ensure_cpp_session_drained()?;
        let row = i32::try_from(row_idx).context("sync_qwen35_packed_decode_batch row overflow")?;
        let left_pad = batch.left_padding[row_idx];
        match &mut state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                // KV caches carry per-column valid-mask positions; strip the
                // left pad so each row's own cache is left-aligned again.
                for (slot, packed) in cpp.kv_flat.iter_mut().zip(batch.packed_kv_flat.iter()) {
                    let new_slot = if left_pad == 0 {
                        slice_row(packed, row)
                    } else {
                        strip_left_padding_from_packed_row(
                            packed,
                            row,
                            left_pad,
                            batch.batch_cache_len,
                            batch.kv_capacity,
                        )
                    };
                    let old = std::mem::replace(slot, new_slot);
                    drop(old);
                }
                // GDR recurrent state is not time-series — no pad to strip.
                for (slot, packed) in cpp.gdr_flat.iter_mut().zip(batch.packed_gdr_flat.iter()) {
                    let old = std::mem::replace(slot, slice_row(packed, row));
                    drop(old);
                }
                state.driver.kv_capacity = batch.kv_capacity;
                state.driver.cache_len = batch.batch_cache_len - left_pad;
            }
            Qwen35StepMode::Rust(_) => {
                bail!("sync_qwen35_packed_decode_batch requires compiled Qwen3.5 state")
            }
        }
    }

    Ok(())
}

fn decode_qwen35_packed_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
    batch: &mut Qwen35PackedDecodeBatch<'a>,
) -> Result<Vec<u32>> {
    ensure!(
        !states.is_empty(),
        "decode_qwen35_packed_batch requires at least one request state"
    );
    ensure!(
        states.len() == batch.batch_size(),
        "decode_qwen35_packed_batch expected {} states, got {}",
        batch.batch_size(),
        states.len()
    );

    if batch.batch_cache_len > 0 && batch.batch_cache_len % KV_CACHE_CHUNK == 0 {
        clear_metal_cache();
    }

    batch.ensure_capacity_for_states(states, batch.batch_cache_len + 1);

    let input_tokens: Vec<u32> = states
        .iter()
        .map(|state| {
            state
                .last_token
                .context("decode_qwen35_packed_batch requires a committed prefill token")
        })
        .collect::<Result<_>>()?;
    let token_values: Vec<i32> = input_tokens.iter().map(|&token| token as i32).collect();
    let token_arr =
        MlxArray::from_slice_i32(
            &token_values,
            &[i32::try_from(states.len())
                .context("decode_qwen35_packed_batch batch size overflow")?],
        );

    // Only materialize the additive attention mask when at least one row is
    // left-padded; same-length batches take the no-mask fast path (identical
    // to pre-varlen behavior).
    let needs_mask = batch.left_padding.iter().any(|&pad| pad != 0);
    let mask_opt: Option<MlxArray> = if needs_mask {
        Some(super::mlx::build_varlen_decode_mask(
            &batch.left_padding,
            batch.batch_cache_len,
        ))
    } else {
        None
    };

    // ALWAYS build per-row RoPE offsets for the packed (batched) decode.
    //
    // Two reasons:
    //   1. Varlen correctness — each row's new Q/K must rotate at its own
    //      logical position `batch_cache_len - left_padding[row]`, not at
    //      the shared `batch_cache_len`.
    //   2. MLX 0.31.1 bug workaround — `fast::rope(..., int offset)` on a
    //      `[B, H, S=1, D]` tensor with `B > 1` silently zeroes out batch
    //      rows > 0. The array-offset overload works for both B=1 and B>1.
    //      So even same-length batches (all offsets equal) must go through
    //      the array path to stay correct. See
    //      docs/experience/errors/2026-04-16-metal-varlen-rope-blocker.md.
    let rope_offsets_data: Vec<i32> = batch
        .left_padding
        .iter()
        .map(|&pad| batch.batch_cache_len - pad)
        .collect();
    let rope_offsets = MlxArray::from_slice_i32(
        &rope_offsets_data,
        &[i32::try_from(rope_offsets_data.len())
            .context("decode_qwen35_packed_batch rope offsets overflow")?],
    );

    let cpp_model = batch
        .weights
        .cpp_model
        .as_ref()
        .context("decode_qwen35_packed_batch requires the compiled Qwen3.5 path")?;
    let logits = cpp_model.step_batch_packed(
        &token_arr,
        i32::try_from(states.len()).context("decode_qwen35_packed_batch batch size overflow")?,
        batch.batch_cache_len,
        &mut batch.packed_kv_flat,
        batch.n_kv_per_request,
        &mut batch.packed_gdr_flat,
        batch.n_gdr_per_request,
        mask_opt.as_ref(),
        Some(&rope_offsets),
    )?;

    let mut eval_refs: Vec<&MlxArray> =
        Vec::with_capacity(1 + batch.packed_kv_flat.len() + batch.packed_gdr_flat.len());
    eval_refs.push(&logits);
    eval_refs.extend(batch.packed_kv_flat.iter());
    eval_refs.extend(batch.packed_gdr_flat.iter());
    async_eval(&eval_refs);

    let logits_shape = logits.shape().to_vec();
    ensure!(
        !logits_shape.is_empty()
            && logits_shape[0]
                == i32::try_from(states.len())
                    .context("decode_qwen35_packed_batch batch shape overflow")?,
        "decode_qwen35_packed_batch expected batched logits, got shape {:?}",
        logits_shape
    );

    batch.batch_cache_len += 1;

    let sampled_tokens = if qwen35_can_batch_sample(states) {
        let sampled = gpu_sample_token_batched(&logits, &states[0].driver.params);
        eval(&[&sampled]);
        let sampled_i32 = sampled.as_slice_i32();
        ensure!(
            sampled_i32.len() == states.len(),
            "decode_qwen35_packed_batch expected {} sampled tokens, got {}",
            states.len(),
            sampled_i32.len()
        );
        sampled_i32.into_iter().map(|token| token as u32).collect()
    } else {
        let mut sampled_arrays = Vec::with_capacity(states.len());
        for (row_idx, state) in states.iter().enumerate() {
            let row = i32::try_from(row_idx).context("decode_qwen35_packed_batch row overflow")?;
            sampled_arrays.push(gpu_sample_token(
                &slice_row(&logits, row),
                &state.driver.params,
            ));
        }
        let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
        eval(&sample_refs);
        sampled_arrays
            .iter()
            .map(|sampled| sampled.item_i32() as u32)
            .collect::<Vec<_>>()
    };

    let mut sampled_tokens_out = Vec::with_capacity(states.len());
    for (row_idx, (state, token)) in states.iter_mut().zip(sampled_tokens).enumerate() {
        // Each row's own logical length = batch_cursor - its own pad, so a
        // row that joined late stays at its shorter length.
        state.driver.cache_len = batch.batch_cache_len - batch.left_padding[row_idx];
        state.driver.kv_capacity = batch.kv_capacity;
        state.record_sampled_token(token)?;
        sampled_tokens_out.push(token);
    }

    Ok(sampled_tokens_out)
}

/// Free-function companion for `MetalRequestState::try_decode_qwen35_dflash_speculative_batch`.
/// See doc comment on the method for eligibility rules and contract.
fn try_decode_qwen35_dflash_speculative_batch<'a>(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>],
) -> Result<Option<DflashBatchOutcome>> {
    if states.len() < 2 {
        return Ok(None);
    }

    // ── 1. Per-row + cross-row eligibility. Partitions into a ready subset
    //      (passed to the batched kernel) and a stale subset (caller runs
    //      scalar). Two-pass structure avoids committing to any single anchor
    //      row until we know at least two rows agree on cross-row predicates.
    //
    // Per-row predicates (any miss → row not ready):
    //   - `phase == Decode`, `Qwen35StepMode::Cpp`
    //   - `dflash.is_some()` with captured `target_hidden`
    //   - `dflash.token_buffer.is_empty()` — a non-empty buffer means the row
    //     is still draining a prior speculative block's tail tokens and the
    //     batched path's cache advance would race the scalar drain.
    //   - `last_token.is_some()` (post-prefill invariant)
    //   - `dflash.runtime.batched_draft_path_eligible()` — scalar
    //     `dflash_draft_forward` routing gates (`DFLASH_DRAFT_CPP=1` AND
    //     `draft_attention_mask != "causal"`).
    //
    // Cross-row predicates form an equivalence relation; we want the LARGEST
    // agreeing subset, not "whoever agrees with the first ready row". Picking
    // the first row as anchor would reintroduce all-or-nothing demotion when
    // the first ready row is the outlier — e.g. `[outlier, compat, compat]`
    // would filter both `compat` rows out. O(n²) majority-equivalence-class
    // scan is fine: bucket sizes are tiny.
    //   - identical `cache_len`
    //   - identical `target_hidden.shape()[0]` (ctx_len axis the batched
    //     block stacks over — partial-accept rebuilds this per row so equal
    //     cache_len does NOT imply equal ctx_len)
    //   - identical `draft_state.active_len()` (per-row `[..len]` slice stacks
    //     cleanly iff this holds)
    //   - identical `weights`/`config`/`arch` pointers and identical DFlash
    //     `runtime`/`config` pointers (shared model handles)
    let mut ready_flags: Vec<bool> = Vec::with_capacity(states.len());
    for state in states.iter() {
        ready_flags.push(row_passes_dflash_batch_per_row_predicates(state));
    }

    let ready_indices = select_dflash_batch_ready_indices(&ready_flags, |anchor, candidate| {
        rows_agree_on_dflash_batch_cross_row_predicates(states[anchor], states[candidate])
    });

    if ready_indices.len() < 2 {
        return Ok(None);
    }

    // Compact `states` down to the ready subset. The existing batched-kernel
    // body below runs against `ready_states` unchanged; stale rows are left
    // untouched for the caller to run scalar.
    let mut ready_states: Vec<&mut ResumableRequestState<Qwen35StepDriver<'a>>> =
        Vec::with_capacity(ready_indices.len());
    let mut ready_cursor = 0usize;
    for (idx, state) in states.iter_mut().enumerate() {
        if ready_indices.get(ready_cursor).copied() == Some(idx) {
            // Reborrow: `states[idx]` is `&mut &mut ResumableRequestState<_>`;
            // deref once, then reborrow mutably to produce a fresh
            // `&mut ResumableRequestState<_>` tied to this function's lifetime.
            ready_states.push(&mut **state);
            ready_cursor += 1;
        }
    }
    let states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'a>>] = &mut ready_states;

    let first_cache_len = states[0].driver.cache_len;

    // Shared model handles anchored on the first ready row; ptr-equality was
    // already enforced by the cross-row filter above.
    let first = &states[0].driver;
    let weights = first.weights;
    let runtime: &MetalDflashRuntime = first
        .dflash
        .as_ref()
        .expect("dflash presence validated above")
        .runtime;
    let target_config: &MetalModelConfig = first
        .dflash
        .as_ref()
        .expect("dflash presence validated above")
        .config;

    let cpp_model: &CppQwen35Model = weights
        .cpp_model
        .as_ref()
        .context("DFlash batched decode requires the compiled Qwen3.5 path")?;

    let block_size = runtime.block_size();
    let block_size_i32 =
        i32::try_from(block_size).context("Qwen3.5 DFlash block_size does not fit i32")?;

    // ── 2. Ensure per-row target KV capacity for batch_cache_len + block_size. ──
    // Mirrors the scalar scheduler path at request_state.rs:2922–2928. We also
    // drain any live cpp session so `cpp_state.kv_flat/gdr_flat` are materialized
    // before we stack them.
    let needed_cap = first_cache_len + block_size_i32;
    for state in states.iter_mut() {
        if needed_cap > state.driver.kv_capacity {
            state.driver.ensure_capacity(needed_cap)?;
        } else {
            state.driver.ensure_cpp_session_drained()?;
        }
    }

    // ── 3. Collect per-row inputs (read-only slices of mutable state). ──
    let batch_i32 = i32::try_from(states.len())
        .context("try_decode_qwen35_dflash_speculative_batch batch size overflow")?;

    let current_tokens: Vec<u32> = states
        .iter()
        .map(|state| state.last_token.expect("validated above"))
        .collect();
    let params_per_row: Vec<SamplingParams> = states
        .iter()
        .map(|state| state.driver.params.clone())
        .collect();
    let target_hidden_per_row: Vec<MlxArray> = states
        .iter()
        .map(|state| {
            state
                .driver
                .dflash
                .as_ref()
                .expect("validated above")
                .target_hidden
                .as_ref()
                .expect("validated above")
                .clone()
        })
        .collect();

    // Stack target KV per-layer across rows. Each row's kv_flat[l] has shape
    // [1, n_kv, kv_capacity, head_dim]; axis-0 concatenate yields
    // [B, n_kv, kv_capacity, head_dim]. Phase 2B = same-length only, so no
    // left-padding (left_padding = [0; B], batch_cache_len = first_cache_len).
    let n_kv_per_request = {
        let Qwen35StepMode::Cpp(cpp) = &states[0].driver.mode else {
            unreachable!("validated above")
        };
        cpp.kv_flat.len()
    };
    let n_gdr_per_request = {
        let Qwen35StepMode::Cpp(cpp) = &states[0].driver.mode else {
            unreachable!("validated above")
        };
        cpp.gdr_flat.len()
    };

    let mut packed_target_kv_flat: Vec<MlxArray> = Vec::with_capacity(n_kv_per_request);
    for l in 0..n_kv_per_request {
        let per_row: Vec<MlxArray> = states
            .iter()
            .map(|state| match &state.driver.mode {
                Qwen35StepMode::Cpp(cpp) => cpp.kv_flat[l].clone(),
                Qwen35StepMode::Rust(_) => unreachable!("validated above"),
            })
            .collect();
        packed_target_kv_flat.push(concatenate_axis(&per_row, 0));
    }
    let mut packed_target_gdr_flat: Vec<MlxArray> = Vec::with_capacity(n_gdr_per_request);
    for g in 0..n_gdr_per_request {
        let per_row: Vec<MlxArray> = states
            .iter()
            .map(|state| match &state.driver.mode {
                Qwen35StepMode::Cpp(cpp) => cpp.gdr_flat[g].clone(),
                Qwen35StepMode::Rust(_) => unreachable!("validated above"),
            })
            .collect();
        packed_target_gdr_flat.push(concatenate_axis(&per_row, 0));
    }

    let mut target_cache_lens: Vec<i32> = vec![first_cache_len; states.len()];
    let left_padding: Vec<i32> = vec![0; states.len()];
    let batch_cache_len = first_cache_len;

    // Detach per-row draft states so we can pass `&mut [ContiguousKvState]` to
    // the batched kernel without holding a simultaneous mutable borrow of the
    // caller's state tree. The kernel mutates in place; we reinstall on exit.
    let mut draft_states: Vec<dflash::ContiguousKvState> = states
        .iter_mut()
        .map(|state| {
            let dflash_state = state.driver.dflash.as_mut().expect("validated above");
            std::mem::replace(
                &mut dflash_state.draft_state,
                dflash::ContiguousKvState::new(1, 1, 1, 1),
            )
        })
        .collect();

    // ── 4. Invoke the batched kernel. ──
    let kernel_result = dflash::qwen35_dflash_speculative_block_batched(
        runtime,
        weights
            .embedding
            .dense()
            .context("Qwen3.5/Qwen3.6 DFlash requires dense target embeddings")?,
        &weights.lm_head,
        target_config,
        cpp_model,
        &params_per_row,
        &current_tokens,
        &target_hidden_per_row,
        &mut packed_target_kv_flat,
        &mut packed_target_gdr_flat,
        &mut target_cache_lens,
        &left_padding,
        batch_cache_len,
        &mut draft_states,
    );

    // ── 5. Reinstall draft states before bailing on error (even on failure
    //      the kernel mutates them in place and the caller's State should
    //      reflect whatever the kernel left behind). ──
    for (state, draft) in states.iter_mut().zip(draft_states) {
        let dflash_state = state.driver.dflash.as_mut().expect("validated above");
        dflash_state.draft_state = draft;
    }

    let block_results = kernel_result?;
    ensure!(
        block_results.len() == states.len(),
        "try_decode_qwen35_dflash_speculative_batch: kernel returned {} rows, expected {}",
        block_results.len(),
        states.len()
    );

    // ── 6. Unstack packed KV/GDR back into each row's cpp_state. ──
    for (row_idx, state) in states.iter_mut().enumerate() {
        let row = i32::try_from(row_idx)
            .context("try_decode_qwen35_dflash_speculative_batch row overflow")?;
        let Qwen35StepMode::Cpp(cpp) = &mut state.driver.mode else {
            unreachable!("validated above")
        };
        for (slot, packed) in cpp.kv_flat.iter_mut().zip(packed_target_kv_flat.iter()) {
            let new_slot = slice_row(packed, row);
            let old = std::mem::replace(slot, new_slot);
            drop(old);
        }
        for (slot, packed) in cpp.gdr_flat.iter_mut().zip(packed_target_gdr_flat.iter()) {
            let new_slot = slice_row(packed, row);
            let old = std::mem::replace(slot, new_slot);
            drop(old);
        }
    }
    let _ = batch_i32; // silence unused warning on path where overflow guard moved

    // ── 7. Per-row scheduler state: cache_len advance + token_buffer fan-out +
    //      updated_target_hidden + acceptance metrics + record first token. ──
    let mut sampled_first_tokens: Vec<u32> = Vec::with_capacity(states.len());
    for (row_idx, (state, block)) in states.iter_mut().zip(block_results).enumerate() {
        ensure!(
            !block.accepted_tokens.is_empty(),
            "DFlash batched block row {row_idx} produced zero accepted tokens"
        );

        // Mirror the scalar path at request_state.rs:2964–2973: advance cache
        // to the kernel-reported value, publish updated target_hidden, push
        // accepted tokens into the buffer, then pop the first for this tick.
        state.driver.cache_len = target_cache_lens[row_idx];
        let dflash_state = state.driver.dflash.as_mut().expect("validated above");
        dflash_state.acceptance_lengths.push(block.accepted_inputs);
        dflash_state.target_hidden = Some(block.updated_target_hidden);
        dflash_state.prefetched_draft = None;
        for t in block.accepted_tokens {
            dflash_state.token_buffer.push_back(t);
        }
        let first_token = dflash_state
            .token_buffer
            .pop_front()
            .context("DFlash batched block produced empty token buffer after push")?;

        // Propagate through ResumableRequestState just like scalar decode_step:
        // record_sampled_token updates last_token + generated_tokens and may
        // transition to Finished.
        state.record_sampled_token(first_token)?;
        sampled_first_tokens.push(first_token);
    }

    Ok(Some(DflashBatchOutcome {
        ready_indices,
        tokens: sampled_first_tokens,
    }))
}

fn select_dflash_batch_ready_indices(
    ready_flags: &[bool],
    mut rows_agree: impl FnMut(usize, usize) -> bool,
) -> Vec<usize> {
    let mut best_anchor: Option<usize> = None;
    let mut best_count: usize = 0;
    for (candidate_idx, &candidate_ready) in ready_flags.iter().enumerate() {
        if !candidate_ready {
            continue;
        }
        let mut count = 0usize;
        for (other_idx, &other_ready) in ready_flags.iter().enumerate() {
            if !other_ready {
                continue;
            }
            if other_idx == candidate_idx || rows_agree(candidate_idx, other_idx) {
                count += 1;
            }
        }
        if count > best_count {
            best_count = count;
            best_anchor = Some(candidate_idx);
        }
    }

    let Some(anchor_idx) = best_anchor else {
        return Vec::new();
    };
    if best_count < 2 {
        return Vec::new();
    }
    ready_flags
        .iter()
        .enumerate()
        .filter_map(|(idx, &ready)| {
            (ready && (idx == anchor_idx || rows_agree(anchor_idx, idx))).then_some(idx)
        })
        .collect()
}

/// Per-row predicate for `try_decode_qwen35_dflash_speculative_batch`. True iff
/// the row can join a DFlash batched-speculative block in isolation (cross-row
/// agreement is checked separately, anchored on the first row that passes
/// this predicate).
fn row_passes_dflash_batch_per_row_predicates(
    state: &ResumableRequestState<Qwen35StepDriver<'_>>,
) -> bool {
    if state.phase != MetalRequestPhase::Decode {
        return false;
    }
    if state.last_token.is_none() {
        return false;
    }
    let driver = &state.driver;
    let Qwen35StepMode::Cpp(_) = driver.mode else {
        return false;
    };
    let Some(dflash) = driver.dflash.as_ref() else {
        return false;
    };
    // Non-empty buffer means the row is still draining a prior speculative
    // block's tail tokens; routing it through the batched path would double-
    // advance the cache.
    if !dflash.token_buffer.is_empty() {
        return false;
    }
    let Some(target_hidden) = dflash.target_hidden.as_ref() else {
        return false;
    };
    if target_hidden.shape().is_empty() {
        return false;
    }
    if !dflash.runtime.batched_draft_path_eligible() {
        return false;
    }
    true
}

/// Cross-row predicate: does `candidate` agree with `anchor` on every axis the
/// batched kernel needs to stack cleanly? Callers must have already confirmed
/// both rows individually pass `row_passes_dflash_batch_per_row_predicates`.
fn rows_agree_on_dflash_batch_cross_row_predicates<'a>(
    anchor: &ResumableRequestState<Qwen35StepDriver<'a>>,
    candidate: &ResumableRequestState<Qwen35StepDriver<'a>>,
) -> bool {
    let anchor_driver = &anchor.driver;
    let candidate_driver = &candidate.driver;
    if anchor_driver.cache_len != candidate_driver.cache_len {
        return false;
    }
    let anchor_dflash = anchor_driver
        .dflash
        .as_ref()
        .expect("anchor per-row predicate guarantees dflash");
    let candidate_dflash = candidate_driver
        .dflash
        .as_ref()
        .expect("candidate per-row predicate guarantees dflash");
    let anchor_target_hidden = anchor_dflash
        .target_hidden
        .as_ref()
        .expect("anchor per-row predicate guarantees target_hidden");
    let candidate_target_hidden = candidate_dflash
        .target_hidden
        .as_ref()
        .expect("candidate per-row predicate guarantees target_hidden");
    // target_hidden is rank-2 `[ctx_len, hidden]`; axis-0 must match across
    // rows to stack. Partial-accept divergence rebuilds this per-row from
    // `accepted_inputs`, so equal `cache_len` does NOT imply equal
    // `target_hidden.shape()[0]`.
    if anchor_target_hidden.shape().first() != candidate_target_hidden.shape().first() {
        return false;
    }
    if anchor_dflash.draft_state.active_len() != candidate_dflash.draft_state.active_len() {
        return false;
    }
    if !std::ptr::eq(anchor_driver.weights, candidate_driver.weights)
        || !std::ptr::eq(anchor_driver.config, candidate_driver.config)
        || !std::ptr::eq(anchor_driver.arch, candidate_driver.arch)
        || !std::ptr::eq(anchor_dflash.runtime, candidate_dflash.runtime)
        || !std::ptr::eq(anchor_dflash.config, candidate_dflash.config)
    {
        return false;
    }
    true
}

fn decode_qwen35_batch(
    states: &mut [&mut ResumableRequestState<Qwen35StepDriver<'_>>],
) -> Result<Vec<u32>> {
    ensure!(
        !states.is_empty(),
        "decode_qwen35_batch requires at least one request state"
    );

    for state in states.iter_mut() {
        state.driver.ensure_cpp_session_drained()?;
    }

    let batch = i32::try_from(states.len()).context("decode_qwen35_batch batch size overflow")?;
    let first = &states[0].driver;
    let weights = first.weights;
    let config = first.config;
    let arch = first.arch;
    let cache_len = first.cache_len;
    let kv_capacity = first.kv_capacity;
    let cpp_model = weights
        .cpp_model
        .as_ref()
        .context("decode_qwen35_batch requires the compiled Qwen3.5 path")?;

    for state in states.iter() {
        ensure!(
            std::ptr::eq(state.driver.weights, weights)
                && std::ptr::eq(state.driver.config, config)
                && std::ptr::eq(state.driver.arch, arch),
            "decode_qwen35_batch requires identical Qwen3.5 model handles"
        );
        ensure!(
            state.driver.cache_len == cache_len && state.driver.kv_capacity == kv_capacity,
            "decode_qwen35_batch requires identical cache_len and kv_capacity"
        );
        ensure!(
            matches!(state.driver.mode, Qwen35StepMode::Cpp(_)),
            "decode_qwen35_batch requires compiled Qwen3.5 state"
        );
    }

    if cache_len > 0 && cache_len % KV_CACHE_CHUNK == 0 {
        clear_metal_cache();
    }

    for state in states.iter_mut() {
        state.driver.ensure_capacity(state.driver.cache_len + 1)?;
    }

    let input_tokens: Vec<u32> = states
        .iter()
        .map(|state| {
            state
                .last_token
                .context("decode_qwen35_batch requires a committed prefill token")
        })
        .collect::<Result<_>>()?;
    let token_values: Vec<i32> = input_tokens.iter().map(|&token| token as i32).collect();
    let token_arr = MlxArray::from_slice_i32(&token_values, &[batch]);

    let n_kv_per_request = match &states[0].driver.mode {
        Qwen35StepMode::Cpp(state) => {
            i32::try_from(state.kv_flat.len()).context("decode_qwen35_batch kv count overflow")?
        }
        Qwen35StepMode::Rust(_) => unreachable!("checked above"),
    };
    let n_gdr_per_request = match &states[0].driver.mode {
        Qwen35StepMode::Cpp(state) => {
            i32::try_from(state.gdr_flat.len()).context("decode_qwen35_batch gdr count overflow")?
        }
        Qwen35StepMode::Rust(_) => unreachable!("checked above"),
    };

    let mut flat_kv = Vec::with_capacity(states.len() * n_kv_per_request as usize);
    let mut flat_gdr = Vec::with_capacity(states.len() * n_gdr_per_request as usize);
    for state in states.iter() {
        match &state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                flat_kv.extend(cpp.kv_flat.iter().cloned());
                flat_gdr.extend(cpp.gdr_flat.iter().cloned());
            }
            Qwen35StepMode::Rust(_) => unreachable!("checked above"),
        }
    }

    // Same MLX 0.31.1 scalar-rope `[B>1, H, S=1, D]` bug workaround as
    // `decode_qwen35_packed_batch`: always feed a per-row rope offsets
    // array. This is a same-length batch so every row shares `cache_len`,
    // but we still need the array path to stay correct for B > 1.
    let rope_offsets_data: Vec<i32> = vec![cache_len; states.len()];
    let rope_offsets = MlxArray::from_slice_i32(&rope_offsets_data, &[batch]);

    let logits = cpp_model.step_batch(
        &token_arr,
        batch,
        cache_len,
        &mut flat_kv,
        n_kv_per_request,
        &mut flat_gdr,
        n_gdr_per_request,
        None,
        Some(&rope_offsets),
    )?;

    let mut step_outputs: Vec<&MlxArray> = Vec::with_capacity(1 + flat_kv.len() + flat_gdr.len());
    step_outputs.push(&logits);
    step_outputs.extend(flat_kv.iter());
    step_outputs.extend(flat_gdr.iter());
    eval(&step_outputs);

    let logits_shape = logits.shape().to_vec();
    ensure!(
        !logits_shape.is_empty() && logits_shape[0] == batch,
        "decode_qwen35_batch expected batched logits, got shape {:?}",
        logits_shape
    );

    let mut sampled_arrays = Vec::with_capacity(states.len());
    for (row_idx, state) in states.iter().enumerate() {
        let row = i32::try_from(row_idx).context("decode_qwen35_batch row overflow")?;
        let mut start = vec![0; logits_shape.len()];
        let mut end = logits_shape.clone();
        let strides = vec![1; logits_shape.len()];
        start[0] = row;
        end[0] = row + 1;
        let row_logits = slice(&logits, &start, &end, &strides);
        sampled_arrays.push(gpu_sample_token(&row_logits, &state.driver.params));
    }
    let sample_refs: Vec<&MlxArray> = sampled_arrays.iter().collect();
    eval(&sample_refs);

    let mut kv_iter = flat_kv.into_iter();
    let mut gdr_iter = flat_gdr.into_iter();
    let mut sampled_tokens = Vec::with_capacity(states.len());

    for (state, sampled) in states.iter_mut().zip(sampled_arrays.iter()) {
        match &mut state.driver.mode {
            Qwen35StepMode::Cpp(cpp) => {
                for slot in &mut cpp.kv_flat {
                    let old = std::mem::replace(
                        slot,
                        kv_iter
                            .next()
                            .context("decode_qwen35_batch missing KV output")?,
                    );
                    drop(old);
                }
                for slot in &mut cpp.gdr_flat {
                    let old = std::mem::replace(
                        slot,
                        gdr_iter
                            .next()
                            .context("decode_qwen35_batch missing GDR output")?,
                    );
                    drop(old);
                }
            }
            Qwen35StepMode::Rust(_) => unreachable!("checked above"),
        }

        let token = sampled.item_i32() as u32;
        state.driver.cache_len += 1;
        state.record_sampled_token(token)?;
        sampled_tokens.push(token);
    }

    ensure!(
        kv_iter.next().is_none() && gdr_iter.next().is_none(),
        "decode_qwen35_batch produced unexpected extra state outputs"
    );

    Ok(sampled_tokens)
}

fn qwen35_can_batch_sample(states: &[&mut ResumableRequestState<Qwen35StepDriver<'_>>]) -> bool {
    let Some((first, rest)) = states.split_first() else {
        return false;
    };
    rest.iter()
        .all(|state| same_sampling_params(&first.driver.params, &state.driver.params))
}

#[allow(clippy::float_cmp)]
fn same_sampling_params(a: &SamplingParams, b: &SamplingParams) -> bool {
    a.temperature == b.temperature
        && a.top_k == b.top_k
        && a.top_p == b.top_p
        && a.min_p == b.min_p
        && a.repetition_penalty == b.repetition_penalty
        && a.frequency_penalty == b.frequency_penalty
        && a.presence_penalty == b.presence_penalty
        && a.seed == b.seed
}

/// Optional DFlash speculative-decode state attached to a `Qwen3StepDriver`.
///
/// When present, `decode_token` runs full DFlash speculative blocks and
/// buffers the accepted tokens; the scheduler still sees one token per step.
/// The DFlash state OWNS the target model's KV cache (`target_state`) instead
/// of the driver's `k_caches`/`v_caches`, because `dflash_speculative_block`
/// needs `&mut ContiguousKvState` for both target and draft.
struct Qwen3DFlashState {
    runtime: &'static MetalDflashRuntime,
    config: &'static MetalModelConfig,
    /// Target model KV state — owned by DFlash, replaces the driver's k/v caches.
    target_state: dflash::ContiguousKvState,
    /// Draft model KV state — owned, separate from target.
    draft_state: dflash::ContiguousKvState,
    /// Target-layer hidden states from the last verified block. Bootstrapped
    /// during prefill via `qwen3_forward_with_hidden_states`.
    target_hidden: Option<MlxArray>,
    /// Multi-token buffer: accepted tokens from the latest speculative block.
    /// `decode_token` pops from here until empty, then runs a new block.
    token_buffer: VecDeque<u32>,
    /// Which target-model layers to capture hidden states from.
    target_layer_ids: Vec<usize>,
    // ── Metrics accumulators (flushed on request completion) ──
    acceptance_lengths: Vec<usize>,
}

struct Qwen3CppPrefillState {
    session_active: bool,
    n_kv: usize,
}

impl Qwen3CppPrefillState {
    fn ensure_session_active(
        &mut self,
        cpp_model: &CppQwen35Model,
        k_caches: &[MlxArray],
        v_caches: &[MlxArray],
    ) -> Result<()> {
        if self.session_active {
            return Ok(());
        }

        let mut kv_flat = Vec::with_capacity(k_caches.len() * 2);
        for (k_cache, v_cache) in k_caches.iter().zip(v_caches.iter()) {
            kv_flat.push(k_cache.clone());
            kv_flat.push(v_cache.clone());
        }
        cpp_model.begin_session(&kv_flat, &[])?;
        self.n_kv = kv_flat.len();
        self.session_active = true;
        Ok(())
    }

    fn ensure_caches_drained(
        &mut self,
        cpp_model: &CppQwen35Model,
        k_caches: &mut [MlxArray],
        v_caches: &mut [MlxArray],
    ) -> Result<()> {
        if !self.session_active {
            return Ok(());
        }

        let (kv_flat, gdr_flat) = cpp_model.end_session(self.n_kv, 0)?;
        ensure!(
            gdr_flat.is_empty(),
            "Qwen3 C++ prefill session unexpectedly returned GDR state"
        );
        ensure!(
            kv_flat.len() == k_caches.len() * 2 && k_caches.len() == v_caches.len(),
            "Qwen3 C++ prefill session returned {} KV tensors for {} layer pairs",
            kv_flat.len(),
            k_caches.len()
        );
        let mut kv_iter = kv_flat.into_iter();
        for (k_cache, v_cache) in k_caches.iter_mut().zip(v_caches.iter_mut()) {
            let old_k = std::mem::replace(
                k_cache,
                kv_iter
                    .next()
                    .context("Qwen3 C++ prefill session missing K cache")?,
            );
            drop(old_k);
            let old_v = std::mem::replace(
                v_cache,
                kv_iter
                    .next()
                    .context("Qwen3 C++ prefill session missing V cache")?,
            );
            drop(old_v);
        }
        ensure!(
            kv_iter.next().is_none(),
            "Qwen3 C++ prefill session returned unexpected extra KV tensors"
        );
        self.session_active = false;
        self.n_kv = 0;
        Ok(())
    }
}

struct Qwen3StepDriver<'a> {
    weights: &'a StandardMetalWeights,
    sample_params: SamplingParams,
    prefill_params: SamplingParams,
    kv_capacity: i32,
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    cpp_prefill: Option<Qwen3CppPrefillState>,
    kv_pool: Option<MetalKVPool>,
    cache_len: i32,
    n_heads: i32,
    n_kv_heads: i32,
    head_dim: i32,
    attn_scale: f32,
    rope_base: f32,
    eps: f32,
    /// DFlash speculative decode state. When `Some`, `decode_token` runs
    /// DFlash blocks and the driver's `k_caches`/`v_caches` are empty stubs
    /// (all KV management goes through `dflash.target_state`).
    dflash: Option<Qwen3DFlashState>,
}

impl<'a> Qwen3StepDriver<'a> {
    fn new(
        weights: &'a StandardMetalWeights,
        config: &'a MetalModelConfig,
        params: &SamplingParams,
        use_kv_pool: bool,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        dflash_runtime: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)>,
    ) -> Result<Self> {
        let n_layers = config.num_hidden_layers;
        let n_heads = config.num_attention_heads as i32;
        let n_kv_heads = config.num_key_value_heads as i32;
        let head_dim = config.head_dim as i32;
        let prefill_len = prompt_tokens.len() as i32;
        let initial_cap =
            ((prefill_len + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK + 1) * KV_CACHE_CHUNK;
        let total_tokens_needed = std::cmp::max(
            initial_cap as usize,
            prompt_tokens.len().saturating_add(max_new_tokens),
        );
        let kv_dtype = weights.layers[0].attention_inputs.kv_dtype();
        // When DFlash is enabled, the driver's k/v caches are unused —
        // DFlash owns its own target_state ContiguousKvState. Skip the
        // allocation to avoid wasting memory on empty cache tensors.
        let is_dflash = dflash_runtime.is_some();
        let (k_caches, v_caches) = if is_dflash {
            (Vec::new(), Vec::new())
        } else {
            let cache_shape = [1i32, n_kv_heads, initial_cap, head_dim];
            let k: Vec<MlxArray> = (0..n_layers)
                .map(|_| zeros(&cache_shape, kv_dtype))
                .collect();
            let v: Vec<MlxArray> = (0..n_layers)
                .map(|_| zeros(&cache_shape, kv_dtype))
                .collect();
            (k, v)
        };

        Ok(Self {
            weights,
            sample_params: params.clone(),
            prefill_params: SamplingParams {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                ignore_eos: params.ignore_eos,
                stop_token_ids: params.stop_token_ids.clone(),
                seed: None,
                max_new_tokens: None,
            },
            kv_capacity: initial_cap,
            k_caches,
            v_caches,
            cpp_prefill: weights
                .cpp_model
                .as_ref()
                .filter(|_| !use_kv_pool && !is_dflash)
                .map(|_| Qwen3CppPrefillState {
                    session_active: false,
                    n_kv: 0,
                }),
            kv_pool: if use_kv_pool {
                Some(
                    MetalKVPool::new(
                        n_layers,
                        n_kv_heads as usize,
                        head_dim as usize,
                        total_tokens_needed,
                        kv_dtype,
                    )
                    .context("pre-alloc MetalKVPool for request state")?,
                )
            } else {
                None
            },
            cache_len: 0,
            n_heads,
            n_kv_heads,
            head_dim,
            attn_scale: 1.0f32 / (head_dim as f32).sqrt(),
            rope_base: config.rope_theta as f32,
            eps: config.rms_norm_eps as f32,
            dflash: match dflash_runtime {
                Some((runtime, static_config)) => {
                    let total_cap = prompt_tokens.len() + max_new_tokens;
                    let target_state = dflash::ContiguousKvState::from_dtype(
                        n_layers, n_kv_heads, head_dim, total_cap, kv_dtype,
                    );
                    let draft_state = dflash::ContiguousKvState::new(
                        runtime.draft_num_hidden_layers(),
                        runtime.draft_n_kv_heads(),
                        runtime.draft_head_dim(),
                        total_cap,
                    );
                    Some(Qwen3DFlashState {
                        runtime,
                        config: static_config,
                        target_state,
                        draft_state,
                        target_hidden: None,
                        token_buffer: VecDeque::new(),
                        target_layer_ids: runtime.target_layer_ids().to_vec(),
                        acceptance_lengths: Vec::new(),
                    })
                }
                None => None,
            },
        })
    }

    fn run_tokens(&mut self, tokens: &[u32], params: &SamplingParams) -> Result<MlxArray> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3 request-state step requires at least one token"
        );
        // Compiled Qwen3 prefill sessions are model-global state on the C++
        // bridge. Any fallback to the Rust graph (single-token tail chunk,
        // decode, prefix import/export paths) must materialize and end that
        // session first so the next request can safely begin its own session.
        self.ensure_cpp_prefill_drained()?;
        let token_count =
            i32::try_from(tokens.len()).context("Qwen3 request-state token count overflow")?;
        self.ensure_capacity(self.cache_len + token_count)?;
        let sampled = build_forward_graph(
            tokens,
            self.weights,
            &mut self.k_caches,
            &mut self.v_caches,
            self.cache_len,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.attn_scale,
            self.rope_base,
            self.eps,
            self.kv_pool.as_mut(),
            METAL_REQUEST_STATE_ID,
            params,
        )?;
        self.cache_len += token_count;
        Ok(sampled)
    }

    fn run_tokens_cpp_prefill(&mut self, tokens: &[u32]) -> Result<MlxArray> {
        ensure!(
            !tokens.is_empty(),
            "Qwen3 C++ prefill session requires at least one token"
        );
        let token_count =
            i32::try_from(tokens.len()).context("Qwen3 request-state token count overflow")?;
        self.ensure_capacity(self.cache_len + token_count)?;
        let token_values: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
        let token_arr = MlxArray::from_slice_i32(&token_values, &[token_count]);
        let cpp_model = self
            .weights
            .cpp_model
            .as_ref()
            .context("Qwen3 C++ prefill session missing compiled model")?;
        let cpp_prefill = self
            .cpp_prefill
            .as_mut()
            .context("Qwen3 C++ prefill session state unavailable")?;
        cpp_prefill.ensure_session_active(cpp_model, &self.k_caches, &self.v_caches)?;
        let logits = cpp_model.prefill_session(&token_arr, token_count, self.cache_len)?;
        self.cache_len += token_count;
        Ok(logits)
    }

    fn ensure_cpp_prefill_drained(&mut self) -> Result<()> {
        let Some(cpp_prefill) = self.cpp_prefill.as_mut() else {
            return Ok(());
        };
        let cpp_model = self
            .weights
            .cpp_model
            .as_ref()
            .context("Qwen3 C++ prefill session missing compiled model")?;
        cpp_prefill.ensure_caches_drained(cpp_model, &mut self.k_caches, &mut self.v_caches)
    }

    fn can_use_cpp_prefill(&self, tokens: &[u32]) -> bool {
        tokens.len() > 1 && self.cpp_prefill.is_some()
    }

    fn run_step(&mut self, token: u32, params: &SamplingParams) -> Result<MlxArray> {
        self.run_tokens(&[token], params)
    }

    fn ensure_capacity(&mut self, needed_tokens: i32) -> Result<()> {
        if self.kv_pool.is_some() {
            return Ok(());
        }
        while needed_tokens > self.kv_capacity {
            self.ensure_cpp_prefill_drained()?;
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            for li in 0..self.k_caches.len() {
                extend_kv_cache(
                    &mut self.k_caches[li],
                    self.n_kv_heads,
                    self.head_dim,
                    new_cap,
                );
                extend_kv_cache(
                    &mut self.v_caches[li],
                    self.n_kv_heads,
                    self.head_dim,
                    new_cap,
                );
            }
            self.kv_capacity = new_cap;
        }
        Ok(())
    }
}

impl StepDriver for Qwen3StepDriver<'_> {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>> {
        let params = if terminal_prompt {
            self.sample_params.clone()
        } else {
            self.prefill_params.clone()
        };
        let sampled = self.run_step(token, &params)?;
        eval(&[&sampled]);
        if terminal_prompt {
            Ok(Some(sampled.item_i32() as u32))
        } else {
            Ok(None)
        }
    }

    fn prefill_tokens(&mut self, tokens: &[u32], terminal_prompt: bool) -> Result<Option<u32>> {
        if tokens.is_empty() {
            return Ok(None);
        }

        // DFlash path: run the full prompt through qwen3_forward_with_hidden_states
        // on the terminal chunk to capture target-layer hidden states for the
        // first speculative block. The budget override in execute_prefill_chunk
        // ensures the entire prompt arrives as one terminal chunk.
        if terminal_prompt && let Some(dflash) = self.dflash.as_mut() {
            let (norm_hidden, target_hidden) = dflash::qwen3_forward_with_hidden_states_on_state(
                tokens,
                self.weights,
                dflash.config,
                &dflash.target_layer_ids,
                &mut dflash.target_state,
            )?;
            dflash.target_hidden = Some(target_hidden);
            let logits = super::ops::linear(&norm_hidden, &self.weights.lm_head);
            let sampled = gpu_sample_token(&logits, &self.sample_params);
            eval(&[&sampled]);
            return Ok(Some(sampled.item_i32() as u32));
        }

        let params = if terminal_prompt {
            self.sample_params.clone()
        } else {
            self.prefill_params.clone()
        };
        let sampled = if self.can_use_cpp_prefill(tokens) {
            // Safe to keep the session alive across prefill chunks because the
            // Metal scheduler admits at most one Prefilling request at a time
            // and keeps chunking that same request until its prompt completes.
            let logits = self.run_tokens_cpp_prefill(tokens)?;
            if terminal_prompt {
                self.ensure_cpp_prefill_drained()?;
                let sampled = gpu_sample_token(&logits, &params);
                let mut outputs: Vec<&MlxArray> =
                    Vec::with_capacity(2 + self.k_caches.len() + self.v_caches.len());
                outputs.push(&logits);
                outputs.push(&sampled);
                outputs.extend(self.k_caches.iter());
                outputs.extend(self.v_caches.iter());
                eval(&outputs);
                sampled
            } else {
                async_eval(&[&logits]);
                logits
            }
        } else {
            let sampled = self.run_tokens(tokens, &params)?;
            eval(&[&sampled]);
            sampled
        };

        if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
            clear_metal_cache();
        }

        if terminal_prompt {
            Ok(Some(sampled.item_i32() as u32))
        } else {
            Ok(None)
        }
    }

    fn decode_token(&mut self, token: u32) -> Result<u32> {
        // ── DFlash speculative path ──────────────────────────────────────
        if let Some(dflash) = self.dflash.as_mut() {
            // 1. Drain buffer first — cheap, no GPU work.
            if let Some(buffered) = dflash.token_buffer.pop_front() {
                return Ok(buffered);
            }

            // 2. Buffer empty → run one full speculative block.
            let target_hidden = dflash
                .target_hidden
                .take()
                .context("DFlash decode_token: target_hidden not set (prefill incomplete?)")?;

            let block = dflash::dflash_speculative_block(
                dflash.runtime,
                token,
                &target_hidden,
                self.weights,
                dflash.config,
                &self.sample_params,
                &mut dflash.target_state,
                &mut dflash.draft_state,
            )?;

            // 3. Update state.
            dflash.acceptance_lengths.push(block.accepted_inputs);
            dflash.target_hidden = Some(block.updated_target_hidden);

            // 4. Push accepted tokens into buffer, pop the first one.
            for &t in &block.accepted_tokens {
                dflash.token_buffer.push_back(t);
            }
            return dflash
                .token_buffer
                .pop_front()
                .context("DFlash speculative block produced zero tokens");
        }

        // ── Standard single-token path ───────────────────────────────────
        if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
            clear_metal_cache();
        }
        let sampled = self.run_step(token, &self.sample_params.clone())?;
        eval(&[&sampled]);
        Ok(sampled.item_i32() as u32)
    }

    fn cleanup(&mut self) -> Result<()> {
        self.ensure_cpp_prefill_drained()?;
        if let Some(pool) = self.kv_pool.as_mut() {
            pool.free_request(METAL_REQUEST_STATE_ID);
        }
        Ok(())
    }
}

/// DFlash speculative decode state for Qwen3.5 hybrid models.
/// Extends the Qwen3 DFlash pattern with GDR recurrent rollback.
struct Qwen35DFlashState {
    runtime: &'static MetalDflashRuntime,
    config: &'static MetalModelConfig,
    /// Draft model KV state (pure transformer, same as Qwen3 DFlash).
    draft_state: dflash::ContiguousKvState,
    /// Target-layer hidden states for the next draft block.
    target_hidden: Option<MlxArray>,
    /// Optional next-block draft block tokens, launched asynchronously after
    /// verify/accept and consumed when the next single-row speculative block starts.
    prefetched_draft: Option<dflash::Qwen35PrefetchedDraft>,
    /// Multi-token buffer from speculative acceptance.
    token_buffer: VecDeque<u32>,
    /// Which target-model layers to capture hidden states from.
    target_layer_ids: Vec<usize>,
    /// Per-block acceptance lengths for metrics.
    acceptance_lengths: Vec<usize>,
}

struct Qwen35CppState {
    kv_flat: Vec<MlxArray>,
    gdr_flat: Vec<MlxArray>,
    session_active: bool,
    n_kv: usize,
    n_gdr: usize,
}

struct Qwen35RustState {
    k_caches: Vec<MlxArray>,
    v_caches: Vec<MlxArray>,
    recurrent: MetalRecurrentState,
}

impl Qwen35CppState {
    fn ensure_session_active(&mut self, cpp_model: &CppQwen35Model) -> Result<()> {
        if self.session_active {
            return Ok(());
        }

        cpp_model.begin_session(&self.kv_flat, &self.gdr_flat)?;
        self.n_kv = self.kv_flat.len();
        self.n_gdr = self.gdr_flat.len();
        self.kv_flat.clear();
        self.gdr_flat.clear();
        self.session_active = true;
        Ok(())
    }

    fn ensure_caches_drained(&mut self, cpp_model: &CppQwen35Model) -> Result<()> {
        if !self.session_active {
            return Ok(());
        }

        let (kv_flat, gdr_flat) = cpp_model.end_session(self.n_kv, self.n_gdr)?;
        self.kv_flat = kv_flat;
        self.gdr_flat = gdr_flat;
        self.session_active = false;
        self.n_kv = 0;
        self.n_gdr = 0;
        Ok(())
    }
}

enum Qwen35StepMode {
    Cpp(Qwen35CppState),
    Rust(Qwen35RustState),
}

struct Qwen35StepDriver<'a> {
    weights: &'a Qwen35MetalWeights,
    config: &'a MetalModelConfig,
    arch: &'a super::config::MetalQwen35ArchConfig,
    params: SamplingParams,
    kv_capacity: i32,
    cache_len: i32,
    mode: Qwen35StepMode,
    /// DFlash speculative decode state (None = standard decode).
    dflash: Option<Qwen35DFlashState>,
    /// Pre-queued sampled token (lazy MlxArray) from the step ahead.
    pending_sampled: Option<MlxArray>,
    /// M_e.1 P1 plumbing — placeholder for the paged-KV pool. Always
    /// `None` today (no flag wired); the field exists so subsequent
    /// commits can land Qwen3.5 dual-write (P2.1) and the kernel
    /// cutover (P3.1) without churning the struct shape under live
    /// callers. Mirrors `Qwen3StepDriver.kv_pool`.
    #[allow(dead_code)]
    kv_pool: Option<MetalKVPool>,
}

impl<'a> Qwen35StepDriver<'a> {
    fn new(
        weights: &'a Qwen35MetalWeights,
        config: &'a MetalModelConfig,
        params: &SamplingParams,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        dflash_runtime: Option<(&'static MetalDflashRuntime, &'static MetalModelConfig)>,
    ) -> Result<Self> {
        let MetalModelArch::Qwen35(arch) = &config.arch else {
            bail!("Qwen3.5 request state requires a Qwen3.5 config");
        };
        let dflash_runtime = dflash_runtime.filter(|_| qwen35_dflash_supported(weights));

        let num_full_layers = arch.num_full_attention_layers();
        let prefill_len = prompt_tokens.len() as i32;
        let total_tokens_needed =
            std::cmp::max(1, prompt_tokens.len().saturating_add(max_new_tokens));
        let initial_cap =
            ((total_tokens_needed as i32 + KV_CACHE_CHUNK - 1) / KV_CACHE_CHUNK) * KV_CACHE_CHUNK;
        let cache_shape = [
            1i32,
            config.num_key_value_heads as i32,
            initial_cap,
            config.head_dim as i32,
        ];
        let k_caches: Vec<MlxArray> = (0..num_full_layers)
            .map(|_| zeros(&cache_shape, super::mlx::Dtype::Bfloat16))
            .collect();
        let v_caches: Vec<MlxArray> = (0..num_full_layers)
            .map(|_| zeros(&cache_shape, super::mlx::Dtype::Bfloat16))
            .collect();
        let recurrent = MetalRecurrentState::new(arch.num_linear_attention_layers(), &arch.linear);

        let mode = if weights.cpp_model.is_some() {
            let kv_flat: Vec<MlxArray> = k_caches
                .iter()
                .zip(v_caches.iter())
                .flat_map(|(k, v)| [k.clone(), v.clone()])
                .collect();
            let gdr_flat: Vec<MlxArray> = recurrent
                .states
                .iter()
                .zip(recurrent.conv_states.iter())
                .flat_map(|(s, c)| [s.clone(), c.clone()])
                .collect();
            Qwen35StepMode::Cpp(Qwen35CppState {
                kv_flat,
                gdr_flat,
                session_active: false,
                n_kv: 0,
                n_gdr: 0,
            })
        } else {
            Qwen35StepMode::Rust(Qwen35RustState {
                k_caches,
                v_caches,
                recurrent,
            })
        };

        let _ = prefill_len;

        let dflash = if let Some((runtime, dflash_config)) = dflash_runtime {
            Some(Qwen35DFlashState {
                runtime,
                config: dflash_config,
                draft_state: dflash::ContiguousKvState::new(
                    runtime.draft_num_hidden_layers(),
                    runtime.draft_n_kv_heads(),
                    runtime.draft_head_dim(),
                    total_tokens_needed,
                ),
                target_hidden: None,
                prefetched_draft: None,
                token_buffer: VecDeque::new(),
                target_layer_ids: runtime.target_layer_ids().to_vec(),
                acceptance_lengths: Vec::new(),
            })
        } else {
            None
        };

        Ok(Self {
            weights,
            config,
            arch,
            params: params.clone(),
            kv_capacity: initial_cap,
            cache_len: 0,
            mode,
            dflash,
            pending_sampled: None,
            kv_pool: None,
        })
    }

    fn append_dflash_target_hidden_chunk(&mut self, captured: Option<MlxArray>) {
        let reset_prefetch = captured.is_some();
        let Some(dflash) = self.dflash.as_mut() else {
            return;
        };
        super::qwen35::append_qwen35_captured_hidden_chunk(&mut dflash.target_hidden, captured);
        if reset_prefetch {
            dflash.prefetched_draft = None;
        }
    }

    fn append_dflash_target_hidden_from_cpp_outputs(
        &mut self,
        cpp_model_raw: *mut std::ffi::c_void,
    ) -> Result<()> {
        let Some(dflash) = self.dflash.as_mut() else {
            return Ok(());
        };
        let captured = super::qwen35::capture_qwen35_hidden_from_cpp_outputs(
            cpp_model_raw,
            dflash.target_layer_ids.len(),
        )?;
        let reset_prefetch = captured.is_some();
        super::qwen35::append_qwen35_captured_hidden_chunk(&mut dflash.target_hidden, captured);
        if reset_prefetch {
            dflash.prefetched_draft = None;
        }
        Ok(())
    }

    fn run_step(&mut self, token: u32) -> Result<MlxArray> {
        self.ensure_capacity(self.cache_len + 1)?;
        let token_arr = MlxArray::from_slice_i32(&[token as i32], &[1]);
        let logits = self.run_step_with_token(&token_arr)?;
        self.cache_len += 1;
        Ok(logits)
    }

    fn run_step_with_token(&mut self, token_arr: &MlxArray) -> Result<MlxArray> {
        let weights = self.weights;
        let config = self.config;
        let arch = self.arch;
        let cache_len = self.cache_len;
        let logits = match &mut self.mode {
            Qwen35StepMode::Cpp(state) => Self::run_cpp_step(weights, cache_len, token_arr, state)?,
            Qwen35StepMode::Rust(state) => {
                Self::run_rust_step(weights, config, arch, cache_len, token_arr, state)
            }
        };
        Ok(logits)
    }

    fn ensure_cpp_session_drained(&mut self) -> Result<()> {
        match &mut self.mode {
            Qwen35StepMode::Cpp(state) => {
                let cpp_model = self
                    .weights
                    .cpp_model
                    .as_ref()
                    .context("Qwen3.5 C++ step path missing compiled model")?;
                state.ensure_caches_drained(cpp_model)
            }
            Qwen35StepMode::Rust(_) => Ok(()),
        }
    }

    fn drain_replay_after_result<T>(&mut self, result: Result<T>, label: &str) -> Result<T> {
        match result {
            Ok(value) => Ok(value),
            Err(err) => {
                if let Err(drain_err) = self.ensure_cpp_session_drained() {
                    return Err(err).context(format!(
                        "{label}; additionally failed to drain replay C++ session: {drain_err:#}"
                    ));
                }
                Err(err)
            }
        }
    }

    fn cpp_session_active(&self) -> bool {
        matches!(&self.mode, Qwen35StepMode::Cpp(state) if state.session_active)
    }

    fn can_import_prefix_snapshot(&self) -> bool {
        matches!(self.mode, Qwen35StepMode::Cpp(_)) && self.weights.cpp_model.is_some()
    }

    fn run_cpp_step(
        weights: &Qwen35MetalWeights,
        cache_len: i32,
        token_arr: &MlxArray,
        state: &mut Qwen35CppState,
    ) -> Result<MlxArray> {
        let cpp_model: &CppQwen35Model = weights
            .cpp_model
            .as_ref()
            .context("Qwen3.5 C++ step path missing compiled model")?;
        state.ensure_session_active(cpp_model)?;
        let logits = cpp_model.step_session(token_arr, cache_len)?;
        async_eval(&[&logits]);
        Ok(logits)
    }

    fn run_rust_step(
        weights: &Qwen35MetalWeights,
        config: &MetalModelConfig,
        arch: &super::config::MetalQwen35ArchConfig,
        cache_len: i32,
        token_arr: &MlxArray,
        state: &mut Qwen35RustState,
    ) -> MlxArray {
        let logits = qwen35_forward_step(
            token_arr,
            weights,
            config,
            arch,
            &mut state.k_caches,
            &mut state.v_caches,
            &mut state.recurrent,
            cache_len,
        );

        let mut step_outputs: Vec<&MlxArray> = Vec::with_capacity(
            1 + state.k_caches.len()
                + state.v_caches.len()
                + state.recurrent.states.len()
                + state.recurrent.conv_states.len(),
        );
        step_outputs.push(&logits);
        step_outputs.extend(state.k_caches.iter());
        step_outputs.extend(state.v_caches.iter());
        step_outputs.extend(state.recurrent.states.iter());
        step_outputs.extend(state.recurrent.conv_states.iter());
        async_eval(&step_outputs);

        state.recurrent.seq_len = (cache_len + 1) as usize;
        logits
    }

    fn run_rust_prefill_hidden_chunk(
        weights: &Qwen35MetalWeights,
        config: &MetalModelConfig,
        arch: &super::config::MetalQwen35ArchConfig,
        cache_len: i32,
        state: &mut Qwen35RustState,
        tokens: &[u32],
        target_layer_ids: &[usize],
    ) -> (MlxArray, MlxArray) {
        let (logits, target_hidden) = qwen35_forward_with_hidden_states(
            tokens,
            weights,
            config,
            arch,
            &mut state.k_caches,
            &mut state.v_caches,
            &mut state.recurrent,
            cache_len,
            target_layer_ids,
        );

        let next_cache_len = cache_len + tokens.len() as i32;
        let mut step_outputs: Vec<&MlxArray> = Vec::with_capacity(
            1 + state.k_caches.len()
                + state.v_caches.len()
                + state.recurrent.states.len()
                + state.recurrent.conv_states.len(),
        );
        step_outputs.push(&logits);
        step_outputs.extend(state.k_caches.iter());
        step_outputs.extend(state.v_caches.iter());
        step_outputs.extend(state.recurrent.states.iter());
        step_outputs.extend(state.recurrent.conv_states.iter());
        async_eval(&step_outputs);

        state.recurrent.seq_len = next_cache_len as usize;
        (logits, target_hidden)
    }

    fn ensure_capacity(&mut self, needed_tokens: i32) -> Result<()> {
        while needed_tokens > self.kv_capacity {
            let new_cap = self.kv_capacity + KV_CACHE_CHUNK;
            match &mut self.mode {
                Qwen35StepMode::Cpp(state) => {
                    let cpp_model = self
                        .weights
                        .cpp_model
                        .as_ref()
                        .context("Qwen3.5 C++ step path missing compiled model")?;
                    state.ensure_caches_drained(cpp_model)?;
                    for li in 0..(state.kv_flat.len() / 2) {
                        extend_kv_cache(
                            &mut state.kv_flat[2 * li],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                        extend_kv_cache(
                            &mut state.kv_flat[2 * li + 1],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                    }
                }
                Qwen35StepMode::Rust(state) => {
                    for li in 0..state.k_caches.len() {
                        extend_kv_cache(
                            &mut state.k_caches[li],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                        extend_kv_cache(
                            &mut state.v_caches[li],
                            self.config.num_key_value_heads as i32,
                            self.config.head_dim as i32,
                            new_cap,
                        );
                    }
                }
            }
            self.kv_capacity = new_cap;
        }
        Ok(())
    }

    fn import_prefix_snapshot(&mut self, snapshot: &Qwen35PrefixSnapshot) -> Result<()> {
        ensure!(
            snapshot.cache_len > 0,
            "Qwen3.5 prefix import requires a non-empty snapshot"
        );
        ensure!(
            snapshot.kv_capacity >= snapshot.cache_len,
            "Qwen3.5 prefix snapshot capacity {} is smaller than cache_len {}",
            snapshot.kv_capacity,
            snapshot.cache_len
        );
        self.ensure_cpp_session_drained()?;
        match &mut self.mode {
            Qwen35StepMode::Cpp(state) => {
                state.kv_flat.clone_from(&snapshot.kv_flat);
                state.gdr_flat.clone_from(&snapshot.gdr_flat);
                self.kv_capacity = snapshot.kv_capacity;
                self.cache_len = snapshot.cache_len;
                self.pending_sampled = None;
                Ok(())
            }
            Qwen35StepMode::Rust(_) => {
                bail!("Qwen3.5 live prefix reuse currently requires the compiled C++ step path")
            }
        }
    }

    fn export_current_cpp_snapshot(&mut self, token_ids: Vec<u32>) -> Result<Qwen35PrefixSnapshot> {
        let cache_len = self.cache_len;
        ensure!(
            cache_len > 0,
            "Qwen3.5 prefix export requires a non-empty cache"
        );
        self.ensure_cpp_session_drained()?;
        match &self.mode {
            Qwen35StepMode::Cpp(state) => Ok(Qwen35PrefixSnapshot {
                token_ids,
                kv_flat: state.kv_flat.clone(),
                gdr_flat: state.gdr_flat.clone(),
                cache_len,
                kv_capacity: self.kv_capacity,
            }),
            Qwen35StepMode::Rust(_) => {
                bail!("Qwen3.5 live prefix export currently requires the compiled C++ step path")
            }
        }
    }

    /// Snapshot the live C++ session in place at the requested cache length
    /// without spinning up a second `Qwen35StepDriver`. The caller must have
    /// established that `target_cache_len == self.cache_len` — Qwen3.5's GDR
    /// recurrent state is processed in stream and cannot be rewound to a
    /// shorter prefix without replay. The session is drained (`end_session`)
    /// so the caller can clone the resident KV+GDR arrays; the next
    /// `prefill_chunk` / `decode_step` re-attaches via `begin_session`.
    fn export_drained_prefix_snapshot(
        &mut self,
        token_ids: Vec<u32>,
        target_cache_len: usize,
    ) -> Result<Qwen35PrefixSnapshot> {
        ensure!(
            !token_ids.is_empty(),
            "Qwen3.5 live prefix snapshot requires at least one token"
        );
        ensure!(
            token_ids.len() == target_cache_len,
            "Qwen3.5 live prefix snapshot token_ids ({}) must match target_cache_len ({target_cache_len})",
            token_ids.len()
        );
        let cache_len = i32::try_from(target_cache_len)
            .context("Qwen3.5 live prefix snapshot cache_len overflow")?;
        ensure!(
            cache_len == self.cache_len,
            "Qwen3.5 live prefix snapshot target cache_len {cache_len} must equal live cache_len {}; recurrent GDR state cannot be truncated",
            self.cache_len
        );
        self.ensure_cpp_session_drained()?;
        match &self.mode {
            Qwen35StepMode::Cpp(state) => Ok(Qwen35PrefixSnapshot {
                token_ids,
                kv_flat: state.kv_flat.clone(),
                gdr_flat: state.gdr_flat.clone(),
                cache_len,
                kv_capacity: self.kv_capacity,
            }),
            Qwen35StepMode::Rust(_) => {
                bail!("Qwen3.5 live prefix snapshot requires the compiled C++ step path")
            }
        }
    }

    fn stream_prefix_snapshots_at_lengths(
        &self,
        prompt_tokens: &[u32],
        block_size: usize,
        target_lens: &[usize],
        mut visit: impl FnMut(Qwen35PrefixSnapshot) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            block_size > 0,
            "Qwen3.5/Qwen3.6 prefix snapshot block size must be > 0"
        );
        ensure!(
            prompt_tokens.len().is_multiple_of(block_size),
            "Qwen3.5/Qwen3.6 prefix snapshot build requires a block-aligned prompt"
        );
        if !matches!(self.mode, Qwen35StepMode::Cpp(_)) || prompt_tokens.is_empty() {
            return Ok(());
        }
        let mut target_lens = target_lens.to_vec();
        target_lens.sort_unstable();
        target_lens.dedup();
        if target_lens.is_empty() {
            return Ok(());
        }
        for &target_len in &target_lens {
            ensure!(
                target_len > 0
                    && target_len <= prompt_tokens.len()
                    && target_len.is_multiple_of(block_size),
                "Qwen3.5/Qwen3.6 selected prefix snapshot target {target_len} must be block-aligned and within {} tokens",
                prompt_tokens.len()
            );
        }

        // The compiled Qwen3.5/Qwen3.6 model owns exactly one live session.
        // A replay driver built on the same `CppQwen35Model` would attempt a
        // nested `session_begin`, which is both invalid and, before the C++
        // fix in this wave, could tear down the active session on error.
        // Prefix publish is opportunistic, so skip replay-based export while
        // the live request still owns the compiled session.
        if self.cpp_session_active() {
            return Ok(());
        }

        let mut replay = Qwen35StepDriver::new(
            self.weights,
            self.config,
            &self.params,
            prompt_tokens,
            1,
            None,
        )
        .context("build replay driver for Qwen3.5/Qwen3.6 prefix snapshots")?;
        let result = (|| -> Result<()> {
            let mut next_target = 0;
            for chunk in prompt_tokens.chunks(block_size) {
                replay
                    .prefill_tokens(chunk, false)
                    .context("replay Qwen3.5/Qwen3.6 prompt chunk for prefix snapshot")?;
                let materialized = replay.cache_len as usize;
                if next_target < target_lens.len() && target_lens[next_target] == materialized {
                    let snapshot = replay
                        .export_current_cpp_snapshot(prompt_tokens[..materialized].to_vec())
                        .context("export replayed Qwen3.5/Qwen3.6 prefix snapshot")?;
                    visit(snapshot)?;
                    next_target += 1;
                }
            }
            ensure!(
                next_target == target_lens.len(),
                "Qwen3.5/Qwen3.6 prefix snapshot replay produced {next_target} of {} requested targets",
                target_lens.len()
            );
            Ok(())
        })();
        replay.drain_replay_after_result(result, "Qwen3.5/Qwen3.6 prefix snapshot replay")
    }
}

impl StepDriver for Qwen35StepDriver<'_> {
    fn prefill_token(&mut self, token: u32, terminal_prompt: bool) -> Result<Option<u32>> {
        let rust_hidden_target_layers = if matches!(self.mode, Qwen35StepMode::Rust(_)) {
            self.dflash
                .as_ref()
                .map(|dflash| dflash.target_layer_ids.clone())
        } else {
            None
        };
        if let Some(target_layer_ids) = rust_hidden_target_layers.as_deref() {
            self.ensure_capacity(self.cache_len + 1)?;
            let (logits, captured) = match &mut self.mode {
                Qwen35StepMode::Rust(state) => Self::run_rust_prefill_hidden_chunk(
                    self.weights,
                    self.config,
                    self.arch,
                    self.cache_len,
                    state,
                    std::slice::from_ref(&token),
                    target_layer_ids,
                ),
                Qwen35StepMode::Cpp(_) => unreachable!("rust hidden prefill selected cpp mode"),
            };
            self.cache_len += 1;
            self.append_dflash_target_hidden_chunk(Some(captured));
            if terminal_prompt {
                let sampled = gpu_sample_token(&logits, &self.params);
                eval(&[&sampled]);
                return Ok(Some(sampled.item_i32() as u32));
            }
            return Ok(None);
        }

        let capture_cpp_hidden =
            self.dflash.is_some() && matches!(self.mode, Qwen35StepMode::Cpp(_));
        let cpp_model_raw = if capture_cpp_hidden {
            Some(
                self.weights
                    .cpp_model
                    .as_ref()
                    .context("Qwen3.5/Qwen3.6 DFlash requires C++ compiled model")?
                    .as_raw(),
            )
        } else {
            None
        };
        let capture_target_layer_ids = if capture_cpp_hidden {
            self.dflash
                .as_ref()
                .map(|dflash| dflash.target_layer_ids.clone())
        } else {
            None
        };
        let logits = if let (Some(raw), Some(target_layer_ids)) =
            (cpp_model_raw, capture_target_layer_ids.as_deref())
        {
            super::qwen35::with_qwen35_capture_layers(raw, target_layer_ids, || {
                self.run_step(token)
            })?
        } else {
            self.run_step(token)?
        };
        if let Some(raw) = cpp_model_raw {
            self.append_dflash_target_hidden_from_cpp_outputs(raw)?;
        }
        if terminal_prompt {
            let sampled = gpu_sample_token(&logits, &self.params);
            eval(&[&sampled]);
            Ok(Some(sampled.item_i32() as u32))
        } else {
            Ok(None)
        }
    }

    fn prefill_tokens(&mut self, tokens: &[u32], terminal_prompt: bool) -> Result<Option<u32>> {
        if tokens.is_empty() {
            return Ok(None);
        }
        self.pending_sampled = None;
        let trace = metal_qwen35_trace_enabled();
        let started = trace.then(Instant::now);
        let cache_len_before = self.cache_len;

        let rust_hidden_target_layers = if matches!(self.mode, Qwen35StepMode::Rust(_)) {
            self.dflash
                .as_ref()
                .map(|dflash| dflash.target_layer_ids.clone())
        } else {
            None
        };
        let use_cpp_batch_prefill = matches!(self.mode, Qwen35StepMode::Cpp(_)) && tokens.len() > 1;
        let use_rust_hidden_prefill = rust_hidden_target_layers.is_some() && tokens.len() > 1;
        if trace {
            let mode = if use_cpp_batch_prefill {
                "cpp_batch_prefill"
            } else if use_rust_hidden_prefill {
                "rust_hidden_prefill"
            } else if matches!(self.mode, Qwen35StepMode::Cpp(_)) {
                "cpp_scalar_prefill"
            } else {
                "rust_scalar_prefill"
            };
            eprintln!(
                "metal_trace[qwen35_prefill_tokens:start]: mode={} tokens={} terminal={} cache_len={}",
                mode,
                tokens.len(),
                terminal_prompt,
                cache_len_before,
            );
        }
        if use_cpp_batch_prefill || use_rust_hidden_prefill {
            self.ensure_capacity(self.cache_len + tokens.len() as i32)?;
        }

        let result = match &mut self.mode {
            Qwen35StepMode::Cpp(state) if tokens.len() > 1 => {
                let token_values: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
                let token_arr = MlxArray::from_slice_i32(&token_values, &[tokens.len() as i32]);
                let cpp_model: &CppQwen35Model = self
                    .weights
                    .cpp_model
                    .as_ref()
                    .context("Qwen3.5/Qwen3.6 C++ prefill path missing compiled model")?;
                state.ensure_session_active(cpp_model)?;
                let logits = if let Some(ref dflash) = self.dflash {
                    super::qwen35::with_qwen35_capture_layers(
                        cpp_model.as_raw(),
                        &dflash.target_layer_ids,
                        || {
                            cpp_model.prefill_session(
                                &token_arr,
                                tokens.len() as i32,
                                self.cache_len,
                            )
                        },
                    )?
                } else {
                    cpp_model.prefill_session(&token_arr, tokens.len() as i32, self.cache_len)?
                };
                async_eval(&[&logits]);
                self.cache_len += tokens.len() as i32;
                if let Some(dflash) = self.dflash.as_mut() {
                    let captured = super::qwen35::capture_qwen35_hidden_from_cpp_outputs(
                        cpp_model.as_raw(),
                        dflash.target_layer_ids.len(),
                    )?;
                    super::qwen35::append_qwen35_captured_hidden_chunk(
                        &mut dflash.target_hidden,
                        captured,
                    );
                    dflash.prefetched_draft = None;
                }

                if terminal_prompt {
                    let sampled = gpu_sample_token(&logits, &self.params);
                    eval(&[&sampled]);
                    Ok(Some(sampled.item_i32() as u32))
                } else {
                    Ok(None)
                }
            }
            Qwen35StepMode::Rust(state) if use_rust_hidden_prefill => {
                let target_layer_ids = rust_hidden_target_layers
                    .as_deref()
                    .context("Qwen3.5/Qwen3.6 rust hidden prefill missing target layers")?;
                let (logits, captured) = Self::run_rust_prefill_hidden_chunk(
                    self.weights,
                    self.config,
                    self.arch,
                    self.cache_len,
                    state,
                    tokens,
                    target_layer_ids,
                );
                self.cache_len += tokens.len() as i32;
                self.append_dflash_target_hidden_chunk(Some(captured));
                if terminal_prompt {
                    let sampled = gpu_sample_token(&logits, &self.params);
                    eval(&[&sampled]);
                    Ok(Some(sampled.item_i32() as u32))
                } else {
                    Ok(None)
                }
            }
            _ => {
                let mut emitted = None;
                for (idx, &token) in tokens.iter().enumerate() {
                    let is_terminal = terminal_prompt && idx + 1 == tokens.len();
                    let sampled = self.prefill_token(token, is_terminal)?;
                    if is_terminal {
                        emitted = sampled;
                    } else if sampled.is_some() {
                        bail!("non-terminal prefill step unexpectedly emitted a sampled token");
                    }
                }
                Ok(emitted)
            }
        };
        if let (true, Some(started), Ok(emitted)) = (trace, started, result.as_ref()) {
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mode = if use_cpp_batch_prefill {
                "cpp_batch_prefill"
            } else if use_rust_hidden_prefill {
                "rust_hidden_prefill"
            } else if matches!(self.mode, Qwen35StepMode::Cpp(_)) {
                "cpp_scalar_prefill"
            } else {
                "rust_scalar_prefill"
            };
            eprintln!(
                "metal_trace[qwen35_prefill_tokens:done]: mode={} tokens={} terminal={} cache_len={}=>{} emitted={} elapsed_ms={:.1}",
                mode,
                tokens.len(),
                terminal_prompt,
                cache_len_before,
                self.cache_len,
                emitted.is_some(),
                elapsed_ms,
            );
        }
        result
    }

    fn decode_token(&mut self, token: u32) -> Result<u32> {
        if self.dflash.is_some() {
            self.pending_sampled = None;
        }

        // ── DFlash speculative path (Qwen3.5/Qwen3.6) ─────────────────────
        // 1. Drain buffer first — cheap, no GPU work, short-lived borrow.
        if let Some(dflash) = self.dflash.as_mut()
            && let Some(buffered) = dflash.token_buffer.pop_front()
        {
            return Ok(buffered);
        }

        // 2. Grow the target KV cache BEFORE running the speculative block.
        //    A prefix-snapshot import may have downsized kv_capacity to the
        //    replay driver's smaller allocation (e.g. 256), and the DFlash
        //    path otherwise bypasses run_step's ensure_capacity; once
        //    cache_len + block_size exceeds kv_capacity the C++ verify_block
        //    produces a malformed slice_update and the forward dies with
        //    "Shapes (1,4,16,256) and (1,4,N,256) cannot be broadcast".
        if let Some(block_size) = self.dflash.as_ref().map(|d| d.runtime.block_size()) {
            let block_size_i32 = i32::try_from(block_size)
                .context("Qwen3.5/Qwen3.6 DFlash block_size does not fit i32")?;
            let needed_cap = self.cache_len + block_size_i32;
            if needed_cap > self.kv_capacity {
                self.ensure_capacity(needed_cap)?;
            }
        }

        if let Some(dflash) = self.dflash.as_mut() {
            if let Qwen35StepMode::Cpp(ref mut cpp_state) = self.mode {
                let Some(target_hidden) = dflash.target_hidden.take() else {
                    // First decode after prefill — target_hidden not captured yet.
                    // Fall through to standard decode.
                    let _ = dflash;
                    let logits = self.run_step(token)?;
                    let sampled = gpu_sample_token(&logits, &self.params);
                    return Ok(sampled.item_i32() as u32);
                };

                let cpp_model = self
                    .weights
                    .cpp_model
                    .as_ref()
                    .context("Qwen3.5/Qwen3.6 DFlash requires C++ compiled model")?;
                cpp_state.ensure_caches_drained(cpp_model)?;

                let block = dflash::qwen35_dflash_speculative_block(
                    dflash.runtime,
                    token,
                    &target_hidden,
                    self.weights
                        .embedding
                        .dense()
                        .context("Qwen3.5/Qwen3.6 DFlash requires dense target embeddings")?,
                    &self.weights.lm_head,
                    dflash.config,
                    cpp_model,
                    &self.params,
                    &mut cpp_state.kv_flat,
                    &mut cpp_state.gdr_flat,
                    &mut self.cache_len,
                    &mut dflash.draft_state,
                    dflash.prefetched_draft.take(),
                )?;

                dflash.acceptance_lengths.push(block.accepted_inputs);
                dflash.target_hidden = Some(block.updated_target_hidden);
                dflash.prefetched_draft = block.prefetched_next_draft;

                for &t in &block.accepted_tokens {
                    dflash.token_buffer.push_back(t);
                }
                return dflash
                    .token_buffer
                    .pop_front()
                    .context("Qwen3.5/Qwen3.6 DFlash block produced zero tokens");
            }
            // Rust mode fallback: fall through to standard decode
        }

        // ── Standard single-token decode ─────────────────────────────────
        //
        // Cross-step double-buffering (mlx_lm pattern). On the hot path the
        // sampled MlxArray from step N is kept as a deferred tensor; we
        // immediately build + async_eval step N+1 *using that tensor* as the
        // input token (no CPU round-trip), and only THEN materialize step N's
        // scalar. This keeps the GPU command queue one step deep so it never
        // idles between steps.
        //
        // `sampled` from gpu_sample_token may be shape `[]` (argmax) or `[1]`
        // (categorical); normalize to `[1]` before feeding it into
        // `run_step_with_token`, which expects a 1-element token array.
        let result = if let Some(prev_sampled) = self.pending_sampled.take() {
            // Fast path: step N was pre-queued on the previous call.
            // Commit its cache slot accounting first.
            self.cache_len += 1;

            // cache_len is post-increment (committed); prequeue only needs one more slot.
            let can_prequeue = self.dflash.is_none() && self.cache_len < self.kv_capacity;
            if can_prequeue {
                if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
                    clear_metal_cache();
                }
                self.ensure_capacity(self.cache_len + 1)?;
                let token_arr = super::mlx::reshape(&prev_sampled, &[1]);
                let next_logits = self.run_step_with_token(&token_arr)?;
                let next_sampled = gpu_sample_token(&next_logits, &self.params);
                async_eval(&[&next_sampled]);
                self.pending_sampled = Some(next_sampled);
            }

            // Materialize step N's token LAST so step N+1 is already queued.
            prev_sampled.item_i32() as u32
        } else {
            // Cold path: no pre-queued step (first decode call, or previous
            // call hit the kv-capacity ceiling and skipped prequeue).
            if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
                clear_metal_cache();
            }
            let logits = self.run_step(token)?;
            let sampled = gpu_sample_token(&logits, &self.params);
            async_eval(&[&sampled]);

            let can_prequeue = self.dflash.is_none() && self.cache_len + 2 <= self.kv_capacity;
            if can_prequeue {
                if self.cache_len > 0 && self.cache_len % KV_CACHE_CHUNK == 0 {
                    clear_metal_cache();
                }
                self.ensure_capacity(self.cache_len + 1)?;
                let token_arr = super::mlx::reshape(&sampled, &[1]);
                let next_logits = self.run_step_with_token(&token_arr)?;
                let next_sampled = gpu_sample_token(&next_logits, &self.params);
                async_eval(&[&next_sampled]);
                self.pending_sampled = Some(next_sampled);
            }

            // Materialize step N AFTER step N+1 has been queued (if any).
            sampled.item_i32() as u32
        };

        Ok(result)
    }

    fn cleanup(&mut self) -> Result<()> {
        self.pending_sampled = None;
        if let Some(dflash) = self.dflash.as_mut() {
            dflash.prefetched_draft = None;
        }
        self.ensure_cpp_session_drained()
    }
}
