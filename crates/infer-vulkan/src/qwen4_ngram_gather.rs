//! The `qwen4_exp` (Qwen3.8-Flash-Next) n-gram table gather — the one part of
//! the forward pass that reads from outside device memory.
//!
//! [`qwen4_ple`](crate::qwen4_ple) turns a token history into row ids. This
//! module turns those row ids into the `[ple_embed_dim]` f32 vector the PLE
//! layer consumes. It owns nothing else: no hashing, no PLE math.
//!
//! # Why this is not "just an indexed read"
//!
//! The table is 320,001,536 rows x 160 FP8 values = **47.68 GiB**, split across
//! 128 `ngram_embedding.shard_<i>.weight` tensors of `[2500012, 160]`. It
//! cannot be resident anywhere: the Vulkan device heap (74.43 GiB) is spoken
//! for by weights, and the host-visible heap is 37.22 GiB. So it stays on disk,
//! mmapped, and every token pays real page faults.
//!
//! One decode step needs 16 rows — 2,560 bytes in total — and those 16 reads
//! are not bandwidth, they are 16 independent storage latencies. The only
//! thing that removes them is having 16 outstanding at once. A `for` loop over
//! the row ids is the single way to get this component wrong, and it is a
//! *silent* way: the answer is bit-identical, only the token is milliseconds
//! slower.
//!
//! `memmap2` exposes no `PrefetchVirtualMemory` on Windows, so there is no way
//! to ask the OS to overlap the faults. The parallelism has to be explicit,
//! hence the persistent worker pool in [`NgramGather`]. It is persistent
//! because spawning threads costs more than the work: thread creation is tens
//! of microseconds each on Windows, and the warm gather is 2.5 us in total.
//!
//! # What the fan-out is actually worth, measured
//!
//! Ryzen AI MAX+ 395, Performance power scheme, checkpoint on a WD PC SN740
//! (DRAM-less NVMe), 16 rows per token, release build. **Cold** means rows
//! never touched before — the number that matters, because hash-scattered ids
//! over a 320M-row table rarely repeat. **Warm** means the same rows again out
//! of the OS page cache.
//!
//! ```text
//!   path                cold us/token    warm us/token
//!   no pool (serial)             2897              2.3
//!    1 worker                    3343              4.3
//!    4 workers                   1079              3.7
//!    8 workers                    764             17.0
//!   16 workers                    667             31.3
//!   32 workers                    653             31.8
//! ```
//!
//! So the fan-out is worth **4.3x**, not the 138x an earlier note recorded:
//! that figure compared a *cold* serial baseline against a *warm* parallel one.
//! Any benchmark here that reuses row ids measures the page cache and nothing
//! else, which is why `real_checkpoint_gather_scales_with_threads` reseeds
//! itself per run and refuses to assert a ratio unless the baseline proves the
//! reads really faulted.
//!
//! Two things fall out of that table. The device saturates around 4x whatever
//! we do — 32 workers buy 2% over 16, and positional `seek_read` in place of
//! the mmap measured no better cold (2234 us/token serial) while costing 7x
//! more warm (1 us of syscall per row against 0.15 us of resident fault). And
//! the pool is a small *regression* on fully warm rows, because waking 16
//! parked threads on Windows costs ~30 us. [`DEFAULT_WORKERS`] is 16 anyway:
//! 16 workers only lose to 4 once the page-cache hit rate passes 94%, and the
//! 28 us they cost when warm is nothing beside the 2.2 ms they save when cold.
//!
//! Reading through the mmap is therefore the measured choice, not an assumed
//! one: a device-local Vulkan allocation does not consume OS RAM on this box,
//! so the page cache stays free to hold whatever of the table it can, and a
//! read that hits it costs a few hundred nanoseconds with no syscall at all.
//!
//! # The scale is not optional
//!
//! The shards are `F8_E4M3` bytes plus **one BF16 scalar**
//! `ngram_embedding.weight_scale` for the entire table (0.000199... on the
//! on-box checkpoint). The checkpoint's own qualification notes say it out
//! loud: a loader that only upcasts the FP8 bytes "will serve wrong PLE
//! embeddings silently" — off by ~5000x, still finite, still plausible.
//! [`NgramTable::read_row`] applies it, and
//! `synthetic_table_applies_the_scalar_weight_scale` fails if it stops.
//!
//! The product is computed in f32 and kept there. That is not a shortcut: an
//! E4M3 element carries 4 significant bits and a BF16 scale carries 8, so the
//! product needs at most 12 and f32's 24 hold it **exactly**. The reference
//! loader rounds the same product to BF16 because that is its compute dtype;
//! rounding here would only throw information away.
//!
//! FP8 codes `0x7F`/`0xFF` decode to NaN, matching PyTorch's `float8_e4m3fn`.
//! No policy is invented for them — a NaN in the table is a corrupt table, and
//! it should reach the logits as a NaN rather than as a quietly zeroed row.
//! (Measured: the on-box shard 0 contains neither code in its first 20,000
//! rows.)
//!
//! # An out-of-range row id is a hash bug
//!
//! Every id this module receives came out of the hash. There is no such thing
//! as a legitimately missing row, so a negative or too-large id is refused with
//! the slot it arrived in rather than filled with zeros. A zero row is the one
//! failure this component could hide, and 47.68 GiB of table means a *wrong*
//! row still looks like plausible numbers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail, ensure};

use infer_gguf::dequant::{bf16_to_f32, fp8_e4m3_to_f32};
use infer_gguf::safetensors::{SafeTensorInfo, SafeTensorsDir};

use crate::qwen4_names::{Qwen4Stream, Qwen4TensorKind, classify_qwen4_tensor};
use crate::qwen4_ple::NGramHash;

/// How long the pool waits for a worker's answer before declaring itself
/// broken. The work it is waiting on is ~1.5 ms at its slowest, so any value
/// this far out means a worker died rather than "the disk was busy". Turning
/// that into an error rather than a `recv()` that never returns is the
/// difference between a diagnosable failure and a hung decode.
const WORKER_TIMEOUT: Duration = Duration::from_secs(60);

/// Upper bound on pool size. Beyond `ngram_heads` (16) there is nothing left
/// to overlap at decode, and the measured table in the module docs shows 32
/// workers buying 2% over 16; the cap is well above both only so prefill,
/// which submits many tokens' rows at once, is not artificially throttled.
const MAX_WORKERS: usize = 64;

/// One worker per n-gram head — the exact decode fan-out, and the point where
/// the measured cold cost stops improving materially. See the module docs for
/// the cold/warm table this comes from, and why fewer workers would be better
/// only on an almost fully cached table.
pub const DEFAULT_WORKERS: usize = 16;

// --------------------------------------------------------------------------
// table
// --------------------------------------------------------------------------

/// One `ngram_embedding.shard_<i>.weight` tensor.
#[derive(Debug, Clone)]
pub struct NgramShard {
    /// The tensor name, kept so reads go back through [`SafeTensorsDir`]
    /// rather than through a slice this struct would have to outlive.
    pub name: String,
    /// Rows this shard holds. Every shard but the last holds
    /// [`NgramTable::rows_per_shard`].
    pub rows: u64,
}

/// The FP8 n-gram embedding table, addressed by global row id.
///
/// Holds an [`Arc<SafeTensorsDir>`] rather than borrowed slices so the worker
/// threads in [`NgramGather`] can be plain `'static` threads. Each read costs
/// one name lookup in the checkpoint's tensor index (~60 ns against a ~100 us
/// page fault), which buys a struct with no lifetime and no unsafe.
pub struct NgramTable {
    st: Arc<SafeTensorsDir>,
    shards: Vec<NgramShard>,
    rows_per_shard: u64,
    row_width: usize,
    total_rows: u64,
    scale: f32,
}

impl NgramTable {
    /// Locate the n-gram table inside an already-open checkpoint.
    ///
    /// The PLE layer index is discovered, not assumed: names are classified by
    /// [`classify_qwen4_tensor`], so this works whichever `layers.<n>.ple.`
    /// the exporter put the table under.
    ///
    /// # Errors
    /// If the shard set is not `0..n` contiguous, if the shards disagree on row
    /// width or dtype, if more than one PLE layer ships a table, or if the
    /// scalar `weight_scale` is missing or non-finite.
    pub fn new(st: Arc<SafeTensorsDir>) -> Result<Self> {
        let mut shard_infos: BTreeMap<u32, &SafeTensorInfo> = BTreeMap::new();
        let mut scale_infos: Vec<&SafeTensorInfo> = Vec::new();
        let mut layers: BTreeSet<Option<usize>> = BTreeSet::new();

        for info in st.tensors() {
            // Cheap reject first: `classify_qwen4_tensor` is a full parse and
            // the on-box checkpoint carries 296,475 names.
            if !info.name.contains("ngram_embedding.") {
                continue;
            }
            let role = classify_qwen4_tensor(&info.name)
                .with_context(|| format!("classifying `{}`", info.name))?;
            if role.stream != Qwen4Stream::Text {
                continue;
            }
            match role.kind {
                Qwen4TensorKind::PleNgramShard => {
                    let idx = role
                        .sub_index
                        .ok_or_else(|| anyhow!("n-gram shard `{}` has no index", info.name))?;
                    if let Some(prev) = shard_infos.insert(idx, info) {
                        bail!(
                            "two n-gram shards claim index {idx}: `{}` and `{}`",
                            prev.name,
                            info.name
                        );
                    }
                    layers.insert(role.layer);
                }
                Qwen4TensorKind::PleNgramWeightScale => {
                    scale_infos.push(info);
                    layers.insert(role.layer);
                }
                _ => {}
            }
        }

        ensure!(
            !shard_infos.is_empty(),
            "checkpoint has no `ngram_embedding.shard_<i>.weight` tensors — this is not a \
             qwen4_exp checkpoint, or the PLE table was stripped"
        );
        ensure!(
            layers.len() == 1,
            "n-gram tensors span {} PLE layers ({:?}); this gather addresses exactly one table",
            layers.len(),
            layers
        );

        // The shard set must be 0..n with no gaps, because `row / rows_per_shard`
        // is the only thing that maps a row to a file. A missing shard would
        // otherwise shift every row after it onto a different, wrong embedding.
        let shard_count = shard_infos.len();
        let expected: BTreeSet<u32> = (0..shard_count as u32).collect();
        let present: BTreeSet<u32> = shard_infos.keys().copied().collect();
        ensure!(
            present == expected,
            "n-gram shard indices are not 0..{shard_count}: missing {:?}, unexpected {:?}",
            expected.difference(&present).collect::<Vec<_>>(),
            present.difference(&expected).collect::<Vec<_>>()
        );

        let first = shard_infos[&0];
        ensure!(
            first.dims.len() == 2,
            "n-gram shard `{}` has rank {}, want 2",
            first.name,
            first.dims.len()
        );
        // `dims` is the safetensors shape REVERSED, so dims[0] is the row width
        // (the contiguous dim) and dims[1] the row count.
        let row_width = usize::try_from(first.dims[0])
            .ok()
            .filter(|&w| w > 0)
            .ok_or_else(|| anyhow!("n-gram row width {} is unusable", first.dims[0]))?;
        let rows_per_shard = first.dims[1];
        ensure!(rows_per_shard > 0, "n-gram shard 0 declares zero rows");

        let mut shards = Vec::with_capacity(shard_count);
        let mut total_rows = 0u64;
        for (&idx, info) in &shard_infos {
            ensure!(
                info.dtype == "F8_E4M3",
                "n-gram shard `{}` is {}, want F8_E4M3",
                info.name,
                info.dtype
            );
            ensure!(
                info.dims.len() == 2 && info.dims[0] == row_width as u64,
                "n-gram shard `{}` has dims {:?}, want [{row_width}, rows]",
                info.name,
                info.dims
            );
            let rows = info.dims[1];
            // Only the LAST shard may be short: a table whose padded row count
            // is not a multiple of the shard count leaves a ragged tail, and
            // `row / rows_per_shard` still lands in the right file as long as
            // every earlier shard is full.
            let last = idx as usize + 1 == shard_count;
            if last {
                ensure!(
                    rows > 0 && rows <= rows_per_shard,
                    "n-gram tail shard `{}` holds {rows} rows, want 1..={rows_per_shard}",
                    info.name
                );
            } else {
                ensure!(
                    rows == rows_per_shard,
                    "n-gram shard `{}` holds {rows} rows but shard 0 holds {rows_per_shard}; \
                     only the last shard may be short",
                    info.name
                );
            }
            let want_bytes = rows
                .checked_mul(row_width as u64)
                .ok_or_else(|| anyhow!("n-gram shard `{}` size overflows", info.name))?;
            ensure!(
                info.len == want_bytes,
                "n-gram shard `{}` declares {} bytes but {rows}x{row_width} FP8 needs {want_bytes}",
                info.name,
                info.len
            );
            // Prove the bytes are actually reachable now, so the hot path is
            // reading a slice this constructor already resolved once.
            let data = st.tensor_data(&info.name)?;
            ensure!(
                data.len() as u64 == want_bytes,
                "n-gram shard `{}` maps {} bytes, want {want_bytes}",
                info.name,
                data.len()
            );
            total_rows += rows;
            shards.push(NgramShard {
                name: info.name.clone(),
                rows,
            });
        }

        ensure!(
            scale_infos.len() == 1,
            "checkpoint has {} `ngram_embedding.weight_scale` tensors, want exactly 1",
            scale_infos.len()
        );
        let scale = read_scalar_scale(&st, scale_infos[0])?;

        Ok(Self {
            st,
            shards,
            rows_per_shard,
            row_width,
            total_rows,
            scale,
        })
    }

    /// Open a checkpoint directory and locate its n-gram table.
    ///
    /// # Errors
    /// If the directory holds no safetensors shards, or [`Self::new`] rejects
    /// what it finds.
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let st = SafeTensorsDir::open_dir(dir)?;
        Self::new(Arc::new(st))
    }

    /// The shard tensors, in shard-index order.
    #[must_use]
    pub fn shards(&self) -> &[NgramShard] {
        &self.shards
    }

    /// Rows in every shard but (possibly) the last.
    #[must_use]
    pub fn rows_per_shard(&self) -> u64 {
        self.rows_per_shard
    }

    /// Values per row — `ple_embed_dim / ngram_heads`, 160 on this checkpoint.
    #[must_use]
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    /// Total addressable rows, i.e. the padded n-gram vocab.
    #[must_use]
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// The one BF16 scalar every element is multiplied by.
    #[must_use]
    pub fn weight_scale(&self) -> f32 {
        self.scale
    }

    /// Check the table against the hash that will address it.
    ///
    /// Worth doing at load: the hash derives its geometry from `config.json`
    /// and the table from the shard headers, and if those two ever disagree
    /// every lookup silently reads the wrong row.
    ///
    /// # Errors
    /// If the row width is not the hash's head dim, or the table's row count is
    /// not the hash's padded vocab.
    pub fn check_against_hash(&self, hash: &NGramHash) -> Result<()> {
        ensure!(
            self.row_width == hash.head_dim(),
            "table rows are {} wide but the hash expects head_dim {}",
            self.row_width,
            hash.head_dim()
        );
        ensure!(
            self.total_rows == hash.padded_vocab_size(),
            "table holds {} rows but the hash addresses a padded vocab of {}",
            self.total_rows,
            hash.padded_vocab_size()
        );
        Ok(())
    }

    /// Dequantize one row into `out`, which must be [`Self::row_width`] long.
    ///
    /// # Errors
    /// If `row` is negative or `>= total_rows` — see the module docs: that is a
    /// hash bug, not a table miss, and it is refused rather than zero-filled.
    pub fn read_row(&self, row: i64, out: &mut [f32]) -> Result<()> {
        ensure!(
            out.len() == self.row_width,
            "n-gram row destination is {} wide, want {}",
            out.len(),
            self.row_width
        );
        let row = u64::try_from(row).map_err(|_| {
            anyhow!(
                "negative n-gram row id {row}: the hash cannot produce one, so this is a hash bug"
            )
        })?;
        ensure!(
            row < self.total_rows,
            "n-gram row id {row} is outside the {}-row table — a hash bug, not a missing row",
            self.total_rows
        );

        let shard = usize::try_from(row / self.rows_per_shard)
            .map_err(|_| anyhow!("n-gram shard index for row {row} overflows"))?;
        let shard = self.shards.get(shard).ok_or_else(|| {
            anyhow!("n-gram row {row} maps to shard {shard}, which does not exist")
        })?;
        let start = usize::try_from((row % self.rows_per_shard) * self.row_width as u64)
            .map_err(|_| anyhow!("n-gram byte offset for row {row} overflows usize"))?;

        let data = self.st.tensor_data(&shard.name)?;
        let src = data
            .get(start..start + self.row_width)
            .ok_or_else(|| anyhow!("n-gram row {row} runs past the end of `{}`", shard.name))?;
        for (dst, &byte) in out.iter_mut().zip(src) {
            *dst = fp8_e4m3_to_f32(byte) * self.scale;
        }
        Ok(())
    }

    /// Dequantize `row_ids` into `out` **serially**.
    ///
    /// `out[r * row_width + j]` is value `j` of `row_ids[r]`, which for one
    /// token's 16 head ids is exactly the reference's `flatten(-2)` layout:
    /// head 0's 160 floats, then head 1's, and so on.
    ///
    /// This is the shape of the trap the module docs describe — correct, and
    /// 138x too slow at decode. It exists as the body each [`NgramGather`]
    /// worker runs on its own slice, and as the oracle the parallel path is
    /// diffed against.
    ///
    /// # Errors
    /// If `out` is not `row_ids.len() * row_width` long, or any id is out of
    /// range.
    pub fn gather_rows(&self, row_ids: &[i64], out: &mut [f32]) -> Result<()> {
        let want = row_ids.len() * self.row_width;
        ensure!(
            out.len() == want,
            "n-gram gather destination is {} long, want {want} for {} rows",
            out.len(),
            row_ids.len()
        );
        for (slot, (&row, dst)) in row_ids
            .iter()
            .zip(out.chunks_exact_mut(self.row_width))
            .enumerate()
        {
            self.read_row(row, dst)
                .with_context(|| format!("n-gram row slot {slot}"))?;
        }
        Ok(())
    }

    /// Refuse the whole batch before any worker starts.
    ///
    /// Hoisting the range check into the caller's thread is what lets the
    /// workers be infallible on the only error they could otherwise hit, which
    /// in turn keeps a bad id from taking down a pool thread mid-token.
    fn validate_ids(&self, row_ids: &[i64]) -> Result<()> {
        for (slot, &row) in row_ids.iter().enumerate() {
            ensure!(
                row >= 0 && (row as u64) < self.total_rows,
                "n-gram row slot {slot} holds id {row}, outside the {}-row table — a hash bug",
                self.total_rows
            );
        }
        Ok(())
    }
}

impl std::fmt::Debug for NgramTable {
    // `SafeTensorsDir` is not `Debug`, and printing 128 shard names would not
    // help anyway: the geometry is what a caller wants to see.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramTable")
            .field("shards", &self.shards.len())
            .field("rows_per_shard", &self.rows_per_shard)
            .field("row_width", &self.row_width)
            .field("total_rows", &self.total_rows)
            .field("weight_scale", &self.scale)
            .finish()
    }
}

/// Read the table's single scalar scale, whatever float dtype it was written in.
fn read_scalar_scale(st: &SafeTensorsDir, info: &SafeTensorInfo) -> Result<f32> {
    ensure!(
        info.element_count() == 1,
        "`{}` holds {} elements, want a single scalar scale",
        info.name,
        info.element_count()
    );
    let bytes = st.tensor_data(&info.name)?;
    let scale = match info.dtype.as_str() {
        "BF16" => {
            let raw: [u8; 2] = bytes
                .try_into()
                .map_err(|_| anyhow!("`{}` is {} bytes, want 2", info.name, bytes.len()))?;
            bf16_to_f32(u16::from_le_bytes(raw))
        }
        "F32" => {
            let raw: [u8; 4] = bytes
                .try_into()
                .map_err(|_| anyhow!("`{}` is {} bytes, want 4", info.name, bytes.len()))?;
            f32::from_le_bytes(raw)
        }
        other => bail!("`{}` has dtype {other}, want BF16 or F32", info.name),
    };
    // A zero or non-finite scale would blank or poison all 47.68 GiB at once.
    ensure!(
        scale.is_finite() && scale != 0.0,
        "n-gram weight_scale is {scale}, which cannot be right"
    );
    Ok(scale)
}

// --------------------------------------------------------------------------
// parallel gather
// --------------------------------------------------------------------------

/// One worker's slice of a gather, and the buffers it borrows to answer.
///
/// The buffers travel with the job in both directions so the pool allocates
/// once and then never again: the caller's steady state is popping a `Chunk`
/// off the free list, filling `rows`, and getting the same allocation back.
struct Chunk {
    /// Index of this chunk's first row within the caller's `row_ids`.
    first: usize,
    rows: Vec<i64>,
    out: Vec<f32>,
    failure: Option<anyhow::Error>,
}

impl Chunk {
    fn empty() -> Self {
        Self {
            first: 0,
            rows: Vec::new(),
            out: Vec::new(),
            failure: None,
        }
    }
}

/// A persistent pool of threads that fault the n-gram rows in parallel.
///
/// One per decode loop; [`Self::gather`] takes `&mut self` because it recycles
/// the job buffers. Dropping it closes the worker inboxes and joins them.
pub struct NgramGather {
    table: Arc<NgramTable>,
    /// Per-worker inbox. Static assignment (chunk `i` -> worker `i`): with 16
    /// equal-cost latency-bound reads there is nothing for work stealing to
    /// balance, and one channel per worker means no contention on dispatch.
    inbox: Vec<Sender<Chunk>>,
    results: Receiver<Chunk>,
    workers: Vec<JoinHandle<()>>,
    free: Vec<Chunk>,
    handled: Vec<Arc<AtomicU64>>,
    /// Set once a worker fails to answer. The pool cannot know which buffers
    /// are still in flight after that, so later calls fail fast instead of
    /// mixing a stale chunk into the next token's embedding.
    poisoned: bool,
}

impl NgramGather {
    /// Spawn `threads` workers over `table`.
    ///
    /// # Errors
    /// If `threads` is 0 or above [`MAX_WORKERS`], or the OS refuses a thread.
    pub fn new(table: Arc<NgramTable>, threads: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_WORKERS).contains(&threads),
            "n-gram gather wants 1..={MAX_WORKERS} workers, got {threads}"
        );
        let (result_tx, results) = channel::<Chunk>();
        let mut inbox = Vec::with_capacity(threads);
        let mut workers = Vec::with_capacity(threads);
        let mut handled = Vec::with_capacity(threads);

        for worker in 0..threads {
            let (job_tx, jobs) = channel::<Chunk>();
            let table = Arc::clone(&table);
            let answers = result_tx.clone();
            let counter = Arc::new(AtomicU64::new(0));
            let mine = Arc::clone(&counter);
            let handle = std::thread::Builder::new()
                .name(format!("arle-ngram-{worker}"))
                .spawn(move || {
                    while let Ok(mut chunk) = jobs.recv() {
                        mine.fetch_add(1, Ordering::Relaxed);
                        // Every fallible step inside is a returned error, never
                        // an index panic: a panicking worker would leave the
                        // caller waiting on a chunk that can never arrive.
                        chunk.failure = table.gather_rows(&chunk.rows, &mut chunk.out).err();
                        if answers.send(chunk).is_err() {
                            break;
                        }
                    }
                })
                .with_context(|| format!("spawn n-gram gather worker {worker}"))?;
            inbox.push(job_tx);
            workers.push(handle);
            handled.push(counter);
        }

        Ok(Self {
            table,
            inbox,
            results,
            workers,
            free: Vec::new(),
            handled,
            poisoned: false,
        })
    }

    /// Open a checkpoint directory and spawn a pool over its n-gram table.
    ///
    /// # Errors
    /// If the table cannot be located, or the pool cannot be spawned.
    pub fn open_dir(dir: impl AsRef<Path>, threads: usize) -> Result<Self> {
        Self::new(Arc::new(NgramTable::open_dir(dir)?), threads)
    }

    #[must_use]
    pub fn table(&self) -> &Arc<NgramTable> {
        &self.table
    }

    /// Pool size.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.inbox.len()
    }

    /// Chunks each worker has handled since the pool was built.
    ///
    /// This is the regression guard the module docs argue for: a gather that
    /// quietly became serial produces identical numbers, so the only cheap
    /// evidence that the fan-out is real is that every worker was given work.
    #[must_use]
    pub fn chunks_per_worker(&self) -> Vec<u64> {
        self.handled
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect()
    }

    /// Dequantize `row_ids` into `out` across the pool.
    ///
    /// Same layout as [`NgramTable::gather_rows`]: `out[r * row_width + j]`.
    /// For decode, `row_ids` is one token's `ngram_heads` ids and `out` is the
    /// `[ple_embed_dim]` vector the PLE layer wants. For prefill, pass every
    /// token's ids at once — the extra rows only deepen the queue.
    ///
    /// # Errors
    /// If `out` is the wrong length, any id is out of range, the pool was
    /// poisoned by an earlier timeout, or a worker fails or stops answering.
    pub fn gather(&mut self, row_ids: &[i64], out: &mut [f32]) -> Result<()> {
        ensure!(
            !self.poisoned,
            "n-gram gather pool was poisoned by an unanswered chunk; rebuild it"
        );
        let width = self.table.row_width;
        let want = row_ids.len() * width;
        ensure!(
            out.len() == want,
            "n-gram gather destination is {} long, want {want} for {} rows",
            out.len(),
            row_ids.len()
        );
        if row_ids.is_empty() {
            return Ok(());
        }
        self.table.validate_ids(row_ids)?;

        // Disjoint field borrows: the dispatch loop reads `inbox` while it pops
        // from `free`.
        let Self {
            inbox,
            free,
            results,
            poisoned,
            ..
        } = self;

        let parts = inbox.len().min(row_ids.len());
        let base = row_ids.len() / parts;
        let extra = row_ids.len() % parts;

        let mut dispatched = 0usize;
        let mut trouble: Option<anyhow::Error> = None;
        let mut start = 0usize;
        for (worker, tx) in inbox.iter().enumerate().take(parts) {
            let take = base + usize::from(worker < extra);
            let mut chunk = free.pop().unwrap_or_else(Chunk::empty);
            chunk.first = start;
            chunk.rows.clear();
            chunk.rows.extend_from_slice(&row_ids[start..start + take]);
            chunk.out.clear();
            chunk.out.resize(take * width, 0.0);
            chunk.failure = None;
            start += take;
            match tx.send(chunk) {
                Ok(()) => dispatched += 1,
                Err(_) => {
                    trouble = Some(anyhow!("n-gram gather worker {worker} is gone"));
                    break;
                }
            }
        }

        // Drain everything that was sent even after a failure, so the free list
        // and the result channel stay in step for the next token.
        for _ in 0..dispatched {
            let mut chunk = match results.recv_timeout(WORKER_TIMEOUT) {
                Ok(chunk) => chunk,
                Err(RecvTimeoutError::Timeout) => {
                    *poisoned = true;
                    trouble.get_or_insert_with(|| {
                        anyhow!(
                            "n-gram gather worker did not answer within {} s",
                            WORKER_TIMEOUT.as_secs()
                        )
                    });
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    *poisoned = true;
                    trouble
                        .get_or_insert_with(|| anyhow!("every n-gram gather worker has stopped"));
                    break;
                }
            };
            if let Some(err) = chunk.failure.take() {
                trouble.get_or_insert(err);
            } else if let Some(dst) = out.get_mut(chunk.first * width..) {
                let n = chunk.out.len();
                if let Some(dst) = dst.get_mut(..n) {
                    dst.copy_from_slice(&chunk.out);
                } else {
                    trouble.get_or_insert_with(|| {
                        anyhow!(
                            "n-gram chunk at row {} overruns the destination",
                            chunk.first
                        )
                    });
                }
            }
            free.push(chunk);
        }

        match trouble {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl Drop for NgramGather {
    fn drop(&mut self) {
        // Closing the inboxes is the shutdown signal; the workers fall out of
        // their `recv()` loop and the joins are immediate.
        self.inbox.clear();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

impl std::fmt::Debug for NgramGather {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NgramGather")
            .field("threads", &self.inbox.len())
            .field("rows", &self.table.total_rows)
            .field("row_width", &self.table.row_width)
            .field("poisoned", &self.poisoned)
            .finish()
    }
}

// --------------------------------------------------------------------------
// tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod synthetic_tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU32;

    const PREFIX: &str = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";

    /// A scale that is not 1.0 and not a power of two, so forgetting it or
    /// applying it twice both show up.
    const SCALE_BITS: u16 = 0x3E4C; // bf16 0.19921875

    fn scale() -> f32 {
        bf16_to_f32(SCALE_BITS)
    }

    fn scratch(tag: &str) -> PathBuf {
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "arle-ngram-{}-{tag}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    /// Write one safetensors shard by hand: 8-byte LE header length, JSON
    /// header, then the blob. Building the file rather than mocking the reader
    /// keeps the test on the same parse path the checkpoint takes.
    fn write_shard(path: &Path, entries: &[(String, Vec<u64>, &str, Vec<u8>)]) {
        let mut json = String::from("{");
        let mut blob: Vec<u8> = Vec::new();
        for (i, (name, shape, dtype, bytes)) in entries.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let start = blob.len();
            blob.extend_from_slice(bytes);
            let shape: Vec<String> = shape.iter().map(u64::to_string).collect();
            json.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
                shape.join(","),
                blob.len()
            ));
        }
        json.push('}');
        let mut file = Vec::new();
        file.extend_from_slice(&(json.len() as u64).to_le_bytes());
        file.extend_from_slice(json.as_bytes());
        file.extend_from_slice(&blob);
        std::fs::write(path, file).expect("write shard");
    }

    /// FP8 byte for row `r`, column `j`. Spread over the low half of the code
    /// space so every value is a distinct finite normal.
    fn byte_of(row: u64, col: usize) -> u8 {
        (((row * 31 + col as u64 * 7) % 120) + 1) as u8
    }

    /// A table of `shard_rows.len()` shards; entry `i` is shard `i`'s row count.
    fn build_table(tag: &str, shard_rows: &[u64], width: usize) -> (PathBuf, NgramTable) {
        let dir = scratch(tag);
        let mut entries: Vec<(String, Vec<u64>, &str, Vec<u8>)> = Vec::new();
        let mut base = 0u64;
        for (idx, &rows) in shard_rows.iter().enumerate() {
            let mut bytes = Vec::with_capacity((rows as usize) * width);
            for r in 0..rows {
                for c in 0..width {
                    bytes.push(byte_of(base + r, c));
                }
            }
            entries.push((
                format!("{PREFIX}shard_{idx}.weight"),
                vec![rows, width as u64],
                "F8_E4M3",
                bytes,
            ));
            base += rows;
        }
        entries.push((
            format!("{PREFIX}weight_scale"),
            vec![1],
            "BF16",
            SCALE_BITS.to_le_bytes().to_vec(),
        ));
        write_shard(&dir.join("model-00000.safetensors"), &entries);
        let table = NgramTable::open_dir(&dir).expect("open synthetic table");
        (dir, table)
    }

    fn expect_row(row: u64, width: usize) -> Vec<f32> {
        (0..width)
            .map(|c| fp8_e4m3_to_f32(byte_of(row, c)) * scale())
            .collect()
    }

    #[test]
    fn synthetic_table_applies_the_scalar_weight_scale() {
        let (dir, table) = build_table("scale", &[4, 4], 6);
        assert_eq!(table.total_rows(), 8);
        assert_eq!(table.row_width(), 6);
        assert_eq!(table.rows_per_shard(), 4);
        assert!((table.weight_scale() - scale()).abs() < 1e-12);

        let mut got = vec![0.0f32; 6];
        table.read_row(5, &mut got).expect("read row 5");
        assert_eq!(got, expect_row(5, 6));
        // The failure this pins: dropping the scale leaves plausible finite
        // numbers ~5x too big here (~5000x on the real table).
        let unscaled: Vec<f32> = (0..6).map(|c| fp8_e4m3_to_f32(byte_of(5, c))).collect();
        assert_ne!(got, unscaled);
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_row_ids_map_across_shard_boundaries() {
        let (dir, table) = build_table("bounds", &[4, 4, 4], 5);
        // The last row of each shard and the first of the next: an off-by-one
        // in `row / rows_per_shard` reads a real, wrong row rather than failing.
        for row in [0u64, 3, 4, 7, 8, 11] {
            let mut got = vec![0.0f32; 5];
            table.read_row(row as i64, &mut got).expect("read");
            assert_eq!(got, expect_row(row, 5), "row {row}");
        }
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_ragged_tail_shard_is_addressed_correctly() {
        let (dir, table) = build_table("ragged", &[4, 4, 2], 3);
        assert_eq!(table.total_rows(), 10);
        assert_eq!(table.rows_per_shard(), 4);
        let mut got = vec![0.0f32; 3];
        table.read_row(9, &mut got).expect("last row");
        assert_eq!(got, expect_row(9, 3));
        assert!(table.read_row(10, &mut got).is_err(), "past the tail");
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_parallel_gather_matches_the_serial_path() {
        let (dir, table) = build_table("parallel", &[8, 8, 8, 8], 4);
        let table = Arc::new(table);
        // Deliberately unsorted and repeating: the pool must preserve the
        // caller's slot order, not the order chunks come back in.
        let ids: Vec<i64> = vec![31, 0, 17, 17, 8, 23, 4, 12, 30, 1, 9, 26, 15, 7, 2, 20];

        let mut serial = vec![0.0f32; ids.len() * 4];
        table.gather_rows(&ids, &mut serial).expect("serial");

        for threads in [1usize, 3, 8, 16] {
            let mut pool = NgramGather::new(Arc::clone(&table), threads).expect("pool");
            let mut parallel = vec![0.0f32; ids.len() * 4];
            pool.gather(&ids, &mut parallel).expect("parallel");
            assert_eq!(parallel, serial, "threads = {threads}");
        }
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_gather_reuses_buffers_and_survives_repeats() {
        let (dir, table) = build_table("repeat", &[8, 8], 4);
        let mut pool = NgramGather::new(Arc::new(table), 4).expect("pool");
        let mut out = vec![0.0f32; 4 * 4];
        for round in 0..8u64 {
            let ids: Vec<i64> = (0..4).map(|k| ((round * 4 + k) % 16) as i64).collect();
            pool.gather(&ids, &mut out).expect("gather");
            for (k, &row) in ids.iter().enumerate() {
                assert_eq!(&out[k * 4..(k + 1) * 4], &expect_row(row as u64, 4)[..]);
            }
        }
        // 8 rounds x 4 chunks, and never a fresh allocation after the first.
        assert_eq!(pool.chunks_per_worker(), vec![8; 4]);
        drop(pool);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The point of the pool. A gather that regressed to a serial loop returns
    /// identical numbers, so this is the cheap thing that would still fail.
    #[test]
    fn synthetic_every_worker_is_given_work() {
        let (dir, table) = build_table("fanout", &[64], 4);
        let mut pool = NgramGather::new(Arc::new(table), 16).expect("pool");
        assert_eq!(pool.threads(), 16);
        let ids: Vec<i64> = (0..16).collect();
        let mut out = vec![0.0f32; 16 * 4];
        pool.gather(&ids, &mut out).expect("gather");
        let counts = pool.chunks_per_worker();
        assert_eq!(counts.len(), 16);
        assert!(
            counts.iter().all(|&c| c == 1),
            "16 rows over 16 workers should be one chunk each, got {counts:?}"
        );
        drop(pool);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_fewer_rows_than_workers_still_covers_every_row() {
        let (dir, table) = build_table("short", &[64], 4);
        let mut pool = NgramGather::new(Arc::new(table), 16).expect("pool");
        let ids: Vec<i64> = vec![63, 5, 40];
        let mut out = vec![0.0f32; 3 * 4];
        pool.gather(&ids, &mut out).expect("gather");
        for (k, &row) in ids.iter().enumerate() {
            assert_eq!(&out[k * 4..(k + 1) * 4], &expect_row(row as u64, 4)[..]);
        }
        assert_eq!(pool.chunks_per_worker().iter().sum::<u64>(), 3);
        drop(pool);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_bad_row_ids_are_loud_not_zero() {
        let (dir, table) = build_table("bad-ids", &[8, 8], 4);
        let table = Arc::new(table);
        let mut pool = NgramGather::new(Arc::clone(&table), 4).expect("pool");
        let mut out = vec![0.0f32; 4 * 4];

        for bad in [-1i64, 16, i64::MIN, i64::MAX] {
            let ids = vec![0, bad, 2, 3];
            let err = pool.gather(&ids, &mut out).expect_err("bad id must fail");
            let text = format!("{err:#}");
            assert!(
                text.contains("slot 1"),
                "message should name the slot: {text}"
            );
            assert!(
                text.contains("hash bug"),
                "message should say what an out-of-range id means: {text}"
            );
            // And the serial path refuses it too, so neither entry point can
            // become the lenient one.
            let mut serial = vec![0.0f32; 4 * 4];
            assert!(table.gather_rows(&ids, &mut serial).is_err(), "id {bad}");
        }
        // The pool is still usable: a rejected batch never reached a worker.
        let ids = vec![0, 1, 2, 3];
        pool.gather(&ids, &mut out).expect("pool still works");
        drop(pool);
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_wrong_destination_length_is_rejected() {
        let (dir, table) = build_table("dest", &[8], 4);
        let table = Arc::new(table);
        let mut pool = NgramGather::new(Arc::clone(&table), 2).expect("pool");
        let ids = vec![0i64, 1];
        assert!(pool.gather(&ids, &mut [0.0; 7]).is_err());
        assert!(table.gather_rows(&ids, &mut [0.0; 9]).is_err());
        assert!(table.read_row(0, &mut [0.0; 3]).is_err());
        drop(pool);
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn synthetic_empty_gather_is_a_no_op() {
        let (dir, table) = build_table("empty", &[8], 4);
        let mut pool = NgramGather::new(Arc::new(table), 4).expect("pool");
        pool.gather(&[], &mut []).expect("empty gather");
        assert_eq!(pool.chunks_per_worker(), vec![0; 4]);
        drop(pool);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_shard_index_is_refused() {
        let dir = scratch("gap");
        let width = 4usize;
        let mut entries: Vec<(String, Vec<u64>, &str, Vec<u8>)> = Vec::new();
        // Shards 0, 1, 3 — index 2 is missing, which would silently shift every
        // row from 8 upward onto the wrong embedding.
        for idx in [0usize, 1, 3] {
            entries.push((
                format!("{PREFIX}shard_{idx}.weight"),
                vec![4, width as u64],
                "F8_E4M3",
                vec![1u8; 4 * width],
            ));
        }
        entries.push((
            format!("{PREFIX}weight_scale"),
            vec![1],
            "BF16",
            SCALE_BITS.to_le_bytes().to_vec(),
        ));
        write_shard(&dir.join("model-00000.safetensors"), &entries);
        let err = NgramTable::open_dir(&dir).expect_err("gap must be refused");
        assert!(format!("{err:#}").contains("not 0.."), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn non_tail_short_shard_is_refused() {
        let dir = scratch("short-mid");
        let width = 4usize;
        let mut entries: Vec<(String, Vec<u64>, &str, Vec<u8>)> = Vec::new();
        for (idx, rows) in [(0usize, 4u64), (1, 2), (2, 4)] {
            entries.push((
                format!("{PREFIX}shard_{idx}.weight"),
                vec![rows, width as u64],
                "F8_E4M3",
                vec![1u8; rows as usize * width],
            ));
        }
        entries.push((
            format!("{PREFIX}weight_scale"),
            vec![1],
            "BF16",
            SCALE_BITS.to_le_bytes().to_vec(),
        ));
        write_shard(&dir.join("model-00000.safetensors"), &entries);
        let err = NgramTable::open_dir(&dir).expect_err("ragged middle must be refused");
        assert!(
            format!("{err:#}").contains("only the last shard may be short"),
            "{err:#}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_weight_scale_is_refused() {
        let dir = scratch("no-scale");
        let entries = vec![(
            format!("{PREFIX}shard_0.weight"),
            vec![4u64, 4],
            "F8_E4M3",
            vec![1u8; 16],
        )];
        write_shard(&dir.join("model-00000.safetensors"), &entries);
        let err = NgramTable::open_dir(&dir).expect_err("scale is not optional");
        assert!(format!("{err:#}").contains("weight_scale"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn worker_count_is_bounded() {
        let (dir, table) = build_table("bounds-threads", &[4], 4);
        let table = Arc::new(table);
        assert!(NgramGather::new(Arc::clone(&table), 0).is_err());
        assert!(NgramGather::new(Arc::clone(&table), MAX_WORKERS + 1).is_err());
        assert!(NgramGather::new(Arc::clone(&table), MAX_WORKERS).is_ok());
        drop(table);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The pool hands `Arc<NgramTable>` to `'static` threads, which only
    /// compiles while the checkpoint reader stays `Send + Sync`.
    #[test]
    fn table_is_shareable_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NgramTable>();
        assert_send_sync::<SafeTensorsDir>();
    }
}

#[cfg(test)]
mod on_box_tests {
    use super::*;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::time::Instant;

    use crate::qwen4_ple::{NGramContext, NGramHashConfig, splitmix64};

    const CKPT_ENV: &str = "ARLE_QWEN4_CKPT";
    const CKPT_DEFAULT: &str = r"C:\Users\Asus\models\qwen3.8-flash-next-nvfp4";

    /// Not `#[ignore]`d, for the reason `qwen4_ple`'s on-box module gives: a
    /// data path whose only real cross-check never runs is unchecked. Off-box
    /// the directory is absent and the test returns.
    fn open() -> Option<Arc<NgramTable>> {
        let dir = std::env::var(CKPT_ENV).unwrap_or_else(|_| CKPT_DEFAULT.into());
        let path = std::path::Path::new(&dir);
        if !path.is_dir() {
            eprintln!("skip: {} not present (set {CKPT_ENV})", path.display());
            return None;
        }
        Some(Arc::new(
            NgramTable::open_dir(path).expect("open n-gram table"),
        ))
    }

    /// Deterministic pseudo-random row ids. Random matters: the table is
    /// 47.68 GiB against 128 GiB of RAM, so scattered ids essentially never hit
    /// the page cache and the timing test measures real faults.
    fn random_rows(seed: u64, count: usize, total: u64) -> Vec<i64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                (splitmix64(state) % total) as i64
            })
            .collect()
    }

    /// Read one row through a completely different path: open the shard file
    /// by hand, re-derive the blob start from its 8-byte header length, and
    /// `seek`/`read_exact` the bytes. No mmap, no `tensor_data`.
    ///
    /// The shard is found by a LINEAR SCAN over the shards' row counts rather
    /// than by dividing, so this is an independent derivation of the row -> file
    /// mapping and not a restatement of `read_row`'s arithmetic.
    fn row_bytes_via_file(st: &SafeTensorsDir, table: &NgramTable, row: u64) -> Vec<u8> {
        let mut base = 0u64;
        let mut hit = None;
        for shard in table.shards() {
            if row < base + shard.rows {
                hit = Some((shard, row - base));
                break;
            }
            base += shard.rows;
        }
        let (shard, within) = hit.unwrap_or_else(|| panic!("row {row} is in no shard"));

        let info = st.tensor(&shard.name).expect("shard tensor");
        let idx = st.tensor_shard(&shard.name).expect("shard file index");
        let path = st.shard_path(idx).expect("shard path");
        let mut file = File::open(path).expect("open shard file");
        let mut header_len = [0u8; 8];
        file.read_exact(&mut header_len).expect("header length");
        let blob = 8 + u64::from_le_bytes(header_len);

        let width = table.row_width() as u64;
        file.seek(SeekFrom::Start(blob + info.offset + within * width))
            .expect("seek");
        let mut bytes = vec![0u8; table.row_width()];
        file.read_exact(&mut bytes).expect("read row");
        bytes
    }

    #[test]
    fn real_checkpoint_table_geometry_matches_the_hash() {
        let Some(table) = open() else { return };
        let hash = NGramHash::new(NGramHashConfig::qwen4_exp()).expect("hash");
        table.check_against_hash(&hash).expect("geometry");

        assert_eq!(table.shards().len(), 128, "split_ngram_parts");
        assert_eq!(table.rows_per_shard(), 2_500_012);
        assert_eq!(table.row_width(), 160);
        assert_eq!(table.total_rows(), 320_001_536);
        // 47.68 GiB, one byte per FP8 element.
        assert_eq!(
            table.total_rows() * table.row_width() as u64,
            51_200_245_760
        );
        // The scalar the qualification notes warn about. Pinned as bits so a
        // drift is unambiguous rather than a rounding argument.
        assert_eq!(
            table.weight_scale().to_bits(),
            bf16_to_f32(0x3951).to_bits(),
            "n-gram weight_scale drifted from the checkpoint's 0x3951 BF16"
        );
    }

    #[test]
    fn real_checkpoint_gather_matches_an_independent_positional_read() {
        let Some(table) = open() else { return };
        let dir = std::env::var(CKPT_ENV).unwrap_or_else(|_| CKPT_DEFAULT.into());
        let st = SafeTensorsDir::open_dir(&dir).expect("second open");

        // First row, last row, both sides of two shard seams, and a scatter.
        let last = table.total_rows() - 1;
        let per = table.rows_per_shard();
        let mut ids: Vec<i64> = vec![
            0,
            1,
            (per - 1) as i64,
            per as i64,
            (per * 64 - 1) as i64,
            (per * 64) as i64,
            last as i64,
        ];
        ids.extend(random_rows(0xA11CE, 25, table.total_rows()));

        let width = table.row_width();
        let mut pool = NgramGather::new(Arc::clone(&table), 16).expect("pool");
        let mut got = vec![0.0f32; ids.len() * width];
        pool.gather(&ids, &mut got).expect("gather");

        let scale = table.weight_scale();
        let mut nonzero_rows = 0usize;
        for (slot, &row) in ids.iter().enumerate() {
            let bytes = row_bytes_via_file(&st, &table, row as u64);
            let want: Vec<f32> = bytes.iter().map(|&b| fp8_e4m3_to_f32(b) * scale).collect();
            let have = &got[slot * width..(slot + 1) * width];
            assert_eq!(have, &want[..], "row {row} (slot {slot})");
            if have.iter().any(|v| *v != 0.0) {
                nonzero_rows += 1;
            }
        }
        // A gather that silently returned zeros would match a zeroed oracle if
        // the oracle were broken the same way; the real table is not zeros.
        assert!(
            nonzero_rows >= ids.len() - 1,
            "only {nonzero_rows}/{} rows had any nonzero value",
            ids.len()
        );
        drop(pool);
    }

    /// End to end: real tokens -> real hash -> real table.
    #[test]
    fn real_checkpoint_gathers_a_hashed_token() {
        let Some(table) = open() else { return };
        let hash = NGramHash::new(NGramHashConfig::qwen4_exp()).expect("hash");
        table.check_against_hash(&hash).expect("geometry");

        let mut context = NGramContext::new(&hash);
        let tokens: Vec<i64> = vec![9707, 11, 1879, 0];
        let ids = hash.row_ids(&context, &tokens).expect("row ids");
        context.push(&tokens);
        assert_eq!(ids.len(), tokens.len() * hash.ngram_heads());

        let mut pool = NgramGather::new(Arc::clone(&table), 16).expect("pool");
        let mut out = vec![0.0f32; ids.len() * table.row_width()];
        pool.gather(&ids, &mut out).expect("gather");

        assert!(out.iter().all(|v| v.is_finite()), "NaN in the n-gram table");
        // One token's slice is `ple_embed_dim` wide and must not be all zeros —
        // that is what a wrong-but-in-range read or a zero-filled miss looks
        // like.
        let embed = hash.ngram_heads() * table.row_width();
        for (t, row) in out.chunks_exact(embed).enumerate() {
            assert_eq!(row.len(), 2560);
            assert!(
                row.iter().any(|v| *v != 0.0),
                "token {t} gathered an all-zero PLE embedding"
            );
        }
        drop(pool);
    }

    /// The regression the module exists to prevent, and the only honest way to
    /// see it.
    ///
    /// The row ids are reseeded from the wall clock on every run. That is not
    /// decoration: with fixed ids the first run faults and every run after it
    /// reads the OS page cache, so a ratio assertion would pass once and then
    /// measure nothing. Reseeding keeps each run on rows this box has never
    /// touched, which is also the regime a real decode lives in — hash-scattered
    /// ids over 320M rows rarely repeat.
    ///
    /// The ratio is only asserted when the serial baseline proves the reads
    /// really faulted. On a fully cached table there is no latency to hide and
    /// the pool is a small, expected loss; asserting there would be asserting
    /// noise.
    #[test]
    fn real_checkpoint_gather_scales_with_threads() {
        let Some(table) = open() else { return };
        let width = table.row_width();
        let heads = 16usize;
        let tokens = 12usize;

        // Wall clock x pid, so two runs — and two test binaries running at
        // once — never share rows.
        let nonce = splitmix64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
                ^ u64::from(std::process::id()) << 40,
        );

        let batches = |salt: u64| -> Vec<Vec<i64>> {
            (0..tokens)
                .map(|t| {
                    random_rows(
                        splitmix64(nonce ^ salt ^ t as u64),
                        heads,
                        table.total_rows(),
                    )
                })
                .collect()
        };

        // The serial baseline goes straight through the table, with no pool at
        // all, so it measures the reads and not a one-worker dispatch.
        let serial_batches = batches(0x5E_21A1);
        let mut out = vec![0.0f32; heads * width];
        let start = Instant::now();
        for ids in &serial_batches {
            table.gather_rows(ids, &mut out).expect("serial gather");
        }
        let serial_ms = start.elapsed().as_secs_f64() * 1e3 / tokens as f64;

        let parallel_batches = batches(0xBE_EF77);
        let mut pool = NgramGather::new(Arc::clone(&table), DEFAULT_WORKERS).expect("pool");
        let start = Instant::now();
        for ids in &parallel_batches {
            pool.gather(ids, &mut out).expect("parallel gather");
        }
        let parallel_ms = start.elapsed().as_secs_f64() * 1e3 / tokens as f64;
        let counts = pool.chunks_per_worker();
        drop(pool);

        eprintln!(
            "n-gram gather ({heads} rows/token, {tokens} tokens, fresh rows): serial {serial_ms:.3} ms/tok, {DEFAULT_WORKERS} workers {parallel_ms:.3} ms/tok ({:.1}x)",
            serial_ms / parallel_ms.max(f64::MIN_POSITIVE)
        );

        // Structural, and independent of how warm the cache was: every worker
        // must have been handed work. A gather that quietly went serial returns
        // identical numbers, so this is the check that still fails.
        assert!(
            counts.iter().all(|&c| c == tokens as u64),
            "each of {DEFAULT_WORKERS} workers should take one chunk per token, got {counts:?}"
        );

        // Measured cold: 2.897 ms/tok serial. Measured warm: 0.041 ms/tok.
        // The gate sits an order of magnitude clear of both, so it selects the
        // cold regime rather than splitting it.
        const COLD_MS: f64 = 0.4;
        if serial_ms > COLD_MS {
            // Measured cold speedup is 4.3x; 2x leaves room for a busy disk.
            assert!(
                parallel_ms * 2.0 < serial_ms,
                "{DEFAULT_WORKERS} workers gave {parallel_ms:.3} ms/tok against {serial_ms:.3} ms/tok serial — the gather is not fanning out"
            );
        } else {
            eprintln!(
                "note: serial gather was {serial_ms:.3} ms/tok, under the {COLD_MS} ms cold gate — these rows were already cached, so the ratio proves nothing and is not asserted"
            );
        }
    }
}
