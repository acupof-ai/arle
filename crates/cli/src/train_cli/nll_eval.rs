use anyhow::{Context, Result, anyhow, bail};
use autograd::TensorStore;

use crate::args::TrainPplArgs;
#[cfg(feature = "cuda")]
use {super::resolve_local_tokenizer_path, std::fs, train::tokenizer::ChatTokenizer};

/// `arle train ppl` — perplexity over a text corpus via teacher-forced logits,
/// to calibrate FP8 / quant checkpoint quality. CUDA-only (the forward path is
/// `LoadedInferenceEngine::forward_token_logits`, which the OPD lane also uses).
#[cfg(feature = "cuda")]
pub(super) fn run_ppl(args: TrainPplArgs) -> Result<()> {
    use infer_api::{EngineLoadConfig, LoadedInferenceEngine};

    let ctx = args.ctx;
    let model_path = args
        .model_path
        .to_str()
        .ok_or_else(|| anyhow!("model path is not valid UTF-8"))?;

    // Tokenize the corpus into one stream with the model's own tokenizer.
    let tokenizer_path = resolve_local_tokenizer_path(&args.model_path)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)
        .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?;
    let corpus = fs::read_to_string(&args.corpus)
        .with_context(|| format!("read corpus {}", args.corpus.display()))?;
    let tokens = tokenizer
        .encode(&corpus, false)
        .context("tokenize corpus")?;
    if tokens.len() < 2 {
        bail!(
            "corpus tokenizes to {} tokens; need at least 2 for a next-token NLL",
            tokens.len()
        );
    }

    let engine = LoadedInferenceEngine::load_with_config(
        model_path,
        /*cuda_graph=*/ true,
        EngineLoadConfig::single_sequence(ctx),
    )
    .with_context(|| format!("load engine from {model_path}"))?;

    // Non-overlapping windows of `ctx`; each contributes len-1 next-token NLLs.
    let mut sum_nll = 0.0f64;
    let mut count = 0usize;
    let mut windows = 0usize;
    for chunk in tokens
        .chunks(ctx)
        .take(args.max_windows.unwrap_or(usize::MAX))
    {
        if chunk.len() < 2 {
            break; // a trailing 1-token window has no next-token target
        }
        let positions: Vec<u32> = (0..chunk.len() as u32).collect();
        let raw = engine
            .forward_token_logits(chunk, &positions)
            .with_context(|| format!("forward window {windows}"))?;
        if raw.seq_len() != chunk.len() {
            bail!(
                "ppl window seq_len mismatch: logits={}, tokens={}",
                raw.seq_len(),
                chunk.len()
            );
        }
        let vocab = raw.vocab_size();
        let host = raw.to_host_f32()?;
        // chunks_exact yields chunk.len() rows; windows(2) yields len-1 pairs →
        // zip drops the final row (no next token), scoring each token once.
        for (row, pair) in host.chunks_exact(vocab).zip(chunk.windows(2)) {
            sum_nll += row_nll(row, pair[1] as usize)?;
            count += 1;
        }
        windows += 1;
    }

    if count == 0 {
        bail!("no scorable positions (corpus shorter than 2 tokens per window)");
    }
    let ppl = (sum_nll / count as f64).exp();

    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "ppl": ppl,
                "tokens": count,
                "windows": windows,
                "ctx": ctx,
                "model_path": model_path,
            })
        );
    } else {
        println!("ppl={ppl} tokens={count} windows={windows} ctx={ctx}");
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
pub(super) fn run_ppl(_args: TrainPplArgs) -> Result<()> {
    bail!("arle train ppl requires the CUDA backend (forward_token_logits is CUDA-only)")
}

/// Numerically stable next-token NLL: `-log_softmax(row)[target]`, subtracting
/// the row max before exp.
fn row_nll(row: &[f32], target: usize) -> Result<f64> {
    let target_logit = *row
        .get(target)
        .ok_or_else(|| anyhow!("target token {target} out of vocab {}", row.len()))?
        as f64;
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum_exp: f64 = row.iter().map(|&l| (l as f64 - max).exp()).sum();
    // NLL = -(logit - max - ln(sum_exp)) = ln(sum_exp) + max - logit
    Ok(sum_exp.ln() + max - target_logit)
}

/// Forward-only held-out NLL: mean next-token cross-entropy of predicting
/// `eval_ids[t+1]` from `eval_ids[..=t]` over a FIXED reference sequence.
/// Tape-off (scratch tape dropped after the forward); non-circular — the
/// reference text is fixed, independent of the moving EMA teacher — so it is a
/// valid SOPD no-regression signal (KL-vs-EMA would be circular).
pub(super) fn heldout_nll(
    student: &train::qwen35::Qwen35Model,
    eval_ids: &[u32],
    vocab: usize,
    store: &mut TensorStore,
) -> Result<f32> {
    use autograd::Tape;
    if eval_ids.len() < 2 {
        bail!("--eval-ids needs at least 2 token ids for a next-token NLL");
    }
    let input: Vec<usize> = eval_ids[..eval_ids.len() - 1]
        .iter()
        .map(|&id| id as usize)
        .collect();
    let seq_len = input.len();
    let mut eval_tape = Tape::new();
    let logits_id = student
        .forward_tokens(&input, store, &mut eval_tape)
        .context("held-out NLL forward")?;
    let host = store.to_host(logits_id)?;
    let expected = seq_len
        .checked_mul(vocab)
        .ok_or_else(|| anyhow!("held-out NLL logits shape overflow"))?;
    if host.len() != expected {
        bail!(
            "held-out NLL logits len {} != seq_len*vocab {} (seq_len={seq_len}, vocab={vocab})",
            host.len(),
            expected
        );
    }
    let mut nll_sum = 0.0_f64;
    for t in 0..seq_len {
        let row = &host[t * vocab..(t + 1) * vocab];
        nll_sum += row_nll(row, eval_ids[t + 1] as usize)?;
    }
    Ok((nll_sum / seq_len as f64) as f32)
}
