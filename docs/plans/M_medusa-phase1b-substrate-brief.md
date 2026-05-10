# M_medusa Phase 1.B substrate brief — codex pickup directive

> **Why now**: Per `0a0d221` audit + `1ccb41f` vLLM prior-art survey,
> Medusa substrate is ~350 LOC Rust + 0 new CUDA kernels. This brief
> specifies the concrete pickup so codex can start the moment user
> approves Option A (Medusa) per `9735b47` REFUTATION pivot.
>
> **Status (2026-05-10)**: PAUSED after user re-scoped pickup away
> from Qwen3/Qwen3.6 and toward Qwen3.5. Do not implement this brief as
> written. Qwen3.5 Step 0 audit found a recurrent-state rollback blocker:
> Medusa verification can roll back paged KV today, but not Qwen3.5
> linear-attention recurrent state. See
> `docs/research/2026-05-10-medusa-phase1b-qwen35-step0-audit.md` and
> `docs/experience/errors/2026-05-10-medusa-qwen35-recurrent-rollback-blocker.md`.

> **Historical note**: the remaining sections below describe the older
> Qwen3/Qwen3.6 pickup shape. Treat them as prior context, not an active
> Qwen3.5 implementation directive.

---

## §1 Acceptance criteria (license-or-kill at end)

| Metric | License | Soft win | Kill |
|---|---|---|---|
| tok/s vs no-spec at agent W3/W4 shape | ≥1.5× | ≥1.2× | <1.0× |
| α (per-head acceptance) | ≥0.55 | ≥0.45 | <0.30 |
| LOC scope | ≤500 Rust + 0 new CUDA kernel | ≤700 Rust | runaway scope |
| Wall-clock substrate | ≤2 days | ≤3 days | >5 days = redesign |
| Greedy correctness vs no-spec | 0.0% diff (greedy mode) | <0.5% diff | >1% diff |

Per `M_medusa-required-path.md` Phase 1.

---

## §2 Substrate scope (Rust LOC, mirrors vLLM v1)

### §2.1 New file: `infer/src/model/medusa.rs` (~250 LOC)

Per `1ccb41f` vLLM prior-art (3 files, ~310 LOC total Python →
~250 Rust translates well):

```rust
// Mirrors vLLM ResidualBlock + Medusa + MedusaConfig
pub struct MedusaConfig {
    pub hidden_size: usize,    // 4096 for Qwen3-4B, 2048 for Qwen3.6 router
    pub vocab_size: usize,     // 151936 for Qwen3-family
    pub num_heads: usize,      // 5 (vLLM default; paper used 4)
    pub num_hidden_layers: usize,  // 1 (vLLM default; ResidualBlock depth)
    pub max_paths: usize,      // 64 for tree-attn (defer to Phase 2)
    pub topk: usize,           // 10 (defer to Phase 2)
}

pub struct ResidualBlock {
    pub layers: Vec<Linear>,   // num_hidden_layers nn.Linear modules
    // SiLU activation (use existing ops::silu)
}

impl ResidualBlock {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            out = out + ops::silu(&layer.forward(&out));
        }
        out
    }
}

pub struct Medusa {
    pub config: MedusaConfig,
    pub blocks: Vec<ResidualBlock>,    // num_heads ResidualBlocks
    pub lm_heads: Vec<Linear>,         // num_heads LM heads (or shared)
}

impl Medusa {
    pub fn forward(&self, hidden_states: &Tensor) -> Vec<Tensor> {
        // Returns Vec<logits> — one per Medusa head
        self.blocks.iter().zip(&self.lm_heads).map(|(blk, head)| {
            head.forward(&blk.forward(hidden_states))
        }).collect()
    }

    pub fn propose(&self, target_hidden_states: &Tensor) -> Vec<u32> {
        // Top-1 only (vLLM v1 simplification)
        // Returns [num_heads] argmax per head
        self.forward(target_hidden_states).iter()
            .map(|logits| ops::argmax(logits).item::<u32>())
            .collect()
    }
}
```

### §2.2 Weight loading: `infer/src/model/medusa/weights.rs` (~80 LOC)

Per `infer/src/model/deepseek/weights.rs` pattern. Medusa heads
ship as separate safetensors (e.g. `medusa_head_0.safetensors` …
`medusa_head_4.safetensors`) OR bundled in single
`medusa_lm_heads.safetensors` per FasterDecoding/Medusa
convention.

```rust
pub fn load_medusa_weights(
    config: &MedusaConfig,
    weights_path: &Path,
    device: Device,
) -> Result<Medusa> {
    let mut blocks = Vec::with_capacity(config.num_heads);
    let mut lm_heads = Vec::with_capacity(config.num_heads);
    for head_idx in 0..config.num_heads {
        let block_weights = load_safetensors(
            weights_path.join(format!("medusa_head_{}_block.safetensors", head_idx))
        )?;
        let head_weights = load_safetensors(
            weights_path.join(format!("medusa_head_{}_lmhead.safetensors", head_idx))
        )?;
        blocks.push(ResidualBlock::from_state_dict(block_weights, device)?);
        lm_heads.push(Linear::from_state_dict(head_weights, device)?);
    }
    Ok(Medusa { config: config.clone(), blocks, lm_heads })
}
```

### §2.3 Integration: extend `infer/src/speculative.rs` (~50 LOC delta)

Replace `MockDraftModel` impl with `MedusaDraftModel`:

```rust
pub struct MedusaDraftModel {
    medusa: Arc<Medusa>,
    target_hidden_capture: Arc<Mutex<HiddenStateCapture>>,
}

impl DraftModel for MedusaDraftModel {
    fn draft_batch(&self, token_ids: &[u32], num_draft_tokens: usize) -> Result<TokenProposal> {
        // 1. Capture target hidden states for last position from prior step
        let hidden = self.target_hidden_capture.lock().unwrap().get_last()?;

        // 2. Propose num_draft_tokens from Medusa (typically num_heads = num_draft_tokens)
        debug_assert_eq!(num_draft_tokens, self.medusa.config.num_heads);
        let draft_tokens = self.medusa.propose(&hidden);

        // 3. Wrap into TokenProposal (existing struct, line 91 of speculative.rs)
        Ok(TokenProposal {
            tokens: draft_tokens,
            // distributions for verify_tokens — need full softmax not just argmax
            // For top-1 vLLM-style, distributions are one-hot; expand later for top-K
            distributions: ...,
        })
    }
}
```

### §2.4 Hidden-state capture: extend `infer/src/model/qwen3/forward.rs` (~30 LOC)

Medusa propose needs access to **target last-position hidden state**
post-final-layernorm (pre-LM-head). Currently the Qwen3 forward
discards this after LM head projection. Add an optional capture hook:

```rust
pub struct ForwardConfig {
    // ... existing fields ...
    pub capture_hidden: Option<Arc<Mutex<HiddenStateCapture>>>,
}

// In forward():
let hidden = self.norm.forward(&hidden_states);
if let Some(cap) = &cfg.capture_hidden {
    cap.lock().unwrap().store(hidden.last_token_view());
}
let logits = self.lm_head.forward(&hidden);
```

---

## §3 What this brief explicitly DEFERS to Phase 2

- **Tree-attention verify path**: vLLM v1 ignores; only needed if
  first-iter α < 1.5× license.
- **Top-K candidates per head**: vLLM v1 uses top-1 only.
- **Token map vocab truncation**: vLLM optimization, ~1.5× minor speedup.
- **Multi-batch propose**: first iter handles batch_size=1 only;
  expand once correctness verified.

---

## §4 Training pipeline (out-of-scope for this brief — separate PR)

Per `M_medusa-phase1a-dataset-directive.md`:
- Dataset: Alpaca (52k, ~2-3 days train) OR lmsys-chat-1m subset
- Wire-up: `arle data download --repo tatsu-lab/alpaca` →
  `arle train medusa --target Qwen3-4B --data ...`
- Storage: ~26M head params + optimizer state ≈ 100 MB

---

## §5 Test plan

| Test | Purpose | Pass criterion |
|---|---|---|
| `tests/test_medusa_load.rs` | Weight loading round-trip | All 5 heads load without shape mismatch |
| `tests/test_medusa_forward.rs` | Numerical match vs Python ref | Max diff < 1e-4 in BF16 |
| `tests/test_medusa_propose_smoke.rs` | Propose path returns 5 tokens | Length match + non-zero tokens |
| `tests/test_medusa_verify_e2e.rs` | End-to-end greedy correctness | 0.0% diff vs no-spec greedy |
| `bench/medusa_alpaca_smoke.sh` | α measurement | α ≥ 0.55 (license) at conc=1 |

---

## §6 Wall-clock budget

| Phase | Hours | Owner |
|---|---:|---|
| §2.1 medusa.rs scaffold | 2-3 | codex |
| §2.2 weight loading | 1-2 | codex |
| §2.3 speculative.rs integration | 1 | codex |
| §2.4 hidden-state capture hook | 1 | codex |
| §5 test suite (without training) | 2-3 | codex |
| Code review (claude codex review) | 0.5 | claude |
| **TOTAL substrate** | **8-11 hr** | — |
| Then: training (Alpaca, 5 heads) | 48-72 hr | GPU + codex |
| Then: license-or-kill bench | 1 hr | claude |

Total wall-clock to first verdict: **~3-4 days** (vs audit's 4-6 days
estimate; substrate is faster, training dominates).

---

## §7 Pickup gate

- [ ] User confirms target model (Qwen3-4B or Qwen3.6)
- [ ] User confirms dataset (Alpaca or lmsys-chat-1m subset)
- [ ] User confirms integration target (CUDA scheduler first)
- [ ] User approves ~3-4 day total wall-clock investment

When all checked, codex pickup begins with §2.1 medusa.rs scaffold.

---

## §8 Cross-references

- `9735b47` REFUTATION wins entry (strategic pivot trigger)
- `0a0d221` Task #28 readiness audit (assumes 500+ LOC; this brief refines down)
- `1ccb41f` vLLM Medusa prior-art survey (~310 LOC Python → ~350 Rust)
- `M_medusa-required-path.md` (Phase 1-4 plan, license thresholds)
- `M_medusa-phase1a-dataset-directive.md` (dataset selection)
- `infer/src/speculative.rs` (existing 721 LOC, MockDraftModel to replace)
- `infer/src/model/qwen3/forward.rs` (hidden-state capture hook target)
- `infer/src/model/deepseek/weights.rs` (pattern reference for weight loading)
- vLLM Medusa source: `vllm/v1/spec_decode/medusa.py`, `vllm/model_executor/models/medusa.py`
