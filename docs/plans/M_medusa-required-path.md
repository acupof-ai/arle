# M_medusa — Medusa multi-head spec-decode REQUIRED path (post-classical-DEAD)

> Master §7.4 P1.1 promoted from "preferred" to **REQUIRED** by 3 classical
> KILLs: `5f26675` (4k self α=7%), `3ac5f4d` (4k ext α=19%),
> `8f2b227` (32k self α=23%). Pattern: classical Leviathan α ≤ 0.25 across
> all tested workloads on Qwen3-4B + sm_89 + ARLE current = structural
> ceiling. Medusa shared-target heads is the architectural change required
> to break α ceiling.
>
> Codex own (training + substrate). Plan ready for pickup.

## Phase 1 — Target

| Field | Value |
|---|---|
| Metric | tok/s on agent W3 / W4 production shape (master §2.1) |
| Baseline | W4A16 Marlin no-spec at agent shape — TBD (gated on `a672b08` admission fix) |
| **License** | tok/s ≥ 1.5× vs no-spec at W3/W4 (master §7.4 threshold) |
| Soft win | tok/s ≥ 1.2× — proceed but flag for K-tuning |
| Kill | tok/s < 1.0× at any agent shape — Medusa training failed |
| Wall-clock budget | 1 week training + 1 day integration + 1 day bench |

## Phase 2 — Hardware constraints

sm_89 RTX 4070 Ti SUPER, 16 GB VRAM. Same as production W4A16 setup.

Medusa-specific:
- Training: needs target model frozen + 4 Medusa heads trained (~8-16M params each)
- Memory at training: target 4B BF16 (8 GB) + 4 heads (~32 MB) + activations
  + optimizer state. Tight on 16 GB; may need gradient checkpointing or
  smaller global batch.
- Inference: target + 4 Medusa heads coexist in VRAM. ~150 MB extra.

## Phase 3 — Binding constraint (formula-grounded)

Medusa speedup formula (per Cai et al. 2024):

```
S = K * α / (1 + K * α - α)   # standard Leviathan
For Medusa K=4 heads with α typically 0.7-0.85 (paper claims):
  α=0.7: S = 4*0.7/(1+4*0.7-0.7) = 2.8/3.1 = 0.90× (LESS than 1×!)

Wait that's wrong... Let me redo:
  α_eff per Medusa: typically tree-attention style where multiple
  candidates verified per step.

Actual Medusa paper (with tree attention on top-K candidates):
  Average tokens accepted per step = 2.0-3.0 (vs 1 for no-spec)
  Throughput = 2.0-3.0× (Medusa head + tree-attn vs 1× no-spec)
```

Refer to Medusa paper §4.1 for exact formula. For Qwen3-4B + 4 heads,
literature predicts 2.0-2.5× tok/s at K=4 with shared-target alignment.

## Phase 4 — Formula prediction (rough)

For Qwen3-4B production W3/W4:
- Baseline tok/s: TBD (need W3 admission fix, est ~150-200 tok/s for c=16 W3 light)
- Medusa K=4: ~2.0-2.5× → 300-500 tok/s
- Medusa K=8 (more heads, deeper tree): potentially 3.0× → 450-600 tok/s

## Phase 5 — Implementation outline (codex own)

### 5.1 Training data (~1 week)

- Source: agent W3/W4 traces from `scripts/data/agent_trace_default.jsonl`
  + master §2.1 representative workload (system prompt + tool calls)
- Augmentation: Qwen3-4B as teacher model generating 100k+ token sequences
- Format: per-token logits + last-N hidden states (input for Medusa heads)

### 5.2 Medusa head architecture (~200 LOC PyTorch)

- 4 heads on top of Qwen3-4B last hidden state
- Each head: Linear(2560 → 2560) + GELU + Linear(2560 → 151936) — predicts
  position +1, +2, +3, +4
- Initialize from base lm_head; fine-tune on training data
- Loss: cross-entropy at each position

### 5.3 ARLE integration (~300 LOC)

- `infer/src/speculative/medusa.rs` — new module:
  - Load Medusa head weights alongside target Qwen3-4B
  - During decode: run target → 4 Medusa head logits in parallel
  - Tree-attention over top-K candidates per head (per Medusa paper)
  - Verify with target run on accepted prefix, accept ≤ K+1 tokens per step
- Sampler integration: replace `--spec-draft-model self/external` with
  `--spec-draft-model medusa:<path-to-heads>`
- TileLang verify kernel reuse from existing `infer/src/speculative.rs`

### 5.4 Tests (~50 LOC)

- `infer/tests/medusa_consistency.rs` — output token-equivalent to no-spec
  (Medusa is verified spec-decode = bit-exact greedy when properly impl)
- Acceptance rate counter in `/v1/stats` (already plumbed)

## Phase 6 — Combinational A/B

Post-license, combine with W4A16 + xgrammar:

| Stack | Expected tok/s |
|---|---|
| BF16 baseline | 150 (4k W3) |
| W4A16 Marlin | 200 (1.33×) |
| W4A16 + Medusa K=4 | 400-500 (2.5-3.3×) |
| **W4A16 + Medusa + xgrammar** | 380-470 (slight grammar overhead) |

## Phase 7 — Tradeoffs

| Axis | Status | Note |
|---|---|---|
| LOC | ⚠ ~500 substrate + training data prep | codex own |
| HW specificity | ✅ none | Medusa is general |
| Memory | ⚠ +150 MB head weights + tree-attn buffers | acceptable on 16 GB |
| **Training risk** | ❌ ~1 week + data quality | the main cost |
| Numerical correctness | ✅ verified spec — bit-exact greedy | matches no-spec |
| Generality | ⚠ workload-trained | retrain per major model update |
| Acceptance ceiling | predicted 0.7-0.85 | breaks classical α ≤ 0.25 |

## Phase 8 — License decision

| Result | Action |
|---|---|
| tok/s ≥ 2× vs no-spec | LAND HARD — production default for agent path |
| tok/s 1.5-2× | LAND incremental + tune K |
| tok/s 1.2-1.5× | hold for retraining or larger heads |
| tok/s < 1.0× | KILL Medusa axis on Qwen3-4B + sm_89 |

## Pre-execution gates

1. **W3 admission fix** (`a672b08`) — production-shape baseline first
2. **Training environment** — Qwen3-4B target + heads fit on 16 GB GPU
3. **bench_agent_trace.py harness** — production-shape measurement

## Cross-references

- Master §7.4 P1.1 update: [`docs/projects/2026-05-07-arle-master-strategy.md`](../projects/2026-05-07-arle-master-strategy.md) (`5acbe94`)
- 3 classical KILL evidence:
  - `5f26675` self-spec K=5 4k random
  - `3ac5f4d` ext-draft Qwen3-0.6B K=5 4k random
  - `8f2b227` self-spec K=5 32k random
- W3 admission blocker: `a672b08`
- Medusa paper: <https://arxiv.org/abs/2401.10774>
- ARLE spec substrate: `infer/src/speculative.rs` (721 LOC)
- Skill v1.3.0: `.claude/skills/kernel-optimization/SKILL.md` (`d09480b`)

## Rule

Medusa is **mandatory** for spec-decode axis on Qwen3-4B + ARLE current.
Three classical KILL evidences proved α ≤ 0.25 ceiling is structural.
The 1-week training cost is now the cheapest path; classical alternatives
all benched dead.

If user/codex wants to defer Medusa training cost, the alternative is
"spec-decode axis OFF for production" — which loses the master §6.1
5-cap moat capability 4 entirely. Master strategy should reflect this
tradeoff explicitly when prioritizing.
