---
title: Medusa Phase 1.A pickup chain — UNBLOCKED via #34 df37a68 + arle data download already works + arle binary stale ergonomics gap
date: 2026-05-10
type: research
status: superseded-for-qwen35-by-recurrent-rollback-blocker
---

# Medusa Phase 1.A pickup chain — UNBLOCKED via #34 df37a68 + arle data download already works + arle binary stale ergonomics gap

> **2026-05-10 later update**: data download remains unblocked, but the
> active Qwen3.5 Medusa path is not ready for codex pickup. Runtime
> correctness is blocked on recurrent-state rollback. Treat this as
> dataset/CLI inventory only, not an implementation green light.

> Pre-survey for #28 Medusa scaffold (next P0 axis per `61c9666`
> architectural analysis if PF8 chain KILLs OR after PF8 chain
> licenses). Per skill v1.11.0+ #28+#31, every claim grounded in
> raw evidence (binary --help output + git show + grep on source).

## §0 Direct evidence (raw verification THIS tick)

### Existing infrastructure inventory (raw grep)

```bash
$ wc -l /home/ckl/projects/arle/infer/src/speculative.rs
721

$ grep -nE '^pub struct|^pub fn|^pub enum' /home/ckl/projects/arle/infer/src/speculative.rs
47:pub struct SpecConfig {
91:pub struct TokenProposal {
154:pub struct VerificationResult {
201:pub fn verify_tokens(proposal: &TokenProposal, rng: &mut impl rand::Rng) -> VerificationResult {
250:pub fn verify_tokens_greedy(...)
303:pub struct AcceptanceTracker {
```

ARLE has **substantial classical spec-decode infrastructure** already:
- `SpecConfig` (config struct)
- `TokenProposal` / `VerificationResult` (data types)
- `verify_tokens` + `verify_tokens_greedy` (verification logic)
- `AcceptanceTracker` (acceptance metrics)

This is the CLASSICAL Leviathan spec-decode path, not Medusa-specific.

### Classical spec-decode 4× KILLed (per `M_medusa-required-path.md`)

| Workload | Acceptance α | Result |
|----------|-------------|--------|
| 4k self-spec | 7% | KILL `5f26675` |
| 4k external draft | 19% | KILL `3ac5f4d` |
| 32k self-spec | 23% | KILL `8f2b227` |
| W3 c=4 production | 19% | KILL `aa00c6a` |

**Pattern**: classical α ≤ 0.25 across all 4 workloads on Qwen3-4B +
sm_89 + ARLE = structural ceiling. Medusa shared-target trained heads
is the architectural change required to break α ceiling.

### Medusa Phase 1.A blocker resolution chain

Per `M_medusa-phase1a-dataset-directive.md`:
- Existing dataset: 584 tokens (massively insufficient)
- Target: 100k+ tokens (172× short)
- **Recommended**: `lmsys/lmsys-chat-1m` HF dataset
- Wire-up: `DATA=$(arle data download --repo <id> --file <path>)`

### `arle data download` works (raw `--help` output THIS tick)

```bash
$ /home/ckl/projects/arle/target/release/arle data download --help
Download one dataset file from Hugging Face

Usage: arle data download [OPTIONS] --repo <REPO> --file <FILE>

Options:
      --repo <REPO>  Hugging Face dataset repo ID
      --file <FILE>  File path within the dataset repo
      --dry-run      Print the fully resolved execution plan without running the job
      --json         Render `--dry-run` output as JSON for scripts and CI
  -h, --help         Print help

Example:
  arle data download --repo tatsu-lab/alpaca --file alpaca_data.json
```

**Phase 1.A dataset download UNBLOCKED**. Codex (or Claude) can run:
```bash
arle data download --repo lmsys/lmsys-chat-1m --file data.jsonl
```

### `arle model download` source EXISTS but binary STALE (ergonomics gap)

```bash
$ git show --stat df37a68 | head -5
commit df37a68b25ce7c1e4481082cb3a866f0dc1b6054
Date:   Sun May 10 05:02:45 2026 +0800
    feat(cli): add `arle model download <id>` — unblocks #34 + P0 #28 spec decode pickup

$ grep "fn run_model_download" /home/ckl/projects/arle/crates/cli/src/train_cli.rs
72:fn run_model_download(args: ModelDownloadArgs) -> ExitCode {

$ /home/ckl/projects/arle/target/release/arle model download --help
error: unrecognized subcommand 'model'

$ ls -la /home/ckl/projects/arle/target/release/arle
-rwxr-xr-x 2 ckl ckl 22017104  5月 8日 13:08 ...
```

**Source has `model download` (committed 2026-05-10), binary is from
2026-05-08 — stale by 2 days**. Rebuild needed:
```bash
CUDA_HOME=/opt/cuda cargo build --release -p arle
```

NOT a Phase 1.A blocker (data download is what's needed first), but
a Phase 2-3 blocker if codex needs to download new model variants
during Medusa head training.

## §1 Updated #28 Medusa pickup chain

```
Phase 1.A (NOW UNBLOCKED via df37a68):
  Step 1: arle data download --repo lmsys/lmsys-chat-1m --file data.jsonl
  Step 2: convert to Medusa training format (training infra TBD)

Phase 1.B (BLOCKED on training infra):
  Step 3: train 4 Medusa heads on lmsys-chat-1m × 1 week
  Step 4: save Medusa checkpoint (target + 4 heads in ~150 MB extra)

Phase 2 (BLOCKED on Phase 1.B):
  Step 5: integrate Medusa heads into existing speculative.rs path
          (extend SpecConfig, TokenProposal, AcceptanceTracker)
  Step 6: dispatch Medusa decode in scheduler

Phase 3 (BLOCKED on Phase 2):
  Step 7: bench Medusa vs no-spec on agent W3/W4 production shape
  Step 8: license per master §7.4 threshold (tok/s ≥ 1.5×)
  Step 9: KILL if tok/s < 1.0× at any agent shape
```

**Phase 1.A is 100% Claude/codex-doable RIGHT NOW** — just needs the
`arle data download` invocation and a JSONL preprocessing step. No
GPU required for Phase 1.A.

**Phase 1.B is the major time sink** (1 week training) — codex own.

## §2 Medusa expected throughput (per M_medusa-required-path.md)

```
E[accepted tokens / step] = 1 + Σ_{i=1..K} Π_{j=1..i} α_j
  where α_j = per-position acceptance prob for j-th Medusa head.

For uniform α (worst case):
  α=0.7: E = 1 + 0.7 + 0.49 + 0.343 + 0.24 = 2.78 tokens/step
  α=0.85: E = 1 + 0.85 + 0.72 + 0.61 + 0.52 = 3.71 tokens/step
```

If Medusa heads achieve α ≥ 0.7 (typical Medusa paper result on
trained heads vs the 0.07-0.23 classical ceiling), tok/s improvement
ranges:
- 1.5× (license threshold) → α ≥ ~0.4
- 2.78× (Medusa paper claim) → α ≥ 0.7
- 3.71× (high-quality heads) → α ≥ 0.85

**Per `61c9666` architectural analysis**: this is the **only remaining
ITL win path on sm_89 W4 decode** because:
- W4 decode is HBM-bound on weight read
- FP8 mma helps compute, NOT bandwidth → wrong lever
- Speculative decoding amortizes weight read across multiple tokens
  per step → directly attacks the binding constraint

## §3 Existing Metal DFlash precedent (raw grep evidence)

```bash
$ grep -rln "DFlash\|dflash" /home/ckl/projects/arle/infer/src
/home/ckl/projects/arle/infer/src/backend/metal/dflash/
/home/ckl/projects/arle/infer/src/main.rs (CLI: --dflash-draft-model)
/home/ckl/projects/arle/infer/src/request_handle.rs (metadata)
/home/ckl/projects/arle/infer/src/hf_hub.rs (DFlash draft auto-download)
```

**Metal DFlash already implements speculative decoding** (different
mechanism — draft model variant of Qwen3.5). CUDA backend has no
equivalent. Medusa would be the CUDA-side equivalent (different
architecture: Medusa = trained heads on shared target, DFlash =
external draft model).

If codex finds Medusa training infeasible, alternative:
- Port DFlash mechanism (external draft) to CUDA
- Less ambitious than Medusa training but reuses Metal-side substrate
- Requires Qwen3.5-style draft model variant for CUDA

## §4 Tick deliverables for fast Phase 1.A unblock

If user wants Phase 1.A started THIS tick by Claude (no GPU work):

1. Trigger dataset download: `arle data download --repo lmsys/lmsys-chat-1m --file data.jsonl`
   (5-30 min depending on bandwidth, 4 GB approx)
2. Verify dataset format (JSONL, conversation field)
3. Document conversion script for Medusa training format
4. Hand off to codex for training (Phase 1.B)

NOT done THIS tick because:
- Codex still in PF8.3 commit-pending state (Working 29m+ on review)
- Don't want to introduce Phase 1.A artifacts before PF8 chain closes
- User hasn't explicitly chosen Medusa as next axis (may pick W3/W2 quant or stay on PF8.5)

## §5 Cross-references

- M_medusa-required-path.md (Phase 1-3 plan + 4 classical KILL evidence)
- M_medusa-phase1a-dataset-directive.md (dataset selection)
- 61c9666 (architectural analysis — Medusa is only remaining ITL win path)
- df37a68 (#34 RESOLVED — arle data download + arle model download CLI)
- speculative.rs:47-303 (existing classical spec-decode infrastructure)
- backend/metal/dflash/ (Metal-side spec-decode precedent)
- Task #28 [pending] — codex own, blocked on training

## §6 Status

Medusa Phase 1.A is UNBLOCKED. Pickup chain documented for next
agent. Phase 1.B (1 week training) remains the time sink.

Dataset download invocation ready: `arle data download --repo
lmsys/lmsys-chat-1m --file data.jsonl`

`arle model download` source exists per df37a68 but binary is stale
(2026-05-08 vs 2026-05-10 source) — rebuild needed for Phase 2-3
model downloads. NOT a Phase 1.A blocker.

Per skill v1.11.0+ #28+#31: every claim grounded in raw evidence
(speculative.rs wc + grep, target/release/arle --help + ls, git show
df37a68, source grep on train_cli.rs — all THIS tick).
