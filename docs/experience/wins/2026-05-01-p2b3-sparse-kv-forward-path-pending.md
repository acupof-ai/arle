# P2.B.3 Sparse-KV Forward Path — pending remote bench, CUDA, 2026-05-01

## Goal

- Verify that MagicDec-style sparse-KV self-spec draft decode can reduce draft attention work without changing full-KV verifier correctness.

## Hypothesis

- Sparse-KV should only affect the draft pass. The verifier remains full-KV, so greedy output correctness should remain bit-identical while P2.B.4/P2.B.6 measure throughput lift.

## Command

```bash
scripts/bench_guidellm.sh p2b3-sparse-kv-forward \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Invoked via: pending-remote.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote CUDA server / H20 or L4 bench slot
- **Commit:** pending commit for P2.B.3
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** `--spec-enabled --spec-draft-mode self-spec --spec-draft-k 5 --spec-sparse-kv-enabled`
- **Server launch:** pending remote bench run

## Canonical params

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh p2b3-sparse-kv-forward`

## Results

Pending remote bench. This entry is opened with the implementation because the diff touches runtime scheduler/model hot paths; P2.B.6 owns the full longctx-32k c=4 throughput run.

## Problems

- Local verification covered Rust type checks and unit tests only. Full CUDA server throughput measurement is pending.

## Learnings

- Sparse-KV draft must stay isolated from verifier execution. Any measured speedup is invalid unless full-KV verifier remains bit-identical.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-05-01-phase1-close-evictable.md`

| metric | baseline | now | Δ% |
|---|---:|---:|---:|
| longctx-32k c=4 effective tok/s | 26.169 | pending | pending |
| verifier correctness | bit-ident target-only | pending | pending |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in the code since baseline: P2.B.3 sparse-KV draft view reaches Qwen3 decode metadata and FlashInfer decode planning.
- Follow-ups: run P2.B.6 longctx-32k c=4 bench after sparse forward integration is committed.
