# P2.B.4 Sparse Draft SpecPath Integration — pending bench, CUDA, 2026-05-01

## Goal

- Lock the end-to-end sparse self-spec scheduler path: sparse-KV draft proposes K tokens from a pruned page view, full-KV verifier remains authoritative, and verifier metrics reflect real scheduler outcomes.

## Hypothesis

- Sparse-KV should not change greedy output because only draft attention is approximate. Full-KV verifier output must stay bit-identical with normal decode.

## Command

```bash
scripts/bench_guidellm.sh p2b4-sparse-draft-spec-path \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-4B \
  --processor models/Qwen3-4B
```

Invoked via: pending; P2.B.6 owns the throughput bench.

## Environment

- **Backend:** cuda
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote CUDA server
- **Commit:** pending commit for P2.B.4
- **Feature set:** `cargo build --release`
- **Non-default flags / env vars:** `--spec-enabled --spec-draft-model self-spec --spec-draft-k 5` plus sparse-KV config
- **Server launch:** pending

## Results

Pending throughput bench. This entry exists to tie the P2.B.4 correctness gate to the later P2.B.6 longctx-32k c=4 measurement.

## Problems

- No local CUDA model bench was run in this tranche.
- `cargo clippy -D warnings` is still blocked by pre-existing lint debt outside the sparse-KV diff.

## Learnings

- Sparse self-spec is valid only when the scheduler can prove draft-only sparsity and full-KV verifier isolation at the same call site.

## Delta vs baseline

- **Baseline:** `docs/experience/wins/2026-05-01-phase1-close-evictable.md`

| metric | baseline | now | delta |
|---|---:|---:|---:|
| longctx-32k c=4 effective tok/s | 26.169 | pending | pending |
| sparse verifier bit-identity | n/a | guarded by `spec_decode_correctness` | pending bench |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace: pending

## Notes

- What changed in code since P2.B.3: `spec_decode_correctness` now exercises sparse self-spec K=5 with a small recent/top-k page budget so the draft view is actually pruned, while metrics assertions confirm the full-KV verifier path ran.
- Follow-ups: P2.B.6 must run the full longctx-32k c=4 bench before claiming lift.
