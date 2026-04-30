# Longctx Phase 1 S4 harness — pending remote guidellm

## Goal

- Benchmark harness: make the ARLE and SGLang longctx-32k Phase 1 S4 runs
  reproducible with explicit workload selection, smoke validation, pinned
  SGLang provenance, and wins/headline artefacts.

## Hypothesis

- The harness changes do not alter the default ARLE GuideLLM workload when no
  workload is selected, while making `longctx-32k` explicit enough for remote
  Phase 1 validation and baseline capture.

## Command

Local validation:

```bash
bash -n scripts/bench_guidellm.sh
bash -n scripts/bench_sglang_longctx.sh
scripts/bench_guidellm.sh --workload longctx-32k --help
scripts/bench_sglang_longctx.sh --help
scripts/bench_sglang_longctx.sh demo --smoke --help
git diff --check
```

Remote smoke and canonical runs:

```bash
scripts/bench_guidellm.sh longctx-s4-smoke --workload longctx-32k --smoke
scripts/bench_sglang_longctx.sh longctx-s4-smoke --smoke

scripts/bench_guidellm.sh longctx-32k-phase1-s4 --workload longctx-32k
scripts/bench_sglang_longctx.sh longctx-32k-phase1-s4
```

## Environment

- **Backend:** CUDA / SGLang
- **Model:** Qwen/Qwen3-4B
- **Hardware:** pending remote NVIDIA L4
- **Commit:** pending; fill after commit lands
- **Feature set:** `CUDA_HOME=/usr/local/cuda cargo build --release`
- **Non-default flags / env vars:** `--workload longctx-32k`
- **Server launch:** ARLE FP8 KV with `--num-slots 16`,
  `--max-seq-len 131072`, `--mem-fraction-static 0.85`; SGLang pin
  `214c35b03184c354acf1f86f99746799e1c9b3a9`

## Results

- Status: `pending-remote`

| check | local result |
|---|---|
| `bash -n scripts/bench_guidellm.sh` | pass |
| `bash -n scripts/bench_sglang_longctx.sh` | pass |
| ARLE `--workload longctx-32k --help` | pass |
| SGLang `--smoke --help` | pass |
| `git diff --check` | pass |

## Problems

- This workspace has no running CUDA ARLE/SGLang service, so the 5s smoke run
  and canonical c=1/c=4 GuideLLM runs remain remote work.

## Learnings

- S4 needs explicit workload selection in the CLI, not only `WORKLOAD=...`,
  so a shell environment cannot silently rewrite a default canonical run.
- Reusing an existing SGLang server must not claim the pinned commit unless
  the caller explicitly provides `SGLANG_SERVER_COMMIT`.

## Delta vs baseline

- **Baseline:** no first-class S4 harness entry; previous pending S1/S2 record:
  `docs/experience/wins/2026-04-30-bench-guidellm-longctx-32k-phase1-s1-s2-pending-remote.md`

| metric | baseline | now | delta |
|---|---|---|---|
| ARLE longctx selector | env-only | `--workload longctx-32k` and env | explicit CLI added |
| 5s smoke path | manual/env only | `--smoke` | first-class |
| SGLang pin provenance with reused server | could be misleading | marked unverified unless `SGLANG_SERVER_COMMIT` matches | fixed |
| SGLang wins seed | absent | seeded after canonical verified run | added |

## Artefacts

- Raw: pending remote
- CSV: pending remote
- HTML: pending remote
- Service trace: pending remote

## Notes

- Code since baseline:
  - `scripts/bench_guidellm.sh` accepts `--workload longctx-32k` and seeds
    workload-resolved params into wins entries.
  - `scripts/bench_guidellm.sh` and `scripts/bench_sglang_longctx.sh` expose
    a 5s `--smoke` path.
  - `scripts/bench_sglang_longctx.sh` marks reused-server provenance as
    unverified unless `SGLANG_SERVER_COMMIT` matches the project pin.
