# Evaluation

ARLE evals are runtime-owned: the candidate answer or patch must be produced by
the ARLE inference engine, and the grader may only score that artifact. The
grader must not repair, rewrite, or judge-improve the model output.

For the OPD/QAT **capability curve** (run the suite below at successive
checkpoints, Δ-vs-baseline with multi-seed CI) and the one serving blocker that
gates non-baseline curve points, see
[opd-capability-curve.md](opd-capability-curve.md). The driver
(`scripts/opd_capability_curve.py`) reuses the per-task evaluators here by
subprocess — it adds no new scorer.

## Result Contract

Every eval run must preserve three artifacts:

- raw engine output
- extracted candidate artifact, if extraction is mechanical
- deterministic scorer output

For capability claims, a result is not SOLID until the artifact and scorer are
both reproducible from the run directory. LLM-as-judge can be useful for
diagnostics, but it is not an authority for MMLU or SWE-bench Pro verdicts.

## MMLU

Use [../scripts/arle_capability_eval.py](../scripts/arle_capability_eval.py).
It talks to `arle serve` through the OpenAI-compatible surface, or to a HF
transformers baseline with `--backend hf`.

Smoke:

```bash
python scripts/arle_capability_eval.py \
  --backend arle \
  --base-url http://localhost:8123 \
  --model-id <served-model-id> \
  --tasks mmlu \
  --n-samples 50 \
  --seed 0 \
  --output bench-output/mmlu-smoke
```

Claim-grade run:

```bash
python scripts/arle_capability_eval.py \
  --backend arle \
  --base-url http://localhost:8123 \
  --model-id <served-model-id> \
  --tasks mmlu \
  --n-samples 500 \
  --seeds 0,1,2,3,4 \
  --output bench-output/mmlu-claim
```

Then compare paired runs with:

```bash
python scripts/analyze_multi_seed.py \
  bench-output/mmlu-claim \
  --paired-vs bench-output/mmlu-baseline \
  --task mmlu
```

`--concurrency N` issues N requests in flight (default 1). The serve batches
continuously, so the default leaves every slot but one idle — a 500-problem
GSM8K seed took 80 minutes at 1-way on a 2-slot engine. **A paired comparison
must use the same `--concurrency` on both arms**: batch composition perturbs MoE
numerics, so a concurrency change is a config change, not a free speedup.

Small MMLU deltas are noisy. The 2026-05-28 retraction stands: a capability
claim under 5 percentage points at small `n` needs multi-seed evidence and a
paired/McNemar or Wilson confidence interval before it can be written as a win.

## SWE-bench Pro

Use [../scripts/arle_swe_pro_eval.py](../scripts/arle_swe_pro_eval.py).

SWE-bench Pro is a patch-generation eval. ARLE sees only the problem statement
and allowed context. The generation prompt explicitly excludes the gold patch,
test patch, `fail_to_pass`, and `pass_to_pass`. The generated `patches.json`
is then scored by the official SWE-bench Pro evaluator.

Prepare a small selection:

```bash
python scripts/arle_swe_pro_eval.py prepare \
  --output bench-output/swepro-smoke \
  --limit 3 \
  --seed 0
```

Generate patches through ARLE:

```bash
python scripts/arle_swe_pro_eval.py generate \
  --output bench-output/swepro-smoke \
  --base-url http://localhost:8123 \
  --model-id <served-model-id> \
  --limit 3 \
  --seed 0
```

Evaluate with the official runner:

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os /tmp/SWE-bench_Pro-os
python scripts/arle_swe_pro_eval.py evaluate \
  --output bench-output/swepro-smoke \
  --eval-repo /tmp/SWE-bench_Pro-os \
  --use-local-docker \
  --docker-platform linux/amd64
```

The official evaluator applies each patch inside the instance image, runs the
instance `run_script.sh`, parses logs through `parser.py`, and marks an instance
resolved only when every `FAIL_TO_PASS` and `PASS_TO_PASS` test is reported as
passed.

## Container Backends

The formal SWE-bench Pro score should use the official Docker/Modal evaluator.
The lighter options are execution backends, not new scorers:

- Modal: lightest local footprint, remote sandbox, official path.
- Docker Desktop, OrbStack, Colima, Podman, or nerdctl: acceptable if they expose
  Docker-compatible behavior and run the official images.
- Host `uv`/venv/micromamba: useful only for ad-hoc smoke. It is not an official
  SWE-bench Pro score because repository dependencies and system packages are no
  longer isolated.
- Nix: possible research direction, but not a first-version replacement for the
  official 41-repository image set.

On Apple Silicon, most official images are Linux amd64. Use
`--docker-platform linux/amd64` when the local OCI backend does not infer that
automatically.
