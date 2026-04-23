# Performance and Correctness Gates

This document defines the minimum validation expectations for changes in
`ARLE`.

The project is both performance-sensitive and correctness-sensitive. A change is
not good enough just because it compiles or appears faster once.

---

## 1. Validation Layers

Use the lightest meaningful layer first, then broaden when risk increases.

1. compile / lint / format
2. unit or contract tests
3. integration or e2e tests
4. benchmark or profiling evidence

---

## 2. Minimum Gates by Change Type

### Docs or non-behavioral cleanup

Require:

- internal consistency check

### CPU-side logic or pure Rust helpers

Require:

- targeted tests
- CPU-only test path when relevant

### API, protocol, or CLI behavior

Require:

- targeted tests
- API compatibility tests where relevant
- docs update when behavior changes

### CUDA, Metal, scheduler runtime, kernels, quantization

Require:

- feature-specific compile checks
- targeted correctness checks
- benchmark evidence for performance claims
- profiling evidence for material runtime or kernel claims

---

## 3. Area-Specific Expectations

### HTTP / protocol changes

Run at least:

- targeted Rust tests
- `tests/test_openai_api.py` when affected
- smoke validation if unit tests are not enough

### Scheduler changes

Run at least:

- scheduler tests
- relevant CPU-only tests
- targeted regression coverage when ordering, fairness, waiting limits, or
  latency policy changes

### Model / kernel / quantization changes

Run at least:

- feature-specific compile checks
- targeted correctness checks
- e2e or baseline comparison when available
- before/after benchmark for claimed wins

### CLI agent changes

Run at least:

- parsing or session tests where applicable
- CLI tests if command behavior changed
- live-model validation when turn loop or tool behavior changed and a model is
  available

---

## 4. Performance Evidence Rule

If a change claims a meaningful performance improvement, include:

- hardware
- model
- command
- before result
- after result
- caveats or variance notes

Preferred evidence sources include existing benchmark scripts and binaries in
this repository.

---

## 5. Correctness Evidence Rule

For correctness-sensitive changes, state the reference used:

- golden baseline
- CPU reference
- HuggingFace parity expectation
- API compatibility expectation
- previous benchmark snapshot

If a baseline changes, explain why.

---

## 6. Merge Blocking Conditions

Do not merge if:

- relevant tests are failing
- a behaviorally significant change has no verification story
- a performance claim has no measurable evidence
- a stable surface changed without docs or compatibility coverage
- baselines changed without explanation

---

## 7. Maintainer Checklist

Before merge, answer:

1. What changed?
2. What stability level does it belong to?
3. What is the minimum meaningful validation?
4. Does this need benchmark evidence?
5. Does this need docs or changelog updates?

Related docs:

- [stability-policy.md](stability-policy.md)
- [compatibility.md](compatibility.md)
- [release-checklist.md](release-checklist.md)
