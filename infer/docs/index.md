# docs index

| Path | TL;DR |
| --- | --- |
| `projects/art-grade-architecture-for-long-agent-infer.md` | Atomic Rust crate decomposition plan and workspace organization for long-sequence, high-throughput, train-infer-consistent engine evolution |
| `projects/nonstandard-attention-milestone.md` | Milestone direction: infer focuses on non-standard attention models, with emphasis on model-family readiness, service experience, framework debt repayment, and disciplined evaluation |
| `projects/qwen35-4b-accuracy.md` | Qwen3.5-4B HF parity work: major decode-state bugs are fixed, `conv1d` now matches HF's bf16 pre-`SiLU` rounding, exact HF matches improved to 11/13, and only two small-logit-drift cases remain |
| `projects/qwen35-4b-optimization.md` | Hybrid 24 linear + 8 full attn. At parity with vLLM: TTFT 225ms, TPOT 11.81ms (+1%). Post-accuracy-fix GDR decode kernel restore (#9) |
| `projects/model-forward-trait.md` | ModelForward trait extraction: weights/state separation, shared generation loop, designed for bs > 1 |
| `projects/runtime-complexity-paydown.md` | Project to reduce model-specific runtime fragmentation; focus shifting to architecture-level abstraction (ModelForward trait) |
| `resources/accuracy-parity-playbook.md` | Accuracy debugging playbook: truth-source rules, first-diff workflow, bf16 rounding traps, and verified Qwen3.5 parity commands |
| `resources/developer-onboarding.md` | New-developer onboarding — toolchain, unified venv, build, tests, benchmark smoke test |
| `resources/profiling-guide.md` | GPU profiling playbook: nsys pitfalls, diagnostic paths, measured kernel comparisons |
| `resources/bench-vs-vllm.md` | infer vs vLLM comparative benchmarking: method, workflow, typical configs, gotchas |
| `resources/model-optimization-pipeline.md` | Per-model optimization methodology: 2 standard profiles, vLLM baseline, e2e dashboard + append-only optimization log |
