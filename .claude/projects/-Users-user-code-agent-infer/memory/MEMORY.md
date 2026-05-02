# Memory Index

- [Commit directly to main is OK](feedback_commit_to_main.md) — no feature branches needed, commit straight to main
- [Done means commit and push](feedback_done_means_commit_and_push.md) — do not call work complete until the relevant slice is committed and pushed to `origin/main`
- [Edit the named target file](feedback_edit_the_named_target_file.md) — when a request relates two files, patch the file the user identified as the target, not the other endpoint
- [Project brand is ARLE](feedback_project_brand_is_arle.md) — user-facing CLI/docs/site/tooling should say `ARLE` / `arle`; keep legacy `agent-infer` names only as explicit compatibility fallbacks
- [git mv + Edit + fmt hook de-stages renames](feedback_git_mv_with_fmt_hook.md) — after batch-Editing renamed files, re-check git status and re-stage with explicit paths before commit
- [No half-states / no jumper wires](feedback_no_half_states.md) — finish a refactor unit fully or revert it; never leave parallel old-and-new code paths in the tree
- [Don't over-delegate simple deterministic work](feedback_no_over_delegation.md) — shell/config/banner edits: Claude executes directly, don't write a Codex brief
- [CLI closure uses real models when available](feedback_cli_use_real_models_for_closure.md) — for CLI DX work, prefer live local models and real train/eval flows over mocks once the user says the machine can run them
- [backend::cuda::prelude must stay minimal](feedback_prelude_minimal.md) — proto-API contract for the eventual cuda-kernels extraction; ≥3 consumers + stable + no infer-internal type leaks gate
- [Bench must run serially on single-machine Apple Silicon](feedback_bench_serial_only.md) — never run two guidellm benches in parallel on the same Mac; GPU/memory contention invalidates both results

- [mlx fast.rope tensor layout](feedback_mlx_rope_layout.md) — fast::rope needs [batch, heads, seq, dim] layout, not [batch, seq, heads, dim]; wrong order = degenerate output
- [mlx fast.rope axis convention](feedback_mlx_rope_axis.md) — T = second-to-last axis; transpose to [B, heads, seq, d] BEFORE rope so T=seq not T=heads
- [Top-tier OSS roadmap](project_top_tier_roadmap.md) — Phased plan: A=foundation (LayerForward trait, WeightLoader, errors), B=quant, C=community, D=second model
- [Architecture optimization roadmap](project_architecture_roadmap.md) — Claude+Codex cross-reviewed P0-P3 priority list for top-tier OSS quality
- [Qwen3.5 Metal optimization](project_metal_qwen35_optimization.md) — 52 tok/s (67% mlx_lm), next: C++ weight registry + prefill + chunked prefill
- [infer/tests/ files with utility-looking names are real tests](project_tests_dir_convention.md) — gen_test_data_*, regen_test_data, bench_prefill are #[test]-bearing by design, don't relocate
- [Qwen3 varlen decode follow-up](project_qwen3_varlen_followup.md) — Qwen3 pure-Rust decode_qwen3_batch still requires same-length; RoPE fix landed but varlen left-pad + mask needs per-layer loop restructuring
