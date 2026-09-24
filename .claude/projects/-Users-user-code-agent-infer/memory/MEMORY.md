# Memory Index

- [Done means commit and push](feedback_done_means_commit_and_push.md) — do not call work complete until the relevant slice is committed and pushed to `origin/main`
- [Edit the named target file](feedback_edit_the_named_target_file.md) — when a request relates two files, patch the file the user identified as the target, not the other endpoint
- [Project brand is ARLE](feedback_project_brand_is_arle.md) — user-facing CLI/docs/site/tooling should say `ARLE` / `arle`; keep legacy `agent-infer` names only as explicit compatibility fallbacks
- [CLI closure uses real models when available](feedback_cli_use_real_models_for_closure.md) — for CLI DX work, prefer live local models and real train/eval flows over mocks once the user says the machine can run them
- [Bench must run serially on single-machine Apple Silicon](feedback_bench_serial_only.md) — never run two guidellm benches in parallel on the same Mac; GPU/memory contention invalidates both results

- [Qwen3 varlen decode follow-up](project_qwen3_varlen_followup.md) — Qwen3 pure-Rust decode_qwen3_batch still requires same-length; RoPE fix landed but varlen left-pad + mask needs per-layer loop restructuring
