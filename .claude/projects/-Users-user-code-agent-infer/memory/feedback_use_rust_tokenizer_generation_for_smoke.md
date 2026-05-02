# Rule

When train/e2e smoke tests need a tiny `tokenizer.json`, generate it with the existing Rust-side `tokenizers` patterns already in `crates/train` instead of ad hoc Python helpers.

# Why

The repo already contains pure-Rust `WordLevel` tokenizer generation paths (for example in `crates/train/src/qwen35_checkpoint.rs`, `crates/train/src/bin/train_sft.rs`, and `crates/train/tests/test_sft_data.rs`). Reintroducing Python for this is unnecessary and violates the project's Rust-first workflow.
