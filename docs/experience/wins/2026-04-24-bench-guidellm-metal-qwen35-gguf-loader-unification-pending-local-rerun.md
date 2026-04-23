# Qwen3.5 GGUF loader unification pending local rerun

## Goal

- Regression-check the Qwen3.5 GGUF loader refactor that moves the model-level
  reorder/reshape logic into one shared host path and makes Metal/CUDA consume
  the same transformed tensors instead of keeping per-backend GGUF variants.

## Hypothesis

- If the shared host loader preserves the exact post-load tensor contract, then
  Metal GGUF generation should remain aligned with `llama.cpp`, the existing
  Metal incremental-decode regression should still pass, and CUDA/CPU compile
  surfaces should stay clean.

## Params

- Label: `metal-qwen35-0p8b-gguf-loader-unification`
- Planned canonical command:
  `GUIDELLM__MP_CONTEXT_TYPE=forkserver ./scripts/bench_guidellm.sh metal-qwen35-0p8b-gguf-loader-unification --target http://127.0.0.1:8010 --model models/Qwen3.5-0.8B-GGUF --processor Qwen/Qwen3.5-0.8B`
- Canonical profile intent: `prompt_tokens=4096,output_tokens=256`

## Env

- Local change date: `2026-04-24`
- Local machine: `Apple M4 Pro / macOS 26.3.1(a)`
- Backend: `metal`
- Model under test: `models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf`
- Feature set validated locally:
  - `cargo check -p infer --no-default-features --features metal,no-cuda`
  - `cargo check -p infer --tests --no-default-features --features cuda,no-cuda`
  - `cargo check -p infer --no-default-features --features cpu,no-cuda`
  - `cargo clippy -p infer --no-default-features --features metal,no-cuda -- -D warnings`

## Results

- Status: `pending-local-rerun`
- Local regression checks passed:
  - `cargo test -p infer --no-default-features --features metal,no-cuda --lib backend::metal::qwen35::tests::compare_qwen35_0p8b_gguf_incremental_decode_vs_full_replay -- --ignored --exact --nocapture`
  - `cargo test -p infer --no-default-features --features metal,no-cuda --lib backend::metal::qwen35::tests::compare_qwen35_0p8b_loaded_tensors_safetensors_vs_gguf -- --ignored --exact --nocapture`
  - `./target/release/metal_request --model models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 8 --temperature 0 --top-k 1`
  - `./target/release/metal_request --model models/Qwen3.5-0.8B --prompt 'Hello' --raw-prompt --warmup 0 --max-new-tokens 4 --temperature 0 --top-k 1`
  - `/opt/homebrew/bin/llama-completion -m models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf -p 'Hello' -n 8 --temp 0 --top-k 1 --seed 0 --no-display-prompt --no-warmup -no-cnv`
- Observed local outputs:
  - Metal GGUF: `Hello -> ",\n\nI am writing a Python script"`
  - `llama.cpp` on the same GGUF: `Hello -> ",\n\nI am writing a Python script"`
  - Metal safetensors smoke still runs: `Hello -> ", I have a"`
- Observed incremental decode replay check:
  - step 0: `,`
  - step 1: `\n\n`
  - step 2: `I`
  - step 3: ` am`

## Problems

- This refactor did not rerun the canonical `guidellm` sweep yet. The prior
  local `0.8B` GGUF sweep was already `pending-local-rerun` because the locked
  `4096/256` profile saturated scheduler admission before producing the normal
  headline artefacts.

## Learnings

- The right unification boundary is not "GGUF parser vs Metal/CUDA loader"; it
  is the model-level post-load tensor contract. The shared layer now owns the
  Qwen3.5-specific V-head reorder, `conv1d` reshape, and `A_log` transform.
- Metal no longer has a second copy of the Qwen3.5 GGUF tensor assembly logic.
  Both safetensors and GGUF now flow through one set of final attention/MLP
  builders, which makes future correctness checks much easier to reason about.
- Deletion-style refactor here also removed stale CUDA-only GGUF reorder helpers
  from `weight_loader.rs`; once the shared host path existed, those special
  wrappers stopped earning their keep.

## Δ vs baseline

- Baseline:
  [2026-04-24-bench-guidellm-qwen35-gguf-kquants-and-decode-correctness-pending-local-rerun.md](2026-04-24-bench-guidellm-qwen35-gguf-kquants-and-decode-correctness-pending-local-rerun.md)
- No new canonical sweep table yet; this entry is the structural-loader follow-up.

## Notes

- What changed in code since baseline:
  - added shared `infer/src/model/qwen35/gguf_host.rs`
  - refactored `infer/src/backend/metal/qwen35.rs` to use one Qwen3.5 weight builder path
  - refactored `infer/src/model/qwen35/weights.rs` CUDA GGUF loading to reuse the shared linear-attention host loader
  - deleted the now-unused specialized GGUF reorder wrappers from `infer/src/weight_loader.rs`
- Follow-up:
  - rerun the canonical `guidellm` profile on a locally stable model/profile or on the larger supported Qwen3.5 checkpoint so this cleanup has a normal benchmark snapshot
