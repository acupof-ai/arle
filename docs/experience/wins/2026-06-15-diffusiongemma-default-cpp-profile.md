# DiffusionGemma C++ profile default-on

## Goal

Make the Metal DiffusionGemma fast-path profile visible by default so every
request explains where wall time went: prompt prefill, denoise forward encode,
entropy accept / renoise encode, host scalar sync, self-conditioning, and final
canvas commit.

## Hypothesis

The prior `diffusion cpp profile` line was opt-in and too coarse. It established
that denoise dominates, but did not separate graph/MLX encode from the host
scalar synchronizations used by adaptive stopping, and it under-counted final
canvas work when generation stopped in the final block.

## What Worked

- `ARLE_DIFFUSION_CPP_PROFILE` is now default-on for the C++/MLX
  DiffusionGemma bridge.
- Operators can still disable the line and its timer collection with
  `ARLE_DIFFUSION_CPP_PROFILE=0`, `false`, `off`, or `no` for clean logs or
  pure wall-clock benchmarks.
- The profile line now includes:
  - `denoise_forward_encode_ms`
  - `denoise_accept_encode_ms`
  - `denoise_scalar_sync_ms`
  - `denoise_selfcond_encode_ms`
  - `scalar_syncs`
  - `final_eval_ms`
- `final_commit_ms` now counts the stopped final block too; before this change,
  stop-at-final-block returned before adding that block's final timing.

## Verification

```bash
cargo build --release --no-default-features --features metal,no-cuda
cargo fmt --all --check
cargo test -p cli --release --no-default-features --features cpu,no-cuda diffusion_backend_uses_direct_chat_template_path -- --nocapture
cargo test -p infer-server --release real_checkpoint_tests::real_diffusion_gemma_external_template_renders_if_cached -- --nocapture
strings target/release/arle | rg "denoise_forward_encode_ms|denoise_scalar_sync_ms|ARLE_DIFFUSION_CPP_PROFILE"
```

The `strings` gate confirmed the new profile fields are present in the rebuilt
root binary:

```text
ARLE_DIFFUSION_CPP_PROFILE
 denoise_forward_encode_ms=
 denoise_scalar_sync_ms=
```

After memory pressure dropped, the same rebuilt binary ran the real 26B
checkpoint:

```bash
./target/release/arle \
  --model-path mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --max-tokens 32 \
  --temperature 0 \
  --non-interactive run --prompt '你好呢' --no-tools
```

Output was visible and correct:

```text
你好！很高兴见到你。请问有什么我可以帮你的吗？或者我们只是随便聊聊天？
```

Default profile line:

```text
diffusion cpp profile: prompt_tokens=15 generated_tokens=23 blocks=1 steps=11 prefill_ms=3968.81 denoise_ms=2867.55 denoise_forward_encode_ms=7.60454 denoise_accept_encode_ms=0.364543 denoise_scalar_sync_ms=2859.38 denoise_selfcond_encode_ms=0.195169 scalar_syncs=11 final_commit_ms=0.345417 final_eval_ms=0.343916 total_ms=6836.78
```

The important reading is not "CPU scalar math takes 2.86 s". MLX is lazy: the
host `.item()` checks for stability/confidence are synchronization points that
force the previously encoded denoise graph to finish. The profile therefore
shows the remaining wall time sitting at the per-step scalar sync barrier, while
the C++ graph-construction slices are tiny. The practical next wall is reducing
per-step host synchronization / branch decisions or reducing the denoise step
count with a quality gate.

## Rule

DiffusionGemma performance claims need a default-visible phase profile. The
profile may be disabled for clean benchmark logs, but the default operator path
should expose whether a request is denoise-bound, host-sync-bound, or final
canvas-bound.
