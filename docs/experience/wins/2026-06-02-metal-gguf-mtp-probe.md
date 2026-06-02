# Metal GGUF MTP Probe

## Context

Follow-up research on Unsloth's Mac-supported Qwen3.6 MTP route showed a
different path from the canonical MLX safetensors model. Unsloth publishes MTP
GGUFs and routes Mac/CPU/GPU inference through Unsloth Studio plus llama.cpp.
Their documented llama.cpp command uses `--spec-type draft-mtp
--spec-draft-n-max 2`, and Studio release notes describe auto MTP speculative
decoding for MTP GGUFs plus Mac/CPU command-shape fixes.

llama.cpp's converter writes GGUF MTP metadata with
`{arch}.nextn_predict_layers` and maps the MTP tensors to `blk.{bid}.nextn.*`
names. ARLE's previous Metal probe handled safetensors only and treated GGUF as
unsupported, so it could not observe the route Unsloth uses on Mac.

## What Worked

Extended the Metal MTP probe to inspect the already-parsed `GgufFile`:

- positive `{arch}.nextn_predict_layers` metadata is reported as an MTP signal;
- GGUF tensor names containing `nextn` / `next_n` are reported as MTP signals;
- safetensors index/header probing remains unchanged;
- decode remains standard Metal decode until native MTP draft/verify is built.

This is still a readiness/probe tranche, not a performance tranche.

## Verification

```text
cargo test -p infer --no-default-features --features metal mtp -- mtp --nocapture
  5 passed
```

The GGUF test constructs a minimal local GGUF header with
`qwen35.nextn_predict_layers=2` and `blk.48.nextn.*` tensor-directory entries,
so no large Unsloth GGUF download is required.

No guidellm run was attached because generation behavior is intentionally
unchanged.

## Rule

For Mac MTP, GGUF readiness is a first-class signal. Do not infer that an MLX
safetensors model is MTP-capable from Unsloth GGUF docs; probe the actual
loaded artifact and keep the decode path unchanged until draft/verify is real.
