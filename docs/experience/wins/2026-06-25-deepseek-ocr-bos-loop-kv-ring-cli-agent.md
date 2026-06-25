# DeepSeek-OCR: the "3B/500M-active is slow" was a missing-BOS non-stop loop, not the MoE — plus KV ring, `arle ocr`, and an agent OCR tool

## Context

User report: "总参数3B，实际激活仅500M 为啥还这么慢，感觉读写的问题" — DeepSeek-OCR
(`sahilchachra/unlimited-ocr-mxfp8-mlx`, 12 layers, hidden 1280, 64 experts top-6,
~500M active, 3.6 GB MXFP8) felt slow on Metal, suspected a read/write (KV) problem.
Also asked to wire OCR into the CLI (auto-download, default-available) and the agent
tools (auto-detect a local OCR model).

Measured on Apple M-series (binary built same day, model cached). Two causes, both
**measured, not inferred** (AGENTS.md §0):

## What Worked

**1. Dominant cause — the in-process path was missing BOS → a non-stopping loop.**
The 2026-06-24 BOS fix only landed in the HTTP `build_deepseek_ocr_prompt`; the
in-process `complete_multimodal_chat` path (used by the CLI) rendered the model's
chat template, which emits **no BOS** (verified: `chat_template.jinja` has none).
Without a position-0 BOS the decoder degenerated into an infinite
`{"text":"image"}{"text":"image"}…` loop that burned the **entire** `max_tokens`
budget on garbage. Measured: N=2048 = 11.8 s of pure garbage. Prepending the BOS
marker made it read the image and **stop at ~70 tokens** ("The quick brown fox jumps
over the lazy dog. OCR TEST 2025. Performance measurement run." + bbox coords). This
was the bulk of the perceived slowness. Fix: one shared
`infer_server::multimodal::build_deepseek_ocr_prompt` (BOS-prefixed, reference
`<image>\n<text>` image-first layout) used by **both** the HTTP server and
`serve_engine::run_multimodal_chat` — killing the two-builder half-state.

**2. Secondary (the user's read/write instinct, confirmed) — per-step KV `concatenate`.**
`mlx_deepseek_ocr_model.cpp` reallocated the full `[1,nkv,len+1,hd]` K/V history
**every** decode step (O(ctx) traffic/token). Replaced with a pre-allocated
`slice_update` ring sized to `prompt+max_new` rounded to a 256 chunk (mirrors the
canonical Qwen35 cache loop), decoding directly against `layer_caches` (dropped the
aliasing `auto next = layer_caches` copies) so the slot write is donated in place.

Clean A/B (same binary, BOS fix in both, dense 1024px full-page image, best of 3,
isolating the KV path):

| context | OLD concatenate | NEW ring |
|---|---|---|
| decode 128→640 | 306 tok/s | 226 tok/s |
| decode 640→1152 | **213 tok/s** (degrading) | **247 tok/s** (flat) |
| N=128 wall | 2.54 s | 2.54 s (wash) |
| N=1536 wall | 9.32 s | 9.12 s (−2%) |

The ring removes the O(ctx) cliff on long-document OCR (rate stays flat vs degrading)
and is never slower at typical lengths. Honest verdict: at short OCR the two are a
wash; the win is the long-decode tail + bounded allocation. Raw decode ~240 tok/s is
*healthy* for 500M-active (faster than Qwen3.6-35B-A3B's 85) — the model was never
fundamentally slow.

**3. Usable output — byte-level decode fix.** The model's `tokenizer.json` ships a
byte-level BPE vocab (`Ġ`=space, `Ċ`=newline) but a mismatched Metaspace/ByteFallback
decoder, so raw decode leaked `Ġ`/`Ċ` glyphs and dropped spaces. Verified the
**reference HF `tokenizers` library reproduces the same bug** (`decode → '{"text":"image"}Ġ{"'`,
roundtrip `'Thequickbrownfox'`) — it's a published-repo bug, not ours. Fix:
`OpenAiTokenizer::force_byte_level_decoder()` overrides the decoder to `ByteLevel`
on the OCR load path only; output is now clean UTF-8.

**4. CLI + agent integration.**
- `arle ocr <image>` (`crates/cli/src/ocr.rs`): free/grounding/markdown modes,
  `--json`, resolves + **auto-downloads** the default model on first use
  (`infer_metal::resolve_model_path` local-first → `download_model_with_progress`).
- Agent `ocr` builtin tool (`crates/tools/src/lib.rs`): checks the HF cache for the
  model, then self-invokes `arle ocr <image> --json` via `current_exe`. Subprocess
  keeps the `tools` crate host-only (no GPU/infer-api dep) and avoids a second
  in-process model; `arle ocr` auto-downloads so it "just works".
- Discoverability: catalog `DEEPSEEK_OCR_MODEL_ID` + `hub_discovery` servable set
  (`model_type=deepseekocr` / `UnlimitedOCR`, Metal).

## Rule

- **A multimodal/VLM path that has TWO prompt builders (HTTP vs in-process) will
  drift** — a fix in one silently misses the other. Converge on one shared builder.
  A missing BOS does not error; it **degenerates into a non-stopping loop** that
  burns the whole token budget and *looks like* "the model is slow." Diff
  `prompt_tokens` and decode the first ~16 generated tokens before blaming the arch.
- **`Ġ`/`Ċ` in decoded output = byte-level vocab with a mismatched decoder.** Verify
  against the reference HF `tokenizers` lib; if it reproduces, it's a repo bug — force
  `ByteLevel`, don't patch the vocab.
- **Per-step KV `concatenate` is O(ctx)/token; a pre-allocated `slice_update` ring is
  O(1)** — but only if you decode directly against the cache (an aliasing copy makes
  `slice_update` copy the whole ring, worse than concat). Confirm the win on a
  *long-decode* workload with the correctness bug fixed in both A/B binaries, else
  early-stop confounds the timing.

## Status

Metal-only model (CUDA N/A). Bench entry exempt for the runtime path: the KV-ring
A/B above is the perf evidence; the rest are correctness + control-plane changes on
the opt-in DeepSeek-OCR surface, default Qwen/DSv4 serving byte-unchanged. Verified
end-to-end: `arle ocr` reads glyph + dense PNGs cleanly and stops early; tools tests
green; cli/infer-server tests green (one modelscope env-race flake, unrelated);
clippy clean. Commits: BOS fix `203eb726`, KV ring `c7112eee`, CLI+decode `c737d056`,
agent tool `164c5a61`.
