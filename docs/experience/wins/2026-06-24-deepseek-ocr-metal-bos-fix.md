# DeepSeek-OCR Metal: OCR fidelity bug was a missing BOS token, not a DeepEncoder numerical bug

## Context

The DeepSeek-OCR (`deepseekocr` / `UnlimitedOCRForCausalLM`) Metal integration
loaded, routed, and decoded coherent text, but image OCR was unfaithful: with a
large clear full-frame image + `<|grounding|>` it produced *structured grounding
output with plausible bounding-box coordinates* yet wrong/garbled text, and short
prompts degenerated (`"You.You."`, leading garbage `Ġ[logoĠ[image`). The standing
hypothesis (carried across a context reset) was a **DeepEncoder numerical bug**,
most likely in the SAM windowed attention + decomposed relative-position
(`get_rel_pos` / `sam_attention`), since the symptom read as "coarse spatial
structure preserved, fine character detail lost".

## What Worked

**Stop inferring, measure (AGENTS.md §0 SOLID + "decode the actual generation").**
Three measured experiments overturned the hypothesis and found the real bug:

1. **Vision tower numerical bisection.** Stood up a minimal MLX venv
   (`mlx 0.31.2`, no PyTorch) running the *reference* `sam.py` / `vision.py`
   against the real mxfp8 weights, dumping per-stage L2 norms on a fixed PNG.
   Added an env-gated `vdbg` L2 trace (`INFER_DSOCR_VDEBUG`) at the same stages in
   `mlx_deepseek_ocr_model.cpp`. **Every stage matched to 4-5 sig figs**
   (sam.patch_embed 1138.7555 = 1138.7555; sam_out 32.0600 = 32.0600;
   clip_out 115.33 ≈ 115.36; proj 71.71 ≈ 71.72). The DeepEncoder is correct —
   the "SAM rel-pos bug" hypothesis was pure inference, never measured, and false.
   The window blocks carry exact `rel_pos_h [27,64]` (=2·14−1) and global blocks
   `[127,64]` (=2·64−1), so the interpolation path is dead code, not a suspect.

2. **Case-as-fact: the synthetic test images had no glyphs.** The "catastrophic"
   outputs came from synthetic gradient/bar PNGs. A full reference end-to-end
   (reference DeepEncoder + a faithful re-impl of `language.py` reading the real
   weights) **also produced empty output on the same synthetic image** — proving
   the test input, not the model, was the artifact. Rendering a real-text PNG
   (Arial glyphs) made the reference read `HELLO WORLD` / `OCR` in global-only
   1024 mode — killing the secondary "global-only can't read text, needs local
   crops" hypothesis too.

3. **Root cause: missing BOS.** With a real-text image the C++ port also read
   `HELLO WORLD`, but with leading garbage and `prompt_tokens=278` vs the
   reference `279`. The reference processor always prepends BOS (id 0); the
   server's `Tokenizer::encode` calls `encode(text, false)`
   (`add_special_tokens=false`), so the manually-built DeepSeek-OCR prompt ran the
   decoder with **no position-0 BOS token**. Fix: prepend the BOS marker string
   `<｜begin▁of▁sentence｜>` (which encodes to id 0 even without special tokens) in
   `build_deepseek_ocr_prompt`. Post-fix `prompt_tokens=279` and the output
   matches the reference token-for-token modulo MoE run-to-run non-determinism
   (`HELLO WORLD`, `OCR TEST 2025`, `The quick brown fox` all faithful; bbox
   coords within ±2px: 319 vs 321).

## Rule

- **A "coarse-right / fine-wrong" vision symptom is config-suspect before
  code-suspect.** Before staring at attention/rel-pos kernels, (a) numerically
  bisect the tower against the reference stage-by-stage (L2 norms on a *fixed
  input*), and (b) confirm the test input actually exercises the failure (render
  real glyphs, don't trust a synthetic gradient). Both the aggregate "metric" and
  the plausible mechanism lied; only the decoded cases + measured norms were true.
- **A prompt-token-count mismatch vs the reference (off by one) is a special-token
  bug until proven otherwise.** The server's `encode` never adds special tokens;
  any hand-built prompt (multimodal splice paths especially) must carry BOS/EOS
  explicitly. Diff `prompt_tokens` against the reference processor as a cheap gate.

## Status

pending-remote: CUDA-side N/A (Metal-only model). Bench/perf A/B deferred — this
is a correctness fix on the opt-in DeepSeek-OCR VLM path; the default Qwen/DSv4
serving paths are byte-unchanged. Verified end-to-end on
`mlx-community`-style real-text PNGs against the mlx-vlm reference re-impl.
