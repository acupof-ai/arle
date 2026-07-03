# #138 decoded: the MTP-off invisible wall is context-129 (SW ring wrap), kernel-independent, lens-maskable

> Status: diagnosis campaign, 2026-07-03, 8×H20 pod (115.190.184.36),
> DSv4-Flash-FP8, TP=4/EP=4 on GPUs 4-7, binary `06e9fc6a`
> (`--features cuda,nccl`, snapshot `arle-p138-bin`), `--spec-type none`,
> greedy T=0, `--probe-out` per case, `chunked_prefill_size=64` (engine default;
> scheduler chunked a162 at 128). Serve cycles A/B/D/E/F/G, one variable each.

## Context

Issue #138: MTP-off eager decode "invisible" above a prompt wall ∈ (123, 162],
while MTP verify is clean at 1795 tokens. This campaign decoded the actual
generations + probe entropy/lens at the boundary.

## Decoded cases (ground truth)

- **"Invisible" = argmax collapse to token id 0** (renders as empty text), and
  the probe shows the logits are **NaN** at those steps (argmax of NaN → 0).
- **The wall is ABSOLUTE context length 129 = sliding_window(128)+1, not prompt
  length.** c50 (prompt 50, count-task, mt 256): coherent decode through pos
  128, token-0/NaN from **pos 129** onward, every step. a123 (prompt 123,
  mt 32): 6 real tokens (pos 123-128), then token-0 — the prior round's
  "prompt 123 visible / 162 invisible" was this same wall sampled at two
  prompt lengths.
- **Chunked prefill breaks at the same threshold**: a162 chunk 1 (pos 0-127)
  all-finite; chunk 2 (pos 128-161) **every position NaN** — first forward row
  whose attention context ≥129. The chunk-1 tail's next-token record (pos 128,
  context 128) is still finite.
- **Kernel-independent (single-flag A/Bs, same binary/session):**
  - FlashMLA decode vs scalar SW fallback (`ARLE_DSV4_FLASHMLA_DECODE=0` +
    `ARLE_DSV4_FLASHMLA_DECODE_ALLOC=1`): identical wall at 129 (d123 = 6 real
    tokens then 0s; d162 all 0s).
  - `ARLE_DSV4_FUSE_ATTN_WINDOW_UPDATE=0`: f50 **token-for-token identical**
    to c50 — wall unmoved. (Deterministic wall; also exonerates the fused ring
    write, with the caveat that byte-identity can't distinguish an equivalent
    path from a silently inert flag.)
  - `ARLE_DSV4_INCREMENTAL_KV=0`: g50 token-identical — wall unmoved (same
    caveat).
  - FlashMLA prefill (chunk 2) is a third kernel lane failing at the same
    threshold.
- **Layer localization (lens 43, contaminated case b162):** decode residual
  stream finite through **layer 37**, NaN at **layers 38-42** every step —
  the corrupted state the decode reads sits at the deep ratio-4/128
  compressed layers. Clean case b123: 43-layer lens 100% finite.
- **Heisenbug: the pure-decode NaN is MASKED by the probe lens.** b123
  (lens 43) decodes REAL tokens at pos 129-130 where a123 (lens 0) emitted
  token-0. The lens's only side effects are per-layer device-buffer keepalive
  (`LENS_STASH`) + an extra D2H/sync per step → the decode-side NaN is
  **timing/lifetime-sensitive** (stream-ordering or buffer-reuse race), not a
  deterministic math bug. The chunk-2 prefill NaN is NOT lens-maskable
  (deterministic in-forward).

## Attribution

All lanes (two decode attention kernels, two window-write paths, FlashMLA
prefill) fail at exactly first-context-≥129 — the one machinery FIRST CONSUMED
at that point is the beyond-window compressed/DSA read (window covers ≤128;
compressed chunks + DSA index engage at 129, which is also the first SW ring
wrap). Combined with lens-masking: **the compressed/DSA state consumed by the
eager lanes is garbage/NaN at first beyond-window read, via a
synchronization/lifetime race in its build→read path** — while the MTP verify
lane (own persistent-stream path) reads sane state. Root-cause hypothesis at
the file:line level still open; next decomposition = instrument the compressed
chunk build vs read on one ratio-4 layer (e.g. layer 38) around pos 129.

## What else fell out

- **`ARLE_DSV4_FLASHMLA_DECODE=0` alone deadlocks serve admission**: no arena
  → `Dsv4LayerKvLayout::flashmla_total_pages()=0` → `effective_total_pages()`
  mirrors a 0-page host admission pool → requests never schedule; rank0
  lockstep spins, ranks 1-3 park in `tcp_recvmsg`, HTTP holds the request
  open forever (20-token prompt hangs; control run without the env returns in
  seconds). Use `ARLE_DSV4_FLASHMLA_DECODE_ALLOC=1` alongside, or better:
  fail-fast at load.
- The env gate itself had never been wired: `dsv4_flashmla_decode_enabled()`
  had only the in-process AtomicI8 override (agent-bench), though lib.rs +
  environment.md documented the env. Wired in `06e9fc6a` (OnceLock-cached,
  default path byte-identical).
- **Scalar SW fallback decode has a separate IN-window quality defect**: d50
  (scalar) rambles/repeats from ~10 tokens in on the same prompt where c50
  (FlashMLA) counts cleanly — the fallback lane is not a valid reference
  (`feedback_unvalidated_path_not_reference`).

## Rule

- A "prompt-length wall" sampled at two prompt lengths is a hypothesis about
  the wrong axis — decode a LONG generation from a SHORT prompt to separate
  absolute-position walls from prefill walls (one request settled it here).
- `token_ids` (`return_token_ids: true`) on the completions API turns
  "invisible output" from a mystery into `argmax(NaN)=0` in one request.
- Probe instrumentation that keeps device buffers alive can MASK
  lifetime/ordering bugs — a lens-on/lens-off flip is itself a high-value
  race detector.
