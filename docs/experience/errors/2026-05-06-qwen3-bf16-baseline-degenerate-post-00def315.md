# Qwen3-4B BF16 Baseline Output Degenerate Post-`00def315`

## Context

Discovered while doing an intermittent GPU smoke during the FP8 KV Tier 1
diagnostic effort (codex's `infer/src/bin/fp8_kv_boundary_diag.rs` + module).

The pre-built release binary at `/tmp/arle-target-release/release/infer`
(built `2026-05-06 05:29:05`, post-`00def315` "fix(cuda): finalize qwen3 fp8
paged prefill kv") was launched on `Qwen3-4B` with **BF16 KV** (`--kv-cache-dtype bf16`)
and exercised with two trivial prompts under greedy `temperature=0`. Both
prompts produced **degenerate output starting at token ~7** — the first
prefill-tail tokens are coherent, then the decode trajectory breaks into
broken-token gibberish. Behavior is independent of `--mem-fraction-static`
(reproduced at both `0.3` and `0.85`).

This makes both world-#1 path-A (close FP8 KV Tier 1 ≥70% match) and path-B
(BF16-both-sides re-measure of the 1.609× SGLang ratio) immediately
unreliable: the BF16 reference itself is corrupt on this binary.

## Command

Server:

```bash
RUST_LOG=warn CUDA_HOME=/usr/local/cuda /tmp/arle-target-release/release/infer \
  --model-path infer/models/Qwen3-4B \
  --port 8090 \
  --kv-cache-dtype bf16 \
  --num-slots 4 \
  --max-seq-len 4096 \
  --mem-fraction-static 0.85 \
  --max-num-batched-tokens 4096 \
  --max-prefill-tokens 4096 \
  --schedule-policy fcfs
```

Probes (greedy, no `ignore_eos`):

```bash
# Probe A
curl http://127.0.0.1:8090/v1/completions -d \
  '{"model":"Qwen3-4B","prompt":"The cat sat on the mat. The dog","max_tokens":256,"temperature":0}'

# Probe B
curl http://127.0.0.1:8090/v1/completions -d \
  '{"model":"Qwen3-4B","prompt":"Write a one-sentence summary of photosynthesis.","max_tokens":128,"temperature":0}'
```

## Environment

- GPU: NVIDIA L4, 23,034 MiB
- CUDA target: `sm_89`
- Model: Qwen3-4B (`infer/models/Qwen3-4B`, weights from
  `/content/drive/MyDrive/dev/project/agent-infer/models/Qwen3-4B`)
- Binary: `/tmp/arle-target-release/release/infer`, built 2026-05-06 05:29:05
  by codex against post-`00def315` HEAD; the binary's git ancestry includes
  PR #43 / #44 / #45 and the FP8 paged-prefill KV finalization patch
  (`00def315`)
- KV dtype: **bf16** (no FP8 path exercised)

## Results

| Probe | mem-fraction | Prompt | max_tokens | completion_tokens | finish_reason | First ~7 tokens | Then |
|---|---|---|---|---|---|---|---|
| A1 | 0.30 | "The cat sat on the mat. The dog" | 256 | 53 | length | "ran into the room. The cat jumped" | "walkedived         What't0 ______0..." |
| A2 | 0.85 | (same) | 256 | 256 | length | "ran into the room. The cat jumped" | "walkedived         What't0 ______0..." (same prefix, then continues degenerate to 256 tokens) |
| B1 | 0.30 | "Write a one-sentence summary of photosynthesis." | 128 | 52 | length | "Photosynthesis is the process by which" | "which greenot which which plants..." |
| B2 | 0.85 | (same) | 128 | 128 | length | "Photosynthesis is the process by which plants" | "thesh so so plants the process. So sentence..." |

GPU during decode: 100% util, throughput ≈ 30 tok/s on c=1 (matches
`project_l4_perf_baseline.md` 30.5 tok/s baseline) — kernels run, but the
data flowing through them is corrupt at or near the first decode boundary.

Cross-reference: codex's earlier in-session smoke
(`/tmp/arle-gpu-smoke-supported-response.json`, prompt "hello", 64 tokens)
returned 64 identical `token_id 5` ("hello") with `logprob = -0.0` for every
step — a degenerate fixed point. At the time it was glossed as "path runs";
in light of the present finding it is the **same regression**, not a benign
echo.

## Resolution Update (2026-05-06 tick 6 — codex 3-way probe)

**True root cause is NOT `00def315`. It is TileLang-attn defaulting into
the Qwen3 HD128 paged-prefill path** (introduced in PR #46
`5a3c0a89 feat(cuda): TileLang AOT generator supports attention + GDR
families`). Disabling TileLang-attn (FlashInfer-only) restores BF16 output
to coherent English without touching `00def315`.

Codex ran the 3-way probe under `bench-output/2026-05-06-bf16-baseline-reprobe/`:

| Probe variant | Probe A output | Probe B output | Verdict |
|---|---|---|---|
| `probe-A-current-head.json` (HEAD, TileLang-attn default-on) | degenerate `walkedived What't0...` | degenerate `thesh so so...` | reproduces my finding |
| `probe-A-revert-00def315.json` (revert `00def315`) | still degenerate | still degenerate | **falsifies H1** |
| `probe-A-flashinfer-only.json` (TileLang disabled, FlashInfer attn) | "ran into the room. The cat jumped off the mat and chased the dog…" coherent | normal photosynthesis / cellular respiration text | **isolates true cause** |

GPU util 100% / mem 19858 MiB across all three — kernels run; only the
TileLang attention path produces broken numerics for Qwen3 HD128
paged-prefill.

**My original H1 (`00def315` BF16 leak) is FALSIFIED.** H2 (stale binary)
also falsified by codex's fresh HEAD build behaving identically to the
05:29 binary on the same TileLang default. H3 (shared prefill→decode KV
write/readback bug) was directionally close — the bug IS at the
prefill→decode boundary — but the specific layer is the new TileLang-attn
prefill kernel, not `decode_prep_paged` shared code or the `00def315`
durable-KV finalization.

The fix lane is now: gate TileLang-attn off-by-default for Qwen3 HD128
paged-prefill (or remove the default entirely until the kernel is
numerically validated against a 32×256 trajectory gate). Codex is landing
the fix as a Cargo.toml feature/default flip across `crates/cuda-kernels/`
and `infer/`. Cross-link the codex writeup once it lands.

Implication for FP8 KV Tier 1 (`2026-05-02-...-fail.md` 0.43% / `2026-05-05-...-still-fail.md`
1.22%): both prior runs were on the same TileLang-attn-default binaries.
Once TileLang-attn is gated, the FP8 vs BF16 trajectory comparison must be
re-run from scratch to see what FP8 KV's true gap is — the published
0.43% / 1.22% numbers may be substantially better when measured against a
non-degenerate BF16 reference.

The "do not stack FP8 patches" rule still holds; it now extends to "do not
stack FP8 OR BF16 numerics analysis on a binary with TileLang-attn
default-on".

---

## Root Cause (original hypothesis — superseded by Resolution Update above)

`00def315 fix(cuda): finalize qwen3 fp8 paged prefill kv` advertises itself
as an FP8-only fix. Stat: it adds 243 / removes 13 lines in
`infer/src/model/qwen3/prefill.rs`, and threads a new
`prefill_token_rows: &CudaSlice<i32>` parameter through the full prefill
call graph: `compute_logits_batch_packed`, `refill_paged_prefill_prefix_if_needed`,
`finalize_paged_prefill_kv_layer`, and the per-format `match pool.format`
branches.

The format-specific behavior changes are inside the `match` arms (FP8 / INT8
/ BF16), but **the wiring of `prefill_token_rows` traverses BF16 call sites
too**. If `prefill_token_rows` is mis-indexed, mis-sized, or written to the
durable BF16 pool with the wrong offset, BF16 prefill / first-decode KV
reads are corrupted at the prefill→decode boundary.

This is consistent with the observed pattern: prefill produces a few
coherent first-decode tokens (because the first generated token uses prefill
KV), then degenerates as soon as the decode loop must re-read the
just-written KV pages — the same shape codex's `static_path_evidence()`
documents for FP8.

**Hypothesis to falsify, in this order:**

1. `00def315` BF16 path leak: bisect by reverting `00def315` (or
   cherry-picking only the FP8-arm changes) and re-running probes A+B. If
   BF16 output normalizes at HEAD~`00def315`, this is the cause.
2. Tokenizer / weights drift: rebuild the binary at HEAD and re-test. If
   degeneration disappears with a fresh build of the same source, the May 6
   05:29 binary itself is stale (less likely — `00def315` is upstream of
   that build).
3. Prefill → first-decode KV write/readback bug shared by FP8 and BF16
   paths; the `static_path_evidence()` finding "BF16 and FP8 share
   `decode_prep_paged` before the format branch" already says the
   pre-quantize path is shared, so a bug in the shared path would appear in
   both.

## Fix

Pending diagnosis. Do **not** patch FP8 KV further until the BF16 baseline
is restored, otherwise codex's `fp8_boundary_diag` tool would compare two
broken paths and emit misleading delta evidence.

Recommended next steps (codex's call on HOW):

1. Quick falsification of H1: `git revert 00def315 --no-commit`, build, run
   probes A + B, compare. If BF16 normalizes, the regression is localized.
2. If H1 confirmed: targeted fix is to gate the new
   `prefill_token_rows`-based durable-KV writeback inside the FP8 / INT8
   `match pool.format` arms so the BF16 arm is byte-identical to pre-patch.
3. After BF16 restored, rerun the 32 × 256 token-trajectory gate from
   `2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md` and the post-patch
   rerun in `2026-05-05-fp8-kv-tier1-still-fail.md` — both numbers may shift
   if the BF16 reference itself was off.

## Rule

**Never call a binary "smoke OK" just because it runs and produces tokens.**
Codex's earlier in-session "supported" smoke returned 64 identical `token_id
5` (`logprob = -0.0`) — that was already a degenerate fixed point, not a
working baseline. A coherence eyeball on the first ~20 tokens of one
probe-prompt is mandatory before claiming a server is "working" — `output ≠
empty` is not sufficient.

Also: **a "FP8-only" fix that touches 250+ lines of a shared prefill path
must verify BF16 is unchanged**. The `00def315` commit's verification block
(build / clippy / cuda-no-cuda check / `flashinfer::` test rebuild / codex
review) covers compile + static gates, not BF16 token-trajectory behavior.

## Artifacts

- Probe A mfrac=0.30: `bench-output/2026-05-06-bf16-baseline-degenerate/probe-A-cat-mat-mfrac0.3.json`
- Probe A mfrac=0.85: `bench-output/2026-05-06-bf16-baseline-degenerate/probe-A-cat-mat-mfrac0.85.json`
- Probe B mfrac=0.30: `bench-output/2026-05-06-bf16-baseline-degenerate/probe-B-photosynthesis-mfrac0.3.json`
- Probe B mfrac=0.85: `bench-output/2026-05-06-bf16-baseline-degenerate/probe-B-photosynthesis-mfrac0.85.json`
- Suspect commit: `00def315 fix(cuda): finalize qwen3 fp8 paged prefill kv` (`infer/src/model/qwen3/prefill.rs +243 -13`)
- Cross-ref: `docs/experience/errors/2026-05-02-qwen3-fp8-kv-numerical-tier1-fail.md`,
  `docs/experience/errors/2026-05-05-fp8-kv-tier1-still-fail.md`,
  memory `project_phase1_correctness_blocker_2026-05-04.md`
