# Multimodal Prefix Cache Identity — guidellm sweep, pending-remote, 2026-05-02

## Goal

- Verify that multimodal content-hash prefix identity does not regress text
  throughput and unlocks image-prefix reuse once the VLM runner is wired.

## Hypothesis

- Text-only traffic should be unchanged because `BlockFingerprint::compute`
  still hashes the same byte stream when no multimodal hashes are supplied.
- Multimodal traffic should be able to reuse prefixes across semantically
  identical image bytes once the image encoder path calls the new helpers.

## Command

```bash
scripts/bench_guidellm.sh metal-qwen3-vl-mm-prefix \
  --target http://localhost:8000 \
  --model Qwen/Qwen3-VL \
  --processor models/Qwen3-VL
```

Invoked via: not run locally; status `pending-remote`.

## Environment

- **Backend:** metal / cuda shared prefix identity
- **Model:** Qwen3-VL family once local runner lands
- **Hardware:** pending remote Apple Silicon / NVIDIA host
- **Commit:** pending at entry creation
- **Feature set:** `cargo build --release --no-default-features --features metal,no-cuda`
- **Non-default flags / env vars:** multimodal runner not yet available
- **Server launch:** pending VLM serve path

## Canonical params (DO NOT CHANGE PER-RUN)

- `--profile sweep`
- `--data prompt_tokens=4096,output_tokens=256`
- `--max-seconds 60`
- `--random-seed 20260416`
- `--outputs json --outputs csv --outputs html`
- Wrapper: `scripts/bench_guidellm.sh <backend-label>`

## Results — sweep headline table

| rate (req/s) | TTFT p50 (ms) | TTFT p99 (ms) | ITL p50 (ms) | ITL p99 (ms) | out tok/s | req/s actual |
|---|---|---|---|---|---|---|
| pending-remote | pending | pending | pending | pending | pending | pending |

## Results — service-side KV / scheduler metrics

| metric | value |
|---|---:|
| peak active | pending |
| peak waiting | pending |
| peak prefill_queue | pending |
| peak kv_util | pending |
| `prefix_hit_rate` | pending |
| `prefix_skip_rate` | pending |
| `kv_fetch_q` | pending |
| `kv_fetch_waiters` | pending |
| `kv_store_q` | pending |
| `kv_store` | pending |
| `kv_bp` | pending |
| `tier_recall` | pending |
| `tier_src` | pending |
| `tier_promoted` | pending |
| `tier_fallback` | pending |

## Results — request accounting

| metric | value |
|---|---:|
| completed input tokens | pending |
| incomplete input tokens | pending |
| completed output tokens | pending |
| incomplete output tokens | pending |

## Problems

- No local multimodal Qwen3-VL runner exists in ARLE yet, so this tranche can
  only verify the hash/cache contracts and the unchanged text-only fingerprint
  path.

## Learnings

- Bind multimodal media by full content hash and represent it in the radix path
  as eight tagged high-bit synthetic tokens, so online lookup and persisted KV
  identity both carry the full 128-bit media hash.

## Δ vs baseline

- **Baseline:** none; first multimodal-prefix-cache identity tranche.

| metric | baseline | now | Δ% |
|---|---|---|---|
| TTFT p50 @ synchronous | n/a | pending | n/a |
| out tok/s @ saturation | n/a | pending | n/a |

## Artefacts

- Raw: pending
- CSV: pending
- HTML: pending
- Service trace (before): pending
- Service trace (during): pending
- Service trace (after): pending
- Service trace (summary): pending

## Notes

- What changed in the code since baseline: added multimodal content hash,
  encoder-cache LRU, OpenAI `image_url` extraction, and a
  `BlockFingerprint::compute_with_multimodal_hashes` helper.
- Suspected cause of any regression: n/a until the VLM path is connected.
- Follow-ups: wire Qwen3-VL/Qwen3.5-VL image encoder output into the scheduler
  prefix admission path and replace this stub with a real guidellm snapshot.
