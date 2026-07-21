#!/usr/bin/env bash
# Start `arle serve` for ThinkingCap-Qwen3.6-27B-FP8 as the OpenAI-v1 endpoint that
# Terminal-Bench's terminus agent (via litellm) will hit.
#
# TOPOLOGY (evidence-grounded 2026-06-26):
#   - Terminal-Bench MUST run on a host with Docker (it spins per-task task
#     containers). The 8xH20 pod is itself a k8s static-pod container WITHOUT
#     docker (`pod 'docker --version'` -> command not found), so TB cannot run
#     on the pod. Docker IS present on the Mac (29.4.0).
#   - The pod container runs with HOST NETWORKING (verified: a port bound inside
#     the container is visible on the node's localhost). So `--bind 0.0.0.0`
#     inside the container is reachable on the node's localhost:PORT, and an
#     `ssh -L PORT:127.0.0.1:PORT` over the existing `127.0.0.1:12222` tunnel
#     forwards it to the Mac's localhost:PORT.
#
# This script runs ON THE POD (GPU). It launches the serve bound to 0.0.0.0 so
# the node (and, via the tunnel local-forward set up by the eval script) the Mac
# can reach it.
#
# Usage (on the pod, e.g. via `pod` / a pod tmux session), TP=1 on a free GPU:
#   CUDA_VISIBLE_DEVICES=4 PORT=8100 scripts/terminal_bench_serve.sh
#
# Env overrides:
#   ARLE_BIN     path to the prebuilt arle binary
#                (default: /host/arle-build/target/release/arle)
#   MODEL_PATH   model dir            (default: /host/ThinkingCap-Qwen3.6-27B-FP8)
#   PORT         listen port          (default: 8000)
#   BIND         bind address         (default: 0.0.0.0 — required for the
#                                       node/Mac to reach it over host-net)
#   CUDA_VISIBLE_DEVICES  pin a single FREE GPU ordinal (e.g. 4). The 27B FP8
#                       weights fit on one H20; serve TP=1 today (TP>1 needs an
#                       HD256 q6_kv1 prefill kernel that the AOT set lacks). The
#                       single-GPU path otherwise targets ordinal 0, which may be
#                       occupied by another workload.
#   EXTRA_FLAGS  extra `arle serve` flags (e.g. "--max-total-tokens 32768")
#
# The served OpenAI model id is the model-dir basename: `ThinkingCap-Qwen3.6-27B-FP8`
# (infer-api `model_id_from_path`). Point TB at it as `openai/ThinkingCap-Qwen3.6-27B-FP8`.
set -euo pipefail

ARLE_BIN="${ARLE_BIN:-/host/arle-build/target/release/arle}"
MODEL_PATH="${MODEL_PATH:-/host/ThinkingCap-Qwen3.6-27B-FP8}"
PORT="${PORT:-8000}"
BIND="${BIND:-0.0.0.0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

if [[ ! -x "$ARLE_BIN" ]]; then
  echo "[tb-serve] error: arle binary not found/executable at $ARLE_BIN" >&2
  echo "[tb-serve]   build it on the pod first, or set ARLE_BIN" >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[tb-serve] error: model dir not found at $MODEL_PATH" >&2
  exit 1
fi

MODEL_ID="$(basename "$MODEL_PATH")"
# DEVICE SELECTION (measured 2026-06-26):
#   - The 27B FP8 weights (~30 GB) fit on ONE H20 (97 GB) — serve TP=1.
#   - TP>1 (multiproc) is BLOCKED today: the AOT kernel set has only the
#     HD256 q24_kv4 prefill kernel; at TP=4 the q6_kv1 shard shape errors
#     `no HD256 paged prefill kernel for q6_kv1` at tick #0.
#   - The single-GPU path IGNORES INFER_CUDA_DEVICES for the ordinal (always
#     targets ordinal 0). Use CUDA_VISIBLE_DEVICES=<free-gpu> to pin a free
#     card (it then appears as ordinal 0). On the shared box GPUs 0-3 are often
#     held by another serve, so e.g. CUDA_VISIBLE_DEVICES=4.
GPU_NOTE="(set CUDA_VISIBLE_DEVICES=<free-gpu> for a single free card)"
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && GPU_NOTE="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[tb-serve] arle:       $ARLE_BIN"
echo "[tb-serve] devices:    ${GPU_NOTE}"
echo "[tb-serve] model:      $MODEL_PATH  (OpenAI model id: $MODEL_ID)"
echo "[tb-serve] endpoint:   http://${BIND}:${PORT}/v1  (chat: /v1/chat/completions)"
echo "[tb-serve] TB --model: openai/${MODEL_ID}   -k api_base=http://localhost:${PORT}/v1"
echo "[tb-serve] starting (cuda, TP=1 single-GPU)…"

# shellcheck disable=SC2086
exec "$ARLE_BIN" serve \
  --backend cuda \
  --model-path "$MODEL_PATH" \
  --bind "$BIND" \
  --port "$PORT" \
  $EXTRA_FLAGS
