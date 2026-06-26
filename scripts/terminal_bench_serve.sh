#!/usr/bin/env bash
# Start `arle serve` for Qwen3.6-27B-FP8 as the OpenAI-v1 endpoint that
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
# Usage (on the pod, e.g. via `pod` / a pod tmux session):
#   ARLE_BIN=/host/arle-build/target/release/arle \
#     scripts/terminal_bench_serve.sh
#
# Env overrides:
#   ARLE_BIN     path to the prebuilt arle binary
#                (default: /host/arle-build/target/release/arle)
#   MODEL_PATH   model dir            (default: /host/Qwen3.6-27B-FP8)
#   PORT         listen port          (default: 8000)
#   BIND         bind address         (default: 0.0.0.0 — required for the
#                                       node/Mac to reach it over host-net)
#   INFER_CUDA_DEVICES  comma list of GPU ordinals to use; its count is the TP
#                       world size (project memory: this is the TP=N trigger).
#                       Set to FREE GPUs (e.g. "4,5,6,7") so the serve does not
#                       collide with another workload on GPUs 0-3.
#   EXTRA_FLAGS  extra `arle serve` flags (e.g. "--max-total-tokens 32768")
#
# The served OpenAI model id is the model-dir basename: `Qwen3.6-27B-FP8`
# (infer-api `model_id_from_path`). Point TB at it as `openai/Qwen3.6-27B-FP8`.
set -euo pipefail

ARLE_BIN="${ARLE_BIN:-/host/arle-build/target/release/arle}"
MODEL_PATH="${MODEL_PATH:-/host/Qwen3.6-27B-FP8}"
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
# Qwen3.6 (model_type=qwen3_5) takes the multiproc TP serve path; its TP world
# size is the count of INFER_CUDA_DEVICES. Default to all 8 H20s; override to a
# free subset (e.g. 4,5,6,7) to avoid colliding with another workload.
export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
echo "[tb-serve] arle:       $ARLE_BIN"
echo "[tb-serve] devices:    INFER_CUDA_DEVICES=$INFER_CUDA_DEVICES  (TP world size = device count)"
echo "[tb-serve] model:      $MODEL_PATH  (OpenAI model id: $MODEL_ID)"
echo "[tb-serve] endpoint:   http://${BIND}:${PORT}/v1  (chat: /v1/chat/completions)"
echo "[tb-serve] TB --model: openai/${MODEL_ID}   -k api_base=http://localhost:${PORT}/v1"
echo "[tb-serve] starting (cuda, 8xH20 multiproc TP)…"

# shellcheck disable=SC2086
exec "$ARLE_BIN" serve \
  --backend cuda \
  --model-path "$MODEL_PATH" \
  --bind "$BIND" \
  --port "$PORT" \
  $EXTRA_FLAGS
