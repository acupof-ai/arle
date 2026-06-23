#!/usr/bin/env bash
# Pod-side RUN runner — runs the built arle binary on ONE specific GPU, detached.
# Launched by scripts/pod.sh (setsid). This is the per-agent isolation path: each
# agent runs its own experiment with its own <label> (own log/pid) on its own
# <gpu>, so concurrent agents don't collide — no lock needed, the GPU + label ARE
# the isolation. Ends the log with RUN_EXIT=<n>.
#
# GPU pinning uses CUDA_VISIBLE_DEVICES (remap the chosen physical GPU to logical
# 0), NOT INFER_CUDA_DEVICE: an OPD run touches BOTH backends, and the autograd
# backend binds cudarc device 0 (it does NOT read INFER_CUDA_DEVICE) — so a bare
# INFER_CUDA_DEVICE=N would land the infer engine on GPU N but the autograd
# student on physical GPU 0 (OOM if 0 is busy). Masking to one GPU pins both.
set -uo pipefail
TREE="${POD_TREE:-/host/arle-build}"
LABEL="${1:?usage: pod-remote-run.sh <label> <gpu> <arle-args...>}"; shift
GPU="${1:?usage: pod-remote-run.sh <label> <gpu> <arle-args...>}"; shift
echo $$ > "/root/run-$LABEL.pid"
source "$TREE/scripts/pod-build-env.sh"
cd "$TREE"
LOG="/root/run-$LABEL.log"
{
  echo "=== RUN START $(date -u) label=$LABEL gpu=$GPU (CUDA_VISIBLE_DEVICES) ==="
  CUDA_VISIBLE_DEVICES="$GPU" INFER_CUDA_DEVICE=0 ./target/release/arle "$@"
  echo "RUN_EXIT=$?"
  echo "=== RUN DONE $(date -u) ==="
} >"$LOG" 2>&1
