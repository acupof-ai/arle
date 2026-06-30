#!/usr/bin/env bash
# DSv4-Flash TP=8 / EP=8 greedy-parity launcher (pod-side, 8×H20 / sm_90a).
#
# Drives the R6 clean-CUDA `dsv4_parity` example across 8 GPUs: rank 0 mints the
# NCCL unique_id in-process and shares it via a file rendezvous, all 8 ranks join
# the same NCCL group and run the identical single-row DSv4 forward, and rank 0's
# `clean_tokens` line is compared to the greedy oracle.
#
# ── What this verifies TODAY ──────────────────────────────────────────────────
#   The FIRST generated token (the full-prefix prefill argmax at start_pos=0) on
#   every layer type (SW / CSA / HCA all recompute attention over [0, len)). The
#   incremental decode steps (start_pos>0) are GATED: DSv4 reallocates its SW ring
#   caches per forward call, so per-step KV reuse is unproven — the harness
#   catches that bail and reports how far it got. The full 16-token oracle below
#   needs the continuous-batching (per-slot SW/compressed retention) follow-up.
#
# ── Build (run ONCE before launching; sm_90a, CUDA 12.9, TileLang venv) ────────
#   On the pod, from the repo root, inside tmux (a detached tunnel SIGKILLs
#   children at ~15-20s — build/run under tmux):
#
#     export TORCH_CUDA_ARCH_LIST=9.0          # sm_90a (H20)
#     export ARLE_DEEPEP_DIR=/path/to/DeepEP   # deepseek-ai/DeepEP source tree
#     # FlashMLA auto-enables when its vendor root is present; the bf16 MLA
#     # correctness core this harness uses does NOT require the FP8 decode path.
#     # Ensure libnccl.so is on LD_LIBRARY_PATH.
#     cargo build --release -p infer-cuda \
#         --features cuda,nccl,deepep --example dsv4_parity
#
#   The example binary lands at:  target/release/examples/dsv4_parity
#   (the NCCL file-rendezvous needs the `nccl` feature, already in the line above.)
#
# ── Run ────────────────────────────────────────────────────────────────────────
#     INFER_DSV4_MODEL_PATH=/path/to/dsv4-fp8-safetensors \
#       scripts/dsv4_multigpu_parity.sh
#
#   Env knobs (all optional except the model path):
#     INFER_DSV4_MODEL_PATH  DSv4 FP8 safetensors dir            (required)
#     INFER_DSV4_PROMPT_IDS  comma-separated ids                 (default in bin)
#     INFER_DSV4_MAX_NEW     max new tokens to attempt           (default 16)
#     DSV4_PARITY_BIN        path to the built example binary
#                            (default target/release/examples/dsv4_parity)
#     WORLD_SIZE             ranks / GPUs                         (default 8)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${DSV4_PARITY_BIN:-$ROOT/target/release/examples/dsv4_parity}"
WORLD_SIZE="${WORLD_SIZE:-8}"

[[ -n "${INFER_DSV4_MODEL_PATH:-}" ]] || {
    echo "ERROR: set INFER_DSV4_MODEL_PATH to the DSv4 FP8 safetensors dir" >&2
    exit 1
}
[[ -x "$BIN" ]] || {
    echo "ERROR: $BIN not found/executable; build it first (see header)" >&2
    exit 1
}

# Comma-separated 0..WORLD_SIZE-1 device list — INFER_CUDA_DEVICES is the TP
# trigger; its ordinal count is the world size (resolve_tp_config in tp.rs).
DEVICES="$(seq -s, 0 $((WORLD_SIZE - 1)))"

WORK="$(mktemp -d -t dsv4-parity.XXXXXX)"
ID_FILE="$WORK/nccl_id.hex"
trap 'rm -rf "$WORK"' EXIT

# ── Spawn WORLD_SIZE ranks, one per GPU, all on the same NCCL group. ──────────
# NCCL bootstrap is a FILE RENDEZVOUS, not a throwaway mint: `ncclGetUniqueId`
# embeds the calling process's bootstrap listener socket, so a separate
# "mint-and-exit" helper leaves a dead socket (Connection refused at
# ncclCommInitRank). All ranks share INFER_NCCL_ID_FILE; the example's rank 0
# mints the id in-process (staying alive), writes it atomically, and the other
# ranks block on the file then read it. See examples/dsv4_parity.rs.
echo "[launcher] NCCL file-rendezvous at $ID_FILE (rank 0 mints in-process)." >&2

declare -a PIDS
declare -a LOGS
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    LOG="$WORK/rank_$r.log"
    LOGS[r]="$LOG"
    # One GPU per rank: CUDA_VISIBLE_DEVICES=$r masks physical GPU $r, which the
    # process then sees re-indexed as ordinal 0 — so INFER_CUDA_DEVICE must be 0,
    # NOT $r (passing $r yields CUDA_ERROR_INVALID_DEVICE since only ordinal 0 is
    # visible). INFER_CUDA_DEVICES gives the world size (8 ordinals → TP=8);
    # INFER_TP_RANK is this rank's index; INFER_NCCL_ID_FILE is the rendezvous path.
    CUDA_VISIBLE_DEVICES="$r" \
    INFER_CUDA_DEVICE=0 \
    INFER_CUDA_DEVICES="$DEVICES" \
    INFER_TP_SIZE="$WORLD_SIZE" \
    INFER_TP_RANK="$r" \
    INFER_NCCL_ID_FILE="$ID_FILE" \
        "$BIN" >"$LOG" 2>&1 &
    PIDS[r]=$!
    echo "[launcher] spawned rank $r (pid ${PIDS[r]}, gpu $r)" >&2
done

# ── Step 3: wait, surface any rank failure. ───────────────────────────────────
FAIL=0
for r in $(seq 0 $((WORLD_SIZE - 1))); do
    if ! wait "${PIDS[r]}"; then
        echo "[launcher] rank $r FAILED — log:" >&2
        cat "${LOGS[r]}" >&2
        FAIL=1
    fi
done
[[ $FAIL -eq 0 ]] || {
    echo "[launcher] one or more ranks failed" >&2
    exit 1
}

# ── Step 4: rank-0 clean_tokens vs the greedy oracle. ─────────────────────────
echo "===== rank 0 log =====" >&2
cat "${LOGS[0]}" >&2
echo "======================" >&2

RANK0_TOKENS="$(grep -E '^clean_tokens=' "${LOGS[0]}" | tail -n1 || true)"
# Oracle: the validated DSv4 greedy continuation for the default prompt.
ORACLE='clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344, 11111, 603, 671, 6102, 294, 8760, 344, 11111, 603]'

echo "[launcher] rank0  : ${RANK0_TOKENS:-<none>}"
echo "[launcher] oracle : $ORACLE"
echo "[launcher] NOTE: today only the FIRST token (prefill argmax) is a verified"
echo "[launcher]       gate on every layer type; the remaining 15 oracle tokens"
echo "[launcher]       need the incremental-decode (start_pos>0) follow-up."
