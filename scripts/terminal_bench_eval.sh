#!/usr/bin/env bash
# Run Terminal-Bench (laude-institute/terminal-bench, https://www.tbench.ai/)
# against OUR model — ThinkingCap-Qwen3.6-27B-FP8 served by `arle serve` — using the
# built-in `terminus` agent driven through litellm at an OpenAI-compatible base
# URL.
#
# RUNS ON THE MAC (Docker present; TB spins per-task Docker containers).
# The serve runs on the 8xH20 pod (no docker there). This script:
#   1. ensures a local-forward `localhost:$PORT -> pod serve` is up (over the
#      existing `127.0.0.1:12222` tn tunnel; host-net pod => node localhost),
#   2. probes `/v1/models` to confirm the serve is reachable,
#   3. runs `tb run --agent terminus --model openai/<id> -k api_base=<url>` on a
#      task SUBSET.
#
# INTEGRATION MECHANISM (grounded in TB source @ main, 2026-06-26):
#   - `tb run --model openai/<name>` + `-k api_base=<url>` flow through
#     `_process_agent_kwargs` -> Harness `_agent_kwargs` -> terminus `__init__(
#     model_name, api_base, ...)` -> `LiteLLM(model_name, api_base)` ->
#     `litellm.completion(model=..., api_base=..., ...)`.
#   - litellm's `openai/` prefix => call the OpenAI-compatible
#     `/v1/chat/completions` at `api_base`. It REQUIRES a non-empty
#     `OPENAI_API_KEY` even for a local server (any dummy string works; our serve
#     does not check it).
#
# Usage:
#   # smoke: 1-2 tasks
#   scripts/terminal_bench_eval.sh --task-id hello-world
#   scripts/terminal_bench_eval.sh -n 2
#   # a named subset (repeat --task-id)
#   scripts/terminal_bench_eval.sh --task-id hello-world --task-id fix-permissions
#   # full core set (~100 tasks) — slow
#   scripts/terminal_bench_eval.sh --all
#
# Pass-through: any flag after the recognized ones is forwarded to `tb run`
# (e.g. --n-concurrent 4, --livestream, --n-attempts 2).
#
# Env overrides:
#   PORT          local + pod serve port           (default: 8000)
#   MODEL_ID      OpenAI model id (basename of pod model dir)
#                                          (default: ThinkingCap-Qwen3.6-27B-FP8)
#   API_BASE      override the serve base URL       (default: http://localhost:$PORT/v1)
#   OPENAI_API_KEY  litellm api key (dummy ok)      (default: arle-local)
#   DATASET_NAME  TB dataset                        (default: terminal-bench-core)
#   DATASET_VER   TB dataset version                (default: 0.1.1)
#   N_CONCURRENT  concurrent trials                 (default: 1 for smoke)
#   OUTPUT_DIR    results dir                       (default: runs/terminal-bench)
#   TB           tb CLI invocation                  (default: auto: `tb`, else `uvx --from terminal-bench tb`)
#   SKIP_TUNNEL=1 skip the ssh -L setup (serve already reachable at API_BASE)
#   SKIP_PROBE=1  skip the /v1/models reachability probe
set -euo pipefail

PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-ThinkingCap-Qwen3.6-27B-FP8}"
API_BASE="${API_BASE:-http://localhost:${PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-arle-local}"
DATASET_NAME="${DATASET_NAME:-terminal-bench-core}"
DATASET_VER="${DATASET_VER:-0.1.1}"
N_CONCURRENT="${N_CONCURRENT:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/terminal-bench}"

# ---- parse the few flags we own; forward the rest to `tb run` -------------
TASK_ARGS=()
N_TASKS=""
RUN_ALL=0
PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id|-t) TASK_ARGS+=(--task-id "$2"); shift 2 ;;
    -n|--n-tasks) N_TASKS="$2"; shift 2 ;;
    --all) RUN_ALL=1; shift ;;
    *) PASSTHRU+=("$1"); shift ;;
  esac
done
if [[ ${#TASK_ARGS[@]} -eq 0 && -z "$N_TASKS" && $RUN_ALL -eq 0 ]]; then
  # default smoke: first 1 task by duration
  N_TASKS=1
  echo "[tb-eval] no subset given; defaulting to a 1-task smoke (override with --task-id / -n / --all)"
fi
[[ -n "$N_TASKS" ]] && TASK_ARGS+=(--n-tasks "$N_TASKS")

# ---- resolve the tb CLI ---------------------------------------------------
if [[ -n "${TB:-}" ]]; then
  read -r -a TB_CMD <<<"$TB"
elif command -v tb >/dev/null 2>&1; then
  TB_CMD=(tb)
elif command -v uvx >/dev/null 2>&1; then
  echo "[tb-eval] tb not on PATH; using ephemeral 'uvx --from terminal-bench tb'"
  TB_CMD=(uvx --from terminal-bench tb)
else
  echo "[tb-eval] error: terminal-bench not installed. Install with one of:" >&2
  echo "[tb-eval]   uv tool install terminal-bench   (then 'tb' is on PATH)" >&2
  echo "[tb-eval]   pip install terminal-bench" >&2
  exit 1
fi

# ---- Docker must be present (TB requirement) ------------------------------
if ! docker info >/dev/null 2>&1; then
  echo "[tb-eval] error: Docker is not available/running. Terminal-Bench needs Docker." >&2
  echo "[tb-eval]   Start Docker Desktop (Mac) and retry." >&2
  exit 1
fi

# ---- ensure the serve is reachable (ssh -L over the tn tunnel) ------------
if [[ "${SKIP_TUNNEL:-0}" != "1" ]]; then
  if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[tb-eval] no local listener on :$PORT — opening ssh -L over the tn tunnel (127.0.0.1:12222 -> node)"
    # The tn 'arle' host == 127.0.0.1:12222 (node SSH via the keepalive tunnel).
    # Host-net pod => the serve bound to 0.0.0.0:$PORT inside the container is
    # the node's localhost:$PORT. Forward it to the Mac's localhost:$PORT.
    # Use /usr/bin/ssh (a shell may wrap `ssh`); the 12222 endpoint's host key
    # rotates as the keepalive reconnects, so don't pin it (it's a localhost
    # tunnel endpoint, not a trust boundary).
    SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
    /usr/bin/ssh -f -N \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i "$SSH_KEY" \
        -L "${PORT}:127.0.0.1:${PORT}" \
        -p 12222 root@127.0.0.1 2>/dev/null \
      || { echo "[tb-eval] ssh -L failed. Is the tn keepalive up (lsof :12222)? Is the serve running on the pod?" >&2;
           echo "[tb-eval]   start the serve on the pod: scripts/terminal_bench_serve.sh" >&2; exit 1; }
    sleep 1
  else
    echo "[tb-eval] local listener on :$PORT already present — reusing it"
  fi
fi

# ---- probe /v1/models -----------------------------------------------------
if [[ "${SKIP_PROBE:-0}" != "1" ]]; then
  echo "[tb-eval] probing ${API_BASE}/models …"
  if ! curl -sf -m 10 "${API_BASE}/models" -H "Authorization: Bearer ${OPENAI_API_KEY}" >/tmp/tb-models.json; then
    echo "[tb-eval] error: serve not reachable at ${API_BASE}. Start it on the pod:" >&2
    echo "[tb-eval]   scripts/terminal_bench_serve.sh   (then re-run; or SKIP_TUNNEL=1 if forwarding yourself)" >&2
    exit 1
  fi
  echo "[tb-eval] serve OK; /v1/models -> $(head -c 300 /tmp/tb-models.json)"
fi

mkdir -p "$OUTPUT_DIR"

echo "[tb-eval] running terminal-bench:"
echo "[tb-eval]   dataset:   ${DATASET_NAME} ${DATASET_VER}"
echo "[tb-eval]   agent:     terminus"
echo "[tb-eval]   model:     openai/${MODEL_ID}  @ ${API_BASE}"
echo "[tb-eval]   subset:    ${TASK_ARGS[*]:-<none>}"
echo "[tb-eval]   output:    ${OUTPUT_DIR}"

# Prefer --dataset-path over a scrubbed local cache: strips HIDS-tripping
# security tasks durably AND dodges the registry SSL-EOF fetch flakiness. Fall
# back to the registry name only on a cold cache (that first fetch can't be
# pre-scrubbed — rerun to scrub).
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/terminal-bench/${DATASET_NAME}/${DATASET_VER}}"
if [[ -d "$CACHE_DIR" ]]; then
  bash "$(dirname "$0")/tb_exclude_security.sh" "$CACHE_DIR"
  DATASET_ARGS=(--dataset-path "$CACHE_DIR")
else
  echo "[tb-eval] WARN: cache $CACHE_DIR absent; using registry (first fetch not pre-scrubbed)"
  DATASET_ARGS=(--dataset "${DATASET_NAME}==${DATASET_VER}")
fi

set -x
# `${arr[@]:-}` keeps empty arrays safe under `set -u` (macOS bash 3.2).
"${TB_CMD[@]}" run \
  "${DATASET_ARGS[@]}" \
  --agent terminus \
  --model "openai/${MODEL_ID}" \
  -k "api_base=${API_BASE}" \
  --n-concurrent "$N_CONCURRENT" \
  --output-path "$OUTPUT_DIR" \
  ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
  ${PASSTHRU[@]+"${PASSTHRU[@]}"}
