#!/usr/bin/env bash
# Capture one real DSv4 decode token under Nsight Systems.
#
# The server must include the CUDA profiler signal handlers from infer/src/main.rs.
# This wrapper starts ARLE under `nsys profile --capture-range=cudaProfilerApi`,
# sends SIGUSR1/SIGUSR2 only to the real infer process, then summarizes CUDA
# runtime APIs and kernels inside `step_decode_kernel_launch` NVTX ranges.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/DeepSeek-V4-Flash}"
SERVER_BIN="${SERVER_BIN:-target/release/infer}"
OUT="${OUT:-docs/trace-artifacts/2026-05-15-dsv4-deepep/nsys-single-decode-token-attention-scratch}"
PORT="${PORT:-18188}"
MAX_TOKENS="${MAX_TOKENS:-2}"
PROMPT="${PROMPT:-Compute 137 + 269. Answer with the number only.}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-V4-Flash}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --out DIR          output directory (default: ${OUT})
  --port PORT        HTTP port (default: ${PORT})
  --model-path DIR   model path (default: ${MODEL_PATH})
  --server-bin PATH  infer binary (default: ${SERVER_BIN})
  --prompt TEXT      prompt for warmup/profile request
  --max-tokens N     request max_tokens; must be >=2 for a decode range
  -h, --help         show this help

Environment:
  CUDA_VISIBLE_DEVICES, INFER_CUDA_DEVICES, ARLE_DSV4_MOE_BACKEND,
  ARLE_DSV4_INCREMENTAL_KV, and ARLE_DSV4_FUSED_DISPATCH_PAYLOAD are read
  from the environment and defaulted for 8xH20 DSv4 DeepEP runs.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --server-bin) SERVER_BIN="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if (( MAX_TOKENS < 2 )); then
    echo "error: --max-tokens must be >=2; max_tokens=1 exits from prefill" >&2
    exit 2
fi

command -v nsys >/dev/null || { echo "error: nsys not found on PATH" >&2; exit 5; }
command -v python3 >/dev/null || { echo "error: python3 not found on PATH" >&2; exit 5; }
[[ -x "$SERVER_BIN" ]] || { echo "error: missing executable server binary: $SERVER_BIN" >&2; exit 5; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export INFER_CUDA_DEVICES="${INFER_CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export ARLE_DSV4_MOE_BACKEND="${ARLE_DSV4_MOE_BACKEND:-deepep}"
export ARLE_DSV4_INCREMENTAL_KV="${ARLE_DSV4_INCREMENTAL_KV:-1}"
export ARLE_DSV4_FUSED_DISPATCH_PAYLOAD="${ARLE_DSV4_FUSED_DISPATCH_PAYLOAD:-1}"

rm -rf "$OUT"
mkdir -p "$OUT"

cat > "$OUT/request_once.py" <<'PY'
import json
import os
import sys
import time
import urllib.request

port = int(sys.argv[1])
out = sys.argv[2]
max_tokens = int(sys.argv[3])
model = os.environ.get("MODEL_NAME", "DeepSeek-V4-Flash")
prompt = os.environ["PROMPT"]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": False,
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=240) as resp:
    body = resp.read()
elapsed = time.perf_counter() - t0
parsed = json.loads(body)
result = {
    "status": 200,
    "elapsed_s": elapsed,
    "usage": parsed.get("usage"),
    "text": parsed["choices"][0]["message"]["content"],
}
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
PY

cat > "$OUT/analyze_decode.py" <<'PY'
import csv
import json
import sqlite3
from pathlib import Path

out = Path(__file__).resolve().parent
conn = sqlite3.connect(out / "trace.sqlite")
cur = conn.cursor()
tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "NVTX_EVENTS" not in tables:
    raise SystemExit("NVTX_EVENTS table missing")

columns = {row[1] for row in cur.execute("PRAGMA table_info(NVTX_EVENTS)")}
if "text" in columns:
    ranges = cur.execute(
        """
        SELECT start, end FROM NVTX_EVENTS
        WHERE text = 'step_decode_kernel_launch' AND end IS NOT NULL
        ORDER BY start
        """
    ).fetchall()
else:
    ranges = []
if not ranges:
    ranges = [
        (start, end)
        for _name, start, end in cur.execute(
            """
            SELECT n.value, e.start, e.end FROM NVTX_EVENTS e
            JOIN StringIds n ON e.textId = n.id
            WHERE n.value = 'step_decode_kernel_launch' AND e.end IS NOT NULL
            ORDER BY e.start
            """
        ).fetchall()
    ]
if not ranges:
    raise SystemExit("decode NVTX ranges missing")

cur.execute("CREATE TEMP TABLE decode_ranges_tmp(start INTEGER, end INTEGER)")
cur.executemany("INSERT INTO decode_ranges_tmp VALUES (?, ?)", ranges)

runtime_rows = cur.execute(
    """
    WITH hits AS (
        SELECT DISTINCT r.rowid AS rid,
               COALESCE(s.value, printf('%d', r.nameId)) AS name,
               (r.end-r.start)/1e6 AS time_ms
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        LEFT JOIN StringIds s ON r.nameId = s.id
        JOIN decode_ranges_tmp d ON r.start >= d.start AND r.end <= d.end
    )
    SELECT name,
           COUNT(*) AS calls,
           SUM(time_ms) AS total_ms,
           AVG(time_ms) AS avg_ms
    FROM hits
    GROUP BY 1 ORDER BY total_ms DESC LIMIT 40
    """
).fetchall()

kernel_rows = cur.execute(
    """
    WITH hits AS (
        SELECT DISTINCT k.rowid AS kid,
               COALESCE(s.value, k.demangledName, k.shortName) AS name,
               (k.end-k.start)/1e6 AS time_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
        JOIN decode_ranges_tmp d ON k.start >= d.start AND k.end <= d.end
    )
    SELECT name,
           COUNT(*) AS calls,
           SUM(time_ms) AS total_ms,
           AVG(time_ms) AS avg_ms
    FROM hits
    GROUP BY 1 ORDER BY total_ms DESC LIMIT 50
    """
).fetchall()

range_ms = [(end - start) / 1e6 for start, end in ranges]
wave_ranges = [ranges]
if len(ranges) % 8 == 0:
    wave_ranges = [ranges[i : i + 8] for i in range(0, len(ranges), 8)]
wave_wall_ms = [
    (max(end for _start, end in wave) - min(start for start, _end in wave)) / 1e6
    for wave in wave_ranges
]
range_count = len(ranges)
summary = {
    "capture": "single profile request, filtered to step_decode_kernel_launch NVTX ranges",
    "decode_ranges": len(ranges),
    "decode_waves": (len(ranges) // 8) if len(ranges) % 8 == 0 else None,
    "decode_wave_wall_ms": wave_wall_ms,
    "decode_wave_wall_ms_max": max(wave_wall_ms),
    "decode_range_ms_min": min(range_ms),
    "decode_range_ms_p50": sorted(range_ms)[len(range_ms) // 2],
    "decode_range_ms_max": max(range_ms),
    "top_runtime_apis": [
        {
            "name": name,
            "time_ms_per_rank_range": total / range_count,
            "total_time_ms_all_ranges": total,
            "calls": calls,
            "avg_ms": avg,
        }
        for name, calls, total, avg in runtime_rows[:15]
    ],
    "top_kernels": [
        {
            "name": name,
            "time_ms_per_rank_range": total / range_count,
            "total_time_ms_all_ranges": total,
            "calls": calls,
            "avg_ms": avg,
        }
        for name, calls, total, avg in kernel_rows[:20]
    ],
}

(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with (out / "decode-only-runtime-api-top.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["name", "calls", "total_ms", "avg_ms"])
    writer.writerows(runtime_rows)
with (out / "decode-only-kernel-top.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["name", "calls", "total_ms", "avg_ms"])
    writer.writerows(kernel_rows)

print(json.dumps(summary, indent=2))
PY

NSYS_CMD=(
    nsys profile
    --trace cuda,nvtx,osrt
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --export=sqlite
    --kill=none
    --force-overwrite=true
    --output "$OUT/trace"
    "$SERVER_BIN"
    --model-path "$MODEL_PATH"
    --port "$PORT"
    --num-slots 1
    --max-seq-len 4096
    --mem-fraction-static 0.10
    --kv-cache-dtype fp8
    --deepseek-distributed-layers 43
)

{
    printf 'CUDA_VISIBLE_DEVICES=%q INFER_CUDA_DEVICES=%q ARLE_DSV4_MOE_BACKEND=%q ARLE_DSV4_INCREMENTAL_KV=%q ARLE_DSV4_FUSED_DISPATCH_PAYLOAD=%q ' \
        "$CUDA_VISIBLE_DEVICES" "$INFER_CUDA_DEVICES" "$ARLE_DSV4_MOE_BACKEND" \
        "$ARLE_DSV4_INCREMENTAL_KV" "$ARLE_DSV4_FUSED_DISPATCH_PAYLOAD"
    printf 'nsys'
    for arg in "${NSYS_CMD[@]:1}"; do printf ' %q' "$arg"; done
    printf '\n'
} > "$OUT/command.txt"

resolve_server_pid() {
    ps -eo pid=,args= | awk -v port="$PORT" '
        $0 ~ /target\/release\/infer/ &&
        $0 ~ ("--port " port) &&
        $0 !~ /nsys profile/ &&
        $0 !~ /profile_dsv4_single_decode_nsys/ &&
        $0 !~ /awk -v/ {
            print $1
            exit
        }'
}

cleanup() {
    set +e
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ -n "${NSYS_PID:-}" ]] && kill -0 "$NSYS_PID" 2>/dev/null; then
        kill -TERM "$NSYS_PID" 2>/dev/null || true
        wait "$NSYS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo ">>> start nsys + infer"
"${NSYS_CMD[@]}" > "$OUT/server.log" 2>&1 &
NSYS_PID=$!
echo "$NSYS_PID" > "$OUT/nsys.pid"

echo ">>> wait for /v1/models on port ${PORT}"
for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" > "$OUT/models.json"; then
        break
    fi
    if ! kill -0 "$NSYS_PID" 2>/dev/null; then
        echo "error: nsys exited before ready" >&2
        tail -200 "$OUT/server.log" >&2 || true
        exit 4
    fi
    sleep 1
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" > "$OUT/models.json"

SERVER_PID="$(resolve_server_pid)"
if [[ -z "$SERVER_PID" ]]; then
    echo "error: failed to resolve infer pid" >&2
    ps -eo pid,ppid,args | grep -E "nsys profile|target/release/infer|${PORT}" >&2 || true
    exit 4
fi
echo "$SERVER_PID" > "$OUT/server.pid"
echo ">>> infer pid: ${SERVER_PID}"

export PROMPT MODEL_NAME
python3 "$OUT/request_once.py" "$PORT" "$OUT/warmup-decode.json" "$MAX_TOKENS"
kill -USR1 "$SERVER_PID"
sleep 0.2
python3 "$OUT/request_once.py" "$PORT" "$OUT/profile-request.json" "$MAX_TOKENS"
sleep 0.2
kill -USR2 "$SERVER_PID"
sleep 2
kill -TERM "$SERVER_PID" 2>/dev/null || true
wait "$NSYS_PID" 2>/dev/null || true
trap - EXIT INT TERM
rm -f "$OUT/nsys.pid" "$OUT/server.pid"

if [[ ! -f "$OUT/trace.nsys-rep" || ! -f "$OUT/trace.sqlite" ]]; then
    echo "error: nsys did not produce trace outputs" >&2
    tail -200 "$OUT/server.log" >&2 || true
    exit 4
fi

nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv --output "$OUT/stats" "$OUT/trace.nsys-rep" > "$OUT/stats.log" 2>&1 || true
python3 "$OUT/analyze_decode.py"
gzip -9f "$OUT/trace.nsys-rep" "$OUT/trace.sqlite" "$OUT/server.log"

echo "PROFILE_REQUEST"
cat "$OUT/profile-request.json"
echo "SUMMARY"
cat "$OUT/summary.json"
