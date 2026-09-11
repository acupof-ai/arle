#!/usr/bin/env bash
# One-command runtime verification for the FlashQLA / DSpark batched-verify
# series (#327/#329/S34): base SHA vs treatment SHA, on the pod, across
# attn_tp 1/2/4/8 as free SM90 GPUs allow.
#
# For EACH arm and attn_tp it runs, on a Qwen3.8-27B-DSpark serve:
#   (a) correctness: scripts/lever_gate.sh needle ladder x3 + concurrent arm,
#       base arm seeds the envelope, treatment is compared against it;
#   (b) matched TTFT / prefill tok/s A/B on long prompts (bench_throughput.py);
#   (c) DSpark draft acceptance A/B at c=1 and c=8 (bench_dspark_accept.py,
#       the existing /v1/stats accepted/drafted measurement; c=8 = eight
#       concurrent clients). Reuses the #292 measurement, does not invent one.
#
# Both SHAs are built --release (a bench-labelled pod-remote-run requires a
# release build receipt) in detached git worktrees. A prereg row is opened and
# closed per arm. Writes results.tsv / results.md. Exits non-zero on any
# correctness (needle/lever) failure; TTFT and acceptance are recorded, not
# gated — this batch MEASURES them.
#
# Required env (model paths; no default invented — nothing is mounted at
# /data00 on every box):
#   MODEL                trunk checkpoint dir (Qwen3.8-27B)
#   DRAFT_MODEL          DSpark draft head dir (.../Qwen3.8-27B-DSpark)
# Optional:
#   ARLE_DSV_TPS         attn_tp grid, default "1 2 4 8"
#   ARLE_DSV_FREE_GPUS   comma list of usable GPU indices (default: discover)
#   LONG_PROMPTS_JSONL   long-prompt file for the matched prefill A/B
#   BENCH_SECONDS        seconds per prefill cell (default 60)
#   ACCEPT_REQUESTS      requests per acceptance client (default 30)
#   EXTRA_SERVE_FLAGS    extra flags for every serve (both arms)
#   GDR_CHUNKED=0        #300 fallback A/B: pass --qwen35-gdr-chunked false to
#                        both arms and verify from /v1/stats that gdr_fq=0
#
# Test seams (shell test only):
#   ARLE_DSV_BIN_BASE / ARLE_DSV_BIN_TREAT  skip git+build, use these binaries
#   ARLE_DSV_FREE_GPUS  GPUs to use (also skips the pick-gpu discovery query)
#   ARLE_DSV_NO_CLAIM=1 skip pick-gpu reserve-set
#   ARLE_DSV_LEVER      replacement lever_gate.sh
#   ARLE_DSV_TOOLS_DIR  dir with needle_gate.py/bench_throughput.py/
#                       bench_dspark_accept.py
#   ARLE_DSV_SKIP_PREREG=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SHA="${1:?usage: dspark_flashqla_verify.sh <base-sha> <treatment-sha> <out-dir>}"
TREAT_SHA="${2:?missing treatment-sha}"
OUT="${3:?missing out-dir}"
TPS="${ARLE_DSV_TPS:-1 2 4 8}"
TOOLS="${ARLE_DSV_TOOLS_DIR:-$ROOT/scripts}"
LEVER="${ARLE_DSV_LEVER:-$ROOT/scripts/lever_gate.sh}"
BENCH_SECONDS="${BENCH_SECONDS:-60}"
ACCEPT_REQUESTS="${ACCEPT_REQUESTS:-30}"
mkdir -p "$OUT/logs"

# Long prompts for the matched TTFT/prefill cell. A caller-provided file wins;
# otherwise generate one 8K-token session set with the existing generator so
# the cell measures prefill rather than the short canned prompts.
LONG_PROMPTS_JSONL="${LONG_PROMPTS_JSONL:-}"
if [ -z "$LONG_PROMPTS_JSONL" ]; then
    generated="$OUT/long-prompts.jsonl"
    if python3 "$TOOLS/gen_bench_prompts.py" "$generated" 4 8000 32 1 \
        >"$OUT/logs/gen-prompts.log" 2>&1; then
        LONG_PROMPTS_JSONL="$generated"
    else
        echo "[verify] long-prompt generation failed; using synthetic prompts" >&2
    fi
fi

if [ "${ARLE_DSV_SKIP_PREREG:-0}" != 1 ]; then
    : "${MODEL:?MODEL (trunk checkpoint dir) required}"
    : "${DRAFT_MODEL:?DRAFT_MODEL (DSpark draft head dir) required}"
fi
# shellcheck disable=SC2206  # intentional word-split of caller passthrough
EXTRA=( ${EXTRA_SERVE_FLAGS:-} )
# #300 fallback A/B: GDR_CHUNKED=0 appends `--qwen35-gdr-chunked false` to BOTH
# arms (cli/src/args.rs:829-830 — the only switch; there is no env var). The
# effect is verified from /v1/stats op_timing, not assumed.
GDR_FLAG=()
case "${GDR_CHUNKED:-}" in 0|false|no) GDR_FLAG=(--qwen35-gdr-chunked false) ;; esac
# ARLE_CUDA_PROFILE=1 makes every serve report per-op timings on /v1/stats,
# which is how the GDR path (gdr_fq vs gdr_recurrent) is verified.
export ARLE_CUDA_PROFILE=1

NAME="dspark-flashqla-verify-$(date +%Y%m%d%H%M%S)"
prereg() {  # $1 = start|close $2 = arm [rest = done fields]
    [ "${ARLE_DSV_SKIP_PREREG:-0}" = 1 ] && return 0
    if [ "$1" = start ]; then
        python3 "$ROOT/scripts/prereg.py" start \
            --name "$NAME-$2" \
            --cmd "scripts/dspark_flashqla_verify.sh $BASE_SHA $TREAT_SHA $OUT" \
            --hypothesis "FlashQLA/DSpark treatment ($TREAT_SHA) stays inside the base needle envelope at attn_tp 1/2/4/8; TTFT/prefill and c=1/c=8 acceptance recorded" \
            >/dev/null
    else
        # shellcheck disable=SC1010  # "done" is the literal prereg.py subcommand
        python3 "$ROOT/scripts/prereg.py" done --name "$NAME-$2" "${@:3}" >/dev/null || true
    fi
}

# ── Materialize + release-build each arm ───────────────────────────────────
TREES="$OUT/trees"
arm_binary() {  # $1 = arm name, $2 = sha
    if [ "$1" = base ] && [ -n "${ARLE_DSV_BIN_BASE:-}" ]; then echo "$ARLE_DSV_BIN_BASE"; return; fi
    if [ "$1" = treatment ] && [ -n "${ARLE_DSV_BIN_TREAT:-}" ]; then echo "$ARLE_DSV_BIN_TREAT"; return; fi
    local tree="$TREES/$1"
    if [ ! -x "$tree/target/release/arle" ]; then
        mkdir -p "$TREES"
        [ -d "$tree" ] || git -C "$ROOT" worktree add --detach "$tree" "$2" >/dev/null
        echo "[verify] $1: release build at $2" >&2
        (cd "$tree" && cargo build --release -p cli) >"$OUT/logs/build-$1.log" 2>&1
    fi
    echo "$tree/target/release/arle"
}

BIN_BASE="$(arm_binary base "$BASE_SHA")"
BIN_TREAT="$(arm_binary treatment "$TREAT_SHA")"

# ── Free GPUs (pick-gpu.sh claims, same scheme as every pod run) ────────────
free_gpus_csv() {
    if [ -n "${ARLE_DSV_FREE_GPUS:-}" ]; then echo "$ARLE_DSV_FREE_GPUS"; return; fi
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 <= 2000 {print $1}' | sort -n | paste -sd, -
}
ALL_FREE="$(free_gpus_csv)"
[ -n "$ALL_FREE" ] || { echo "[verify] no free GPUs" >&2; exit 3; }
IFS=',' read -r -a FREE_IDX <<< "$ALL_FREE"

claim_set() {  # $1 = csv ; echoes op id on success
    local op="dsv-verify-$$-$RANDOM"
    if [ "${ARLE_DSV_NO_CLAIM:-0}" = 1 ]; then echo "$op"; return; fi
    ARLE_OP_ID="$op" ARLE_OWNER="$(id -u):$(id -un)" ARLE_CLAIM_PID="$$" \
        bash "$ROOT/scripts/pick-gpu.sh" reserve-set "$1" >/dev/null && echo "$op"
}
release_set() {  # $1 = csv $2 = op
    [ "${ARLE_DSV_NO_CLAIM:-0}" = 1 ] && return 0
    local IFS=','
    for g in $1; do
        c="${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$g"
        [ "$(awk -F= '$1=="op" {print $2}' "$c" 2>/dev/null)" = "$2" ] && rm -f "$c"
    done
}

TSV="$OUT/results.tsv"; MD="$OUT/results.md"
printf 'arm\tattn_tp\tphase\tmetric\tstatus\tdetail\n' >"$TSV"
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >>"$TSV"; }

# bench_throughput.py reports prefill as TTFT (no blended tok/s field):
# points[0].summary.ttft.{mean,p50}_ms plus mean prompt length.
prefill_metrics() {
    python3 - "$1" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
    s = r["points"][-1]["summary"]
    t = s.get("ttft") or {}
    mean = t.get("mean_ms"); p50 = t.get("p50_ms")
    plen = s["prompt_tokens"] / s["complete"] if s.get("complete") else 0
    print(f"ttft_mean_ms={mean:.1f} ttft_p50_ms={p50:.1f} prompt_tok_mean={plen:.0f}"
          if mean is not None else "ttft_unavailable")
except (OSError, KeyError, IndexError, ValueError):
    print("parse_failed")
PY
}

# ── One serve for the bench phases (lever boots its own) ────────────────────
serve_up() { for _ in $(seq 1 120); do curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

run_arm_tp() {  # $1=arm $2=bin $3=tp $4=gpu-csv $5=baseline-log-or-""
    local arm="$1" bin="$2" tp="$3" gpus="$4" baseline="$5"
    local port=$((18300 + RANDOM % 200))
    local logdir="$OUT/logs/${arm}-tp${tp}"; mkdir -p "$logdir"
    local label="dsv-${arm}-tp${tp}"

    # Production TP launch is the CLI flag --tensor-parallel-size N
    # (scripts/serve_dsv4_tp4.sh; crates/cli/src/args.rs:741). The multiproc
    # coordinator spawns N workers and sets INFER_TP_SIZE per rank itself
    # (serve_multiproc.rs); CUDA_VISIBLE_DEVICES remaps the physical claim to
    # logical 0..N-1. attn_tp = N (CP/DP default 1).
    local tp_flag=(--tensor-parallel-size "$tp")
    local serve_flags=(--spec-type dspark --mtp-draft-model "$DRAFT_MODEL" "${tp_flag[@]}"
        ${GDR_FLAG[@]+"${GDR_FLAG[@]}"} ${EXTRA[@]+"${EXTRA[@]}"})

    # (a) correctness: lever boots and tears down its own serve.
    local lever_rc=0 base_env=()
    [ -z "$baseline" ] || base_env=(BASELINE_LOG="$baseline")
    env CUDA_VISIBLE_DEVICES="$gpus" PORT="$port" \
        MODEL="$MODEL" GATE_PROFILE=generic BIN="$bin" \
        LEVER_GATE_SKIP_TEMP=1 LEVER_GATE_ALLOW_NO_BASELINE=1 \
        OUT="$logdir/needle.log" STATS_OUT="$logdir/stats.json" \
        SERVE_FLAGS="${serve_flags[*]}" \
        ${base_env[@]+"${base_env[@]}"} bash "$LEVER" "$label" >"$logdir/lever.log" 2>&1 || lever_rc=$?
    if [ "$lever_rc" -eq 0 ]; then
        row "$arm" "$tp" needle "ladder x3 + concurrent" PASS "$logdir/needle.log"
    else
        row "$arm" "$tp" needle "-" FAIL "lever rc=$lever_rc $logdir/lever.log"
    fi

    # Bench serve on a second port (lever's serve is gone by now).
    local bport=$((port + 1)) serve_pid
    CUDA_VISIBLE_DEVICES="$gpus" \
        RUST_LOG=info "$bin" serve --backend cuda --model-path "$MODEL" --port "$bport" \
        "${serve_flags[@]}" >"$logdir/serve.log" 2>&1 &
    serve_pid=$!
    trap 'kill "$serve_pid" 2>/dev/null || true' RETURN
    if ! serve_up "$bport"; then
        row "$arm" "$tp" launch "-" FAIL "bench serve never ready $logdir/serve.log"
        row "$arm" "$tp" prefill "-" FAIL "serve down"
        row "$arm" "$tp" accept-c1 "-" FAIL "serve down"
        row "$arm" "$tp" accept-c8 "-" FAIL "serve down"
        kill "$serve_pid" 2>/dev/null || true
        return 0
    fi

    # Confirm the serve actually came up at the requested TP — a silent
    # fallback to single-process would invalidate every measurement in the cell.
    # Coordinator logs "all N worker engines ready"; tp=1 is single-process.
    local obs_tp=""
    if [ "$tp" -gt 1 ]; then
        obs_tp="$(grep -oE 'all [0-9]+ worker engines ready' "$logdir/serve.log" | head -1 | grep -oE '[0-9]+' || true)"
    else
        grep -q 'serving single-process' "$logdir/serve.log" && obs_tp=1
    fi
    if [ "$obs_tp" = "$tp" ]; then
        row "$arm" "$tp" launch "observed workers=$obs_tp (attn_tp=$tp)" PASS "$logdir/serve.log"
    else
        row "$arm" "$tp" launch "expected tp=$tp observed=${obs_tp:-none}" FAIL "$logdir/serve.log"
        kill "$serve_pid" 2>/dev/null || true
        trap - RETURN
        return 0
    fi

    # (b) matched TTFT / prefill on long prompts, c=1 cell (prefill-bound).
    local tp_args=(--url "http://127.0.0.1:$bport" --concurrency-grid 1
        --seconds-per-concurrency "$BENCH_SECONDS" --max-tokens 32
        --output "$logdir/throughput")
    if [ -n "${LONG_PROMPTS_JSONL:-}" ]; then tp_args+=(--prompts-jsonl "$LONG_PROMPTS_JSONL"); fi
    if python3 "$TOOLS/bench_throughput.py" "${tp_args[@]}" >"$logdir/throughput.log" 2>&1; then
        local metrics
        metrics="$(prefill_metrics "$logdir/throughput.json")"
        row "$arm" "$tp" prefill "$metrics" PASS "$logdir/throughput.json"
    else
        row "$arm" "$tp" prefill "-" FAIL "$logdir/throughput.log"
    fi

    # (c) acceptance, existing bench_dspark_accept.py measurement. One process
    # per c with a thread pool and a SINGLE global stats pair (the tool's
    # counters are server-global; separate processes would overcount).
    accept_run() {  # $1=c (1|8)
        local c="$1"
        local o="$logdir/accept-c${c}.json"
        if python3 "$TOOLS/bench_dspark_accept.py" --port "$bport" \
            --concurrency "$c" --measure-requests "$ACCEPT_REQUESTS" --max-tokens 64 \
            --output "$o" >"$logdir/accept-c${c}.log" 2>&1; then
            python3 - "$o" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"{r['accepted']}/{r['drafted']}")
PY
        else echo FAIL; fi
    }
    local r1 r8
    r1="$(accept_run 1)"
    r8="$(accept_run 8)"
    if [ "$r1" != FAIL ]; then row "$arm" "$tp" accept-c1 "$r1" PASS "$logdir/accept-c1.json"; else row "$arm" "$tp" accept-c1 "-" FAIL "$logdir/accept-c1.log"; fi
    if [ "$r8" != FAIL ]; then row "$arm" "$tp" accept-c8 "$r8" PASS "$logdir/accept-c8.json"; else row "$arm" "$tp" accept-c8 "-" FAIL "$logdir/accept-c8.log"; fi

    # GDR path effect check from /v1/stats op_timing (ARLE_CUDA_PROFILE=1):
    # chunked FlashQLA logs "linear/gdr_fq", the varlen fallback logs
    # "linear/gdr_recurrent". GDR_CHUNKED=0 must force gdr_fq count to 0, so
    # a misspelled switch can't pass as a silent no-op.
    local gdr_path
    gdr_path="$(python3 - "$bport" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/v1/stats", timeout=10) as r:
    ops = json.load(r).get("op_timing", {}).get("ops", [])
fq = sum(o.get("count", 0) for o in ops if o.get("name") == "linear/gdr_fq")
rec = sum(o.get("count", 0) for o in ops if o.get("name") == "linear/gdr_recurrent")
print(f"gdr_fq={fq} gdr_recurrent={rec}")
PY
)"
    if [ "${#GDR_FLAG[@]}" -gt 0 ]; then
        local fqcnt="${gdr_path#gdr_fq=}"; fqcnt="${fqcnt%% *}"
        if [ "$fqcnt" = 0 ]; then
            row "$arm" "$tp" gdr-path "$gdr_path (chunked forced off)" PASS "$logdir/serve.log"
        else
            row "$arm" "$tp" gdr-path "$gdr_path (expected gdr_fq=0)" FAIL "$logdir/serve.log"
        fi
    else
        row "$arm" "$tp" gdr-path "$gdr_path" INFO "$logdir/serve.log"
    fi

    kill "$serve_pid" 2>/dev/null || true; wait "$serve_pid" 2>/dev/null || true
    trap - RETURN
}

CORRECT_FAIL=0
prereg start base; prereg start treatment
for tp in $TPS; do
    [ "$tp" -le "${#FREE_IDX[@]}" ] || { echo "[verify] attn_tp=$tp needs $tp free GPUs, have ${#FREE_IDX[@]} — SKIP" >&2
        row base "$tp" all "-" SKIP "insufficient free GPUs"; continue; }
    csv="$(IFS=','; echo "${FREE_IDX[*]:0:$tp}")"
    op="$(claim_set "$csv")" || { echo "[verify] GPUs $csv not claimable — SKIP tp=$tp" >&2
        row base "$tp" all "-" SKIP "GPUs busy"; continue; }
    echo "[verify] attn_tp=$tp gpus=$csv" >&2

    run_arm_tp base "$BIN_BASE" "$tp" "$csv" ""
    baseline="$OUT/logs/base-tp${tp}/needle.log"
    [ -f "$baseline" ] || baseline=""
    run_arm_tp treatment "$BIN_TREAT" "$tp" "$csv" "$baseline"

    release_set "$csv" "$op"
    if awk -F'\t' -v tp="$tp" 'NR>1 && $2==tp && ($3=="needle" || $3=="launch" || $3=="gdr-path") && $5=="FAIL"{bad=1} END{exit !bad}' "$TSV"; then
        CORRECT_FAIL=1
    fi
done

# ── results.md ─────────────────────────────────────────────────────────────
{
    echo "# FlashQLA / DSpark verify A/B — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "base \`$BASE_SHA\` vs treatment \`$TREAT_SHA\` · gpus=${ALL_FREE}"
    echo
    echo "| arm | attn_tp | phase | metric | status | detail |"
    echo "|---|---|---|---|---|---|"
    tail -n +2 "$TSV" | while IFS=$'\t' read -r arm t phase metric st detail; do
        echo "| $arm | $t | $phase | $metric | $st | $detail |"
    done
    echo
    echo "Needle FAIL = correctness failure (exit 1). prefill/accept rows are"
    echo "recorded measurements, not gates."
} >"$MD"

if [ "$CORRECT_FAIL" -eq 0 ]; then
    prereg close base --status ok --result "see $TSV" --finding "needle/lever green at every runnable attn_tp" --decision "correctness holds; read TTFT/acceptance deltas in results.md"
    prereg close treatment --status ok --result "see $TSV" --finding "treatment inside the base envelope" --decision "GPU runtime verification passed"
    echo "[verify] correctness PASS — $TSV $MD"
    exit 0
fi
prereg close base --status ok --result "see $TSV" --finding "base arm completed" --decision "compare against failing treatment"
prereg close treatment --status rejected --result "needle FAIL in $TSV" --finding "treatment outside the base needle envelope" --decision "do not ship"
echo "[verify] correctness FAIL — $TSV" >&2
exit 1
