#!/usr/bin/env bash
# bench_ab.sh control-arm guarantees:
#  1. --vs-no-spec refuses a treatment command carrying no spec flags, and
#     refuses BEFORE deriving/launching the control arm.
#  2. The cross-arm diff refuses a single arm (one arm has no data).
#
# Why a reverse control for #1: the guard called die() before its definition.
# bench_ab.sh runs under `set -uo pipefail` (no -e), so an undefined die in
# `[[ ... ]] || die` only prints "command not found" (127) and execution
# CONTINUES — both arms then ran the identical command. Moving die() below its
# first use must reproduce that bypass; the red signal is the control-arm
# derivation being printed (plus the command-not-found line), NOT the exit code
# (a later, by-then-defined die also exits 3 on the dead fake server).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

NOSPEC='some-binary serve --model-path M --port 8000 &'
ARGS=(lbl --vs-no-spec --cmd-treatment "$NOSPEC" --seconds-per-concurrency 5)

# ---- 1a. GREEN: the fixed script refuses at the guard, exit 3 --------------
rc=0
( cd "$TMP" && bash "$ROOT/scripts/bench_ab.sh" "${ARGS[@]}" ) > "$TMP/out" 2>&1 || rc=$?
[ "$rc" = 3 ] || { echo "FAIL: expected exit 3, got $rc" >&2; cat "$TMP/out" >&2; exit 1; }
grep -q 'no spec flags found in treatment command' "$TMP/out" \
    || { echo "FAIL: guard message missing" >&2; cat "$TMP/out" >&2; exit 1; }
! grep -q 'control arm derived by stripping spec flags' "$TMP/out" \
    || { echo "FAIL: guard did not stop the run" >&2; cat "$TMP/out" >&2; exit 1; }

# ---- 1b. REVERSE CONTROL: move die() below its first use --------------------
# The guard must then be inert: the script prints "command not found" AND walks
# straight past the guard into deriving the control arm. Kill it as soon as
# that bypass marker appears, so the dead fake server is never polled.
DIE='die() { echo "error: $*" >&2; exit 3; }'
python3 - "$ROOT/scripts/bench_ab.sh" "$TMP/inert.sh" "$DIE" <<'PY'
import sys, pathlib
src, dst, die = sys.argv[1], sys.argv[2], sys.argv[3] + "\n"
s = pathlib.Path(src).read_text()
assert s.count(die) == 1, "die() definition not found verbatim"
s = s.replace("\n" + die, "\n", 1)
marker = "# Kill only serves:"
assert marker in s
s = s.replace(marker, die + "\n" + marker, 1)
pathlib.Path(dst).write_text(s)
PY

( cd "$TMP" && bash "$TMP/inert.sh" "${ARGS[@]}" ) > "$TMP/inert.out" 2>&1 &
pid=$!
bypassed=false
for _ in $(seq 1 30); do
    if grep -q 'control arm derived by stripping spec flags' "$TMP/inert.out" 2>/dev/null; then
        bypassed=true; break
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
kill -9 "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
[ "$bypassed" = true ] \
    || { echo "FAIL: reverse control did not bypass the guard (test proves nothing)" >&2; cat "$TMP/inert.out" >&2; exit 1; }
grep -q 'die: command not found' "$TMP/inert.out" \
    || { echo "FAIL: reverse control did not reproduce the undefined-die path" >&2; cat "$TMP/inert.out" >&2; exit 1; }

# ---- 2. The diff refuses a single arm ---------------------------------------
DIFF="$ROOT/scripts/bench_ab_diff.py"
mkdir -p "$TMP/A" "$TMP/B"
cat > "$TMP/A/bench_throughput.json" <<'JSON'
{"points":[{"summary":{"concurrency":1,"ttft":{"p50_ms":10.0},"itl":{"p50_ms":20.0,"mean_ms":20.0}}}]}
JSON

# 2a. B has no data -> nonzero, explicit refusal, no diff file.
rm -f "$TMP/diff.md"
if python3 "$DIFF" "$TMP/A" "$TMP/B" A B "$TMP/diff.md" > "$TMP/diff.out" 2>"$TMP/diff.err"; then
    echo "FAIL: diff accepted a missing B arm" >&2; exit 1
fi
grep -q 'refusing to diff without both arms' "$TMP/diff.err" \
    || { echo "FAIL: single-arm refusal message missing" >&2; cat "$TMP/diff.err" >&2; exit 1; }
[ ! -e "$TMP/diff.md" ] || { echo "FAIL: diff file written despite a missing arm" >&2; exit 1; }

# 2b. A missing, B present -> refuses the other direction too.
if python3 "$DIFF" "$TMP/B" "$TMP/A" B A "$TMP/diff2.md" >/dev/null 2>"$TMP/diff2.err"; then
    echo "FAIL: diff accepted a missing A arm" >&2; exit 1
fi
grep -q 'refusing to diff without both arms' "$TMP/diff2.err" \
    || { echo "FAIL: reverse single-arm refusal message missing" >&2; exit 1; }

# 2c. Both arms present -> exit 0 and a diff file is produced.
cat > "$TMP/B/bench_throughput.json" <<'JSON'
{"points":[{"summary":{"concurrency":1,"ttft":{"p50_ms":9.0},"itl":{"p50_ms":18.0,"mean_ms":18.0}}}]}
JSON
python3 "$DIFF" "$TMP/A" "$TMP/B" A B "$TMP/diffok.md" > /dev/null
grep -q '^# A/B diff — A vs B' "$TMP/diffok.md" \
    || { echo "FAIL: both-arm run did not write the diff" >&2; exit 1; }

echo "PASS: bench_ab control arm (no-spec guard + reverse control + single-arm diff refusal)"
