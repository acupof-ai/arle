#!/usr/bin/env bash
# Mock-binary test for scripts/parity_gpu_batch.sh: a clean world exits 0 with
# a TSV+markdown report; a positive FAIL and a --negative-control run that
# does not trip both force exit 1. No cargo, GPU, or prereg ledger involved.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BIN="$TMP/bin"
mkdir -p "$BIN"

# Every mock answers --kernel-build-id; behavior past that is per gate.
make_mock() {  # $1 = name, then the kind: pass | posfail | negdead | plain
    local name="$1" kind="$2"
    cat > "$BIN/$name" <<SH
#!/usr/bin/env bash
if [ "\${1:-}" = "--kernel-build-id" ]; then echo "mock-kernel-1"; exit 0; fi
case "$kind" in
  pass)
    if [ "\${1:-}" = "--negative-control" ]; then
      echo "[mock] NEGATIVE CONTROL OK (all families failed as required)"; exit 0
    fi
    echo "[mock] ALL PASS" ;;
  posfail)
    if [ "\${1:-}" = "--negative-control" ]; then
      echo "[mock] NEGATIVE CONTROL OK"; exit 0
    fi
    echo "[mock] FAIL family-x relL2=9"; exit 1 ;;
  negdead)
    if [ "\${1:-}" = "--negative-control" ]; then
      echo "[mock] ALL PASS"; exit 0   # teeth did NOT trip, still rc 0
    fi
    echo "[mock] ALL PASS" ;;
  plain)
    if [ "\${1:-}" = "--negative-control" ]; then exit 2; fi
    echo "[mock] ALL PASS" ;;
esac
SH
    chmod +x "$BIN/$name"
}

run_batch() {  # $1 = gate list, $2 = out dir
    ARLE_PARITY_BIN_DIR="$BIN" \
    ARLE_PARITY_GATE_LIST="$1" \
    ARLE_PARITY_GPU=7 \
    ARLE_PARITY_NO_NEG_ALLOWLIST="gate_d" \
    ARLE_PARITY_SKIP_PREREG=1 \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$2"
}

# ── Clean world: one pass gate (SM90-tagged) and one model-only SKIP ──
make_mock gate_a pass
make_mock gate_d pass
OUT1="$TMP/out-ok"
if ! run_batch "gate_a:sm90 gate_d:model" "$OUT1" >"$TMP/ok.log" 2>&1; then
    echo "FAIL: clean world exited non-zero" >&2; cat "$TMP/ok.log" >&2; exit 1
fi
[ -f "$OUT1/results.tsv" ] || { echo "FAIL: results.tsv missing" >&2; exit 1; }
[ -f "$OUT1/results.md" ] || { echo "FAIL: results.md missing" >&2; exit 1; }
grep -qE '^gate_a\tyes\trequired\t0\t.*ALL PASS.*\t0\t.*NEGATIVE CONTROL OK' "$OUT1/results.tsv" \
    || { echo "FAIL: gate_a row missing/incorrect" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -qE '^gate_d\tno\tallowlisted\t-\t-\t-\t-\tSKIP\t' "$OUT1/results.tsv" \
    || { echo "FAIL: model gate not recorded SKIP/allowlisted" >&2; cat "$OUT1/results.tsv" >&2; exit 1; }
grep -q 'pass=1 fail=0 skip=1' "$TMP/ok.log" \
    || { echo "FAIL: summary counts wrong" >&2; cat "$TMP/ok.log" >&2; exit 1; }
grep -q '| gate_a | yes | required |' "$OUT1/results.md" \
    || { echo "FAIL: markdown table missing gate_a" >&2; exit 1; }
grep -q 'mock-kernel-1' "$OUT1/kernel-build-id.txt" \
    || { echo "FAIL: kernel build id not recorded" >&2; exit 1; }

# ── Gate that rejects --negative-control and is NOT allowlisted: FAIL ──
make_mock gate_noneg plain
OUT_NN="$TMP/out-noneg"
if ARLE_PARITY_BIN_DIR="$BIN" ARLE_PARITY_GATE_LIST="gate_noneg" ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$OUT_NN" >"$TMP/noneg.log" 2>&1; then
    echo "FAIL: non-allowlisted gate without negative mode exited 0" >&2; cat "$TMP/noneg.log" >&2; exit 1
fi
grep -qE '^gate_noneg\tno\trequired\t0\t.*\t2\t' "$OUT_NN/results.tsv" \
    && grep -q 'no negative control' "$TMP/noneg.log" \
    || { echo "FAIL: missing-negative-mode gate not reported FAIL (no negative control)" >&2
         cat "$OUT_NN/results.tsv" >&2; cat "$TMP/noneg.log" >&2; exit 1; }

# Same gate allowlisted: negative column n/a, run green.
OUT_AL="$TMP/out-allowlist"
ARLE_PARITY_BIN_DIR="$BIN" ARLE_PARITY_GATE_LIST="gate_noneg" ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 ARLE_PARITY_NO_NEG_ALLOWLIST="gate_noneg" \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$OUT_AL" >"$TMP/allow.log" 2>&1 \
    || { echo "FAIL: allowlisted gate did not run green" >&2; cat "$TMP/allow.log" >&2; exit 1; }
grep -qE '^gate_noneg\tno\tallowlisted\t0\t.*\t-\t' "$OUT_AL/results.tsv" \
    || { echo "FAIL: allowlisted row wrong" >&2; cat "$OUT_AL/results.tsv" >&2; exit 1; }

# ── Device-gated example: rc 0 and a SKIP line but NO pass marker, in both
# the positive and --negative-control worlds. Must be SKIP (one n_skip),
# never PASS (silent) nor FAIL (false). ──
cat > "$BIN/gate_skip" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = "--kernel-build-id" ]; then echo mock-kernel-1; exit 0; fi
# No ALL PASS / NEGATIVE CONTROL OK marker in either world.
echo "SKIP: requires sm_70, device is sm_90"
exit 0
SH
chmod +x "$BIN/gate_skip"
OUT_SK="$TMP/out-skip"
if ARLE_PARITY_BIN_DIR="$BIN" ARLE_PARITY_GATE_LIST="gate_skip" ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$OUT_SK" >"$TMP/skip.log" 2>&1; then
    :
else
    echo "FAIL: SKIP world exited non-zero" >&2; cat "$TMP/skip.log" >&2; exit 1
fi
# Status column must be SKIP for both worlds, rc 0, with the reason captured.
grep -qE $'^gate_skip\tno\trequired\t0\tSKIP: requires sm_70.*\t0\tSKIP: requires sm_70.*\tSKIP\t' \
    "$OUT_SK/results.tsv" \
    || { echo "FAIL: skip gate row not SKIP/SKIP with reason" >&2
         cat "$OUT_SK/results.tsv" >&2; exit 1; }
if grep -qE 'PASS|FAIL' "$OUT_SK/results.tsv"; then
    echo "FAIL: a SKIP gate leaked a PASS/FAIL verdict" >&2
    cat "$OUT_SK/results.tsv" >&2; exit 1
fi
grep -q 'pass=0 fail=0 skip=1' "$TMP/skip.log" \
    || { echo "FAIL: skip counted wrong (must be pass=0 fail=0 skip=1)" >&2
         cat "$TMP/skip.log" >&2; exit 1; }
grep -qE '^\| gate_skip .* \| SKIP \| SKIP \|' "$OUT_SK/results.md" \
    || { echo "FAIL: markdown did not render SKIP/SKIP" >&2
         cat "$OUT_SK/results.md" >&2; exit 1; }

# ── Dirty world: positive FAIL and dead negative control ──
make_mock gate_b posfail
make_mock gate_c negdead
OUT2="$TMP/out-bad"
if run_batch "gate_a gate_b gate_c" "$OUT2" >"$TMP/bad.log" 2>&1; then
    echo "FAIL: dirty world exited 0" >&2; cat "$TMP/bad.log" >&2; exit 1
fi
grep -qE '^gate_b\t.*\tFAIL\t' "$OUT2/results.tsv" \
    || { echo "FAIL: gate_b positive failure not recorded" >&2; cat "$OUT2/results.tsv" >&2; exit 1; }
grep -qE '^gate_c\t.*\tFAIL\t' "$OUT2/results.tsv" \
    || { echo "FAIL: gate_c dead negative control not recorded" >&2; cat "$OUT2/results.tsv" >&2; exit 1; }
# gate_a must still be green in the same report.
grep -qE '^gate_a\tno\trequired\t0\t.*\t0\t' "$OUT2/results.tsv" \
    || { echo "FAIL: gate_a not green in dirty report" >&2; exit 1; }
grep -q 'family-x' "$OUT2/results.tsv" \
    || { echo "FAIL: verdict FAIL lines not captured into TSV" >&2; exit 1; }

# ── Log byte cap ──
cat > "$BIN/gate_big" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = "--kernel-build-id" ]; then echo mock-kernel-1; exit 0; fi
yes "0123456789abcdef" 2>/dev/null | head -n 200000
if [ "${1:-}" = "--negative-control" ]; then
  echo "[mock] NEGATIVE CONTROL OK"; exit 0
fi
echo "[mock] ALL PASS"
SH
chmod +x "$BIN/gate_big"
OUT3="$TMP/out-cap"
ARLE_PARITY_BIN_DIR="$BIN" ARLE_PARITY_GATE_LIST="gate_big" ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 ARLE_PARITY_LOG_CAP_BYTES=65536 \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$OUT3" >/dev/null 2>&1 \
    || { echo "FAIL: capped run exited non-zero" >&2; exit 1; }
size="$(wc -c <"$OUT3/logs/gate_big.positive.log" | tr -d ' ')"
[ "$size" -le 65536 ] || { echo "FAIL: log not capped ($size bytes)" >&2; exit 1; }

echo "PASS: parity_gpu_batch mocks (clean green; positive fail and dead negative control red; log capped)"
