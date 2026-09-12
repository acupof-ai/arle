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

run_batch() {  # $1 = gate list, $2 = out dir; extra env via HOST_UNITS_RC
    ARLE_PARITY_BIN_DIR="$BIN" \
    ARLE_PARITY_GATE_LIST="$1" \
    ARLE_PARITY_GPU=7 \
    ARLE_PARITY_NO_NEG_ALLOWLIST="gate_d" \
    ARLE_PARITY_SKIP_PREREG=1 \
    ARLE_PARITY_HOST_UNITS_CMD="${ARLE_PARITY_HOST_UNITS_CMD:-true}" \
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
grep -q 'host-only unit tests PASS' "$TMP/ok.log" \
    || { echo "FAIL: host-only unit phase not reported" >&2; cat "$TMP/ok.log" >&2; exit 1; }
# The model-gated gate must be named as never executed in stdout and the MD.
grep -q 'NEVER EXECUTED on this device' "$TMP/ok.log" \
    || { echo "FAIL: clean run with a skip missing the NEVER-EXECUTED stdout line" >&2
         cat "$TMP/ok.log" >&2; exit 1; }
grep -qE '^- gate_d\(set INFER_DSV4_MODEL_PATH' "$OUT1/results.md" \
    || { echo "FAIL: results.md never-executed section missing model gate" >&2
         cat "$OUT1/results.md" >&2; exit 1; }
grep -q '## Never executed on this device' "$OUT1/results.md" \
    || { echo "FAIL: results.md missing the never-executed heading" >&2
         cat "$OUT1/results.md" >&2; exit 1; }
grep -q '| gate_a | yes | required |' "$OUT1/results.md" \
    || { echo "FAIL: markdown table missing gate_a" >&2; exit 1; }
grep -q 'mock-kernel-1' "$OUT1/kernel-build-id.txt" \
    || { echo "FAIL: kernel build id not recorded" >&2; exit 1; }

# ── Model gate with INFER_DSV4_MODEL_PATH: the multi-rank launcher is invoked.
# The shell-test seam (ARLE_PARITY_TEST_NO_CLAIM) skips GPU claims and runs the
# real dsv4_multigpu_parity.sh, so its first-token verdict is exercised.
DSV4_LAUNCH="$ROOT/scripts/dsv4_multigpu_parity.sh"
run_model_batch() {  # $1 gate list, $2 out dir
    INFER_DSV4_MODEL_PATH="$TMP/fake-model" \
    ARLE_PARITY_DSV4_WORLD=2 \
    ARLE_PARITY_TEST_NO_CLAIM=1 \
    ARLE_PARITY_BIN_DIR="$BIN" \
    ARLE_PARITY_GATE_LIST="$1" \
    ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 \
    ARLE_PARITY_HOST_UNITS_CMD=true \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$2"
}
# Mock binary: print the right first token; the launcher prints ALL PASS.
cat > "$BIN/dsv4_parity" <<'SH'
#!/usr/bin/env bash
echo "clean_tokens=[11111, 603, 671, 6102, 294, 8760, 344]"
SH
chmod +x "$BIN/dsv4_parity"
# Launcher must exist in the repo tree the test runs against.
[ -x "$DSV4_LAUNCH" ] || { echo "FAIL: $DSV4_LAUNCH not executable" >&2; exit 1; }
OUT_MR="$TMP/out-model-run"
if run_model_batch "dsv4_parity:model" "$OUT_MR" >"$TMP/model-run.log" 2>&1; then :; else
    echo "FAIL: model gate with valid oracle exited non-zero" >&2
    cat "$TMP/model-run.log" >&2; exit 1
fi
grep -qE $'^dsv4_parity\tno\tallowlisted\t0\tALL PASS\tn/a\tn/a\tPASS\t' "$OUT_MR/results.tsv" \
    || { echo "FAIL: model gate not PASS with the correct oracle" >&2
         cat "$OUT_MR/results.tsv" >&2; exit 1; }
grep -q 'multi-rank on physical GPUs' "$TMP/model-run.log" \
    || { echo "FAIL: model phase launch line missing" >&2; cat "$TMP/model-run.log" >&2; exit 1; }
# Wrong first token: launcher FAIL and the batch exits non-zero.
cat > "$BIN/dsv4_parity" <<'SH'
#!/usr/bin/env bash
echo "clean_tokens=[99999, 603]"
SH
chmod +x "$BIN/dsv4_parity"
OUT_MF="$TMP/out-model-fail"
if run_model_batch "dsv4_parity:model" "$OUT_MF" >"$TMP/model-fail.log" 2>&1; then
    echo "FAIL: wrong-oracle model gate exited 0" >&2; cat "$TMP/model-fail.log" >&2; exit 1
fi
grep -qE $'^dsv4_parity\t.*\tFAIL\t' "$OUT_MF/results.tsv" \
    || { echo "FAIL: wrong-oracle model gate not recorded FAIL" >&2
         cat "$OUT_MF/results.tsv" >&2; exit 1; }
grep -q 'FAIL: rank0 first token 99999 != 11111' "$OUT_MF/logs/dsv4_parity.positive.log" \
    || { echo "FAIL: launcher verdict line missing" >&2
         cat "$OUT_MF/logs/dsv4_parity.positive.log" >&2; exit 1; }

# ── Gate that rejects --negative-control and is NOT allowlisted: FAIL ──
make_mock gate_noneg plain
OUT_NN="$TMP/out-noneg"
if ARLE_PARITY_BIN_DIR="$BIN" ARLE_PARITY_GATE_LIST="gate_noneg" ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 ARLE_PARITY_HOST_UNITS_CMD=true \
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
    ARLE_PARITY_HOST_UNITS_CMD=true \
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
    ARLE_PARITY_SKIP_PREREG=1 ARLE_PARITY_HOST_UNITS_CMD=true \
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
# Device-gated gate must also be named in the never-executed section.
grep -q -- '- gate_skip(requires sm_70' "$OUT_SK/results.md" \
    || { echo "FAIL: device-gated gate not named as never executed" >&2
         cat "$OUT_SK/results.md" >&2; exit 1; }
grep -q 'NEVER EXECUTED on this device' "$TMP/skip.log" \
    || { echo "FAIL: skip run missing NEVER-EXECUTED stdout line" >&2
         cat "$TMP/skip.log" >&2; exit 1; }

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

# ── Host-unit failure aborts before any GPU gate runs ──
OUT_HU="$TMP/out-hu"
if ARLE_PARITY_BIN_DIR="$BIN" ARLE_PARITY_GATE_LIST="gate_a" ARLE_PARITY_GPU=7 \
    ARLE_PARITY_SKIP_PREREG=1 ARLE_PARITY_HOST_UNITS_CMD='echo host fail; exit 1' \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$OUT_HU" >"$TMP/hu.log" 2>&1; then
    echo "FAIL: host-unit failure exited 0" >&2; cat "$TMP/hu.log" >&2; exit 1
fi
grep -q 'host-only unit tests failed' "$TMP/hu.log" \
    || { echo "FAIL: host-unit failure not reported" >&2; cat "$TMP/hu.log" >&2; exit 1; }
grep -q 'host fail' "$OUT_HU/host-units.log" \
    || { echo "FAIL: host-unit output not captured to host-units.log" >&2; exit 1; }
[ ! -f "$OUT_HU/results.tsv" ] \
    || { echo "FAIL: GPU gates ran after host-unit failure (results.tsv exists)" >&2; exit 1; }

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
    ARLE_PARITY_HOST_UNITS_CMD=true \
    bash "$ROOT/scripts/parity_gpu_batch.sh" "$OUT3" >/dev/null 2>&1 \
    || { echo "FAIL: capped run exited non-zero" >&2; exit 1; }
size="$(wc -c <"$OUT3/logs/gate_big.positive.log" | tr -d ' ')"
[ "$size" -le 65536 ] || { echo "FAIL: log not capped ($size bytes)" >&2; exit 1; }

echo "PASS: parity_gpu_batch mocks (host units green+red; clean green; positive fail and dead negative control red; log capped)"
