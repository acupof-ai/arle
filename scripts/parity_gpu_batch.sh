#!/usr/bin/env bash
# One-command GPU batch for every standalone parity example listed in
# operators/registry.toml correctness_gate. Runs in the pod tree:
#   1. builds every example in ONE `cargo build --release -p infer-cuda
#      --features cuda,nccl --example …` call (new gates picked up from the
#      registry, no script edit);
#   2. records --kernel-build-id once;
#   3. claims one free SM90 GPU with scripts/pick-gpu.sh (the same claim the
#      pod run scripts use) and runs each example positive then with
#      --negative-control;
#   4. writes results.tsv and a wins/errors-ready results.md under the
#      caller-given output dir, with per-run capped logs.
# Before the GPU gates it runs the cuda-kernels host-only unit tests with an
# explicit name filter (no GPU claim); a failure aborts before any gate runs.
#
# Exit non-zero if any positive run fails or any --negative-control run does
# not print NEGATIVE CONTROL OK. Multi-rank model gates (dsv4_parity,
# :model in ARLE_PARITY_GATE_LIST) need INFER_DSV4_MODEL_PATH and N free SM90
# GPUs: the single-card claim is released for their phase and reclaimed after;
# without the model path or enough free cards they are recorded SKIP.
#
# Test seams (shell test only, never set in production):
#   ARLE_PARITY_BIN_DIR   run mock binaries from this dir instead of cargo build
#   ARLE_PARITY_GATE_LIST gate list as name[:sm90][:model], space-separated
#   ARLE_PARITY_GPU       skip the pick-gpu.sh claim and use this index
#   ARLE_PARITY_NO_NEG_ALLOWLIST  gates allowed to run WITHOUT --negative-control
#   ARLE_PARITY_SKIP_PREREG=1  do not open/close a prereg row
#   ARLE_PARITY_DSV4_GPUS  pinned physical GPU csv for the dsv4 multi-rank phase
#   ARLE_PARITY_DSV4_WORLD ranks to launch (default 8; unused without the model)
#
# dsv4_parity runs allreduce MoE under the batch's cuda,nccl build; the DeepEP
# transport is not part of this gate (its first-token parity is transport-
# independent only by construction, not measured).
set -euo pipefail

# Gates allowed to have no --negative-control mode. A source grep is not used:
# a gate that silently loses its negative mode would then run positive-only.
# Everything NOT listed here is run with the flag, and a binary that rejects
# it is reported FAIL ("no negative control").
NEG_ALLOWLIST="${ARLE_PARITY_NO_NEG_ALLOWLIST:-dsv4_parity}"
neg_exempt() { case " $NEG_ALLOWLIST " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: parity_gpu_batch.sh <output-dir>}"
LOG_CAP_BYTES="${ARLE_PARITY_LOG_CAP_BYTES:-$((1024 * 1024))}"
mkdir -p "$OUT/logs"

NAME="parity-gpu-batch-$(date +%Y%m%d%H%M%S)"
if [ "${ARLE_PARITY_SKIP_PREREG:-0}" != 1 ]; then
    python3 "$ROOT/scripts/prereg.py" start \
        --name "$NAME" \
        --cmd "scripts/parity_gpu_batch.sh $OUT" \
        --hypothesis "every registry-listed parity gate passes positive and trips under --negative-control on one free SM90 GPU" \
        >/dev/null
fi

prereg_done() {  # $1 = status, rest = result/finding/decision already set via globals
    [ "${ARLE_PARITY_SKIP_PREREG:-0}" = 1 ] && return 0
    python3 "$ROOT/scripts/prereg.py" done \
        --name "$NAME" --status "$1" \
        --result "$PREREG_RESULT" --finding "$PREREG_FINDING" --decision "$PREREG_DECISION" \
        >/dev/null || true
}

# Gate metadata is derived, never hardcoded. Stdout lines: name<TAB>flags,
# where flags is a comma list from {neg,sm90,model}.
derive_gates() {
    if [ -n "${ARLE_PARITY_GATE_LIST:-}" ]; then
        for entry in $ARLE_PARITY_GATE_LIST; do
            name="${entry%%:*}"; tag="${entry#"$name"}"; tag="${tag#:}"
            flags=""
            case ":$tag:" in *:sm90:*) flags="${flags}sm90," ;; esac
            case ":$tag:" in *:model:*) flags="${flags}model," ;; esac
            printf '%s\t%s\n' "$name" "${flags%,}"
        done
        return
    fi
    local registry="$ROOT/operators/registry.toml"
    sed -n 's/^correctness_gate[[:space:]]*=[[:space:]]*"\(.*\)"$/\1/p' "$registry" \
        | tr ';' '\n' \
        | grep -oE 'crates/infer-cuda/examples/[A-Za-z0-9_]+\.rs' | sort -u \
        | while IFS= read -r src_rel; do
            src="$ROOT/$src_rel"
            [ -f "$src" ] || continue
            name="$(basename "$src" .rs)"
            flags=""
            head -n 60 "$src" | grep -Eiq 'SM90|sm_90|sm90' && flags="${flags}sm90,"
            grep -q 'INFER_DSV4_MODEL_PATH' "$src" && flags="${flags}model,"
            printf '%s\t%s\n' "$name" "${flags%,}"
        done
}

# ── Build (one cargo call for the whole list) ──────────────────────────────
if [ -n "${ARLE_PARITY_BIN_DIR:-}" ]; then
    BIN_DIR="$ARLE_PARITY_BIN_DIR"
    echo "parity-batch: mock binaries from $BIN_DIR"
else
    gates_list="$(derive_gates)"
    examples=()
    while IFS=$'\t' read -r gname _; do examples+=("--example" "$gname"); done <<< "$gates_list"
    [ "${#examples[@]}" -gt 0 ] || { echo "no parity gates found in registry" >&2; exit 2; }
    echo "parity-batch: building ${#examples[@]} examples in one cargo call"
    set +e
    cargo build --release -p infer-cuda --features cuda,nccl "${examples[@]}" \
        >"$OUT/build.log" 2>&1
    build_rc=$?
    set -e
    if [ "$build_rc" -ne 0 ]; then
        tail -n 40 "$OUT/build.log" >&2
        PREREG_RESULT="build failed rc=$build_rc"
        PREREG_FINDING="release build of the registry parity examples failed; no gate ran"
        PREREG_DECISION="fix the build before reading any parity signal"
        prereg_done killed
        exit "$build_rc"
    fi
    # Resolve through cargo so CARGO_TARGET_DIR / .cargo/config are honored.
    target_dir="$(cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT/Cargo.toml" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
    BIN_DIR="$target_dir/release/examples"
fi

# ── cuda-kernels HOST-only unit tests (no GPU claim, run before gates) ──────
# These functions are behind feature cuda but touch no device (shape
# validation, env parsing), so they cannot join the toolchain-free CI lane.
# The device functions are NOT listed: a pass that depends on the runner
# lacking a GPU is not a gate. Explicit filters — never the whole crate.
run_host_units() {
    # Seam for the shell test (mock binaries, no toolchain): command that
    # stands in for the cargo test invocation.
    if [ -n "${ARLE_PARITY_HOST_UNITS_CMD:-}" ]; then
        bash -c "$ARLE_PARITY_HOST_UNITS_CMD"
        return
    fi
    # Name filters OR-match; they go after `--` (cargo accepts one positional
    # filter itself, the rest must be forwarded to libtest).
    cargo test --release -p cuda-kernels --features cuda --lib -- \
        'tensor::weight_format::tests::' \
        'tensor::device_context::tests::parse_device_ordinal'
}
echo "parity-batch: cuda-kernels host-only unit tests"
if ! run_host_units >"$OUT/host-units.log" 2>&1; then
    echo "parity-batch: host-only unit tests failed" >&2
    tail -n 40 "$OUT/host-units.log" >&2
    PREREG_RESULT="host unit tests failed"
    PREREG_FINDING="cuda-kernels host-only unit tests failed before any GPU gate ran; see host-units.log"
    PREREG_DECISION="fix the host failure before reading any parity signal"
    prereg_done rejected
    exit 1
fi
echo "parity-batch: host-only unit tests PASS"

# ── Kernel build id, once ──────────────────────────────────────────────────
kernel_id=""
gates=()
while IFS= read -r gline; do gates+=("$gline"); done < <(derive_gates)
for line in "${gates[@]}"; do
    name="${line%%$'\t'*}"
    if [ -x "$BIN_DIR/$name" ] && "$BIN_DIR/$name" --kernel-build-id >"$OUT/.kernel-id.tmp" 2>/dev/null; then
        kernel_id="$(cat "$OUT/.kernel-id.tmp")"
        break
    fi
done
rm -f "$OUT/.kernel-id.tmp"
printf '%s\n' "$kernel_id" >"$OUT/kernel-build-id.txt"

# ── Claim one free GPU through the existing pick-gpu.sh ───────────────────
CLAIM_ENV=(ARLE_OP_ID="$NAME" ARLE_OWNER="$(id -u):$(id -un)" ARLE_CLAIM_PID="$$")
if [ -n "${ARLE_PARITY_GPU:-}" ]; then
    GPU="$ARLE_PARITY_GPU"
    CLAIM=""
else
    GPU="$(env "${CLAIM_ENV[@]}" bash "$ROOT/scripts/pick-gpu.sh")" || GPU="NONE"
    [ "$GPU" != "NONE" ] || {
        echo "parity-batch: no free GPU" >&2
        PREREG_RESULT="no free GPU"
        PREREG_FINDING="pick-gpu.sh offered no free SM90 card"
        PREREG_DECISION="rerun when a card is free"
        prereg_done killed
        exit 3
    }
    CLAIM="${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$GPU"
fi
cleanup() {
    [ -z "${CLAIM:-}" ] || rm -f "$CLAIM"
    for g in ${MODEL_CLAIMS:-}; do rm -f "${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$g" 2>/dev/null || true; done
}
trap cleanup EXIT
echo "parity-batch: gpu=$GPU kernel_build_id=$kernel_id gates=${#gates[@]}"

# Keep the tail of a log (verdict lines print at the end).
cap_log() {
    local f="$1"
    [ -f "$f" ] || return 0
    local size; size="$(wc -c <"$f" | tr -d ' ')"
    if [ "$size" -gt "$LOG_CAP_BYTES" ]; then
        tail -c "$LOG_CAP_BYTES" "$f" >"$f.tmp" && mv "$f.tmp" "$f"
    fi
}

# Verdict extraction: positive needs rc 0 plus ALL PASS; negative needs
# rc 0 AND the NEGATIVE CONTROL OK line.
pos_marker() { grep -F 'ALL PASS' "$1" | tail -n 1 || true; }
neg_marker() { grep -F 'NEGATIVE CONTROL OK' "$1" | tail -n 1 || true; }
# A gate that cannot run on this device prints a single `SKIP: <reason>`
# line with rc 0 and NO pass marker. Recognize it explicitly so rc=0 alone
# is neither a false FAIL nor a silent PASS.
skip_marker() { grep -E '^SKIP: ' "$1" | tail -n 1 || true; }
skip_reason() { sed -n 's/^SKIP: //p' "$1" | tail -n 1 | tr '\t' ' ' | cut -c1-200; }
fail_lines() {  # FAIL / family-teeth lines, TSV-safe single line
    grep -E 'FAIL|negative teeth OK' "$1" | tail -n 20 \
        | tr '\t\n' '  ' | sed 's/  */ /g' | cut -c1-800 || true
}

TSV="$OUT/results.tsv"
MD="$OUT/results.md"
printf 'example\tsm90_only\tnegative_supported\tpositive_exit\tpositive_verdict\tnegative_exit\tnegative_verdict\tstatus\tfail_lines\n' >"$TSV"

n_pass=0; n_fail=0; n_skip=0; fail_names=""; notrun_list=""
# Record a gate that produced no verdict this run. $1=name $2=reason; the list
# is one "name<TAB>reason" record per line so spaces inside a reason survive.
mark_notrun() {
    local r
    r="$(printf '%s' "$2" | tr '[:space:]' ' ' | sed 's/  */ /g; s/^ //; s/ $//')"
    notrun_list="${notrun_list}${1}"$'\t'"${r}"$'\n'
}

# Multi-rank model gate (today: dsv4_parity at TP=N). Needs INFER_DSV4_MODEL_PATH
# and N free SM90 GPUs; ARLE_PARITY_DSV4_GPUS pins a physical set, otherwise a
# free contiguous N-set is claimed via pick-gpu.sh reserve-set.
# Echoes "RUN <csv>" or "SKIP <reason>".
re_claim_single() {
    [ -n "${ARLE_PARITY_GPU:-}" ] && { GPU="$ARLE_PARITY_GPU"; CLAIM=""; return; }
    GPU="$(env "${CLAIM_ENV[@]}" bash "$ROOT/scripts/pick-gpu.sh")" || GPU="NONE"
    if [ "$GPU" != NONE ]; then
        CLAIM="${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$GPU"
    else
        CLAIM=""
    fi
}
dsv4_reserve_gpu_set() {
    local n="$1" csv
    # Test seam: accept the pinned/default set without calling pick-gpu.sh.
    if [ "${ARLE_PARITY_TEST_NO_CLAIM:-0}" = 1 ]; then
        if [ -n "${ARLE_PARITY_DSV4_GPUS:-}" ]; then echo "RUN $ARLE_PARITY_DSV4_GPUS"; else
            local i s=""; for ((i=0;i<n;i++)); do s+="$i,"; done; echo "RUN ${s%,}"; fi
        return
    fi
    if [ -n "${ARLE_PARITY_DSV4_GPUS:-}" ]; then
        csv="$ARLE_PARITY_DSV4_GPUS"
        if env "${CLAIM_ENV[@]}" bash "$ROOT/scripts/pick-gpu.sh" reserve-set "$csv" >/dev/null 2>&1; then
            echo "RUN $csv"
        else
            echo "SKIP reserved GPUs $csv not all free SM90"
        fi
        return
    fi
    local indices
    indices="$(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', ' '$2 == "9.0" {print $1}' | sort -n | paste -sd, -)"
    if [ -z "$indices" ]; then echo "SKIP no SM90 GPUs visible"; return; fi
    local IFS=,; set -f; read -r -a all <<< "$indices"; set +f; unset IFS
    local lo window
    for ((lo=0; lo+n<=${#all[@]}; lo++)); do
        window="$(printf '%s,' "${all[@]:$lo:$n}" | sed 's/,$//')"
        if env "${CLAIM_ENV[@]}" bash "$ROOT/scripts/pick-gpu.sh" reserve-set "$window" >/dev/null 2>&1; then
            echo "RUN $window"; return
        fi
    done
    echo "SKIP no $n free SM90 GPUs for the multi-rank launcher"
}

run_model_gate() {  # $1=name $2=sm90 $3=neg_mode
    local name="$1" sm90="$2" neg_mode="$3"
    local pos_log="$OUT/logs/$name.positive.log"
    if [ -z "${INFER_DSV4_MODEL_PATH:-}" ]; then
        printf '%s\t%s\t%s\t-\t-\t-\t-\tSKIP\tset INFER_DSV4_MODEL_PATH (multi-rank launcher)\n' \
            "$name" "$sm90" "$neg_mode" >>"$TSV"
        n_skip=$((n_skip + 1)); mark_notrun "$name" "set INFER_DSV4_MODEL_PATH (multi-rank launcher)"
        return
    fi
    local world="${ARLE_PARITY_DSV4_WORLD:-8}"
    # The model gate needs N whole GPUs; free the single-card claim so the set
    # search sees every card, and re-claim one afterward for the rest of the
    # single-card gates (skipped in the no-claim shell-test seam).
    if [ "${ARLE_PARITY_TEST_NO_CLAIM:-0}" != 1 ]; then
        if [ -n "${CLAIM:-}" ]; then rm -f "$CLAIM"; CLAIM=""; fi
        GPU=""
    fi
    local decision; decision="$(dsv4_reserve_gpu_set "$world")"
    if [ "${decision%% *}" = SKIP ]; then
        local reason="${decision#SKIP }"
        printf '%s\t%s\t%s\t-\t-\t-\t-\tSKIP\t%s\n' "$name" "$sm90" "$neg_mode" "$reason" >>"$TSV"
        n_skip=$((n_skip + 1)); mark_notrun "$name" "$reason"
        if [ "${ARLE_PARITY_TEST_NO_CLAIM:-0}" != 1 ]; then re_claim_single; fi
        return
    fi
    local model_gpus="${decision#RUN }"
    MODEL_CLAIMS="${MODEL_CLAIMS:-} ${model_gpus//,/ }"
    echo "parity-batch: $name multi-rank on physical GPUs $model_gpus (world=$world)" >&2
    set +e
    INFER_DSV4_MODEL_PATH="$INFER_DSV4_MODEL_PATH" \
    DSV4_PARITY_GPUS="$model_gpus" \
    DSV4_PARITY_BIN="$BIN_DIR/$name" \
        bash "$ROOT/scripts/dsv4_multigpu_parity.sh" >"$pos_log" 2>&1
    local pos_rc=$?
    set -e
    if [ "${ARLE_PARITY_TEST_NO_CLAIM:-0}" != 1 ]; then
        for g in ${model_gpus//,/ }; do rm -f "${ARLE_GPU_CLAIMS:-/tmp/arle-gpu-claims}/$g"; done
        MODEL_CLAIMS=""
    fi
    cap_log "$pos_log"
    local pos_verdict fails pos_status
    pos_verdict="$(pos_marker "$pos_log")"
    if [ "$pos_rc" -eq 0 ] && [ -n "$pos_verdict" ]; then
        pos_status=PASS; n_pass=$((n_pass + 1))
    else
        pos_status=FAIL; n_fail=$((n_fail + 1)); fail_names="$fail_names $name(multi-rank rc=$pos_rc)"
    fi
    fails="$(fail_lines "$pos_log")"
    printf '%s\t%s\t%s\t%s\t%s\tn/a\tn/a\t%s\t%s\n' \
        "$name" "$sm90" "$neg_mode" "$pos_rc" "${pos_verdict:-—}" "$pos_status" "${fails:-—}" >>"$TSV"
    if [ "${ARLE_PARITY_TEST_NO_CLAIM:-0}" != 1 ]; then re_claim_single; fi
}
for line in "${gates[@]}"; do
    name="${line%%$'\t'*}"; flags="${line#*$'\t'}"
    sm90=no; model=no; neg=yes; neg_mode="required"
    case ",$flags," in *,sm90,*) sm90=yes ;; esac
    case ",$flags," in *,model,*) model=yes ;; esac
    if neg_exempt "$name"; then neg=no; neg_mode="allowlisted"; fi

    if [ "$model" = yes ]; then
        run_model_gate "$name" "$sm90" "$neg_mode"
        continue
    fi
    bin="$BIN_DIR/$name"
    if [ "$GPU" = NONE ]; then
        printf '%s\t%s\t%s\t-\t-\t-\t-\tSKIP\tno free single GPU after the multi-rank phase\n' \
            "$name" "$sm90" "$neg_mode" >>"$TSV"
        n_skip=$((n_skip + 1)); mark_notrun "$name" "no free single GPU after the multi-rank phase"
        continue
    fi
    if [ ! -x "$bin" ]; then
        printf '%s\t%s\t%s\t-\t-\t-\t-\tFAIL\tbinary missing after build\n' \
            "$name" "$sm90" "$neg_mode" >>"$TSV"
        n_fail=$((n_fail + 1)); fail_names="$fail_names $name(missing-binary)"
        continue
    fi

    pos_log="$OUT/logs/$name.positive.log"
    set +e
    INFER_CUDA_DEVICE="$GPU" "$bin" >"$pos_log" 2>&1
    pos_rc=$?
    set -e
    cap_log "$pos_log"
    pos_verdict="$(pos_marker "$pos_log")"
    pos_skip="$(skip_marker "$pos_log")"
    if [ -n "$pos_skip" ]; then
        pos_status=SKIP
        pos_verdict="$pos_skip"
        n_skip=$((n_skip + 1))
        mark_notrun "$name" "$(skip_reason "$pos_log")"
    elif [ "$pos_rc" -eq 0 ] && [ -n "$pos_verdict" ]; then
        pos_status=PASS; n_pass=$((n_pass + 1))
    else
        pos_status=FAIL; n_fail=$((n_fail + 1)); fail_names="$fail_names $name(positive rc=$pos_rc)"
    fi

    neg_rc="-"; neg_verdict="-"; neg_status="n/a"; neg_log=""
    if [ "$neg" = yes ]; then
        neg_log="$OUT/logs/$name.negative.log"
        set +e
        INFER_CUDA_DEVICE="$GPU" "$bin" --negative-control >"$neg_log" 2>&1
        neg_rc=$?
        set -e
        cap_log "$neg_log"
        neg_verdict="$(neg_marker "$neg_log")"
        neg_skip="$(skip_marker "$neg_log")"
        if [ -n "$neg_skip" ]; then
            # Same device gate skips in both worlds; counted once (the
            # positive SKIP above) so a skipped gate yields exactly one
            # n_skip, not two.
            neg_status=SKIP
            neg_verdict="$neg_skip"
        elif [ "$neg_rc" -eq 0 ] && [ -n "$neg_verdict" ]; then
            neg_status=PASS
        else
            neg_status=FAIL
            n_fail=$((n_fail + 1))
            if [ "$neg_rc" -ne 0 ]; then
                fail_names="$fail_names $name(no negative control)"
            else
                fail_names="$fail_names $name(negative-control)"
            fi
        fi
    fi

    fails="$(fail_lines "$pos_log")"
    if [ "$neg_status" = FAIL ]; then fails="$fails | $(fail_lines "${neg_log:-/dev/null}")"; fi
    status="$pos_status"
    if [ "$neg_status" = FAIL ] || { [ "$pos_status" = FAIL ] && [ "$neg_status" != SKIP ]; }; then
        status=FAIL
    fi
    tsv_reason="${fails:-—}"
    if [ "$pos_status" = SKIP ]; then tsv_reason="$(skip_reason "$pos_log")"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$sm90" "$neg_mode" "$pos_rc" "${pos_verdict:-—}" \
        "$neg_rc" "${neg_verdict:-—}" "$status" "$tsv_reason" >>"$TSV"
done

# ── Markdown table (wins/errors-ready) ─────────────────────────────────────
{
    echo "# Parity GPU batch — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "kernel_build_id: \`${kernel_id:-unknown}\` · GPU: ${GPU} · pass ${n_pass} · fail ${n_fail} · skip ${n_skip}"
    echo
    echo "| Example | SM90-only | Negative mode | Positive | Negative control | FAIL lines (tail) |"
    echo "|---|---|---|---|---|---|"
    tail -n +2 "$TSV" | while IFS=$'\t' read -r ex sm neg_s p_rc p_v n_rc n_v st fails; do
        [ "$st" = SKIP ] && { echo "| ${ex} | ${sm} | ${neg_s} | SKIP | SKIP | ${fails} |"; continue; }
        p_cell="rc=${p_rc} ${p_v}"; n_cell="n/a"
        if [ "$neg_s" = required ]; then n_cell="rc=${n_rc} ${n_v}"; fi
        echo "| ${ex} | ${sm} | ${neg_s} | ${p_cell} | ${n_cell} | ${fails} |"
    done
    if [ "$n_skip" -gt 0 ]; then
        echo
        echo "## Never executed on this device (${n_skip}): no positive or negative verdict"
        echo
        echo "These gates were SKIP, not PASS — neither their positive arm nor a"
        echo "negative control ran here, so this batch says nothing about their"
        echo "correctness on GPU ${GPU}. Run them on the named device/launcher."
        echo
        printf '%s' "$notrun_list" | while IFS=$'\t' read -r n r; do
            [ -n "$n" ] || continue
            printf -- '- %s(%s)\n' "$n" "$r"
        done
    fi
} >"$MD"

echo "parity-batch: pass=$n_pass fail=$n_fail skip=$n_skip"
[ -z "$fail_names" ] || echo "parity-batch: FAILING:$fail_names"
if [ "$n_skip" -gt 0 ]; then
    notrun_display="$(printf '%s' "$notrun_list" | sed '/^$/d; s/\t/(/; s/$/)/' | paste -sd ' ' -)"
    echo "parity-batch: NEVER EXECUTED on this device ($n_skip; no positive/negative verdict): $notrun_display"
fi
echo "parity-batch: $TSV"
echo "parity-batch: $MD"

if [ "$n_fail" -eq 0 ]; then
    PREREG_RESULT="pass=$n_pass fail=0 skip=$n_skip"
    if [ "$n_skip" -eq 0 ]; then
        PREREG_FINDING="every registry-listed parity gate printed its positive verdict and every negative-control run printed NEGATIVE CONTROL OK"
    else
        PREREG_FINDING="$n_pass gates green; $n_skip gates never executed on this device (no verdict): $notrun_display"
    fi
    PREREG_DECISION="gates accepted on this build; results.md lists the non-executed gates and their required device/launcher"
    prereg_done ok
    exit 0
fi
PREREG_RESULT="pass=$n_pass fail=$n_fail skip=$n_skip"
PREREG_FINDING="failing gates:$fail_names"
PREREG_DECISION="do not trust the gated operators on this build until the listed gates are green"
prereg_done rejected
exit 1
