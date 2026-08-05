#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d -t pod-flow-XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
LOCAL="$TMP/local"; NODE="$TMP/node/tree"; TREE="$TMP/pod/tree"; STATE="$TMP/state"; BIN="$TMP/bin"
mkdir -p "$LOCAL/scripts" "$TMP/node" "$TMP/pod" "$STATE" "$BIN"
cp "$ROOT/.gitignore" "$LOCAL/"
cp "$ROOT/scripts/"{pod.sh,pod-remote-build.sh,pod-remote-run.sh,pick-gpu.sh,reap_run.py,pod-build-env.sh,pod-tilelang-env.sh,cuda_prebuilt_manifest.sh,kernel_artifacts.sh} "$LOCAL/scripts/"

git -C "$LOCAL" init -q
git -C "$LOCAL" config user.email test@example.com
git -C "$LOCAL" config user.name test
printf old > "$LOCAL/delete me"
printf rename > "$LOCAL/old name"
git -C "$LOCAL" add . && git -C "$LOCAL" commit -qm base
mv "$LOCAL/old name" "$LOCAL/new name"
rm "$LOCAL/delete me"
printf untracked > "$LOCAL/untracked space"

cat > "$BIN/pod" <<'SH'
#!/usr/bin/env bash
NODE_TREE="${NODE_TREE:?}" POD_TREE="${POD_TREE:?}" bash -c '
command=${1//"$NODE_TREE"/"$POD_TREE"}
bash -c "$command"
' _ "$1"
SH
cat > "$BIN/tn" <<'SH'
#!/usr/bin/env bash
[ "${TN_FAIL_AT:-}" != "${TN_COUNT:-0}" ] || exit 1
src=$2; dst=$3
pod_dst="${POD_TREE}${dst#"$NODE_TREE"}"
mkdir -p "$(dirname "$pod_dst")"
cp "$src" "$pod_dst"
SH
cat > "$BIN/python3" <<'SH'
#!/usr/bin/env bash
if [[ "$1" == *reap_run.py ]]; then
  shift
  if [ "${2:-}" = --argv-file ]; then op=$1; file=$3; cmd=$4; exec /usr/bin/python3 - "$file" "$cmd" <<'PY'
import os, sys
raw = open(sys.argv[1], "rb").read().split(b"\0")
if raw and not raw[-1]: raw.pop()
os.execv(sys.argv[2], [sys.argv[2], *(os.fsdecode(x) for x in raw)])
PY
  fi
fi
exec /usr/bin/python3 "$@"
SH
cat > "$BIN/cargo" <<'SH'
#!/usr/bin/env bash
[ "${CARGO_CASE:-success}" != fail ] || exit 7
out="$POD_TREE/target/test-manifest"
exe="$POD_TREE/target/release/arle"
mkdir -p "$(dirname "$exe")" "$out" "$POD_TREE/target/stale"
id=kernel-123; [ "${CARGO_CASE:-}" != embedded-mismatch ] || id=kernel-wrong
# Two lines, like the real binary: comparing the whole output against the
# manifest id failed every green build until 06a27527e's `capabilities:` line
# was accounted for, and a one-line fake could never have caught it.
printf '#!/usr/bin/env bash\nif [ "$1" = --kernel-build-id ]; then echo %s; echo capabilities:fa3,flashmla; exit; fi\nexit 0\n' "$id" > "$exe"
chmod +x "$exe"
printf 'schema=1\nkernel_build_id=kernel-123\n' > "$out/arle-cuda-kernels.manifest"
printf 'schema=1\nkernel_build_id=stale\n' > "$POD_TREE/target/stale/arle-cuda-kernels.manifest"
python3 - "$exe" "$out" "${CARGO_CASE:-success}" <<'PY'
import json, os, pathlib, sys
exe, out, case = sys.argv[1:]
root = pathlib.Path(os.environ["ARLE_CARGO_WORKSPACE_ROOT"])
package = (root / "crates/cuda-kernels").resolve().as_uri()
artifact = {"reason":"compiler-artifact","target":{"name":"arle","kind":["bin"]},"executable":exe}
script = {"reason":"build-script-executed","package_id":f"path+{package}#0.1.0","out_dir":out}
events = [] if case == "missing-exe" else [artifact]
if case == "duplicate-exe": events.append(artifact)
if case != "missing-out": events.append(script)
if case == "duplicate-out": events.append(script)
for event in events: print(json.dumps(event))
PY
SH
cat > "$BIN/rustup" <<'SH'
#!/usr/bin/env bash
exit 0
SH
cat > "$BIN/setsid" <<'SH'
#!/usr/bin/env bash
exec "$@"
SH
cat > "$BIN/flock" <<'SH'
#!/usr/bin/env bash
lock=$1; shift
case "$lock" in [0-9]*) lock="$(readlink "/dev/fd/$lock")";; esac
mkdir "$lock.d" 2>/dev/null || { while ! mkdir "$lock.d" 2>/dev/null; do sleep .05; done; }
trap 'rmdir "$lock.d"' EXIT
[ $# -eq 0 ] || "$@"
SH
cat > "$BIN/nvidia-smi" <<'SH'
#!/usr/bin/env bash
case "$*" in
  *--query-compute-apps=gpu_uuid*) printf '%b' "${SMI_COMPUTE:-}" ;;
  *--query-gpu=index,uuid,memory.used,compute_cap*) printf '%b\n' "${SMI:-0, GPU-0, 0, 9.0\n1, GPU-1, 0, 9.0}" ;;
  *) printf '%b\n' "${SMI:-0, GPU-0, 0, 9.0\n1, GPU-1, 0, 9.0}" ;;
esac
SH
chmod +x "$BIN/"*
export PATH="$BIN:$PATH" POD="$BIN/pod" TN="$BIN/tn" NODE_TREE="$NODE" POD_TREE="$TREE" POD_STATE="$STATE"

mkdir -p "$TREE/scripts" "$TREE/target" "$TREE/crates/cuda-kernels/tools/tilelang/.venv" "$TREE/bench-output"
printf sentinel > "$TREE/sentinel"
printf target > "$TREE/target/keep"
printf venv > "$TREE/crates/cuda-kernels/tools/tilelang/.venv/keep"
printf bench > "$TREE/bench-output/keep"
cp "$LOCAL/scripts/pod-remote-build.sh" "$TREE/scripts/"
TN_FAIL_AT=0 "$LOCAL/scripts/pod.sh" sync >/dev/null 2>&1 && exit 1 || true
[ "$(cat "$TREE/sentinel")" = sentinel ]
grep -Fq 'COPYFILE_DISABLE=1 tar ' "$LOCAL/scripts/pod.sh"

"$LOCAL/scripts/pod.sh" sync >/dev/null
[ -f "$TREE/new name" ] && [ -f "$TREE/untracked space" ] && [ ! -e "$TREE/delete me" ] && [ ! -e "$TREE/old name" ]
[ "$(cat "$TREE/target/keep")" = target ] && [ "$(cat "$TREE/crates/cuda-kernels/tools/tilelang/.venv/keep")" = venv ] && [ "$(cat "$TREE/bench-output/keep")" = bench ]

digest_before="$(awk -F= '$1=="digest" {print $2}' "$TREE/.arle-source-receipt")"
printf second > "$LOCAL/untracked space"
"$LOCAL/scripts/pod.sh" sync >/dev/null
digest_second="$(awk -F= '$1=="digest" {print $2}' "$TREE/.arle-source-receipt")"
[ "$digest_second" != "$digest_before" ]
printf third > "$LOCAL/untracked space"
"$LOCAL/scripts/pod.sh" sync >/dev/null
digest_third="$(awk -F= '$1=="digest" {print $2}' "$TREE/.arle-source-receipt")"
[ "$digest_third" != "$digest_second" ]
git -C "$LOCAL" add "untracked space"
"$LOCAL/scripts/pod.sh" sync >/dev/null
[ "$(awk -F= '$1=="digest" {print $2}' "$TREE/.arle-source-receipt")" = "$digest_third" ]
# shellcheck disable=SC2016
lock_expr='TREE_LOCK="/tmp/arle-build$(printf '\''%s'\'' "$TREE" | tr '\''/.'\'' '\''__'\'').lock"'
grep -Fq "$lock_expr" "$LOCAL/scripts/pod-remote-build.sh"
grep -Fq 'flock 9' "$LOCAL/scripts/pod-remote-build.sh"
"$LOCAL/scripts/pod.sh" sync >/dev/null
digest_before="$(awk -F= '$1=="digest" {print $2}' "$TREE/.arle-source-receipt")"

mkdir -p "$STATE/builds/good" "$TREE/target/release"
cat > "$TREE/target/release/arle" <<'SH'
#!/usr/bin/env bash
printf '%s\0' "$@" > "$ARGV_OUT"
[ -z "${ENV_OUT:-}" ] || env | grep -E '^(CUDA_VISIBLE_DEVICES|INFER_CUDA_DEVICE|INFER_CUDA_DEVICES|INFER_TP_SIZE)=' | sort > "$ENV_OUT"
[ -z "${CLAIM_SWAP:-}" ] || printf 'schema=arle-gpu-claim-v1\nop=foreign\n' > "$CLAIM_SWAP"
SH
chmod +x "$TREE/target/release/arle"
bsha="$(sha256sum "$TREE/target/release/arle" | cut -d' ' -f1)"
head="$(git -C "$TREE" rev-parse HEAD)"
printf 'schema=arle-build-v1\nexit=0\nbinary=%s\nbinary_sha=%s\nsource_head=%s\nsource_digest=%s\n' "$TREE/target/release/arle" "$bsha" "$head" "$digest_before" > "$STATE/builds/good/receipt"
[ "$(awk -F= '$1=="binary_sha" {print $2}' "$STATE/builds/good/receipt")" = "$bsha" ]

printf changed >> "$TREE/new name"
printf '\0' > "$TMP/empty-argv"
set +e
POD_TREE="$TREE" POD_STATE="$STATE" setsid bash "$TREE/scripts/pod-remote-run.sh" run good changed auto op-change "$TMP/empty-argv" >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && grep -q 'source changed since build' "$STATE/runs/changed/log"
grep -Fq "pod-remote-run.sh' run '\$build'" "$LOCAL/scripts/pod.sh"
printf rename > "$TREE/new name"
digest_now="$(POD_TREE="$TREE" bash "$TREE/scripts/pod-remote-build.sh" source-digest "$TREE")"
python3 - "$TREE/.arle-source-receipt" "$digest_now" <<'PY'
import sys
p, digest = sys.argv[1:]
lines = open(p).read().splitlines()
open(p, "w").write("\n".join(f"digest={digest}" if x.startswith("digest=") else x for x in lines) + "\n")
PY
printf 'schema=arle-build-v1\nexit=0\nbinary=%s\nbinary_sha=%s\nsource_head=%s\nsource_digest=%s\n' "$TREE/target/release/arle" "$bsha" "$head" "$digest_now" > "$STATE/builds/good/receipt"

printf '%s\0' '' 'a b' 'q"uote' '*' > "$TMP/argv"
export ARGV_OUT="$TMP/seen"
set +e
POD_TREE="$TREE" POD_STATE="$STATE" SMI='0, GPU-0, 0, 9.0' setsid bash "$TREE/scripts/pod-remote-run.sh" run good exact auto op-exact "$TMP/argv" >/dev/null
rc=$?; set -e
[ "$rc" -eq 0 ] || { command cat "$STATE/runs/exact/log" >&2; exit 1; }
[ -f "$TMP/seen" ] || { command cat "$STATE/runs/exact/log" >&2; exit 1; }
cmp "$STATE/runs/exact/argv.nul" "$TMP/seen"
[ "$(sha256sum "$TREE/target/release/arle" | cut -d' ' -f1)" = "$bsha" ]

TP4_SET='0, GPU-0, 0, 9.0\n1, GPU-1, 0, 9.0\n2, GPU-2, 0, 9.0\n3, GPU-3, 0, 9.0'
printf '%s\0' serve > "$TMP/tp4-argv"
export ENV_OUT="$TMP/tp4-env" ARGV_OUT="$TMP/tp4-seen"
TP4_CLAIMS="$TMP/tp4-claims"
POD_TREE="$TREE" POD_STATE="$STATE" ARLE_GPU_CLAIMS="$TP4_CLAIMS" SMI="$TP4_SET" setsid bash "$TREE/scripts/pod-remote-run.sh" run good tp4 0,1,2,3 op-tp4 "$TMP/tp4-argv" >/dev/null
receipt="$STATE/runs/tp4/receipt"
grep -Fxq 'gpu=0,1,2,3' "$receipt"
grep -Fxq 'CUDA_VISIBLE_DEVICES=0,1,2,3' "$ENV_OUT"
grep -Fxq 'INFER_CUDA_DEVICES=0,1,2,3' "$ENV_OUT"
grep -Fxq 'INFER_TP_SIZE=4' "$ENV_OUT"
grep -q '^INFER_CUDA_DEVICE=' "$ENV_OUT" && exit 1 || true
for gpu in 0 1 2 3; do [ ! -e "$TP4_CLAIMS/$gpu" ]; done

printf '%s\0' serve > "$TMP/tp4-foreign-argv"
export ENV_OUT="$TMP/tp4-foreign-env" CLAIM_SWAP="$TP4_CLAIMS/2"
POD_TREE="$TREE" POD_STATE="$STATE" ARLE_GPU_CLAIMS="$TP4_CLAIMS" SMI="$TP4_SET" setsid bash "$TREE/scripts/pod-remote-run.sh" run good tp4-foreign 0,1,2,3 op-tp4-foreign "$TMP/tp4-foreign-argv" >/dev/null
[ "$(awk -F= '$1=="op" {print $2}' "$TP4_CLAIMS/2")" = foreign ]
for gpu in 0 1 3; do [ ! -e "$TP4_CLAIMS/$gpu" ]; done
unset CLAIM_SWAP ENV_OUT

mkdir -p "$STATE/runs/stale"
printf 'op=foreign\npid=%s\npgid=%s\nstart=0\n' "$$" "$$" > "$STATE/runs/stale/process"
set +e
POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-run.sh" kill stale >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && kill -0 $$

set +e
"$LOCAL/scripts/pod.sh" build bad --release --profile release-fast --bin arle >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && [ ! -e "$STATE/builds/bad" ]
printf '%s\0' --release --message-format json --bin arle > "$TMP/reserved-format"
set +e
POD_TREE="$TREE" bash "$TREE/scripts/pod-remote-build.sh" validate-build-args "$TMP/reserved-format" >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ]
printf '%s\0' --release --message-format=json --bin arle > "$TMP/reserved-format-equals"
set +e
POD_TREE="$TREE" bash "$TREE/scripts/pod-remote-build.sh" validate-build-args "$TMP/reserved-format-equals" >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ]

out1="$("$LOCAL/scripts/pod.sh" build 2>/dev/null)"; out2="$("$LOCAL/scripts/pod.sh" build 2>/dev/null)"
label1="$(printf '%s' "$out1" | awk -F"'" '{print $2}')"; label2="$(printf '%s' "$out2" | awk -F"'" '{print $2}')"
[ -n "$label1" ] && [ "$label1" != "$label2" ]

printf '%s\0' --release --features cuda --bin arle > "$TMP/build-argv"
POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-build.sh" build manifest op-manifest "$TMP/build-argv" >/dev/null
receipt="$STATE/builds/manifest/receipt"
[ "$(awk -F= '$1=="kernel_id" {print $2}' "$receipt")" = kernel-123 ]
[ "$(awk -F= '$1=="embedded_id" {print $2}' "$receipt")" = kernel-123 ]
# The success case asserted every field except the one that says it succeeded,
# so a green build reporting exit=1 went unnoticed (06a27527e's second
# `--kernel-build-id` line broke the embedded-id comparison).
[ "$(awk -F= '$1=="exit" {print $2}' "$receipt")" = 0 ]
[ "$(awk -F= '$1=="cargo_out_dir" {print $2}' "$receipt")" = "$TREE/target/test-manifest" ]
grep -Fq "producer_manifest=$TREE/target/test-manifest/arle-cuda-kernels.manifest" "$receipt"
[ "$(awk -F= '$1=="argv_sha" {print $2}' "$receipt")" = "$(sha256sum "$STATE/builds/manifest/argv.nul" | cut -d' ' -f1)" ]

for case in missing-exe duplicate-exe missing-out duplicate-out embedded-mismatch fail; do
  label="build-$case"; printf '%s\0' --release --features cuda --bin arle > "$TMP/$label"
  set +e
  CARGO_CASE="$case" POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-build.sh" build "$label" "op-$case" "$TMP/$label" >/dev/null
  rc=$?; set -e
  [ "$rc" -ne 0 ] && [ "$(awk -F= '$1=="exit" {print $2}' "$STATE/builds/$label/receipt")" -ne 0 ]
done

printf stale >> "$TREE/target/release/arle"
printf '\0' > "$TMP/stale-run-argv"
set +e
POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-run.sh" run manifest stale-binary auto op-stale "$TMP/stale-run-argv" >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && grep -q 'binary SHA mismatch' "$STATE/runs/stale-binary/log"
printf '%s\0' --release --features cuda --bin arle > "$TMP/failed-run-argv"
set +e
POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-run.sh" run build-fail failed-build auto op-failed "$TMP/failed-run-argv" >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && grep -q 'successful build receipt required' "$STATE/runs/failed-build/log"

mkdir -p "$STATE/builds/shared" "$STATE/runs/shared" "$TMP/proc/4242"
printf 'exit=0\n' > "$STATE/builds/shared/receipt"
printf '%s' 'done' > "$STATE/builds/shared/log"
printf '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12345\n' > "$TMP/proc/4242/stat"
printf '%s\0' bash "$TREE/scripts/pod-remote-run.sh" run shared active-run > "$TMP/proc/4242/cmdline"
printf 'schema=arle-run-v1\nbinary=/tmp/arle\nstate=running-unobserved\n' > "$STATE/runs/shared/receipt"
printf 'schema=arle-process-v1\nkind=run\nexpected_helper=%s\noperation=active-run\npid=4242\npgid=4242\nstart=12345\nexpected_binary=/tmp/arle\n' "$TREE/scripts/pod-remote-run.sh" > "$STATE/runs/shared/process"
cat > "$BIN/ps" <<'SH'
#!/usr/bin/env bash
printf '4242\n'
SH
cat > "$BIN/mock-kill" <<'SH'
#!/usr/bin/env bash
printf killed > "$KILL_MARKER"
SH
chmod +x "$BIN/ps" "$BIN/mock-kill"
export KILL_MARKER="$TMP/killed"
PROC_ROOT="$TMP/proc" KILL_CMD="$BIN/mock-kill" POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-run.sh" kill shared >/dev/null
[ -f "$KILL_MARKER" ]

mkdir -p "$TMP/claims"
printf 'schema=arle-gpu-claim-v1\nop=foreign\nowner=other\npid=%s\nstart=%s\n' "$$" "$(awk '{print $22}' /proc/$$/stat)" > "$TMP/claims/0"
gpu="$(ARLE_GPU_CLAIMS="$TMP/claims" ARLE_OP_ID=ours ARLE_OWNER=test SMI='0, GPU-0, 0, 9.0\n1, GPU-1, 0, 9.0' bash "$TREE/scripts/pick-gpu.sh")"
[ "$gpu" = 1 ] && kill -0 $$

SM90_SET='0, GPU-0, 0, 9.0\n1, GPU-1, 0, 9.0\n2, GPU-2, 0, 9.0\n3, GPU-3, 0, 9.0\n4, GPU-4, 0, 9.0\n5, GPU-5, 0, 9.0\n6, GPU-6, 0, 9.0\n7, GPU-7, 0, 9.0'
SMI="$SM90_SET" ARLE_GPU_CLAIMS="$TMP/free-set" bash "$TREE/scripts/pick-gpu.sh" check-free-set 0,1,2,3,4,5,6,7 >/dev/null
reserve_op=reserve-test
SMI="$SM90_SET" ARLE_GPU_CLAIMS="$TMP/reserved-set" ARLE_OP_ID="$reserve_op" ARLE_OWNER=test \
  ARLE_CLAIM_PID="$$" ARLE_CLAIM_START="$(awk '{print $22}' /proc/$$/stat)" \
  bash "$TREE/scripts/pick-gpu.sh" reserve-set 0,1,2,3,4,5,6,7 >/dev/null
for gpu in $(seq 0 7); do [ "$(awk -F= '$1=="op" {print $2}' "$TMP/reserved-set/$gpu")" = "$reserve_op" ]; done
set +e
SMI="$SM90_SET" SMI_COMPUTE='GPU-3\n' ARLE_GPU_CLAIMS="$TMP/free-set" bash "$TREE/scripts/pick-gpu.sh" check-free-set 0,1,2,3,4,5,6,7 >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ]
set +e
SMI="$SM90_SET" ARLE_GPU_CLAIMS="$TMP/free-set" bash "$TREE/scripts/pick-gpu.sh" check-free-set 0,1,1,2 >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ]
mkdir -p "$TMP/tp4-busy"
printf 'schema=arle-gpu-claim-v1\nop=foreign\npid=%s\nstart=%s\n' "$$" "$(awk '{print $22}' /proc/$$/stat)" > "$TMP/tp4-busy/2"
set +e
SMI="$TP4_SET" ARLE_GPU_CLAIMS="$TMP/tp4-busy" bash "$TREE/scripts/pick-gpu.sh" check-free-set 0,1,2,3 >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && [ ! -e "$TMP/tp4-busy/0" ] && [ ! -e "$TMP/tp4-busy/1" ] && [ ! -e "$TMP/tp4-busy/3" ]
set +e
SMI="$TP4_SET" SMI_COMPUTE='GPU-2\n' ARLE_GPU_CLAIMS="$TMP/tp4-free" bash "$TREE/scripts/pick-gpu.sh" reserve-set 0,1,2,3 >/dev/null 2>&1
rc=$?; set -e
[ "$rc" -ne 0 ] && [ ! -e "$TMP/tp4-free/0" ] && [ ! -e "$TMP/tp4-free/1" ] && [ ! -e "$TMP/tp4-free/2" ] && [ ! -e "$TMP/tp4-free/3" ]
low_compute="$(ARLE_GPU_CLAIMS="$TMP/low-compute" ARLE_OP_ID=ours ARLE_OWNER=test SMI='0, GPU-0, 1, 9.0\n1, GPU-1, 0, 9.0' SMI_COMPUTE='GPU-0\n' bash "$TREE/scripts/pick-gpu.sh")"
[ "$low_compute" = 1 ]

for mismatch in kind expected_helper operation start pgid; do
  cp "$STATE/runs/shared/process" "$TMP/process.good"
  case "$mismatch" in
    kind) key=kind; value=build ;;
    expected_helper) key=expected_helper; value=/wrong/helper ;;
    operation) key=operation; value=wrong-op ;;
    start) key=start; value=0 ;;
    pgid) key=pgid; value=0 ;;
  esac
  python3 - "$STATE/runs/shared/process" "$key" "$value" <<'PY'
import sys
path, key, value = sys.argv[1:]
lines = open(path).read().splitlines()
open(path, "w").write("\n".join(f"{key}={value}" if line.startswith(key + "=") else line for line in lines) + "\n")
PY
  rm -f "$KILL_MARKER"
  set +e
  PROC_ROOT="$TMP/proc" KILL_CMD="$BIN/mock-kill" POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-run.sh" status shared >/dev/null 2>&1
  status_rc=$?
  PROC_ROOT="$TMP/proc" KILL_CMD="$BIN/mock-kill" POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-run.sh" kill shared >/dev/null 2>&1
  kill_rc=$?; set -e
  [ "$status_rc" -ne 0 ] && [ "$kill_rc" -ne 0 ] && [ ! -e "$KILL_MARKER" ]
  mv "$TMP/process.good" "$STATE/runs/shared/process"
done

echo "pod flow tests: PASS"
