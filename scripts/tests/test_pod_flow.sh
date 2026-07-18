#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d -t pod-flow-XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
LOCAL="$TMP/local"; NODE="$TMP/node/tree"; TREE="$TMP/pod/tree"; STATE="$TMP/state"; BIN="$TMP/bin"
mkdir -p "$LOCAL/scripts" "$TMP/node" "$TMP/pod" "$STATE" "$BIN"
cp "$ROOT/.gitignore" "$LOCAL/"
cp "$ROOT/scripts/"{pod.sh,pod-remote-build.sh,pod-remote-run.sh,pick-gpu.sh,reap_run.py,pod-build-env.sh,pod-tilelang-env.sh,cuda_prebuilt_manifest.sh} "$LOCAL/scripts/"

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
mkdir -p "$POD_TREE/target/release" "$POD_TREE/target/test-manifest"
printf '#!/usr/bin/env bash\nexit 0\n' > "$POD_TREE/target/release/arle"
chmod +x "$POD_TREE/target/release/arle"
printf 'schema=1\nkernel_build_id=kernel-123\n' > "$POD_TREE/target/test-manifest/arle-cuda-kernels.manifest"
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
printf '%s\n' "${SMI:-0, 0 MiB\n1, 0 MiB}"
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
POD_TREE="$TREE" POD_STATE="$STATE" SMI='0, 0 MiB' setsid bash "$TREE/scripts/pod-remote-run.sh" run good exact auto op-exact "$TMP/argv" >/dev/null
rc=$?; set -e
[ "$rc" -eq 0 ] || { command cat "$STATE/runs/exact/log" >&2; exit 1; }
[ -f "$TMP/seen" ] || { command cat "$STATE/runs/exact/log" >&2; exit 1; }
cmp "$STATE/runs/exact/argv.nul" "$TMP/seen"
[ "$(sha256sum "$TREE/target/release/arle" | cut -d' ' -f1)" = "$bsha" ]

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

out1="$("$LOCAL/scripts/pod.sh" build 2>/dev/null)"; out2="$("$LOCAL/scripts/pod.sh" build 2>/dev/null)"
label1="$(printf '%s' "$out1" | awk -F"'" '{print $2}')"; label2="$(printf '%s' "$out2" | awk -F"'" '{print $2}')"
[ -n "$label1" ] && [ "$label1" != "$label2" ]

printf '%s\0' --release --features cuda --bin arle > "$TMP/build-argv"
POD_TREE="$TREE" POD_STATE="$STATE" bash "$TREE/scripts/pod-remote-build.sh" build manifest op-manifest "$TMP/build-argv" >/dev/null
[ "$(awk -F= '$1=="kernel_id" {print $2}' "$STATE/builds/manifest/receipt")" = kernel-123 ]

mkdir -p "$STATE/builds/shared" "$STATE/runs/shared" "$TMP/proc/4242"
printf 'exit=0\n' > "$STATE/builds/shared/receipt"
printf '%s' 'done' > "$STATE/builds/shared/log"
printf '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 12345\n' > "$TMP/proc/4242/stat"
printf 'pod-remote-run.sh active-run\0' > "$TMP/proc/4242/cmdline"
printf 'op=active-run\npid=4242\npgid=4242\nstart=12345\n' > "$STATE/runs/shared/process"
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
gpu="$(ARLE_GPU_CLAIMS="$TMP/claims" ARLE_OP_ID=ours ARLE_OWNER=test SMI='0, 0 MiB\n1, 0 MiB' bash "$TREE/scripts/pick-gpu.sh")"
[ "$gpu" = 1 ] && kill -0 $$

echo "pod flow tests: PASS"
