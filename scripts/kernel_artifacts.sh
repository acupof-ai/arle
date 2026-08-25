#!/usr/bin/env bash
# Immutable TileLang AOT bundles on the `kernel-artifacts` GitHub Release.
# Asset names are exact source/toolchain identities; bundle and file SHA-256
# metadata reject partial, stale, or overwritten content.
# `sync`: pre-build — pull the current source's bundle into generated/, else no-op.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/crates/cuda-kernels/generated"
REL="${ARLE_KERNEL_RELEASE_TAG:-kernel-artifacts}"
REPO="${GITHUB_REPOSITORY:-cklxx/arle}"
LANE="${ARLE_KERNEL_BUNDLE_LANE:-t1}"
ARCHS="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
CUDA_CONTRACT="${ARLE_KERNEL_BUNDLE_CUDA_CONTRACT:-12.8.0}"
CORRECTNESS_STATUS="${ARLE_KERNEL_CORRECTNESS_STATUS:-not-run}"
CORRECTNESS_EVIDENCE="${ARLE_KERNEL_CORRECTNESS_EVIDENCE:-}"
TESTED_ARTIFACT_SHA256="${ARLE_KERNEL_TESTED_ARTIFACT_SHA256:-}"
source "$ROOT/scripts/cuda_prebuilt_manifest.sh"

validate_correctness_status() {
    case "$CORRECTNESS_STATUS" in
        not-run|passed|failed) ;;
        *)
            echo "invalid ARLE_KERNEL_CORRECTNESS_STATUS: $CORRECTNESS_STATUS (expected not-run|passed|failed)" >&2
            return 1
            ;;
    esac
}

source_commit() {
    git -C "$ROOT" rev-parse HEAD
}

validate_correctness_evidence() {
    local id="$1" commit
    commit="$(source_commit)"
    [[ -n "$CORRECTNESS_EVIDENCE" && -f "$CORRECTNESS_EVIDENCE" ]] || {
        echo "$CORRECTNESS_STATUS requires ARLE_KERNEL_CORRECTNESS_EVIDENCE" >&2
        return 1
    }
    jq -e --arg status "$CORRECTNESS_STATUS" --arg id "$id" --arg commit "$commit" \
        --arg artifact_sha "$TESTED_ARTIFACT_SHA256" \
        --arg tested_sms "$ARCHS" --arg capabilities "fa3" '
        .schema == 2 and .status == $status and .bundle_id == $id and
        .source_commit == $commit and .artifact_sha256 == $artifact_sha and
        ($artifact_sha | test("^[0-9a-f]{64}$")) and
        .tested_sms == ($tested_sms | split(";")) and
        .capabilities == ($capabilities | split(",")) and
        (keys | sort == ["artifact_sha256", "bundle_id", "capabilities", "schema", "source_commit", "status", "tested_sms"])
    ' "$CORRECTNESS_EVIDENCE" >/dev/null || {
        echo "correctness evidence is not bound to status=$CORRECTNESS_STATUS bundle_id=$id source_commit=$commit" >&2
        return 1
    }
}

stage_correctness_evidence() {
    local stage="$1" id="$2" commit
    commit="$(source_commit)"
    if [[ "$CORRECTNESS_STATUS" == not-run ]]; then
        printf '{"schema":2,"status":"not-run","bundle_id":"%s","source_commit":"%s","artifact_sha256":null,"tested_sms":[],"capabilities":[]}\n' \
            "$id" "$commit" >"$stage/correctness-evidence.json"
        return
    fi
    validate_correctness_evidence "$id"
    jq -cS . "$CORRECTNESS_EVIDENCE" >"$stage/correctness-evidence.json"
}

# Source trees whose content can change kernel bytes. infer-cuda/src is the
# consumer, not an input: codegen never reads it, and the FFI contract is
# guarded by ffi/ + kernels.toml's abi hash + the symbol allowlist. Including
# it forced a full ~1h rebuild on every runtime change.
KERNEL_INPUT_PATHS=(
    crates/cuda-kernels/build.rs
    crates/cuda-kernels/kernels.toml
    crates/cuda-kernels/tools/tilelang
    crates/cuda-kernels/csrc
    crates/cuda-kernels/src
    crates/cuda-kernels/vendor
    requirements-build.txt
)

kernel_bundle_identity() {
    # Hash tracked working-tree content; generated and untracked files cannot
    # change the identity, while dirty inputs cannot masquerade as HEAD.
    local tilelang_inputs
    tilelang_inputs=$(
        for path in "${KERNEL_INPUT_PATHS[@]}"; do
            printf 'input\t%s\t%s\n' "$path" "$(cuda_prebuilt_tracked_hash "$path")"
        done | cuda_prebuilt_hash_stream
    )
    cat <<EOF
schema=7
lane=$LANE
arches=$ARCHS
tilelang_inputs=$tilelang_inputs
cuda_contract=$CUDA_CONTRACT
EOF
}

kernel_bundle_manifest() {
    local id="$1" abi_hash="$2" symbols_hash="$3" evidence_hash="$4" nvcc="$CUDA_HOME/bin/nvcc"
    [[ -x "$nvcc" ]] || nvcc="$(command -v nvcc 2>/dev/null || true)"
    cat <<EOF
{
  "schema": 3,
  "bundle_id": "$id",
  "lane": "$LANE",
  "arches": "$ARCHS",
  "cuda_contract": "$CUDA_CONTRACT",
  "tilelang_inputs_sha256": "$(cd "$ROOT" && cuda_prebuilt_files_hash "${KERNEL_INPUT_PATHS[@]}")",
  "nvcc_sha256": "$(if [[ -n "$nvcc" ]]; then cuda_prebuilt_command_id "$nvcc" --version; else printf missing; fi)",
  "host_compiler_sha256": "$(cuda_prebuilt_command_id "${NVCC_CCBIN:-g++}" --version)",
  "python_sha256": "$(cuda_prebuilt_command_id python3 --version)",
  "abi_sha256": "$abi_hash",
  "symbol_allowlist": "symbols.txt",
  "symbol_allowlist_sha256": "$symbols_hash",
  "correctness_status": "$CORRECTNESS_STATUS",
  "correctness_evidence": "correctness-evidence.json",
  "correctness_evidence_sha256": "$evidence_hash",
  "source_commit": "$(source_commit)",
  "files": "SHA256SUMS"
}
EOF
}

kernel_bundle_id() {
    (cd "$ROOT" && kernel_bundle_identity) | cuda_prebuilt_hash_stream
}

write_tree_checksums() {
    local root="$1"
    (
        cd "$root"
        find . -type f ! -path './SHA256SUMS' -print0 |
            LC_ALL=C sort -z |
            while IFS= read -r -d '' file; do
                [[ "$file" != *$'\n'* ]] || { echo "newline in bundle path: $file" >&2; exit 1; }
                printf '%s  %s\n' "$(cuda_prebuilt_hash_file "$file")" "${file#./}"
            done
    ) >"$root/SHA256SUMS"
}

verify_tree_checksums() {
    local root="$1"
    local expected file actual
    while IFS='  ' read -r expected file; do
        file="${file# }"
        [[ -n "$expected" && -f "$root/$file" ]] || {
            echo "bundle file missing: $file" >&2
            return 1
        }
        actual="$(cuda_prebuilt_hash_file "$root/$file")"
        [[ "$actual" == "$expected" ]] || {
            echo "bundle file checksum mismatch: $file" >&2
            return 1
        }
    done <"$root/SHA256SUMS"
    while IFS= read -r -d '' file; do
        file="${file#"$root/"}"
        awk -v file="$file" 'substr($0, 67) == file { found = 1 } END { exit !found }' \
            "$root/SHA256SUMS" || {
            echo "bundle file absent from SHA256SUMS: $file" >&2
            return 1
        }
    done < <(find "$root" -type f ! -path "$root/SHA256SUMS" -print0)
}

verify_archive_checksum() {
    local archive="$1" checksum="$2" expected actual
    expected="$(awk 'NR == 1 {print $1}' "$checksum")"
    actual="$(cuda_prebuilt_hash_file "$archive")"
    [[ "$actual" == "$expected" ]] || {
        echo "bundle archive checksum mismatch: $(basename "$archive")" >&2
        return 1
    }
}

verify_archive() {
    local archive="$1" checksum="$2" expected_id="$3" require_passed="${4:-0}"
    local tmp entry recorded_id recorded_abi recorded_symbols actual_symbols
    local recorded_status recorded_commit recorded_evidence_hash actual_evidence_hash
    verify_archive_checksum "$archive" "$checksum"
    while IFS= read -r entry; do
        [[ "$entry" != /* && "/$entry/" != *'/../'* ]] || {
            echo "unsafe bundle path: $entry" >&2
            return 1
        }
    done < <(tar -tzf "$archive")
    tmp="$(mktemp -d)"
    tar -xzf "$archive" -C "$tmp"
    recorded_id="$(sed -n 's/^  "bundle_id": "\([^"]*\)",$/\1/p' "$tmp/manifest.json")"
    [[ "$recorded_id" == "$expected_id" ]] || {
        echo "bundle identity mismatch: expected=$expected_id actual=$recorded_id" >&2
        rm -rf "$tmp"
        return 1
    }
    recorded_abi="$(sed -n 's/^  "abi_sha256": "\([^"]*\)",$/\1/p' "$tmp/manifest.json")"
    [[ "$recorded_abi" == "$(cuda_prebuilt_hash_file "$ROOT/crates/cuda-kernels/kernels.toml")" ]] || {
        echo "bundle ABI hash mismatch" >&2
        rm -rf "$tmp"
        return 1
    }
    recorded_symbols="$(sed -n 's/^  "symbol_allowlist_sha256": "\([^"]*\)",$/\1/p' "$tmp/manifest.json")"
    actual_symbols="$(cuda_prebuilt_hash_file "$tmp/symbols.txt")"
    [[ "$recorded_symbols" == "$actual_symbols" ]] || {
        echo "bundle symbol allowlist hash mismatch" >&2
        rm -rf "$tmp"
        return 1
    }
    recorded_status="$(sed -n 's/^  "correctness_status": "\([^"]*\)",$/\1/p' "$tmp/manifest.json")"
    recorded_commit="$(sed -n 's/^  "source_commit": "\([^"]*\)",$/\1/p' "$tmp/manifest.json")"
    recorded_evidence_hash="$(sed -n 's/^  "correctness_evidence_sha256": "\([^"]*\)",$/\1/p' "$tmp/manifest.json")"
    actual_evidence_hash="$(cuda_prebuilt_hash_file "$tmp/correctness-evidence.json")"
    [[ "$recorded_evidence_hash" == "$actual_evidence_hash" ]] || {
        echo "bundle correctness evidence checksum mismatch" >&2
        rm -rf "$tmp"
        return 1
    }
    jq -e --arg status "$recorded_status" --arg id "$expected_id" --arg commit "$recorded_commit" \
        --arg arches "$ARCHS" '
        .schema == 2 and .status == $status and .bundle_id == $id and
        .source_commit == $commit and
        (if $status == "not-run" then
           .artifact_sha256 == null and .tested_sms == [] and .capabilities == []
         else (.artifact_sha256 | test("^[0-9a-f]{64}$")) and
           .tested_sms == ($arches | split(";")) and .capabilities == ["fa3"] end) and
        (keys | sort == ["artifact_sha256", "bundle_id", "capabilities", "schema", "source_commit", "status", "tested_sms"])
    ' "$tmp/correctness-evidence.json" >/dev/null || {
        echo "bundle correctness evidence binding mismatch" >&2
        rm -rf "$tmp"
        return 1
    }
    if [[ "$require_passed" == 1 && "$recorded_status" != passed ]]; then
        echo "formal kernel bundle requires passed correctness evidence, got $recorded_status" >&2
        rm -rf "$tmp"
        return 1
    fi
    verify_tree_checksums "$tmp"
    rm -rf "$tmp"
}

pack_bundle() {
    local id="$1" file stage epoch abi_hash symbols_hash evidence_hash
    validate_correctness_status
    [[ -d "$GEN" ]] || {
        echo "no $GEN - build with ARLE_KERNEL_VENDOR=1 first" >&2
        return 1
    }
    find "$GEN" -name meta.txt -type f -print -quit | grep -q . || {
        echo "$GEN contains no TileLang metadata" >&2
        return 1
    }
    id="$(kernel_bundle_id)"
    file="arle-kernels-$LANE-$id.tar.gz"
    stage="$(mktemp -d)"
    cp -R "$GEN/." "$stage/"
    rm -f "$stage/manifest.json" "$stage/SHA256SUMS" "$stage/symbols.txt" \
        "$stage/correctness-evidence.json"
    find "$stage" -name meta.txt -type f -exec sed -n 's/^FUNC_NAME=//p' {} + |
        LC_ALL=C sort -u >"$stage/symbols.txt"
    [[ -s "$stage/symbols.txt" ]] || {
        echo "bundle contains no exported TileLang symbols" >&2
        rm -rf "$stage"
        return 1
    }
    abi_hash="$(cuda_prebuilt_hash_file "$ROOT/crates/cuda-kernels/kernels.toml")"
    symbols_hash="$(cuda_prebuilt_hash_file "$stage/symbols.txt")"
    stage_correctness_evidence "$stage" "$id"
    evidence_hash="$(cuda_prebuilt_hash_file "$stage/correctness-evidence.json")"
    kernel_bundle_manifest "$id" "$abi_hash" "$symbols_hash" "$evidence_hash" >"$stage/manifest.json"
    write_tree_checksums "$stage"
    epoch="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || printf '0')}"
    if tar --version 2>/dev/null | grep -q 'GNU tar'; then
        tar -C "$stage" --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner -cf - . |
            gzip -n >"$ROOT/$file"
    else
        COPYFILE_DISABLE=1 tar -C "$stage" -cf - . | gzip -n >"$ROOT/$file"
    fi
    rm -rf "$stage"
    printf '%s  %s\n' "$(cuda_prebuilt_hash_file "$ROOT/$file")" "$file" >"$ROOT/$file.sha256"
    verify_archive "$ROOT/$file" "$ROOT/$file.sha256" "$id"
    printf '%s\n' "$file"
}

qualification_candidate() {
    local candidate="$1" tmp computed_sha
    if [[ -d "$candidate" ]]; then
        : "${ARLE_KERNEL_CANDIDATE_ARCHIVE_SHA256:?directory candidates require ARLE_KERNEL_CANDIDATE_ARCHIVE_SHA256}"
        [[ "$ARLE_KERNEL_CANDIDATE_ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
            echo "invalid ARLE_KERNEL_CANDIDATE_ARCHIVE_SHA256" >&2
            return 1
        }
        QUALIFICATION_TREE="$candidate"
        printf -v "$2" '%s' "$ARLE_KERNEL_CANDIDATE_ARCHIVE_SHA256"
        return
    fi
    [[ -f "$candidate" ]] || { echo "qualification candidate not found: $candidate" >&2; return 1; }
    computed_sha="$(cuda_prebuilt_hash_file "$candidate")"
    tmp="$(mktemp -d)"
    while IFS= read -r entry; do
        [[ "$entry" != /* && "/$entry/" != *'/../'* ]] || {
            echo "unsafe bundle path: $entry" >&2
            rm -rf "$tmp"
            return 1
        }
    done < <(tar -tzf "$candidate")
    tar -xzf "$candidate" -C "$tmp"
    QUALIFICATION_TREE="$tmp"
    printf -v "$2" '%s' "$computed_sha"
}

qualification_manifest_value() {
    local manifest="$1" key="$2"
    jq -er --arg key "$key" '.[$key] | select(type == "string" and length > 0)' "$manifest"
}

qualification_fragment() {
    local candidate="$1" stats="$2" output="$3" archive_sha tree producer binary_sha stats_sha
    local bundle_id source_commit kernel_id capabilities cleanup=""
    : "${ARLE_KERNEL_TEST_BINARY:?ARLE_KERNEL_TEST_BINARY is required}"
    : "${ARLE_KERNEL_TESTED_SM:?ARLE_KERNEL_TESTED_SM is required}"
    : "${ARLE_KERNEL_QUALIFICATION_PROFILE:?ARLE_KERNEL_QUALIFICATION_PROFILE is required}"
    [[ ${ARLE_KERNEL_TESTED_CAPABILITIES+x} ]] || {
        echo "ARLE_KERNEL_TESTED_CAPABILITIES is required (empty is valid)" >&2
        return 1
    }
    [[ -x "$ARLE_KERNEL_TEST_BINARY" ]] || { echo "test binary is not executable: $ARLE_KERNEL_TEST_BINARY" >&2; return 1; }
    [[ -f "$stats" ]] || { echo "stats JSON not found: $stats" >&2; return 1; }
    qualification_candidate "$candidate" archive_sha
    tree="$QUALIFICATION_TREE"
    [[ "$tree" == "$candidate" ]] || cleanup="$tree"
    trap '[[ -z "${cleanup:-}" ]] || rm -rf "$cleanup"' RETURN
    [[ -f "$tree/manifest.json" && -f "$tree/correctness-evidence.json" && -f "$tree/SHA256SUMS" ]] || {
        echo "candidate lacks pack metadata" >&2; return 1;
    }
    verify_tree_checksums "$tree"
    jq -e '.schema == 2 and .status == "not-run" and .artifact_sha256 == null and
        .tested_sms == [] and .capabilities == []' "$tree/correctness-evidence.json" >/dev/null || {
        echo "candidate is not an unqualified pack archive" >&2; return 1;
    }
    bundle_id="$(qualification_manifest_value "$tree/manifest.json" bundle_id)"
    source_commit="$(qualification_manifest_value "$tree/manifest.json" source_commit)"
    producer="$(find "$tree" -name arle-cuda-kernels.manifest -type f -print)"
    [[ "$(wc -l <<<"$producer" | tr -d ' ')" == 1 ]] || {
        echo "candidate must contain exactly one arle-cuda-kernels.manifest" >&2; return 1;
    }
    cuda_prebuilt_manifest_validate "$producer"
    kernel_id="$(cuda_prebuilt_manifest_value "$producer" kernel_build_id)"
    capabilities="$(cuda_prebuilt_manifest_value "$producer" capabilities)"
    binary_sha="$(cuda_prebuilt_hash_file "$ARLE_KERNEL_TEST_BINARY")"
    stats_sha="$(jq -er '.build_identity.product_binary_sha256 | select(test("^sha256:[0-9a-f]{64}$")) | sub("^sha256:"; "")' "$stats")"
    [[ "$binary_sha" == "$stats_sha" ]] || { echo "test binary SHA does not match /v1/stats product SHA" >&2; return 1; }
    [[ "$(jq -er '.build_identity.kernel_bundle_id' "$stats")" == "$kernel_id" ]] || {
        echo "runtime kernel bundle ID does not match candidate producer manifest" >&2; return 1;
    }
    python3 - "$archive_sha" "$bundle_id" "$source_commit" "$kernel_id" "$capabilities" \
        "$binary_sha" "$ARLE_KERNEL_TESTED_SM" "$ARLE_KERNEL_QUALIFICATION_PROFILE" \
        "$ARLE_KERNEL_TESTED_CAPABILITIES" "$output" <<'PY'
import json, re, sys
archive_sha, bundle_id, commit, kernel_id, bundle_csv, binary_sha, sm, profile, tested_csv, output = sys.argv[1:]
def csv(value): return [] if not value else value.split(',')
def valid(items): return len(items) == len(set(items)) and all(re.fullmatch(r'[a-z0-9-]+', x) for x in items)
bundle_caps, tested_caps = csv(bundle_csv), csv(tested_csv)
if not valid(bundle_caps) or not valid(tested_caps): raise SystemExit('invalid or duplicate capability list')
if profile not in {'qwen', 'qwen-fa3', 'dsv4'}: raise SystemExit('invalid qualification profile')
if sm not in {'8.0', '8.6', '8.9', '9.0'}: raise SystemExit('invalid tested SM')
allowed = {'qwen': set(), 'qwen-fa3': {'fa3'}, 'dsv4': {'flashmla', 'deepgemm-native'}}[profile]
if not set(tested_caps) <= allowed: raise SystemExit('false capability/profile claim')
if not set(tested_caps) <= set(bundle_caps): raise SystemExit('tested capability absent from bundle')
if profile == 'qwen' and sm not in {'8.0', '8.6', '8.9'}: raise SystemExit('generic qwen profile is Ampere/Ada only')
if profile == 'qwen-fa3' and (sm != '9.0' or tested_caps != ['fa3']): raise SystemExit('qwen-fa3 requires sm_90 and exactly fa3')
if profile == 'dsv4' and sm != '9.0': raise SystemExit('dsv4 qualification requires sm_90')
fragment = {'schema': 1, 'candidate_archive_sha256': archive_sha, 'bundle_id': bundle_id,
    'source_commit': commit, 'kernel_build_id': kernel_id, 'bundle_capabilities': sorted(bundle_caps),
    'product_binary_sha256': binary_sha, 'tested_sm': sm, 'profile': profile,
    'tested_capabilities': sorted(tested_caps)}
with open(output, 'w') as f: json.dump(fragment, f, sort_keys=True, separators=(',', ':')); f.write('\n')
PY
}

qualification_aggregate() {
    local candidate="$1" output="$2" archive_sha tree producer bundle_id source_commit kernel_id capabilities cleanup=""
    shift 2
    (( $# > 0 )) || { echo "aggregate requires evidence fragments" >&2; return 1; }
    qualification_candidate "$candidate" archive_sha
    tree="$QUALIFICATION_TREE"
    [[ "$tree" == "$candidate" ]] || cleanup="$tree"
    trap '[[ -z "${cleanup:-}" ]] || rm -rf "$cleanup"' RETURN
    verify_tree_checksums "$tree"
    bundle_id="$(qualification_manifest_value "$tree/manifest.json" bundle_id)"
    source_commit="$(qualification_manifest_value "$tree/manifest.json" source_commit)"
    producer="$(find "$tree" -name arle-cuda-kernels.manifest -type f -print)"
    [[ "$(wc -l <<<"$producer" | tr -d ' ')" == 1 ]] || { echo "candidate must contain exactly one producer manifest" >&2; return 1; }
    cuda_prebuilt_manifest_validate "$producer"
    kernel_id="$(cuda_prebuilt_manifest_value "$producer" kernel_build_id)"
    capabilities="$(cuda_prebuilt_manifest_value "$producer" capabilities)"
    python3 - "$archive_sha" "$bundle_id" "$source_commit" "$kernel_id" "$capabilities" "$output" "$@" <<'PY'
import json, re, sys
archive_sha, bundle_id, commit, kernel_id, caps_csv, output, *paths = sys.argv[1:]
keys = {'schema','candidate_archive_sha256','bundle_id','source_commit','kernel_build_id','bundle_capabilities','product_binary_sha256','tested_sm','profile','tested_capabilities'}
bundle_caps = sorted([] if not caps_csv else caps_csv.split(','))
rows, seen = [], set()
for path in paths:
    with open(path) as f: row = json.load(f)
    if set(row) != keys or row['schema'] != 1: raise SystemExit(f'invalid fragment schema: {path}')
    identity = (row['candidate_archive_sha256'], row['bundle_id'], row['source_commit'], row['kernel_build_id'], row['bundle_capabilities'])
    if identity != (archive_sha, bundle_id, commit, kernel_id, bundle_caps): raise SystemExit(f'mixed candidate evidence: {path}')
    if not re.fullmatch(r'[0-9a-f]{64}', row['product_binary_sha256']): raise SystemExit(f'invalid product SHA: {path}')
    key = (row['tested_sm'], row['profile'])
    if key in seen: raise SystemExit(f'duplicate tested_sm/profile: {key}')
    seen.add(key); rows.append(row)
required = {('8.0','qwen'), ('8.6','qwen'), ('8.9','qwen'), ('9.0','qwen-fa3')}
if not required <= seen: raise SystemExit('incomplete T1 hardware coverage')
for row in rows:
    allowed = {'qwen': set(), 'qwen-fa3': {'fa3'}, 'dsv4': {'flashmla','deepgemm-native'}}.get(row['profile'])
    if allowed is None or not set(row['tested_capabilities']) <= allowed: raise SystemExit('false capability/profile claim')
    if not set(row['tested_capabilities']) <= set(bundle_caps): raise SystemExit('capability overclaim')
    if row['profile'] == 'qwen' and (row['tested_sm'] not in {'8.0','8.6','8.9'} or row['tested_capabilities']): raise SystemExit('invalid generic qwen evidence')
    if row['profile'] == 'qwen-fa3' and (row['tested_sm'] != '9.0' or row['tested_capabilities'] != ['fa3']): raise SystemExit('invalid qwen-fa3 evidence')
    if row['profile'] == 'dsv4' and row['tested_sm'] != '9.0': raise SystemExit('invalid dsv4 evidence')
for cap in {'flashmla','deepgemm-native'} & set(bundle_caps):
    if not any(r['profile'] == 'dsv4' and cap in r['tested_capabilities'] for r in rows):
        raise SystemExit(f'missing dsv4 evidence for {cap}')
result = {'schema': 1, 'status': 'passed', 'candidate_archive_sha256': archive_sha,
    'bundle_id': bundle_id, 'source_commit': commit, 'kernel_build_id': kernel_id,
    'bundle_capabilities': bundle_caps,
    'observations': sorted(rows, key=lambda r: (r['tested_sm'], r['profile'], r['product_binary_sha256']))}
for row in result['observations']:
    for key in ('schema','candidate_archive_sha256','bundle_id','source_commit','kernel_build_id','bundle_capabilities'):
        del row[key]
with open(output, 'w') as f: json.dump(result, f, sort_keys=True, separators=(',', ':')); f.write('\n')
PY
}

qualification_policy_validate() (
    local candidate="$1" aggregate="$2" tmp
    [[ -f "$aggregate" ]] || { echo "aggregate qualification JSON not found: $aggregate" >&2; return 1; }
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    python3 - "$aggregate" "$tmp" <<'PY'
import json, os, sys
path, output = sys.argv[1:]
with open(path) as f: result = json.load(f)
keys = {'schema','status','candidate_archive_sha256','bundle_id','source_commit','kernel_build_id','bundle_capabilities','observations'}
if not isinstance(result, dict) or set(result) != keys or result['schema'] != 1 or result['status'] != 'passed':
    raise SystemExit('invalid aggregate schema or status')
if not isinstance(result['observations'], list): raise SystemExit('aggregate observations must be an array')
identity = {key: result[key] for key in ('candidate_archive_sha256','bundle_id','source_commit','kernel_build_id','bundle_capabilities')}
for index, observation in enumerate(result['observations']):
    if not isinstance(observation, dict): raise SystemExit('aggregate observation must be an object')
    row = {'schema': 1, **identity, **observation}
    with open(os.path.join(output, f'{index}.json'), 'w') as f:
        json.dump(row, f, sort_keys=True, separators=(',', ':')); f.write('\n')
PY
    qualification_aggregate "$candidate" "$tmp/rebuilt.json" "$tmp"/[0-9]*.json
    cmp "$aggregate" "$tmp/rebuilt.json" >/dev/null || {
        echo "aggregate qualification is not canonical or policy-valid" >&2
        return 1
    }
)

promotion_archive() {
    local candidate="$1" found
    if [[ -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
    fi
    [[ -d "$candidate" ]] || { echo "promotion candidate not found: $candidate" >&2; return 1; }
    found="$(find "$candidate" -maxdepth 1 -type f -name '*.tar.gz' -print)"
    [[ "$(wc -l <<<"$found" | tr -d ' ')" == 1 && -n "$found" ]] || {
        echo "promotion directory must contain exactly one .tar.gz candidate" >&2
        return 1
    }
    printf '%s\n' "$found"
}

qualification_publish() (
    local candidate="$1" aggregate="$2" archive file checksum sidecar assets present tmp target source_checksum
    archive="$(promotion_archive "$candidate")"
    source_checksum="$archive.sha256"
    [[ -f "$source_checksum" ]] || { echo "candidate checksum not found: $source_checksum" >&2; return 1; }
    verify_archive_checksum "$archive" "$source_checksum"
    qualification_policy_validate "$archive" "$aggregate"
    file="$(basename "$archive")"
    checksum="$file.sha256"
    sidecar="$file.qualification.json"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    cp "$source_checksum" "$tmp/$checksum"
    cp "$aggregate" "$tmp/$sidecar"

    if [[ -n "${ARLE_KERNEL_PROMOTE_DIR:-}" ]]; then
        mkdir -p "$ARLE_KERNEL_PROMOTE_DIR"
        present=0
        for target in "$file" "$checksum" "$sidecar"; do
            [[ -e "$ARLE_KERNEL_PROMOTE_DIR/$target" ]] && ((present += 1))
        done
        [[ "$present" == 0 || "$present" == 3 ]] || { echo "partial immutable qualification already promoted: $file" >&2; return 1; }
        if [[ "$present" == 0 ]]; then
            cp "$archive" "$ARLE_KERNEL_PROMOTE_DIR/$file"
            cp "$tmp/$checksum" "$tmp/$sidecar" "$ARLE_KERNEL_PROMOTE_DIR/"
        fi
        cmp "$archive" "$ARLE_KERNEL_PROMOTE_DIR/$file"
        cmp "$tmp/$checksum" "$ARLE_KERNEL_PROMOTE_DIR/$checksum"
        cmp "$tmp/$sidecar" "$ARLE_KERNEL_PROMOTE_DIR/$sidecar"
        verify_archive_checksum "$ARLE_KERNEL_PROMOTE_DIR/$file" "$ARLE_KERNEL_PROMOTE_DIR/$checksum"
        qualification_policy_validate "$ARLE_KERNEL_PROMOTE_DIR/$file" "$ARLE_KERNEL_PROMOTE_DIR/$sidecar"
        echo "promoted qualified candidate $file -> $ARLE_KERNEL_PROMOTE_DIR"
        return
    fi

    assets="$(remote_assets)"
    present=0
    for target in "$file" "$checksum" "$sidecar"; do
        grep -Fxq "$target" <<<"$assets" && ((present += 1))
    done
    [[ "$present" == 0 || "$present" == 3 ]] || { echo "partial immutable qualification already published: $file" >&2; return 1; }
    if [[ "$present" == 0 ]]; then
        gh release upload "$REL" -R "$REPO" "$archive" "$tmp/$checksum" "$tmp/$sidecar"
        echo "published qualified candidate $file -> release $REL"
        return
    fi
    mkdir "$tmp/remote"
    gh release download "$REL" -R "$REPO" -p "$file" -p "$checksum" -p "$sidecar" -D "$tmp/remote"
    cmp "$archive" "$tmp/remote/$file"
    cmp "$tmp/$checksum" "$tmp/remote/$checksum"
    cmp "$tmp/$sidecar" "$tmp/remote/$sidecar"
    verify_archive_checksum "$tmp/remote/$file" "$tmp/remote/$checksum"
    qualification_policy_validate "$tmp/remote/$file" "$tmp/remote/$sidecar"
    echo "verified existing qualified candidate $file"
)

remote_assets() {
    gh release view "$REL" -R "$REPO" --json assets --jq '.assets[].name'
}

case "${1:-help}" in
    sync)
        # Pull the current source's bundle into generated/ (build.rs then skips
        # TileLang codegen; nvcc still runs), or no-op → build from source.
        # Caller-run pre-build step: needs network + gh; build.rs never calls it.
        cd "$ROOT"
        id="$(kernel_bundle_id)"
        file="arle-kernels-$LANE-$id.tar.gz"
        if gh release view "$REL" -R "$REPO" --json assets \
            --jq '.assets[].name' 2>/dev/null | grep -Fxq "$file"; then
            "$0" fetch
        else
            echo "no published bundle for $id ($file); build from source" >&2
        fi
        ;;
    id)
        kernel_bundle_id
        ;;
    pack)
        cd "$ROOT"
        pack_bundle "$(kernel_bundle_id)"
        ;;
    qualify-fragment)
        [[ $# == 4 ]] || { echo "usage: $0 qualify-fragment CANDIDATE STATS_JSON OUTPUT_JSON" >&2; exit 1; }
        qualification_fragment "$2" "$3" "$4"
        ;;
    aggregate-qualification)
        (( $# >= 4 )) || { echo "usage: $0 aggregate-qualification CANDIDATE OUTPUT_JSON FRAGMENT..." >&2; exit 1; }
        qualification_aggregate "$2" "$3" "${@:4}"
        ;;
    qualify-publish)
        [[ $# == 3 ]] || { echo "usage: $0 qualify-publish CANDIDATE_ARCHIVE_OR_DIR AGGREGATE_JSON" >&2; exit 1; }
        qualification_publish "$2" "$3"
        ;;
    publish)
        cd "$ROOT"
        case "$CORRECTNESS_STATUS" in
            passed) id="$(kernel_bundle_id)"; validate_correctness_evidence "$id" ;;
            not-run) id="$(kernel_bundle_id)" ;;  # candidate: no GPU evidence yet
            *) echo "publish requires ARLE_KERNEL_CORRECTNESS_STATUS=passed or not-run" >&2; exit 1 ;;
        esac
        tar --version 2>/dev/null | grep -q 'GNU tar' || {
            echo "publishing requires GNU tar for canonical metadata" >&2
            exit 1
        }
        file="$(pack_bundle "$id")"
        checksum="$file.sha256"
        gh release view "$REL" -R "$REPO" >/dev/null 2>&1 ||
            gh release create "$REL" -R "$REPO" --prerelease \
                --title "Immutable TileLang kernel artifacts" \
                --notes "Exact, source-addressed TileLang AOT bundles."
        assets="$(remote_assets)"
        have_file=0
        have_checksum=0
        grep -Fxq "$file" <<<"$assets" && have_file=1
        grep -Fxq "$checksum" <<<"$assets" && have_checksum=1
        if [[ "$have_file" == 1 || "$have_checksum" == 1 ]]; then
            [[ "$have_file" == 1 && "$have_checksum" == 1 ]] || {
                echo "partial immutable bundle already published: $file" >&2
                exit 1
            }
            tmp="$(mktemp -d)"
            gh release download "$REL" -R "$REPO" -p "$file" -p "$checksum" -D "$tmp"
            # require_passed matches publish mode: candidate accepts not-run; qualified requires passed.
            require_passed=0
            [[ "$CORRECTNESS_STATUS" == passed ]] && require_passed=1
            verify_archive "$tmp/$file" "$tmp/$checksum" "$id" "$require_passed"
            cmp "$checksum" "$tmp/$checksum" >/dev/null || {
                echo "immutable bundle identity collision: $file" >&2
                rm -rf "$tmp"
                exit 1
            }
            rm -rf "$tmp"
            echo "verified existing immutable bundle $file"
            exit 0
        fi
        gh release upload "$REL" -R "$REPO" "$file" "$checksum"
        echo "published immutable $file -> release $REL"
        ;;
    fetch|fetch-qualified)
        cd "$ROOT"
        qualified=0
        [[ "$1" == fetch-qualified ]] && qualified=1
        id="$(kernel_bundle_id)"
        file="arle-kernels-$LANE-$id.tar.gz"
        checksum="$file.sha256"
        sidecar="$file.qualification.json"
        tmp="$(mktemp -d)"
        source_ref="${2:-$REL}"
        if [[ -d "$source_ref" ]]; then
            cp "$source_ref/$file" "$source_ref/$checksum" "$tmp/"
            [[ "$qualified" == 0 ]] || cp "$source_ref/$sidecar" "$tmp/"
        else
            patterns=(-p "$file" -p "$checksum")
            [[ "$qualified" == 0 ]] || patterns+=(-p "$sidecar")
            gh release download "$source_ref" -R "$REPO" "${patterns[@]}" -D "$tmp"
        fi
        verify_archive "$tmp/$file" "$tmp/$checksum" "$id"
        [[ "$qualified" == 0 ]] || qualification_policy_validate "$tmp/$file" "$tmp/$sidecar"
        stage="$ROOT/crates/cuda-kernels/generated.fetch.$$"
        rm -rf "$stage"
        mkdir -p "$stage"
        tar -xzf "$tmp/$file" -C "$stage"
        rm -rf "$GEN"
        mv "$stage" "$GEN"
        rm -rf "$tmp"
        count="$(find "$GEN" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
        echo "exact kernel bundle $id -> $GEN ($count artifact dirs)"
        ;;
    *)
        sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
        ;;
esac
