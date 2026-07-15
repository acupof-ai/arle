#!/usr/bin/env bash
# Immutable TileLang AOT bundles on the `kernel-artifacts` GitHub Release.
# Asset names are exact source/toolchain identities; bundle and file SHA-256
# metadata reject partial, stale, or overwritten content.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN="$ROOT/crates/cuda-kernels/generated"
REL="${ARLE_KERNEL_RELEASE_TAG:-kernel-artifacts}"
REPO="${GITHUB_REPOSITORY:-cklxx/arle}"
LANE="${ARLE_KERNEL_BUNDLE_LANE:-t1}"
ARCHS="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
CUDA_CONTRACT="${ARLE_KERNEL_BUNDLE_CUDA_CONTRACT:-12.8.0}"
CORRECTNESS_STATUS="${ARLE_KERNEL_CORRECTNESS_STATUS:-unverified-local}"
source "$ROOT/scripts/cuda_prebuilt_manifest.sh"

kernel_bundle_identity() {
    # Hash git-TRACKED content (blob/tree object per path), not the working tree:
    # `find -type f` pulls in gitignored __pycache__/*.pyc that any codegen run
    # drops into tools/tilelang, so the id differed between the publish job (post
    # codegen, has .pyc) and a pristine consumer (no .pyc) — the bundle was
    # unfetchable. `git rev-parse HEAD:<path>` is deterministic and pyc-immune
    # (same helper the prebuilt manifest already uses).
    local tilelang_inputs
    tilelang_inputs=$(
        for path in crates/cuda-kernels/build.rs crates/cuda-kernels/kernels.toml \
            crates/cuda-kernels/tools/tilelang requirements-build.txt; do
            printf 'input\t%s\t%s\n' "$path" "$(cuda_prebuilt_tree_hash "$path")"
        done | cuda_prebuilt_hash_stream
    )
    cat <<EOF
schema=4
lane=$LANE
arches=$ARCHS
tilelang_inputs=$tilelang_inputs
cuda_contract=$CUDA_CONTRACT
flashqla_gdr=${ARLE_CUDA_ENABLE_FLASHQLA_GDR:-}
EOF
}

kernel_bundle_manifest() {
    local id="$1" abi_hash="$2" symbols_hash="$3" nvcc="$CUDA_HOME/bin/nvcc"
    [[ -x "$nvcc" ]] || nvcc="$(command -v nvcc 2>/dev/null || true)"
    cat <<EOF
{
  "schema": 3,
  "bundle_id": "$id",
  "lane": "$LANE",
  "arches": "$ARCHS",
  "cuda_contract": "$CUDA_CONTRACT",
  "tilelang_inputs_sha256": "$(cd "$ROOT" && cuda_prebuilt_files_hash crates/cuda-kernels/build.rs crates/cuda-kernels/kernels.toml crates/cuda-kernels/tools/tilelang requirements-build.txt)",
  "nvcc_sha256": "$(if [[ -n "$nvcc" ]]; then cuda_prebuilt_command_id "$nvcc" --version; else printf missing; fi)",
  "host_compiler_sha256": "$(cuda_prebuilt_command_id "${NVCC_CCBIN:-g++}" --version)",
  "python_sha256": "$(cuda_prebuilt_command_id python3 --version)",
  "flashqla_gdr": "${ARLE_CUDA_ENABLE_FLASHQLA_GDR:-}",
  "abi_sha256": "$abi_hash",
  "symbol_allowlist": "symbols.txt",
  "symbol_allowlist_sha256": "$symbols_hash",
  "correctness_status": "$CORRECTNESS_STATUS",
  "files": "SHA256SUMS"
}
EOF
}

kernel_bundle_id() {
    (cd "$ROOT" && kernel_bundle_identity) | cuda_prebuilt_hash_stream
}

bundle_name() {
    printf 'arle-kernels-%s-%s.tar.gz\n' "$LANE" "$(kernel_bundle_id)"
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
    local archive="$1" checksum="$2" expected_id="$3"
    local tmp entry recorded_id recorded_abi recorded_symbols actual_symbols
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
    verify_tree_checksums "$tmp"
    rm -rf "$tmp"
}

pack_bundle() {
    local id file stage epoch abi_hash symbols_hash
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
    rm -f "$stage/manifest.json" "$stage/SHA256SUMS" "$stage/symbols.txt"
    find "$stage" -name meta.txt -type f -exec sed -n 's/^FUNC_NAME=//p' {} + |
        LC_ALL=C sort -u >"$stage/symbols.txt"
    [[ -s "$stage/symbols.txt" ]] || {
        echo "bundle contains no exported TileLang symbols" >&2
        rm -rf "$stage"
        return 1
    }
    abi_hash="$(cuda_prebuilt_hash_file "$ROOT/crates/cuda-kernels/kernels.toml")"
    symbols_hash="$(cuda_prebuilt_hash_file "$stage/symbols.txt")"
    kernel_bundle_manifest "$id" "$abi_hash" "$symbols_hash" >"$stage/manifest.json"
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

remote_assets() {
    gh release view "$REL" -R "$REPO" --json assets --jq '.assets[].name'
}

case "${1:-help}" in
    id)
        kernel_bundle_id
        ;;
    pack)
        cd "$ROOT"
        pack_bundle
        ;;
    publish)
        cd "$ROOT"
        case "$CORRECTNESS_STATUS" in
            ""|unverified-local|failed)
                echo "publish requires explicit non-failing ARLE_KERNEL_CORRECTNESS_STATUS" >&2
                exit 1
                ;;
        esac
        tar --version 2>/dev/null | grep -q 'GNU tar' || {
            echo "publishing requires GNU tar for canonical metadata" >&2
            exit 1
        }
        file="$(pack_bundle)"
        checksum="$file.sha256"
        id="$(kernel_bundle_id)"
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
            verify_archive "$tmp/$file" "$tmp/$checksum" "$id"
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
    fetch)
        cd "$ROOT"
        id="$(kernel_bundle_id)"
        file="arle-kernels-$LANE-$id.tar.gz"
        checksum="$file.sha256"
        tmp="$(mktemp -d)"
        source_ref="${2:-$REL}"
        if [[ -d "$source_ref" ]]; then
            cp "$source_ref/$file" "$source_ref/$checksum" "$tmp/"
        else
            gh release download "$source_ref" -R "$REPO" -p "$file" -p "$checksum" -D "$tmp"
        fi
        verify_archive "$tmp/$file" "$tmp/$checksum" "$id"
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
