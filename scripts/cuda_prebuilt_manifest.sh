#!/usr/bin/env bash

cuda_prebuilt_hash_stream() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum | awk '{print $1}'
    else
        shasum -a 256 | awk '{print $1}'
    fi
}

cuda_prebuilt_hash_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    else
        shasum -a 256 "$path" | awk '{print $1}'
    fi
}

cuda_prebuilt_files_hash() {
    local path
    {
        for path in "$@"; do
            if [[ -f "$path" ]]; then
                printf 'file\t%s\t%s\n' "$path" "$(cuda_prebuilt_hash_file "$path")"
            elif [[ -d "$path" ]]; then
                find "$path" -type f -print0 |
                    LC_ALL=C sort -z |
                    while IFS= read -r -d '' file; do
                        printf 'file\t%s\t%s\n' "$file" "$(cuda_prebuilt_hash_file "$file")"
                    done
            else
                printf 'missing\t%s\n' "$path"
            fi
        done
    } | cuda_prebuilt_hash_stream
}

cuda_prebuilt_command_id() {
    local command="$1"
    shift
    if ! command -v "$command" >/dev/null 2>&1; then
        printf 'missing'
        return
    fi
    {
        command -v "$command"
        "$command" "$@" 2>&1 || true
    } | cuda_prebuilt_hash_stream
}

cuda_prebuilt_tracked_hash() {
    local path="$1" root relative list merged hasher
    root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    [[ -n "$root" ]] || { cuda_prebuilt_files_hash "$path"; return; }
    relative="$path"
    [[ "$path" != /* ]] || relative="${path#"$root"/}"
    if command -v sha256sum >/dev/null 2>&1; then
        hasher=(sha256sum)
    else
        hasher=(shasum -a 256)
    fi
    list="$(mktemp)"
    merged="$(mktemp)"
    # Existing files are hashed in one batched invocation; per-file forking
    # made `kernel_artifacts.sh id` O(10s) on macOS (shasum is a Perl script).
    git ls-files -z -- "$relative" | LC_ALL=C sort -z |
        while IFS= read -r -d '' file; do
            if [[ -f "$root/$file" ]]; then
                printf 'F\t%s\n' "$file"
            else
                printf 'M\t%s\n' "$file"
            fi
        done >"$list"
    {
        grep '^F	' "$list" | cut -f2- | (cd "$root" && xargs -r "${hasher[@]}") |
            awk '{printf "F\t%s\t%s\n", $2, $1}'
        grep '^M	' "$list" || true
    } | LC_ALL=C sort -t$'\t' -k2,2 >"$merged"
    while IFS=$'\t' read -r tag file hash; do
        if [[ "$tag" == F ]]; then
            printf 'file\t%s\t%s\n' "$file" "$hash"
        else
            printf 'missing\t%s\n' "$file"
        fi
    done <"$merged" | cuda_prebuilt_hash_stream
    rm -f "$list" "$merged"
}

cuda_prebuilt_manifest_validate() {
    local manifest="$1" line key seen=$'\n'
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" == *=* ]] || { echo "CUDA prebuilt manifest line is not key=value: $line" >&2; return 1; }
        key="${line%%=*}"
        [[ -n "$key" ]] || { echo "CUDA prebuilt manifest has an empty key" >&2; return 1; }
        [[ "$seen" != *$'\n'"$key"$'\n'* ]] || {
            echo "CUDA prebuilt manifest key $key is duplicated" >&2
            return 1
        }
        seen+="$key"$'\n'
    done <"$manifest"
}

cuda_prebuilt_manifest_value() {
    local manifest="$1" key="$2" count line
    count="$(grep -c "^${key}=" "$manifest" || true)"
    [[ "$count" == 1 ]] || {
        echo "CUDA prebuilt manifest key $key must occur exactly once (found $count)" >&2
        return 1
    }
    line="$(grep "^${key}=" "$manifest")"
    printf '%s\n' "${line#*=}"
}

cuda_prebuilt_archive_symbols() {
    nm -g --defined-only "$1" 2>/dev/null |
        while read -r -a fields; do
            (( ${#fields[@]} > 0 )) && printf '%s\n' "${fields[${#fields[@]} - 1]}"
        done |
        LC_ALL=C sort -u
}

cuda_prebuilt_archive_has_symbol() {
    cuda_prebuilt_archive_symbols "$1" | grep -Fxq "$2"
}
