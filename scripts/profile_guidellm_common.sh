#!/usr/bin/env bash

set -euo pipefail

profile_repo_root() {
    local script_dir="$1"
    cd "$script_dir/.." && pwd
}

profile_require_command() {
    local name="$1"
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "error: required command not found on PATH: $name" >&2
        exit 2
    fi
}

profile_unique_dir() {
    local base="$1"
    local candidate="$base"
    local run=1
    while [[ -e "$candidate" ]]; do
        run=$((run + 1))
        candidate="${base}-run${run}"
    done
    printf '%s\n' "$candidate"
}

profile_target_port() {
    python3 - "$1" <<'PY'
from urllib.parse import urlparse
import sys

target = sys.argv[1]
parsed = urlparse(target)
if not parsed.scheme or not parsed.hostname:
    raise SystemExit(f"invalid target URL: {target}")
if parsed.port is not None:
    print(parsed.port)
elif parsed.scheme == "https":
    print(443)
else:
    print(80)
PY
}

profile_resolve_server_pid() {
    local target="$1"
    local explicit_pid="${2:-}"
    if [[ -n "$explicit_pid" ]]; then
        printf '%s\n' "$explicit_pid"
        return 0
    fi

    profile_require_command lsof

    local port
    port="$(profile_target_port "$target")"

    local pids
    pids="$(lsof -n -P -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -z "$pids" ]]; then
        echo "error: could not find a listening server PID on port $port (target: $target)" >&2
        echo "       pass --server-pid explicitly if the server is already running." >&2
        exit 2
    fi

    local pid
    pid="$(printf '%s\n' "$pids" | head -n1)"
    if [[ "$(printf '%s\n' "$pids" | wc -l | tr -d ' ')" -gt 1 ]]; then
        echo "warning: multiple listening PIDs found on port $port; using $pid" >&2
    fi
    printf '%s\n' "$pid"
}

profile_extract_output_dir_from_log() {
    python3 - "$1" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
patterns = [
    r"raw artefacts:\s*(\S+)",
    r"output\s*:\s*(\S+)",
]
matches = []
for pattern in patterns:
    matches = re.findall(pattern, text)
    if matches:
        break
if not matches:
    raise SystemExit("failed to locate bench output dir in log")
print(matches[-1])
PY
}

profile_sha256() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file"
    else
        echo "sha256-unavailable $file"
    fi
}

profile_build_guidellm_replay_script() {
    local command_txt="$1"
    local replay_output_dir="$2"
    local replay_script="$3"

    python3 - "$command_txt" "$replay_output_dir" "$replay_script" <<'PY'
import pathlib
import shlex
import sys

command_path = pathlib.Path(sys.argv[1])
output_dir = sys.argv[2]
script_path = pathlib.Path(sys.argv[3])

lines = [line.strip() for line in command_path.read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"empty command file: {command_path}")

env_pairs = {}
command_line = None
for line in lines:
    if "=" in line and " " not in line.split("=", 1)[0]:
        key, value = line.split("=", 1)
        env_pairs[key] = value
        continue
    if line.startswith("guidellm benchmark run "):
        command_line = line
        break

if command_line is None:
    raise SystemExit(f"missing `guidellm benchmark run ...` line in {command_path}")

tokens = shlex.split(command_line)
rewritten = []
skip_next = False
for idx, token in enumerate(tokens):
    if skip_next:
        skip_next = False
        continue
    if token == "--output-dir":
        rewritten.extend(["--output-dir", output_dir])
        skip_next = True
        continue
    rewritten.append(token)

script_lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    f"mkdir -p {shlex.quote(output_dir)}",
]
for key, value in env_pairs.items():
    script_lines.append(f"export {key}={shlex.quote(value)}")
script_lines.append("exec " + " ".join(shlex.quote(token) for token in rewritten))

script_path.write_text("\n".join(script_lines) + "\n")
PY

    chmod +x "$replay_script"
}
