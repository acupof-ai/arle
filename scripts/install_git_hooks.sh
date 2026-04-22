#!/usr/bin/env bash
#
# Configure Git to use the repo-managed hooks in .githooks/.
#
# Usage:
#   scripts/install_git_hooks.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

git -C "${REPO_ROOT}" config core.hooksPath .githooks
echo "[install-hooks] configured core.hooksPath=.githooks"
