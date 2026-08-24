#!/usr/bin/env python3
"""Sweep untracked build/run residue from the working tree.

Plan-then-delete, modeled on deepseek-harness's scripts/clean.ts: every
target is validated before any deletion, tracked files are never touched,
and every target must resolve inside the repository boundary. One failed
validation aborts the whole run with nothing deleted.

Usage:
    python3 scripts/clean_repo.py          # dry-run: print the plan
    python3 scripts/clean_repo.py --apply  # delete
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Walked past, never swept: expensive-to-rebuild state and local config.
PRUNE_DIRS = {
    ".git",
    ".claude",
    ".toolchains",
    ".venv",
    "models",
    "node_modules",
    "target",
    "target-off",
    "vendor",
}

# Cache directories swept anywhere in the tree.
CACHE_DIRS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".eggs"}

# Junk files swept anywhere in the tree.
JUNK_FILE_RE = re.compile(
    r"(?:"
    r"\.DS_Store"
    r"|\._.*"
    r".*\.pyc"
    r"|.*\.pyo"
    r"|.*\.egg"
    r"|.*\.profraw"
    r"|.*\.profdata"
    r"|\.codex"
    r")$"
)

# Directories swept only at the repository root (or under web/).
ROOT_DIRS = ["bench-output", "opd-output", "runs", "observe-data", "dist", "build", ".eggs"]
WEB_DIRS = ["web/dist", "web/.astro"]


def tracked_files() -> set[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return set(output.splitlines())


def has_tracked_under(rel_dir: str, tracked: set[str]) -> bool:
    prefix = rel_dir + "/"
    return any(path == rel_dir or path.startswith(prefix) for path in tracked)


def inside_root(path: Path) -> bool:
    root = ROOT.resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def collect_targets(tracked: set[str]) -> tuple[list[Path], list[str]]:
    targets: list[Path] = []
    errors: list[str] = []

    def consider(path: Path, *, allow_tracked_under: bool = False) -> None:
        if not path.exists() and not path.is_symlink():
            return
        rel = path.relative_to(ROOT).as_posix()
        if not inside_root(path):
            errors.append(f"refusing target outside repository: {rel}")
            return
        if rel in tracked:
            errors.append(f"refusing tracked file: {rel}")
            return
        if path.is_dir() and not allow_tracked_under and has_tracked_under(rel, tracked):
            errors.append(f"refusing directory with tracked content: {rel}")
            return
        targets.append(path)

    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [name for name in dirnames if name not in PRUNE_DIRS]
        base = Path(dirpath)
        for name in list(dirnames):
            if name in CACHE_DIRS:
                consider(base / name)
                dirnames.remove(name)
        for name in filenames:
            if JUNK_FILE_RE.match(name):
                consider(base / name)

    for rel in ROOT_DIRS + WEB_DIRS:
        consider(ROOT / rel)

    # Deduplicate (a root dir may also have been found by the walk) and sort.
    unique = sorted({Path(t).resolve() for t in targets})
    return unique, errors


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    tracked = tracked_files()
    targets, errors = collect_targets(tracked)

    if errors:
        print("clean: aborted, nothing deleted", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if not targets:
        print("clean: already clean")
        return 0

    for target in targets:
        print(target.relative_to(ROOT).as_posix())

    if not apply:
        print(f"\nclean: dry-run, {len(targets)} paths would be removed (pass --apply to delete)")
        return 0

    for target in targets:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
    print(f"\nclean: removed {len(targets)} paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
