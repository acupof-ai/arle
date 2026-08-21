#!/usr/bin/env python3
"""Guardrails for public docs, templates, and repository hygiene.

This checker stays intentionally lightweight:
- public/governance docs and GitHub templates
- workspace-members <-> codebase-map truth-surface sync (refactor roadmap R0.2)
- repo-wide banned-marker scan on tracked text files
- docs/experience entry inventory caps
- no external dependencies
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_DOCS = [
    Path("README.md"),
    Path("README.zh-CN.md"),
    Path("CONTRIBUTING.md"),
    Path("CHANGELOG.md"),
]

GOVERNANCE_DOCS = [
    Path("docs/http-api.md"),
    Path("docs/support-matrix.md"),
    Path("docs/stability-policy.md"),
    Path("docs/perf-and-correctness-gates.md"),
    Path("docs/release-checklist.md"),
    Path("docs/environment.md"),
    Path("docs/bench-and-trace-spec.md"),
    Path("docs/index.md"),
]

TEMPLATE_DOCS = [
    Path(".github/PULL_REQUEST_TEMPLATE.md"),
    Path(".github/ISSUE_TEMPLATE/bug_report.md"),
    Path(".github/ISSUE_TEMPLATE/feature_request.md"),
]

PUBLIC_CHECK_FILES = PUBLIC_DOCS + GOVERNANCE_DOCS + TEMPLATE_DOCS

PR_TEMPLATE_REQUIRED_HEADINGS = [
    "## Summary",
    "## Why",
    "## Surface Area",
    "## Stability / Support / Compatibility",
    "## Docs Updated",
    "## Validation",
    "## Benchmark / Profiling Evidence",
    "## Migration Notes",
]

PR_TEMPLATE_REQUIRED_DOC_REFS = [
    "docs/support-matrix.md",
    "docs/stability-policy.md",
    "docs/perf-and-correctness-gates.md",
    "docs/release-checklist.md",
]

BUG_TEMPLATE_REQUIRED_FIELDS = [
    "## Surface",
    "## Steps to Reproduce",
    "## Expected Behavior",
    "## Actual Behavior",
    "## Environment",
    "## Evidence",
    "- **Backend**:",
    "- **Command / server flags**:",
]

FEATURE_TEMPLATE_REQUIRED_FIELDS = [
    "## Problem",
    "## Proposed Surface",
    "## Proposed Solution",
    "## Alternatives Considered",
    "## Compatibility / Migration Impact",
    "## Success Criteria",
]

DISALLOWED_PUBLIC_MARKERS = [
    ".claude/",
    "/Users/",
    "/content/workspace/",
    "file://",
]

MAX_EXPERIENCE_ENTRIES = {
    Path("docs/experience/wins"): 790,
    Path("docs/experience/errors"): 296,
}

REPO_WIDE_DISALLOWED_MARKERS = [
    "/Users/",
    "PEGAINFER",
    "release/infer",
]

JUNK_PATH_RE = re.compile(r"(^|/)(\.DS_Store|Thumbs\.db|__pycache__/|.*\.pyc)$")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

WORKSPACE_MANIFEST = Path("Cargo.toml")
CODEBASE_MAP = Path("docs/codebase-map.md")
WORKSPACE_MEMBER_RE = re.compile(r'^\s*"crates/([A-Za-z0-9_-]+)"\s*,?\s*$')
# Includes `*` so wildcard mentions like `crates/infer-*` are captured whole
# (and then skipped) instead of truncating to a non-existent crate name.
CRATE_REF_RE = re.compile(r"crates/([A-Za-z0-9_*-]+)")


def repo_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_required_files() -> list[str]:
    errors = []
    for rel_path in PUBLIC_CHECK_FILES:
        abs_path = ROOT / rel_path
        if not abs_path.exists():
            errors.append(f"missing required file: {rel_path}")
    return errors


def normalize_link_target(doc_path: Path, target: str) -> Path | None:
    if not target or target.startswith(("http://", "https://", "mailto:", "#")):
        return None

    target = target.split("#", 1)[0].strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if not target:
        return None

    if ":" in target and not target.startswith(("/", "./", "../")):
        maybe_file = target.split(":", 1)[0]
        maybe_path = (doc_path.parent / maybe_file).resolve()
        if maybe_path.exists():
            target = maybe_file

    candidate = Path(target)
    if candidate.is_absolute():
        return candidate
    return (doc_path.parent / candidate).resolve()


def check_markdown_links(paths: list[Path]) -> list[str]:
    errors = []
    for rel_path in paths:
        abs_path = ROOT / rel_path
        text = load_text(abs_path)
        for match in MARKDOWN_LINK_RE.finditer(text):
            target = match.group(1).strip()
            resolved = normalize_link_target(abs_path, target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(f"{rel_path}: broken local link -> {target}")
    return errors


def check_disallowed_markers(paths: list[Path]) -> list[str]:
    errors = []
    for rel_path in paths:
        text = load_text(ROOT / rel_path)
        for marker in DISALLOWED_PUBLIC_MARKERS:
            if marker in text:
                errors.append(f"{rel_path}: contains private/local path marker {marker!r}")
    return errors


def check_template(path: Path, required_strings: list[str]) -> list[str]:
    text = load_text(ROOT / path)
    missing = [item for item in required_strings if item not in text]
    if not missing:
        return []
    joined = ", ".join(missing)
    return [f"{path}: missing required template fields: {joined}"]


def parse_workspace_members() -> list[str]:
    members: list[str] = []
    in_members = False
    for line in load_text(ROOT / WORKSPACE_MANIFEST).splitlines():
        stripped = line.strip()
        if stripped.startswith("members = ["):
            in_members = True
            continue
        if in_members:
            if stripped.startswith("]"):
                break
            match = WORKSPACE_MEMBER_RE.match(line)
            if match:
                members.append(match.group(1))
    return members


def check_workspace_truth_surface() -> list[str]:
    """codebase-map.md is the canonical workspace topology; keep it mechanically
    in sync with the Cargo workspace so a new crate cannot land undocumented."""
    errors = []
    members = parse_workspace_members()
    if not members:
        return [f"{WORKSPACE_MANIFEST}: could not parse [workspace] members"]
    map_text = load_text(ROOT / CODEBASE_MAP)
    for name in members:
        if f"crates/{name}" not in map_text:
            errors.append(
                f"{CODEBASE_MAP}: workspace member crates/{name} is undocumented"
            )
    for name in sorted(set(CRATE_REF_RE.findall(map_text))):
        if "*" in name:
            continue
        if not (ROOT / "crates" / name).is_dir():
            errors.append(
                f"{CODEBASE_MAP}: references crates/{name}, which does not exist in the tree"
            )
    return errors


def check_git_tracked_junk() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        candidates = output.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        candidates = [
            repo_path(path)
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]

    offenders = [line for line in candidates if JUNK_PATH_RE.search(line)]
    if not offenders:
        return []
    return [f"tracked junk file: {path}" for path in offenders]


def list_git_tracked_files(*paths: Path) -> list[str]:
    command = ["git", "ls-files", "--"]
    command.extend(str(path) for path in paths)
    try:
        output = subprocess.check_output(
            command,
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        tracked = []
        for path in paths:
            abs_path = ROOT / path
            if abs_path.is_file():
                tracked.append(repo_path(abs_path))
                continue
            if abs_path.is_dir():
                tracked.extend(
                    repo_path(candidate)
                    for candidate in abs_path.rglob("*")
                    if candidate.is_file()
                )
        return sorted(tracked)
    return [line for line in output.splitlines() if line]


def list_experience_entries(path: Path) -> list[str]:
    return [
        rel_path
        for rel_path in list_git_tracked_files(path)
        if Path(rel_path).parent == path and Path(rel_path).suffix == ".md"
    ]


def check_repo_wide_disallowed_markers() -> list[str]:
    own_path = repo_path(Path(__file__).resolve())
    try:
        command = ["git", "grep", "-I", "-n"]
        for marker in REPO_WIDE_DISALLOWED_MARKERS:
            command.extend(["-e", marker])
        command.extend(["--", "."])
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        output = None
    else:
        if result.returncode == 0:
            output = result.stdout
        elif result.returncode == 1:
            output = ""
        else:
            output = None

    if output is not None:
        errors = set()
        for line in output.splitlines():
            path_str, _, content = line.partition(":")
            if not content or path_str == own_path:
                continue
            for marker in REPO_WIDE_DISALLOWED_MARKERS:
                if marker in content:
                    errors.add(
                        f"{path_str}: contains repo-wide banned marker {marker!r}"
                    )
        return sorted(errors)

    errors = []
    for rel_path in list_git_tracked_files(Path(".")):
        if rel_path == own_path:
            continue
        abs_path = ROOT / rel_path
        try:
            text = abs_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except UnicodeDecodeError:
            continue
        for marker in REPO_WIDE_DISALLOWED_MARKERS:
            if marker in text:
                errors.append(
                    f"{rel_path}: contains repo-wide banned marker {marker!r}"
                )
    return errors


# Production consumers call CUDA kernels through the typed launchers in
# crates/cuda-kernels/src/<family>.rs; a direct `ffi::<symbol>(` call outside
# that crate bypasses the shape/pointer guards and the registry. Examples and
# benches are exempt (they are probes, not serving paths).
LAUNCHER_BOUNDARY_PATHS = ("crates/infer-cuda/src", "crates/infer-api/src", "crates/cli/src", "crates/train/src")


def check_launcher_boundary() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "grep", "-I", "-n", "-E", r"ffi::[a-z][a-z0-9_]*\(", "--", *LAUNCHER_BOUNDARY_PATHS],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    hits = [line for line in result.stdout.splitlines() if "/ffi/" not in line and "/ffi.rs:" not in line]
    return [f"raw CUDA FFI call outside cuda-kernels (use a typed launcher): {hit}" for hit in hits]


# Every implementation id the runtime can report through /v1/stats
# (`implementation_hits`) must be a registry row, so a counter name and the
# registry never drift apart: the registry is what the receipts are read against.
REGISTRY_PATH = Path("operators/registry.toml")
RUNTIME_COUNTER_PATHS = ("crates/infer-cuda/src",)


def check_registry_covers_runtime_counters() -> list[str]:
    registry = load_text(ROOT / REGISTRY_PATH)
    registry_ids = set(re.findall(r'^id = "([^"]+)"', registry, re.MULTILINE))
    try:
        result = subprocess.run(
            ["git", "grep", "-I", "-h", "-o", "-E", r'"cuda\.[a-z0-9_]+(\.[a-z0-9_]+)+"', "--", *RUNTIME_COUNTER_PATHS],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    runtime_ids = {hit.strip('"') for hit in result.stdout.split()}
    return [
        f"runtime implementation id {rid!r} has no row in {REGISTRY_PATH}"
        for rid in sorted(runtime_ids - registry_ids)
    ]


def check_experience_doc_inventory() -> list[str]:
    errors = []
    for rel_path, max_entries in MAX_EXPERIENCE_ENTRIES.items():
        count = len(list_experience_entries(rel_path))
        if count > max_entries:
            errors.append(
                f"{rel_path}: top-level markdown entry count {count} exceeds cap {max_entries}; archive or consolidate old entries before adding more"
            )
    return errors


def main() -> int:
    errors: list[str] = []

    errors.extend(check_required_files())
    errors.extend(check_markdown_links(PUBLIC_CHECK_FILES))
    errors.extend(check_disallowed_markers(PUBLIC_CHECK_FILES))
    errors.extend(check_template(Path(".github/PULL_REQUEST_TEMPLATE.md"), PR_TEMPLATE_REQUIRED_HEADINGS))
    errors.extend(check_template(Path(".github/PULL_REQUEST_TEMPLATE.md"), PR_TEMPLATE_REQUIRED_DOC_REFS))
    errors.extend(check_template(Path(".github/ISSUE_TEMPLATE/bug_report.md"), BUG_TEMPLATE_REQUIRED_FIELDS))
    errors.extend(check_template(Path(".github/ISSUE_TEMPLATE/feature_request.md"), FEATURE_TEMPLATE_REQUIRED_FIELDS))
    errors.extend(check_git_tracked_junk())
    errors.extend(check_experience_doc_inventory())
    errors.extend(check_repo_wide_disallowed_markers())
    errors.extend(check_workspace_truth_surface())
    errors.extend(check_launcher_boundary())
    errors.extend(check_registry_covers_runtime_counters())

    if errors:
        print("[repo-hygiene] FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("[repo-hygiene] OK")
    print(
        "[repo-hygiene] public docs, templates, local links, tracked junk, "
        "repo-wide marker bans, experience entry caps, workspace "
        "truth-surface, CUDA launcher-boundary, and registry-coverage checks all passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
