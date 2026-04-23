#!/usr/bin/env python3
"""Guardrails for public docs, templates, and repository hygiene.

This checker stays intentionally narrow:
- only public/governance docs and GitHub templates
- no deep validation of historical plans / experience logs
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
    Path("docs/compatibility.md"),
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
    "docs/compatibility.md",
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

JUNK_PATH_RE = re.compile(r"(^|/)(\.DS_Store|Thumbs\.db|__pycache__/|.*\.pyc)$")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


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


def check_latest_updates(path: Path, marker: str, max_entries: int) -> list[str]:
    text = load_text(ROOT / path)
    if marker not in text:
        return [f"{path}: missing marker {marker!r}"]
    after_marker = text.split(marker, 1)[1]
    entries = []
    for line in after_marker.splitlines():
        stripped = line.strip()
        if stripped.startswith("Full history:") or stripped.startswith("完整历史："):
            break
        if stripped.startswith("- **"):
            entries.append(stripped)
    if not entries:
        return [f"{path}: latest updates section has no entries"]
    if len(entries) > max_entries:
        return [f"{path}: latest updates section has {len(entries)} entries (max {max_entries})"]
    return []


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


def check_git_tracked_junk() -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return [f"git ls-files failed: {exc}"]

    offenders = [line for line in output.splitlines() if JUNK_PATH_RE.search(line)]
    if not offenders:
        return []
    return [f"tracked junk file: {path}" for path in offenders]


def main() -> int:
    errors: list[str] = []

    errors.extend(check_required_files())
    errors.extend(check_latest_updates(Path("README.md"), "## 📰 Latest Updates", 3))
    errors.extend(check_latest_updates(Path("README.zh-CN.md"), "## 📰 最新动态", 3))
    errors.extend(check_markdown_links(PUBLIC_CHECK_FILES))
    errors.extend(check_disallowed_markers(PUBLIC_CHECK_FILES))
    errors.extend(check_template(Path(".github/PULL_REQUEST_TEMPLATE.md"), PR_TEMPLATE_REQUIRED_HEADINGS))
    errors.extend(check_template(Path(".github/PULL_REQUEST_TEMPLATE.md"), PR_TEMPLATE_REQUIRED_DOC_REFS))
    errors.extend(check_template(Path(".github/ISSUE_TEMPLATE/bug_report.md"), BUG_TEMPLATE_REQUIRED_FIELDS))
    errors.extend(check_template(Path(".github/ISSUE_TEMPLATE/feature_request.md"), FEATURE_TEMPLATE_REQUIRED_FIELDS))
    errors.extend(check_git_tracked_junk())

    if errors:
        print("[repo-hygiene] FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("[repo-hygiene] OK")
    print(
        "[repo-hygiene] public docs, templates, local links, and tracked junk checks all passed"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
