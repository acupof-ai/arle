#!/usr/bin/env python3
"""Guardrails for public docs, templates, and repository hygiene.

This checker stays intentionally lightweight:
- public/governance docs and GitHub templates
- workspace-members <-> codebase-map truth-surface sync (refactor roadmap R0.2)
- repo-wide banned-marker scan on tracked text files
- docs/experience entry inventory caps
- frozen docs/experience/archived seal (manifest hash drift = fail)
- no external dependencies
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# This script importlib-loads prereg.py and agenda.py to validate their ledgers.
# Writing their .pyc would create the junk that check_git_tracked_junk then
# reports, because the pre-push snapshot is not a git repo and that check falls
# back to walking the filesystem.
sys.dont_write_bytecode = True

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


def git_grep(args: list[str]) -> list[str] | None:
    """Lines from `git grep <args>`; [] when nothing matches, None when git is unusable."""
    try:
        result = subprocess.run(
            ["git", "grep", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode == 0:
        return result.stdout.splitlines()
    return [] if result.returncode == 1 else None


# A perf entry on/after this date must record the configuration its numbers came
# from, or waive it. Cost of not having this: a 2026-08-21 entry's numbers were
# used as a c-sweep baseline for three weeks; when the comparison finally had to
# be matched, the entry had no Parameters section, the run directory was gone,
# and there was no prereg row — so the baseline arm was re-run on the GPU.
WINS_PARAMS_CUTOFF = "2026-09-09"
_PARAMS_RE = re.compile(r"^#+\s*Parameters", re.MULTILINE)


def check_wins_parameters() -> list[str]:
    errors = []
    for rel in list_experience_entries(Path("docs/experience/wins")):
        if Path(rel).name[:10] < WINS_PARAMS_CUTOFF:
            continue
        text = (ROOT / rel).read_text(errors="ignore")
        if (
            len(_PERF_RE.findall(text)) >= 3
            and not _PARAMS_RE.search(text)
            and not _WAIVER_RE.search(text)
        ):
            errors.append(
                f"{rel}: an entry that reports numbers needs a Parameters section "
                f"recording the configuration they came from, or a 'no baseline' "
                f"waiver; without it the numbers cannot be a baseline later"
            )
    return errors


def check_repo_wide_disallowed_markers() -> list[str]:
    own_path = f"scripts/{Path(__file__).name}"
    command = ["-I", "-n"]
    for marker in REPO_WIDE_DISALLOWED_MARKERS:
        command.extend(["-e", marker])
    command.extend(["--", "."])
    lines = git_grep(command)
    output = None if lines is None else "\n".join(lines)

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
    # Textual by necessity: autograd and the probe examples consume
    # `cuda_kernels::ffi` directly, so the module cannot be pub(crate) yet.
    lines = git_grep(["-I", "-n", "-E", r"ffi::[a-z][a-z0-9_]*\(", "--", *LAUNCHER_BOUNDARY_PATHS]) or []
    hits = [line for line in lines if "/ffi/" not in line and "/ffi.rs:" not in line]
    return [f"raw CUDA FFI call outside cuda-kernels (use a typed launcher): {hit}" for hit in hits]


# Every implementation id the runtime can report through /v1/stats
# (`implementation_hits`) must be a registry row, so a counter name and the
# registry never drift apart: the registry is what the receipts are read against.
REGISTRY_PATH = Path("operators/registry.toml")
RUNTIME_COUNTER_PATHS = ("crates/infer-cuda/src",)


def check_registry_covers_runtime_counters() -> list[str]:
    registry = load_text(ROOT / REGISTRY_PATH)
    registry_ids = set(re.findall(r'^id = "([^"]+)"', registry, re.MULTILINE))
    lines = git_grep(["-I", "-h", "-o", "-E", r'"cuda\.[a-z0-9_]+(\.[a-z0-9_]+)+"', "--", *RUNTIME_COUNTER_PATHS]) or []
    runtime_ids = {line.strip().strip('"') for line in lines if line.strip()}
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


ARCHIVED_ROOT = Path("docs/experience/archived")
ARCHIVED_MANIFEST = ARCHIVED_ROOT / "manifest.json"
ARCHIVED_NAME_RE = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md")
ARCHIVED_CLASSES = ("wins", "errors")


def check_archived_experience() -> list[str]:
    """Sealed entries are frozen: every archived file must match its manifest
    hash, every manifest row must exist on disk, and the tree may hold nothing
    except wins/errors entries named YYYY-MM-DD-slug.md."""
    archived = ROOT / ARCHIVED_ROOT
    if not archived.is_dir():
        return []
    manifest_path = ROOT / ARCHIVED_MANIFEST
    if not manifest_path.is_file():
        return [f"{ARCHIVED_MANIFEST}: missing (seal entries with scripts/archive_experience.py)"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{ARCHIVED_MANIFEST}: invalid JSON: {exc}"]
    if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
        return [f"{ARCHIVED_MANIFEST}: expected {{'version': 1, 'files': {{...}}}}"]

    sealed: dict[str, str] = manifest["files"]
    on_disk: set[str] = set()
    errors: list[str] = []
    for path in sorted(archived.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(archived).as_posix()
        if rel == "manifest.json":
            continue
        parts = path.relative_to(archived).parts
        if len(parts) != 2 or parts[0] not in ARCHIVED_CLASSES:
            errors.append(f"{ARCHIVED_ROOT / rel}: only wins/ and errors/ entries may live in the archive")
            continue
        if not ARCHIVED_NAME_RE.fullmatch(parts[1]):
            errors.append(f"{ARCHIVED_ROOT / rel}: archived entry name must be YYYY-MM-DD-slug.md")
            continue
        on_disk.add(rel)
        expected = sealed.get(rel)
        if expected is None:
            errors.append(
                f"{ARCHIVED_ROOT / rel}: unsealed file in archive "
                f"(seal it with scripts/archive_experience.py --write)"
            )
        elif "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"{ARCHIVED_ROOT / rel}: sealed entry modified — frozen entries never change")
    for rel in sorted(set(sealed) - on_disk):
        errors.append(f"{ARCHIVED_ROOT / rel}: sealed entry missing from the tree")
    return errors


# Entries before this date are grandfathered; perf-claim entries on/after it
# must say in words what the baseline was, or carry an explicit waiver line.
# A bare commit hash satisfied this until 2026-09-09 and no longer does:
# commit references were removed from the docs corpus, and a hash never said
# what the number was compared against.
WINS_BASELINE_CUTOFF = "2026-08-23"
_PERF_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:tok/s|tokens/s|req/s|ms|us|ns|GB|GiB|MiB|KiB|MB|KB|TB|TFLOPS|GFLOPS|GB/s|TB/s|%)"
)
_HASH_RE = re.compile(r"[0-9a-f]{7,40}")
_BASELINE_RE = re.compile(r"baseline", re.IGNORECASE)
_WAIVER_RE = re.compile(r"no.baseline|without.baseline|pending.remote", re.IGNORECASE)


def check_wins_baseline_citations() -> list[str]:
    errors = []
    for rel in list_experience_entries(Path("docs/experience/wins")):
        if Path(rel).name[:10] < WINS_BASELINE_CUTOFF:
            continue
        text = (ROOT / rel).read_text(errors="ignore")
        if (
            len(_PERF_RE.findall(text)) >= 3
            and not _BASELINE_RE.search(text)
            and not _WAIVER_RE.search(text)
        ):
            errors.append(
                f"{rel}: perf-claim wins entry must name its baseline in words "
                f"or carry a 'no baseline' waiver; see docs/experience/wins/TEMPLATE-bench.md"
            )
    return errors


PREREG_WRITER = Path("scripts/prereg.py")


def load_prereg():
    """The ledger's own reader, so the format has exactly one definition."""
    spec = importlib.util.spec_from_file_location("prereg", ROOT / PREREG_WRITER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_prereg_no_stale_running() -> list[str]:
    """A row left `running` past 24h is a job that died without closing it; the
    ledger is the only place that death is recorded."""
    if not (ROOT / PREREG_WRITER).is_file():
        return [f"missing required file: {PREREG_WRITER}"]
    return [
        f"docs/experience/prereg.jsonl: {stale} — close it with scripts/prereg.py done"
        for stale in load_prereg().stale_running()
    ]


AGENDA_WRITER = Path("scripts/agenda.py")


def load_agenda():
    """The ledger's own reader, so the format has exactly one definition."""
    spec = importlib.util.spec_from_file_location("agenda", ROOT / AGENDA_WRITER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_agenda_ledger() -> list[str]:
    """A task with no exit gate closes on someone's opinion; a task left open and
    untouched is a lane nobody is running; a done task with no entry is a claim
    with no evidence."""
    if not (ROOT / AGENDA_WRITER).is_file():
        return [f"missing required file: {AGENDA_WRITER}"]
    return [f"docs/agenda.jsonl: {item}" for item in load_agenda().defects()]


# --- selftest -------------------------------------------------------------
#
# A check that cannot fail is not a check. Five of the checks above reach the
# tree through `git grep` / `git ls-files`, which report "nothing matched" and
# "git is unusable" through the same empty result, so each of them can pass by
# not running at all. `--selftest` runs those five against a world built by
# COPYING the real artifact and breaking it, and asserts FAIL there; the same
# world unbroken must PASS, and that half is what proves the scan ran. A
# hand-written world shares the check's own assumptions and certifies nothing.

LAUNCHER_FIXTURE = "crates/infer-cuda/src/ops/quant_linear.rs"
REGISTRY_FIXTURE = "operators/registry.toml"
MARKER_FIXTURE = "CONTRIBUTING.md"
WINS_FIXTURE = "docs/experience/wins/2026-09-02-metal-prefix-restore-survives-turns.md"
WINS_PARAMS_FIXTURE = "docs/experience/wins/2026-09-09-quant-decode-splits-ceiling-64.md"
ARCHIVE_FIXTURE = "docs/experience/archived"


def disown_ambient_git() -> None:
    """Hooks run with GIT_DIR and GIT_WORK_TREE exported. A `git init` that
    inherits them reinitializes the REAL repository instead of the world, and
    because the world's directory is not that repository's work tree, git
    records `core.bare = true` — which makes every checkout and every worktree
    fail with "this operation must be run in a work tree". Measured 2026-09-09:
    this selftest, running inside pre-push, broke the main checkout four times.
    The world's own `git grep` / `git ls-files` would also have read the real
    repository, so the checks would have passed against the wrong tree."""
    for name in [k for k in os.environ if k.startswith("GIT_")]:
        del os.environ[name]


def build_world(*rel_paths: str) -> Path:
    """A git repo holding real copies of the named paths, staged so `git grep`
    and `git ls-files` see them as tracked."""
    root = Path(tempfile.mkdtemp(prefix="hygiene-world-"))
    for rel in rel_paths:
        src = ROOT / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "core.excludesfile=", "add", "-Af"], cwd=root, check=True)
    return root


@contextlib.contextmanager
def rooted(root: Path):
    global ROOT
    saved = ROOT
    ROOT = root
    try:
        yield
    finally:
        ROOT = saved


def break_launcher_boundary(root: Path) -> None:
    path = root / LAUNCHER_FIXTURE
    path.write_text(path.read_text() + "\nfn _world() { unsafe { ffi::launch_quant_linear(); } }\n")


def break_registry_coverage(root: Path) -> None:
    path = root / REGISTRY_FIXTURE
    ids = re.findall(r'"(cuda\.[a-z0-9_.]+)"', (root / LAUNCHER_FIXTURE).read_text())
    if not ids:
        raise AssertionError(f"{LAUNCHER_FIXTURE} holds no cuda.* implementation id to unseat")
    kept = [line for line in path.read_text().splitlines(keepends=True) if f'id = "{ids[0]}"' not in line]
    path.write_text("".join(kept))


def break_repo_wide_markers(root: Path) -> None:
    path = root / MARKER_FIXTURE
    path.write_text(path.read_text() + "\nbuilt at /Users/someone/code/agent-infer\n")


def break_wins_baseline(root: Path) -> None:
    path = root / WINS_FIXTURE
    text = _HASH_RE.sub("XXXXXXX", path.read_text())
    text = _BASELINE_RE.sub("reference", text)
    path.write_text(_WAIVER_RE.sub("unmeasured", text))


def break_wins_parameters(root: Path) -> None:
    path = root / WINS_PARAMS_FIXTURE
    text = _PARAMS_RE.sub("## Setup", path.read_text())
    path.write_text(_WAIVER_RE.sub("unmeasured", text))


def break_archive_seal(root: Path) -> None:
    entries = sorted((root / ARCHIVE_FIXTURE).rglob("*.md"))
    if not entries:
        raise AssertionError(f"{ARCHIVE_FIXTURE} holds no sealed entry to modify")
    entries[0].write_text(entries[0].read_text() + "\n")


def break_prereg_stale_running(root: Path) -> None:
    """Written by the real writer, then back-dated: the world exercises the same
    start/read path the ledger uses, so a format drift between them shows up here."""
    subprocess.run(
        [sys.executable, str(root / PREREG_WRITER), "start", "--name", "world",
         "--cmd", "true", "--hypothesis", "a row left open is caught"],
        cwd=root, check=True, stdout=subprocess.DEVNULL,
    )
    ledger = root / "docs/experience/prereg.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    rows[-1]["started"] = stale.strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))


def break_agenda_stale_task(root: Path) -> None:
    """Written by the real writer, then back-dated: the world exercises the same
    add/read path the ledger uses, so a format drift between them shows up here."""
    def run(*argv):
        subprocess.run([sys.executable, str(root / AGENDA_WRITER), *argv],
                       cwd=root, check=True, stdout=subprocess.DEVNULL)
    run("goal", "add", "--id", "w", "--statement", "s", "--north-star", "m")
    run("task", "add", "--id", "t", "--goal", "w", "--exit", "a named measurable event")
    ledger = root / "docs/agenda.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    stale = datetime.now(timezone.utc) - timedelta(days=7)
    rows[-1]["updated"] = stale.strftime("%Y-%m-%dT%H:%M:%SZ")
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))


SELFTEST_WORLDS = [
    ("launcher_boundary", check_launcher_boundary, (LAUNCHER_FIXTURE,), break_launcher_boundary),
    ("registry_covers_runtime_counters", check_registry_covers_runtime_counters,
     (REGISTRY_FIXTURE, LAUNCHER_FIXTURE), break_registry_coverage),
    ("repo_wide_disallowed_markers", check_repo_wide_disallowed_markers,
     (MARKER_FIXTURE,), break_repo_wide_markers),
    ("wins_baseline_citations", check_wins_baseline_citations, (WINS_FIXTURE,), break_wins_baseline),
    ("wins_parameters", check_wins_parameters, (WINS_PARAMS_FIXTURE,), break_wins_parameters),
    ("archived_experience", check_archived_experience, (ARCHIVE_FIXTURE,), break_archive_seal),
    ("prereg_no_stale_running", check_prereg_no_stale_running,
     (str(PREREG_WRITER),), break_prereg_stale_running),
    ("agenda_ledger", check_agenda_ledger, (str(AGENDA_WRITER),), break_agenda_stale_task),
]


def selftest() -> int:
    disown_ambient_git()
    failures: list[str] = []
    for name, check, fixtures, break_it in SELFTEST_WORLDS:
        root = build_world(*fixtures)
        try:
            with rooted(root):
                clean = check()
                if clean:
                    failures.append(f"{name}: the unbroken world already FAILs ({clean[0]}) — the world is wrong, not the check")
                    continue
                break_it(root)
                broken = check()
            if not broken:
                failures.append(f"{name}: PASSes on its broken world — the check cannot fail")
            else:
                print(f"[selftest] {name}: FAILs on its broken world -> {broken[0]}")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("[selftest] FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"[selftest] OK — {len(SELFTEST_WORLDS)} checks proved they can fail")
    return 0


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
    errors.extend(check_archived_experience())
    errors.extend(check_wins_baseline_citations())
    errors.extend(check_wins_parameters())
    errors.extend(check_repo_wide_disallowed_markers())
    errors.extend(check_workspace_truth_surface())
    errors.extend(check_launcher_boundary())
    errors.extend(check_registry_covers_runtime_counters())
    errors.extend(check_prereg_no_stale_running())
    errors.extend(check_agenda_ledger())

    if errors:
        print("[repo-hygiene] FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("[repo-hygiene] OK")
    print(
        "[repo-hygiene] public docs, templates, local links, tracked junk, "
        "repo-wide marker bans, experience entry caps, wins parameters, frozen archive seal, "
        "workspace truth-surface, CUDA launcher-boundary, registry-coverage, "
        "prereg-ledger, and agenda-ledger checks all passed"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="assert every git-backed check FAILs on a broken copy of its real artifact",
    )
    sys.exit(selftest() if parser.parse_args().selftest else main())
