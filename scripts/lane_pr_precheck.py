#!/usr/bin/env python3
"""Content/marker gates, run in two places over one shared range computation.

Two entry points:

- `lane.sh pr` (default): all rules below against merge-base origin/main..HEAD,
  including the three PR-BODY marker rules (BUILD/CUDA/CLIPPY_EXIT) that need a
  PR body a push does not have.
- the pre-push hook (`--push-content BASE..HEAD`): only the body-less CONTENT
  rules (1,2,3,6,8 below), over the exact pushed range. This is what closes the
  direct-to-main hole — a push to main has no PR and otherwise bypassed
  everything except the checker's selftest.

The content rules are 1 bench-entry, 2 comment refs, 3 abs paths, 6 gate
registry, 8 dead docs shas. The marker rules are 4 build-exit, 5 cuda-check,
7 clippy-exit.

1. A commit touching crates/, scripts/bench_*/, or src/ has a
   "Bench-entry exemption" body line or the PR adds a
   docs/experience/{wins,errors}/ entry.
2. Added code comments carry no PR number (#123) and no 7+ hex SHA.
3. No added line contains a machine-local home/container/data/mount/pod path.
4. An examples/ or benches/ change needs a BUILD_EXIT=0 line in the PR body.
5. Rust changed in a no-cuda-gated crate, or anything under
   crates/cuda-kernels/, needs a CUDA_CHECK_EXIT=0 line from a pod
   `cargo check --features cuda,nccl` run without no-cuda.
6. A parity gate under crates/infer-cuda/examples/ is named in an
   operators/registry.toml correctness_gate and prints the negative-control
   marker scripts/parity_gpu_batch.sh greps for.
7. Same trigger as rule 5, but a `cargo check` passes while clippy lints are
   denied (`-D warnings` deprecated/unused-mut lints are invisible to check),
   so the body also needs a CLIPPY_EXIT=0 from a pod
   `cargo clippy --workspace --all-targets --features cuda,nccl
   -- -D warnings` run without no-cuda.
8. An ADDED line under docs/ may not cite a backticked short sha (7-12 hex)
   that does not resolve to a commit in THIS repository. Existing lines are
   not scanned, so historical dead shas are left alone; a sha attributed to an
   upstream project is allowed via an explicit upstream cue. Known limit: a
   dead sha whose short form is all decimal digits cannot be distinguished
   from an ordinary number and is intentionally not matched.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUNTIME_PATH = re.compile(r"^(crates/[^/]+/|scripts/bench_[^/]*/|src/)")
EXAMPLE_OR_BENCH_PATH = re.compile(r"^crates/[^/]+/(examples|benches)/")
EXPERIENCE_ENTRY = re.compile(r"^docs/experience/(wins|errors)/.+\.md$")
DATED_EXPERIENCE = re.compile(r"^docs/experience/(wins|errors)/\d{4}-\d{2}-\d{2}-")
PR_REF = re.compile(r"#\d{2,}")
# A git SHA needs hex with at least one a-f letter and one digit; a plain
# decimal like 1048576 must not match.
LONG_SHA = re.compile(r"(?=\b[0-9a-f]*[0-9][0-9a-f]*\b)(?=\b[0-9a-f]*[a-f][0-9a-f]*\b)\b[0-9a-f]{7,40}\b")
# Banned path literals are assembled, not written: check_repo_hygiene bans
# those strings in tracked text. ABS_PATH uses a strict lookbehind (no hyphen)
# so the shell overridable-default idiom ${VAR:-<build-tree>} cannot hide a
# literal even behind the `:-`; check_abs_paths re-allows that form solely
# inside executable .sh code, never in a comment and never outside shell.
_ABS_SEGMENTS = "|".join(["/" + s for s in ("Users/", "root/", "data0", "mnt/", "host/")])
ABS_PATH = re.compile(r"(?<![\w./])(" + _ABS_SEGMENTS + r")")
SHELL_PATH = re.compile(r"\.sh$")
PARITY_EXAMPLE_PATH = re.compile(r"^crates/infer-cuda/examples/[A-Za-z0-9_]+\.rs$")
# parity_gpu_batch.sh derives its gate list from registry correctness_gate
# values and reads these two markers out of each run's log.
NEG_FLAG = "--negative-control"
NEG_MARKER = "NEGATIVE CONTROL OK"
REGISTRY_REL = "operators/registry.toml"

BUILD_EXIT_OK = re.compile(r"^BUILD_EXIT=0\s*$", re.MULTILINE)
CUDA_CHECK_EXIT_OK = re.compile(r"^CUDA_CHECK_EXIT=0\s*$", re.MULTILINE)
CLIPPY_EXIT_OK = re.compile(r"^CLIPPY_EXIT=0\s*$", re.MULTILINE)
CRATE_RUST_PATH = re.compile(r"^crates/([^/]+)/.*\.rs$")
ROOT_RUST_PATH = re.compile(r"^src/.*\.rs$")
CUDA_KERNELS_PATH = re.compile(r"^crates/cuda-kernels/")
# A crate's [features] table declaring no-cuda, e.g. `no-cuda = ["..."]`.
NO_CUDA_FEATURE = re.compile(r"(?m)^\s*no-cuda\s*=")

# Only code-file comments count: `//`/`///` for C-family sources, `#` for
# hash-comment sources. Markdown headings and C preprocessor lines are not
# comments for this rule.
SLASH_FILES = re.compile(r"\.(rs|cu|cuh|h|cc|cpp)$")
HASH_FILES = re.compile(r"\.(py|sh|toml|yml|yaml)$")
SLASH_COMMENT = re.compile(r"^\+\s*///?\s?\S")
HASH_COMMENT = re.compile(r"^\+\s*#\s?\S")

# Rule 8: a backticked 7-12 hex code span in an added docs line is treated as a
# commit citation. The {7,12} bound deliberately excludes the 16-char per-row
# block/data hashes in analysis tables, and the backticks keep it to a
# deliberate citation rather than prose. Both a digit and an a-f letter are
# required (a bare decimal must not match).
DOC_MD_PATH = re.compile(r"^docs/.*\.md$")
DOC_SHORT_SHA = re.compile(
    r"`(?=[0-9a-f]{7,12}`)(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,12}`"
)
# A sha attributed to another repository is not supposed to resolve here. The
# cue must be explicit on the same line; this is the only carve-out.
UPSTREAM_CUE = re.compile(
    r"upstream|NVlabs|mlc-ai|llama\.cpp|cuda-oxide|xgrammar", re.IGNORECASE
)


# Named abs-path exemptions, by exact path. The two ledgers are run-provenance
# records whose purpose is to record where a run happened; the two scripts must
# contain the path strings they print or test for. CHANGELOG prose is
# deliberately absent: a ledger paragraph is exactly what this rule must catch.
ABS_PATH_EXEMPT_FILES = frozenset(
    {
        "docs/agenda.jsonl",
        "docs/experience/prereg.jsonl",
        "scripts/lane.sh",
        "scripts/check_repo_hygiene.py",
    }
)
# comment-ref exempts upstream CI config (its comments cite upstream PR numbers).
COMMENT_REF_EXEMPT_PREFIX = ".github/"
# These four paths are the COMPLETE exemption set: the two ledgers exist to
# record run locations, the two scripts must contain the strings they print or
# test for. A fifth entry means the rule itself is wrong and should be
# rethought — do not grow the whitelist.


def git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def added_lines(repo: Path, base: str, head: str) -> dict[str, list[str]]:
    # Explicit endpoints. `base..head` is the set of objects the caller is
    # pushing or proposing; callers pass a base already resolved for their
    # context (remote tip for a push, merge-base for a PR), so this is the one
    # range computation both paths share.
    files: dict[str, list[str]] = {}
    current: str | None = None
    for raw in git(["diff", "--unified=0", "--no-color", f"{base}..{head}"], repo).splitlines():
        if raw.startswith("+++ b/"):
            current = raw[6:]
        elif current and raw.startswith("+") and not raw.startswith("+++"):
            files.setdefault(current, []).append(raw[1:])
    return files


def changed_files(repo: Path, base: str, head: str) -> list[str]:
    return [
        p
        for p in git(["diff", "--name-only", f"{base}..{head}"], repo).splitlines()
        if p
    ]


def check_bench_exemption(repo: Path, base: str, head: str, adds_experience: bool) -> list[str]:
    if adds_experience:
        return []
    failures: list[str] = []
    for commit in git(["rev-list", f"{base}..{head}"], repo).split():
        touched = git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit], repo)
        if not any(RUNTIME_PATH.match(p) for p in touched.splitlines() if p):
            continue
        if "Bench-entry exemption" in git(["log", "-1", "--format=%B", commit], repo):
            continue
        subject = git(["log", "-1", "--format=%s", commit], repo).strip()
        failures.append(
            f"bench-entry: {commit[:9]} ({subject}) touches runtime code with no "
            "'Bench-entry exemption' line and no docs/experience entry"
        )
    return failures


def check_comments(added: dict[str, list[str]]) -> list[str]:
    failures: list[str] = []
    for path, lines in added.items():
        if path.startswith(COMMENT_REF_EXEMPT_PREFIX):
            continue  # upstream CI config legitimately cites upstream PR numbers
        if DATED_EXPERIENCE.match(path):
            continue
        if SLASH_FILES.search(path):
            is_comment = lambda ln: bool(SLASH_COMMENT.match("+" + ln))
        elif HASH_FILES.search(path):
            is_comment = lambda ln: bool(HASH_COMMENT.match("+" + ln))
        else:
            continue
        for n, line in enumerate(lines, 1):
            if not is_comment(line):
                continue
            if PR_REF.search(line):
                failures.append(f"comment-ref: {path}:{n} comment cites a PR number")
            if LONG_SHA.search(line):
                failures.append(f"comment-ref: {path}:{n} comment embeds a commit SHA")
    return failures


def check_abs_paths(added: dict[str, list[str]]) -> list[str]:
    failures = []
    for path, lines in added.items():
        if path in ABS_PATH_EXEMPT_FILES:
            continue  # named run-provenance / machinery files; see set comment
        shell = bool(SHELL_PATH.search(path))
        for n, line in enumerate(lines, 1):
            for m in ABS_PATH.finditer(line):
                # The only carve-out from the strict rule is the shell
                # parameter-expansion default in executable .sh code — never
                # in a .sh comment, and nowhere outside shell.
                if (
                    shell
                    and not HASH_COMMENT.match("+" + line)
                    and line[max(0, m.start() - 2):m.start()] == ":-"
                ):
                    continue
                failures.append(
                    f"abs-path: {path}:{n} adds a machine-local absolute path"
                )
    return failures


def crate_has_no_cuda(repo: Path, crate: str) -> bool:
    manifest = repo / "crates" / crate / "Cargo.toml"
    try:
        return bool(NO_CUDA_FEATURE.search(manifest.read_text()))
    except OSError:
        return False


def root_has_no_cuda(repo: Path) -> bool:
    try:
        return bool(NO_CUDA_FEATURE.search((repo / "Cargo.toml").read_text()))
    except OSError:
        return False


def cuda_gate_triggered(repo: Path, changed: list[str]) -> bool:
    """Rules 5 and 7 share one trigger: CUDA-only code the no-cuda lint hides."""
    if any(CUDA_KERNELS_PATH.match(p) for p in changed):
        return True
    crates = {m.group(1) for p in changed if (m := CRATE_RUST_PATH.match(p))}
    if any(crate_has_no_cuda(repo, c) for c in crates):
        return True
    return any(ROOT_RUST_PATH.match(p) for p in changed) and root_has_no_cuda(repo)


def check_cuda_check_exit(repo: Path, changed: list[str], pr_body: str) -> list[str]:
    if CUDA_CHECK_EXIT_OK.search(pr_body) or not cuda_gate_triggered(repo, changed):
        return []
    return [
        "cuda-check: diff touches Rust in a no-cuda-gated crate (or crates/cuda-kernels/); "
        "run a pod `cargo check --features cuda,nccl` WITHOUT no-cuda and put a "
        "CUDA_CHECK_EXIT=0 line in the PR body"
    ]


def check_clippy_exit(repo: Path, changed: list[str], pr_body: str) -> list[str]:
    if CLIPPY_EXIT_OK.search(pr_body) or not cuda_gate_triggered(repo, changed):
        return []
    return [
        "clippy-exit: diff touches Rust in a no-cuda-gated crate (or crates/cuda-kernels/); "
        "`cargo check` runs no clippy lints, so run a pod "
        "`cargo clippy --workspace --all-targets --features cuda,nccl -- -D warnings` "
        "WITHOUT no-cuda and put a CLIPPY_EXIT=0 line in the PR body"
    ]


def check_build_exit(added: dict[str, list[str]], pr_body: str) -> list[str]:
    if not any(EXAMPLE_OR_BENCH_PATH.match(p) for p in added) or BUILD_EXIT_OK.search(pr_body):
        return []
    return ["build-exit: examples/ or benches/ changed but the PR body has no BUILD_EXIT=0 line"]


def _sha_in_main(repo: Path, sha: str, base: str) -> bool:
    """True if sha resolves AND is reachable from base (origin/main merge-base).

    A sha that resolves only on the feature branch is the rebase/squash death
    the rule targets: it is a real commit now, but after merge it is not, so a
    doc citing it is already stale.
    """
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, base],
        cwd=repo,
        capture_output=True,
    )
    return r.returncode == 0


def check_doc_dead_sha(repo: Path, added: dict[str, list[str]], base: str) -> list[str]:
    failures: list[str] = []
    for path, lines in added.items():
        if not DOC_MD_PATH.match(path):
            continue
        for n, line in enumerate(lines, 1):
            if UPSTREAM_CUE.search(line):
                continue
            for m in DOC_SHORT_SHA.finditer(line):
                sha = m.group(0).strip("`")
                if _sha_in_main(repo, sha, base):
                    continue
                failures.append(
                    f"dead-sha: {path}:{n} cites `{sha}` which does not resolve to a "
                    "commit on origin/main (rebased/squashed lane sha?). Cite the PR "
                    "number, or mark an upstream-repo sha with its project on the line"
                )
    return failures


def check_gate_registry(repo: Path, changed: list[str]) -> list[str]:
    gates = [
        rel for rel in changed
        if PARITY_EXAMPLE_PATH.match(rel)
        and (repo / rel).exists()
        and NEG_FLAG in (repo / rel).read_text()
    ]
    if not gates:
        return []
    registry = repo / REGISTRY_REL
    listed = registry.read_text() if registry.exists() else ""
    changed_text = "\n".join(
        (repo / rel).read_text() for rel in changed if (repo / rel).is_file()
    )
    failures = []
    for rel in gates:
        if rel not in listed:
            failures.append(
                f"gate-registry: {rel} takes {NEG_FLAG} but no correctness_gate in "
                f"{REGISTRY_REL} names it; parity_gpu_batch.sh derives its gate list "
                "from that file, so the gate is never built and never run"
            )
        if NEG_MARKER not in changed_text:
            failures.append(
                f"gate-registry: {rel} takes {NEG_FLAG} but nothing in this PR prints "
                f"'{NEG_MARKER}'; parity_gpu_batch.sh records the negative arm as FAIL "
                "without that line, even when the controls fire"
            )
    return failures


def run_content(repo: Path, base: str, head: str) -> list[str]:
    """The body-less content rules, run over an explicit range.

    Shared by the PR path (base = merge-base with main, head = HEAD) and the
    pre-push hook (base = remote tip, head = pushed tip). These rules need no PR
    body, so they are the ones that can and must also gate a direct-to-main
    push. The three body-marker rules (build/cuda/clippy) are PR-only and live
    in `run_pr`, because a push has no body to carry the marker.
    """
    added = added_lines(repo, base, head)
    changed = changed_files(repo, base, head)
    failures = check_bench_exemption(
        repo, base, head, any(EXPERIENCE_ENTRY.match(p) for p in added)
    )
    failures += check_comments(added)
    failures += check_abs_paths(added)
    failures += check_gate_registry(repo, changed)
    failures += check_doc_dead_sha(repo, added, base)
    return failures


def run_pr(repo: Path, pr_body: str) -> list[str]:
    """PR-time check: all content rules plus the three PR-body marker rules."""
    base = git(["merge-base", "origin/main", "HEAD"], repo).strip()
    added = added_lines(repo, base, "HEAD")
    changed = changed_files(repo, base, "HEAD")
    failures = run_content(repo, base, "HEAD")
    failures += check_build_exit(added, pr_body)
    failures += check_cuda_check_exit(repo, changed, pr_body)
    failures += check_clippy_exit(repo, changed, pr_body)
    return failures


def run_push(repo: Path, base: str, head: str) -> list[str]:
    """Pre-push content gate over the exact bytes being pushed."""
    return run_content(repo, base, head)


# --- selftest: one clean world, one broken fixture per rule ----------------

EXEMPT_BODY = "feat(x): thing\n\nBench-entry exemption: dev-only tooling.\n"
BARE_BODY = "feat(x): thing\n"
RUNTIME_FIXTURE = "crates/x/src/lib.rs"
EXAMPLE_FIXTURE = "crates/x/examples/demo.rs"
HISTORY_FIXTURE = "docs/experience/wins/2026-09-12-precheck-selftest.md"
GATE_FIXTURE = "crates/infer-cuda/examples/foo_parity.rs"
GATE_SRC = 'fn main() { let neg = arg("--negative-control"); println!("NEGATIVE CONTROL OK"); }\n'
GATE_SRC_NO_MARKER = 'fn main() { let neg = arg("--negative-control"); println!("ALL PASS"); }\n'
REGISTRY_LISTING = 'correctness_gate = "crates/infer-cuda/examples/foo_parity.rs"\n'
REGISTRY_OTHER = 'correctness_gate = "crates/infer-cuda/examples/other_parity.rs"\n'



# Selftest worlds pin commit dates: a sha prefix the cases cite is derived from
# the commit timestamp, so wall-clock dates make a world nondeterministic — when
# the 9-char prefix happens to be all decimal digits the sha regex (which must
# reject plain decimals) stops matching and the dead-sha arm spuriously passes.
PINNED_GIT_ENV = {
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def world(files: dict[str, str], body: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix="precheck-world-"))
    env = dict(os.environ, **PINNED_GIT_ENV)
    g = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, env=env)
    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (root / ".base").write_text("b")
    g("add", ".base")
    g("commit", "-q", "-m", "base")
    g("branch", "-M", "main")
    g("branch", "origin/main")
    g("checkout", "-q", "-b", "lane/x")
    base_sha = git(["rev-parse", "HEAD"], root).strip()[:9]
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content.replace("{{BASE_SHA}}", base_sha))
        g("add", rel)
    g("commit", "-q", "-m", body.splitlines()[0], "-m", body)
    return root


def selftest() -> int:
    _root = "/" + "root/"
    _host_tree = "/" + "host/arle-build"
    _root_tree = "/" + "root/arle-build"
    cases = [
        ("clean world", {RUNTIME_FIXTURE: "pub fn x() {}\n", EXAMPLE_FIXTURE: "fn main() {}\n"},
         EXEMPT_BODY, "BUILD_EXIT=0\n", None),
        ("bench-entry", {RUNTIME_FIXTURE: "pub fn x() {}\n"}, BARE_BODY, "", "bench-entry"),
        ("comment PR ref", {RUNTIME_FIXTURE: "// fixed in #123\npub fn x() {}\n"}, EXEMPT_BODY, "", "comment-ref"),
        ("comment SHA", {RUNTIME_FIXTURE: "// ported from abcdef1\npub fn x() {}\n"}, EXEMPT_BODY, "", "comment-ref"),
        ("decimal not a SHA", {RUNTIME_FIXTURE: "// 1048576 bytes\npub fn x() {}\n"}, EXEMPT_BODY, "", None),
        ("hex word not a SHA", {RUNTIME_FIXTURE: "// deadbeef table\npub fn x() {}\n"}, EXEMPT_BODY, "", None),
        ("markdown heading", {"README.md": "# see #123 and abcdef1\n"}, BARE_BODY, "", None),
        ("hash script PR ref", {"scripts/x.sh": "# fixed in #123\n"}, EXEMPT_BODY, "", "comment-ref"),
        ("history entry", {HISTORY_FIXTURE: "# Title\n\nSuperseded by #123 (abcdef12).\n"}, BARE_BODY, "", None),
        ("abs path", {RUNTIME_FIXTURE: f'pub const L: &str = "{_root}x";\n'}, EXEMPT_BODY, "", "abs-path"),
        ("abs path shell override idiom", {"scripts/x.sh": f'TREE="${{TREE:-{_host_tree}}}"\n'}, EXEMPT_BODY, "", None),
        ("abs path shell plain literal", {"scripts/x.sh": f'cd {_root_tree}\n'}, EXEMPT_BODY, "", "abs-path"),
        ("abs path shell comment override", {"scripts/x.sh": f'# default is {_host_tree} here\n'}, EXEMPT_BODY, "", "abs-path"),
        ("abs path prose override form", {"README.md": f"use `${{TREE:-{_host_tree}}}` as the tree\n"}, BARE_BODY, "", "abs-path"),
        ("hyphen non-path passes", {"scripts/x.sh": 'flag --x-root /none-such\n'}, EXEMPT_BODY, "", None),
        ("build-exit", {EXAMPLE_FIXTURE: "fn main() {}\n"}, EXEMPT_BODY, "", "build-exit"),
        (
            "cuda-check missing",
            {
                "crates/x/src/lib.rs": "pub fn x() {}\n",
                "crates/x/Cargo.toml": "[features]\nno-cuda = []\n",
            },
            EXEMPT_BODY,
            "",
            "cuda-check",
        ),
        (
            "cuda-check present",
            {
                "crates/x/src/lib.rs": "pub fn x() {}\n",
                "crates/x/Cargo.toml": "[features]\nno-cuda = []\n",
            },
            EXEMPT_BODY,
            "CUDA_CHECK_EXIT=0\nCLIPPY_EXIT=0\n",
            None,
        ),
        (
            "cuda-check non-gated crate passes",
            {"crates/x/src/lib.rs": "pub fn x() {}\n",
             "crates/x/Cargo.toml": "[features]\ncuda = []\n"},
            EXEMPT_BODY,
            "",
            None,
        ),
        (
            "cuda-check cuda-kernels path",
            {"crates/cuda-kernels/csrc/x.cu": "// x\n"},
            EXEMPT_BODY,
            "",
            "cuda-check",
        ),
        (
            "cuda-check root crate missing",
            {
                "src/main.rs": "fn main() {}\n",
                "Cargo.toml": "[features]\nno-cuda = [\"cli/no-cuda\"]\n",
            },
            EXEMPT_BODY,
            "",
            "cuda-check",
        ),
        (
            "cuda-check root crate present",
            {
                "src/main.rs": "fn main() {}\n",
                "Cargo.toml": "[features]\nno-cuda = [\"cli/no-cuda\"]\n",
            },
            EXEMPT_BODY,
            "CUDA_CHECK_EXIT=0\nCLIPPY_EXIT=0\n",
            None,
        ),
        (
            "cuda-check root crate without feature passes",
            {"src/main.rs": "fn main() {}\n", "Cargo.toml": "[features]\ncuda = []\n"},
            EXEMPT_BODY,
            "",
            None,
        ),
        (
            "clippy-exit missing despite check marker",
            {
                "crates/x/src/lib.rs": "pub fn x() {}\n",
                "crates/x/Cargo.toml": "[features]\nno-cuda = []\n",
            },
            EXEMPT_BODY,
            "CUDA_CHECK_EXIT=0\n",
            "clippy-exit",
        ),
        (
            "clippy-exit present",
            {
                "crates/x/src/lib.rs": "pub fn x() {}\n",
                "crates/x/Cargo.toml": "[features]\nno-cuda = []\n",
            },
            EXEMPT_BODY,
            "CUDA_CHECK_EXIT=0\nCLIPPY_EXIT=0\n",
            None,
        ),
        (
            "clippy-exit non-cuda change unaffected",
            {"crates/x/src/lib.rs": "pub fn x() {}\n",
             "crates/x/Cargo.toml": "[features]\ncuda = []\n"},
            EXEMPT_BODY,
            "",
            None,
        ),
    ]
    cases += [
        ("gate registered",
         {GATE_FIXTURE: GATE_SRC, "operators/registry.toml": REGISTRY_LISTING},
         EXEMPT_BODY, "BUILD_EXIT=0\n", None),
        ("gate unregistered",
         {GATE_FIXTURE: GATE_SRC, "operators/registry.toml": REGISTRY_OTHER},
         EXEMPT_BODY, "BUILD_EXIT=0\n", "gate-registry"),
        ("gate without negative marker",
         {GATE_FIXTURE: GATE_SRC_NO_MARKER, "operators/registry.toml": REGISTRY_LISTING},
         EXEMPT_BODY, "BUILD_EXIT=0\n", "gate-registry"),
        ("non-gate example passes",
         {"crates/infer-cuda/examples/bar.rs": "fn main() {}\n"},
         EXEMPT_BODY, "BUILD_EXIT=0\n", None),
    ]
    cases += [
        ("dead-sha on main commit passes",
         {"docs/note.md": "See commit `{{BASE_SHA}}` for the fix.\n"},
         BARE_BODY, "", None),
        ("dead-sha unresolvable fails",
         {"docs/note.md": "Landed in `0123abcd`.\n"},
         BARE_BODY, "", "dead-sha"),
        ("dead-sha upstream cue passes",
         {"docs/research/x.md": "Upstream pinned at `0123abcd` (NVlabs/cuda-oxide).\n"},
         BARE_BODY, "", None),
        ("dead-sha 16-char data hash passes",
         {"docs/note.md": "Block hash `0123456789abcdef` in the table.\n"},
         BARE_BODY, "", None),
        ("dead-sha decimal passes",
         {"docs/note.md": "Counter `1048576` and word `abcdef` not a sha.\n"},
         BARE_BODY, "", None),
        ("dead-sha outside docs passes",
         {"README-notes.txt": "Landed in `0123abcd`.\n"},
         BARE_BODY, "", None),
    ]
    failures = []
    for name, files, body, pr_body, expected in cases:
        root = world(files, body)
        try:
            got = run_pr(root, pr_body)
        except Exception as exc:  # a crash is a selftest failure
            got = [f"checker raised {exc!r}"]
        shutil.rmtree(root, ignore_errors=True)
        hit = next((f for f in got if expected and f.startswith(expected + ":")), None)
        if expected and not hit:
            failures.append(f"{name}: expected {expected!r} failure, got {got}")
        elif not expected and got:
            failures.append(f"{name}: expected pass, got {got}")
        else:
            print(f"[selftest] {name}: {'FAILs as designed -> ' + hit if hit else 'PASS'}")

    # Branch-only sha world: a real commit that lives only on the lane, not on
    # the merge-base, is the rebase/squash death. Two lane commits — a work
    # commit then a doc that cites it — so the cited sha exists in the object
    # store but is not an ancestor of origin/main. Dates are pinned because the
    # cited 9-char prefix derives from the commit timestamp; an all-decimal
    # prefix (~1.5%) would not match the sha regex, which must reject decimals.
    root = Path(tempfile.mkdtemp(prefix="precheck-world-"))
    try:
        env = dict(os.environ, **PINNED_GIT_ENV)
        g = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, env=env)
        g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
        (root / ".base").write_text("b"); g("add", ".base")
        g("commit", "-q", "-m", "base"); g("branch", "-M", "main"); g("branch", "origin/main")
        g("checkout", "-q", "-b", "lane/x")
        (root / "crates_x").write_text("x"); g("add", "crates_x")
        g("commit", "-q", "-m", "lane work")
        lane_sha = git(["rev-parse", "HEAD"], root).strip()[:9]
        (root / "docs").mkdir(parents=True)
        (root / "docs" / "note.md").write_text(f"Depends on `{lane_sha}` (lane-only).\n")
        g("add", "docs/note.md"); g("commit", "-q", "-m", "doc cites lane sha")
        got = run_pr(root, "")
        hit = next((f for f in got if f.startswith("dead-sha:")), None)
        name = "dead-sha branch-only commit fails"
        if hit:
            print(f"[selftest] {name}: FAILs as designed -> {hit}")
        else:
            failures.append(f"{name}: expected dead-sha failure, got {got}")
    except Exception as exc:
        failures.append(f"branch-sha world raised {exc!r}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("[lane-precheck][selftest] FAIL")
        print("\n".join(f"- {f}" for f in failures))
        return 1
    print("[lane-precheck][selftest] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--pr-body")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument(
        "--push-content",
        metavar="BASE..HEAD",
        help=(
            "Run only the body-less content rules over an explicit pushed range "
            "(the pre-push hook uses this with remote-tip..pushed-tip)."
        ),
    )
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.push_content:
        if ".." not in args.push_content:
            print("push-content expects BASE..HEAD", file=sys.stderr)
            return 2
        base, head = args.push_content.split("..", 1)
        failures = run_push(Path(args.repo).resolve(), base, head)
        label = "push-content"
    else:
        pr_body = os.environ.get("ARLE_PR_BODY", "")
        if args.pr_body:
            pr_body = Path(args.pr_body).read_text()
        elif not sys.stdin.isatty():
            pr_body = sys.stdin.read()
        failures = run_pr(Path(args.repo).resolve(), pr_body)
        label = "lane-precheck"

    if failures:
        print(f"[{label}] FAIL — refusing:")
        print("\n".join(f"- {f}" for f in failures))
        return 1
    print(f"[{label}] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
