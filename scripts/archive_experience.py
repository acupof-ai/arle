#!/usr/bin/env python3
"""Seal docs/experience entries into the frozen archive.

Modeled on deepseek-harness's archived-agent-notes flow: sealed entries are
frozen forever (check_repo_hygiene.py fails on any hash drift or deletion),
and the manifest is append-only. Inbound links to each sealed entry are
rewritten to its archived path in the same change.

Usage:
    python3 scripts/archive_experience.py [--write] <entry.md>...
    python3 scripts/archive_experience.py --delete <entry.md> [--retarget-to <owner.md>] [--write]

Default is a dry-run plan; --write performs the change. All entries are
validated before anything touches disk — one bad entry aborts the whole run.

Seal mode moves an entry into the frozen archive. Delete mode removes a
fully-superseded live entry: every unique fact must already live in the
owner entry, and every inbound link is retargeted to the owner. Sealed
entries are never deleted.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = Path("docs/experience")
ARCHIVED = EXPERIENCE / "archived"
MANIFEST = ARCHIVED / "manifest.json"
CLASSES = ("wins", "errors")
NAME_RE = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if not (ROOT / MANIFEST).exists():
        return {"version": 1, "files": {}}
    data = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("files"), dict):
        sys.exit(f"archive: malformed manifest {MANIFEST}")
    return data


def validate_entry(arg: str) -> tuple[Path, Path, str]:
    src = Path(arg)
    if not src.is_absolute():
        src = ROOT / src
    rel = src.relative_to(ROOT).as_posix()
    parts = Path(rel).parts
    if len(parts) != 4 or parts[0] != "docs" or parts[1] != "experience" or parts[2] not in CLASSES:
        sys.exit(f"archive: entries must live under docs/experience/{{wins,errors}}/: {rel}")
    if not NAME_RE.fullmatch(Path(rel).name):
        sys.exit(f"archive: entry name must be YYYY-MM-DD-slug.md: {rel}")
    if not src.is_file():
        sys.exit(f"archive: no such entry: {rel}")
    cls = parts[2]
    dst = ROOT / ARCHIVED / cls / Path(rel).name
    if dst.exists():
        sys.exit(f"archive: already archived: {rel}")
    key = f"{cls}/{Path(rel).name}"
    return src, dst, key


def inbound_links(old_rel: str, new_rel: str) -> list[tuple[Path, str, str]]:
    try:
        output = subprocess.check_output(
            ["git", "grep", "-l", "-F", old_rel, "--", "."],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            return []
        raise
    rewrites = []
    for path_str in output.splitlines():
        if path_str == old_rel:
            continue
        path = ROOT / path_str
        text = path.read_text(encoding="utf-8")
        if old_rel in text:
            rewrites.append((path, old_rel, new_rel))
    return rewrites


def validate_live_entry(arg: str) -> Path:
    src = Path(arg)
    if not src.is_absolute():
        src = ROOT / src
    rel = src.relative_to(ROOT).as_posix()
    parts = Path(rel).parts
    if len(parts) != 4 or parts[0] != "docs" or parts[1] != "experience" or parts[2] not in CLASSES:
        sys.exit(f"archive: entries must live under docs/experience/{{wins,errors}}/: {rel}")
    if not NAME_RE.fullmatch(Path(rel).name):
        sys.exit(f"archive: entry name must be YYYY-MM-DD-slug.md: {rel}")
    if not src.is_file():
        sys.exit(f"archive: no such entry: {rel}")
    return src


def main() -> int:
    write = False
    delete = False
    retarget_to = None
    args = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--write":
            write = True
        elif a == "--delete":
            delete = True
        elif a == "--retarget-to":
            retarget_to = next(it, None)
        else:
            args.append(a)
    if not args:
        print(__doc__)
        return 1

    if delete:
        if len(args) != 1:
            sys.exit("archive: --delete takes exactly one entry")
        src = validate_live_entry(args[0])
        old_rel = src.relative_to(ROOT).as_posix()
        owner_rel = None
        if retarget_to:
            owner = validate_live_entry(retarget_to)
            owner_rel = owner.relative_to(ROOT).as_posix()
        rewrites = inbound_links(old_rel, owner_rel or old_rel)
        if rewrites and not owner_rel:
            print(f"archive: {old_rel} has inbound links; pass --retarget-to <owner.md>:", file=sys.stderr)
            for path, _, _ in rewrites:
                print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
            return 1
        print(f"delete {old_rel}")
        for path, _, new_rel in rewrites:
            print(f"  link retarget -> {Path(new_rel).name} ({path.relative_to(ROOT)})")
        if not write:
            print("\narchive: dry-run, entry would be deleted (pass --write to apply)")
            return 0
        for path, old, new in rewrites:
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(old, new), encoding="utf-8")
        src.unlink()
        print(f"\narchive: deleted {old_rel}")
        return 0

    plan = []
    for arg in args:
        src, dst, key = validate_entry(arg)
        old_rel = src.relative_to(ROOT).as_posix()
        new_rel = dst.relative_to(ROOT).as_posix()
        plan.append((src, dst, key, inbound_links(old_rel, new_rel)))

    for src, dst, key, rewrites in plan:
        print(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
        for path, _, _ in rewrites:
            print(f"  link rewrite: {path.relative_to(ROOT)}")

    if not write:
        print(f"\narchive: dry-run, {len(plan)} entr(ies) would be sealed (pass --write to seal)")
        return 0

    manifest = load_manifest()
    for src, dst, key, rewrites in plan:
        for path, old_rel, new_rel in rewrites:
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace(old_rel, new_rel), encoding="utf-8")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        manifest["files"][key] = sha256(dst)

    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\narchive: sealed {len(plan)} entr(ies); manifest now has {len(manifest['files'])} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
