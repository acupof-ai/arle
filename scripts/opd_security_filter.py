#!/usr/bin/env python3
"""Shared security-content blocklist for the OPD rollout corpora.

Every corpus entry point (gen_agent_opd_tasks.py, gen_terminal_tasks.py,
stage_swe_pro.py select) runs its generated/selected text through
`is_security_flagged` and DROPS anything that matches — no security
offense/defense content may reach a rollout task, fixture, or staged repo.
Ambiguous words (payload, credential, injection, vault, ...) are on the list
on purpose: corpus supply is not scarce, so we err heavily on dropping.

API:    is_security_flagged(text) -> Optional[str]   # matched rule name
CLI:    python3 scripts/opd_security_filter.py            # fixture self-check
        python3 scripts/opd_security_filter.py --scan DIR # report flagged files
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

_I = re.IGNORECASE

# (rule-name, compiled pattern). Word boundaries keep short tokens from
# matching inside ordinary words (force !~ RCE, ids.py !~ IDS).
_RULES = [
    # --- offense tooling / actions ---
    ("exploit", re.compile(r"exploit", _I)),
    ("vulnerability", re.compile(r"vulnerab", _I)),
    ("cve", re.compile(r"\bCVE-\d", _I)),
    ("pentest", re.compile(r"pentest|penetration[\s-]?test", _I)),
    ("offense-tool", re.compile(
        r"metasploit|\bnmap\b|\bburp\b|\bhydra\b|wireshark|sqlmap", _I)),
    ("shellcode", re.compile(r"shellcode", _I)),
    ("payload", re.compile(r"\bpayloads?\b", _I)),
    ("reverse-shell", re.compile(r"reverse[\s-]?shell|bind[\s-]?shell", _I)),
    ("backdoor", re.compile(r"backdoor", _I)),
    ("rootkit", re.compile(r"rootkit", _I)),
    ("keylogger", re.compile(r"keylog", _I)),
    ("ransomware", re.compile(r"\bransom", _I)),
    ("malware", re.compile(r"malware|botnet|trojan|spyware", _I)),
    ("phishing", re.compile(r"phish", _I)),
    ("spoofing", re.compile(r"spoof", _I)),
    ("brute-force", re.compile(r"brute[\s-]?forc", _I)),
    ("password", re.compile(r"\bpassword", _I)),
    ("credential", re.compile(r"credential", _I)),
    ("secret", re.compile(r"\bsecrets?\b", _I)),
    ("priv-esc", re.compile(
        r"privilege[\s-]?escalat|priv[\s-]?esc|escalate[\s-]privileges?", _I)),
    ("sandbox-escape", re.compile(r"sandbox[\s-]?escape", _I)),
    ("jailbreak", re.compile(r"jailbreak", _I)),
    ("rce", re.compile(r"\bRCE\b|remote[\s-]code[\s-]execution", _I)),
    ("attack", re.compile(r"\battack", _I)),
    # --- vulnerability classes ---
    ("injection", re.compile(r"\binject", _I)),
    ("xss", re.compile(r"\bXSS\b|cross[\s-]site[\s-]script", _I)),
    ("csrf", re.compile(r"\bCSRF\b|\bSSRF\b", _I)),
    ("buffer-overflow", re.compile(r"buffer[\s-]?overflow|heap[\s-]?overflow", _I)),
    ("use-after-free", re.compile(r"use[\s-]after[\s-]free", _I)),
    ("path-traversal", re.compile(r"path[\s-]?traversal|directory[\s-]?traversal", _I)),
    ("deserialization", re.compile(r"deserializ", _I)),
    ("fuzzing", re.compile(r"\bfuzz", _I)),
    # --- crypto-attack flavored (plain base64/hex ENCODING wording is fine) ---
    ("decrypt", re.compile(r"decrypt", _I)),
    ("crack", re.compile(r"\bcrack", _I)),
    ("cipher", re.compile(r"\bcipher", _I)),
    ("keygen", re.compile(r"\bkeygen\b", _I)),
    # --- defense tooling ---
    ("firewall", re.compile(r"firewall|iptables|nftables", _I)),
    ("mac-hardening", re.compile(r"selinux|apparmor", _I)),
    ("antivirus", re.compile(r"antivirus|anti-virus", _I)),
    ("ids", re.compile(r"\bIDS\b")),  # case-sensitive: ids.py / short_id are fine
    ("intrusion", re.compile(r"intrusion", _I)),
    ("tamper", re.compile(r"tamper", _I)),
    ("vault", re.compile(r"\bvault", _I)),
]


def is_security_flagged(text: str) -> Optional[str]:
    """Name of the first matching blocklist rule, or None when clean."""
    for name, pat in _RULES:
        if pat.search(text):
            return name
    return None


# Files whose CONTENT is not code the model ever trains on: binary assets and
# documentation prose. Their bytes (an image matching "xss") or prose (a
# CHANGELOG line "fixed a vulnerability") produce keyword false positives that
# whole-repo-ban otherwise-clean repos. The file-TREE scans skip these; the text
# scan (is_security_flagged over .py code, test patches, task statements) is
# unchanged — real security-offense code is still caught.
_SKIP_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".pdf",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".zip", ".gz", ".bz2", ".xz", ".tar", ".tgz", ".whl", ".jar",
    ".so", ".pyc", ".pyd", ".dll", ".dylib", ".o", ".a", ".mo",
    ".dat", ".bin", ".pickle", ".pkl", ".npy", ".npz", ".parquet",
    ".rst",  # reStructuredText documentation
})
_SKIP_DOC_STEMS = frozenset({
    "changelog", "changes", "history", "readme", "authors", "news",
    "contributing", "license", "copying", "notice",
})


def scan_skips(path: Path) -> bool:
    """True iff `path`'s content must NOT be security-scanned — binary assets +
    documentation prose (keyword-false-positive sources, never trained on).
    Code (.py, .cfg, .txt, .yml, test fixtures) stays scanned. `.svg` stays
    scanned (XML text can carry real script)."""
    return path.suffix.lower() in _SKIP_SUFFIXES or path.stem.lower() in _SKIP_DOC_STEMS


# ------------------------------------------------------------------- CLI ---

_POSITIVE = [
    "exploit the vulnerability described in CVE-2024-1234",
    "run an nmap scan then start metasploit",
    "craft a reverse shell payload",
    "brute-force the admin password",
    "the login form is vulnerable to SQL injection",
    "decrypt the ciphertext without the key",
    "escalate privileges via a priv-esc chain",
    "configure iptables firewall rules to block the port",
    "use ansible-vault to store the credentials",
    "tamper with the audit log to hide the intrusion",
    "an XSS attack vector in the comment field",
    "fuzz the parser until it segfaults",
]

_NEGATIVE = [
    "base64-encode the file and write it to encoded.b64",
    "compute the SHA-256 checksum of report.dat",
    "the hex digest must be 64 lowercase characters",
    "count the ERROR lines and write the error rate to a file",
    "p99 latency regressed after the scheduler change",
    "sort the distinct words in ascending order",
    "count login sessions per user from the events log",
    "the invoice balance is overdue by 30 days",
    "short_id uses hashlib; see ids.py for the helpers",
    "force-merge the branches after the tests pass",
]


def _self_check() -> int:
    bad = 0
    for text in _POSITIVE:
        rule = is_security_flagged(text)
        status = f"flagged({rule})" if rule else "MISSED"
        bad += rule is None
        print(f"  positive {status:24s} {text[:60]}")
    for text in _NEGATIVE:
        rule = is_security_flagged(text)
        status = f"FALSE-FLAG({rule})" if rule else "clean"
        bad += rule is not None
        print(f"  negative {status:24s} {text[:60]}")
    n = len(_POSITIVE) + len(_NEGATIVE)
    print(f"self-check: {n - bad}/{n} fixtures correct")
    return 1 if bad else 0


def _scan(root: Path) -> int:
    flagged = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if scan_skips(path):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        rule = is_security_flagged(text)
        if rule:
            flagged += 1
            print(f"FLAGGED [{rule}] {path}")
    print(f"scan: {flagged} flagged files under {root}")
    return 1 if flagged else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--scan":
        sys.exit(_scan(Path(sys.argv[2])))
    sys.exit(_self_check())
