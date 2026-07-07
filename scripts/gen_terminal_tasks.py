#!/usr/bin/env python3
"""Compositional, self-verifying Terminal-Bench task GENERATOR (Tmax §5.1).

WHY: the agentic-OPD loop plateaus (pass@1 6->7->7) because the fixed hand-picked
task set isn't difficulty-calibrated — once the model learns them the rest are
always-pass (0 gradient) or always-fail (0 gradient). Tmax's fix: sample over
ORTHOGONAL difficulty axes so a "sweet-spot" band (sometimes-pass) always exists.

Axes (sampled + combined per task):
  DOMAIN      {file-ops, text-processing, data-transform, config-parsing,
               crypto-encoding, log-analysis, service}
  COMPLEXITY  {1: bash+coreutils, 2: bash+python(stdlib), 3: a background service}
  VERIFIER    {exact-file-content, metric-threshold, multi-file-state,
               output-format-json}
  N_STEPS     {short: 2-4 cmds, medium: 5-10 cmds}
Difficulty (task.yaml enum) is a deterministic function of complexity x n_steps.

Each emitted task is a standard Terminal-Bench task dir loadable via
`tb run --dataset-path <out>`:
  task.yaml            instruction + metadata (difficulty, axis tags)
  Dockerfile           FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest
  docker-compose.yaml  canonical TB client service
  run-tests.sh         uv add pytest; run tests/test_outputs.py
  solution.sh          reference oracle that solves the fixture
  tests/test_outputs.py  pytest checking the SOLVED end-state (not re-solving)
  fixture/             input files COPYed into the container WORKDIR /app

Tests resolve the working dir via APP_DIR (default /app, the container WORKDIR),
so the --self-check gate can run them in a local temp dir WITHOUT Docker:
for every task it verifies the UNSOLVED fixture FAILS the tests and the
solution-applied fixture PASSES; only passing tasks are kept. Self-check imports
the test module directly (no local pytest needed) and runs the `test_*` fns —
exactly what PytestParser does in the container. Service (complexity-3) tasks use
a localhost background process, so they self-verify locally too (a real `tb run`
still needs the container); they are tagged `service` in task.yaml.

CLI:
  python3 scripts/gen_terminal_tasks.py --out <dir> --n <count> --seed 0 \
      [--self-check] [--max-complexity 2]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# Shared fixture vocabulary (ascii-only so GNU `sort` == python `sorted`).
# --------------------------------------------------------------------------

WORDS = (
    "time year people way day man thing woman life child world school state "
    "family student group country problem hand part place case week company "
    "system program question work government number night point home water "
    "room mother area money story fact month lot right study book eye job word "
    "business issue side kind head house service friend father power hour game "
    "line member law car city community name team minute idea body back door "
    "health person art war history party result"
).split()

REGIONS = ["north", "south", "east", "west", "central"]
CATEGORIES = ["alpha", "beta", "gamma", "delta"]
LEVELS = ["INFO", "WARN", "DEBUG"]

# --------------------------------------------------------------------------
# Static task-dir scaffolding (identical for every task).
# --------------------------------------------------------------------------

DOCKERFILE = (
    "FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:latest\n"
    "WORKDIR /app\n"
    "COPY fixture/ /app/\n"
)

DOCKER_COMPOSE = """services:
  client:
    build:
      dockerfile: Dockerfile
    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}
    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}
    command: [ "sh", "-c", "sleep infinity" ]
    environment:
      - TEST_DIR=${T_BENCH_TEST_DIR}
    volumes:
      - ${T_BENCH_TASK_LOGS_PATH}:${T_BENCH_CONTAINER_LOGS_PATH}
      - ${T_BENCH_TASK_AGENT_LOGS_PATH}:${T_BENCH_CONTAINER_AGENT_LOGS_PATH}
"""

RUN_TESTS = """#!/bin/bash
# Install uv (isolated venv), then run the pytest end-state checks.
set -uo pipefail
apt-get update >/dev/null 2>&1 || true
apt-get install -y curl >/dev/null 2>&1 || true
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || true
  source "$HOME/.local/bin/env" 2>/dev/null || true
fi
cd "${TEST_DIR:-/tests}"
uv init >/dev/null 2>&1 || true
uv add "pytest==8.4.1" >/dev/null 2>&1
uv run pytest "${TEST_DIR:-/tests}/test_outputs.py" -rA
"""

TEST_HEADER = (
    "import json\n"
    "import os\n"
    "from pathlib import Path\n\n"
    "# The dir the agent operated in (container WORKDIR /app; overridable for\n"
    "# Docker-free self-check via APP_DIR).\n"
    'APP = Path(os.environ.get("APP_DIR", "/app"))\n\n\n'
)

# Portable across GNU (container) and BSD (macOS self-check).
SHA256_CMD = (
    "if command -v sha256sum >/dev/null 2>&1; then\n"
    "  sha256sum {f} | awk '{{print $1}}' > {out}\n"
    "else\n"
    "  shasum -a 256 {f} | awk '{{print $1}}' > {out}\n"
    "fi\n"
)
# Read from stdin so the same command works on GNU (-d) and BSD/macOS (-D).
BASE64_DECODE = (
    "base64 -d < {f} > {out} 2>/dev/null "
    "|| base64 -D < {f} > {out}\n"
)


def py_heredoc(body: str) -> str:
    """Wrap python source as a quoted heredoc (no host var expansion)."""
    return "python3 - <<'PY'\n" + body + "PY\n"


# --------------------------------------------------------------------------
# Task builders. Each takes an rng, returns a spec dict:
#   domain, complexity, verifier, n_steps, instruction,
#   fixtures {relpath: text}, solution (bash body), test_body (pytest source).
# Expected values are computed here and BAKED as literals into the test — the
# test checks the required end-state, it never re-runs the solution's logic.
# --------------------------------------------------------------------------


def b_linecount(rng):
    n = rng.randint(8, 40)
    lines = [f"record-{i}-{rng.randint(100, 999)}" for i in range(n)]
    return dict(
        domain="file-ops", complexity=1, verifier="metric-threshold",
        n_steps="short",
        instruction=(
            "The file data.txt contains one record per line.\n"
            "Count the records and write ONLY that integer (no other text) to "
            "line_count.txt."
        ),
        fixtures={"data.txt": "\n".join(lines) + "\n"},
        solution="wc -l < data.txt | tr -d '[:space:]' > line_count.txt\n",
        test_body=(
            "def test_line_count():\n"
            "    p = APP / 'line_count.txt'\n"
            "    assert p.exists(), 'line_count.txt missing'\n"
            f"    assert p.read_text().strip() == {str(n)!r}\n"
        ),
    )


def b_grepcount(rng):
    n_err = rng.randint(3, 10)
    n_other = rng.randint(12, 30)
    lines = [f"{rng.choice(LEVELS)} handler ok {rng.randint(0,9999)}"
             for _ in range(n_other)]
    lines += [f"ERROR failure code {rng.randint(0,9999)}" for _ in range(n_err)]
    rng.shuffle(lines)
    return dict(
        domain="log-analysis", complexity=1, verifier="metric-threshold",
        n_steps="short",
        instruction=(
            "app.log is a server log with one event per line.\n"
            'Count how many lines contain the substring "ERROR" and write ONLY '
            "that number to error_count.txt."
        ),
        fixtures={"app.log": "\n".join(lines) + "\n"},
        solution="grep -c ERROR app.log > error_count.txt\n",
        test_body=(
            "def test_error_count():\n"
            "    p = APP / 'error_count.txt'\n"
            "    assert p.exists(), 'error_count.txt missing'\n"
            f"    assert p.read_text().strip() == {str(n_err)!r}\n"
        ),
    )


def b_sortuniq(rng):
    words = [rng.choice(WORDS) for _ in range(rng.randint(18, 34))]
    expected = sorted(set(words))
    return dict(
        domain="text-processing", complexity=1, verifier="exact-file-content",
        n_steps="medium",
        instruction=(
            "words.txt contains one lowercase word per line, with duplicates.\n"
            "Write sorted_unique.txt containing each DISTINCT word exactly once, "
            "sorted in ascending order, one per line."
        ),
        fixtures={"words.txt": "\n".join(words) + "\n"},
        solution="LC_ALL=C sort -u words.txt > sorted_unique.txt\n",
        test_body=(
            "def test_sorted_unique():\n"
            "    p = APP / 'sorted_unique.txt'\n"
            "    assert p.exists(), 'sorted_unique.txt missing'\n"
            f"    assert p.read_text().splitlines() == {expected!r}\n"
        ),
    )


def b_sha256(rng):
    msg = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14)))
    digest = hashlib.sha256(msg.encode()).hexdigest()
    return dict(
        domain="crypto-encoding", complexity=1, verifier="exact-file-content",
        n_steps="short",
        instruction=(
            "Compute the SHA-256 checksum of the file payload.dat and write ONLY "
            "the 64-character lowercase hex digest to payload.sha256."
        ),
        fixtures={"payload.dat": msg},
        solution=SHA256_CMD.format(f="payload.dat", out="payload.sha256"),
        test_body=(
            "def test_sha256():\n"
            "    p = APP / 'payload.sha256'\n"
            "    assert p.exists(), 'payload.sha256 missing'\n"
            f"    assert p.read_text().strip() == {digest!r}\n"
        ),
    )


def b_base64(rng):
    msg = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14)))
    enc = base64.b64encode(msg.encode()).decode()
    return dict(
        domain="crypto-encoding", complexity=1, verifier="exact-file-content",
        n_steps="short",
        instruction=(
            "secret.b64 contains Base64-encoded text.\n"
            "Decode it and write the original text to secret.txt (exact bytes; "
            "do not add a trailing newline)."
        ),
        fixtures={"secret.b64": enc + "\n"},
        solution=BASE64_DECODE.format(f="secret.b64", out="secret.txt"),
        test_body=(
            "def test_decoded():\n"
            "    p = APP / 'secret.txt'\n"
            "    assert p.exists(), 'secret.txt missing'\n"
            f"    assert p.read_text() == {msg!r}\n"
        ),
    )


def b_split(rng):
    cats = rng.sample(CATEGORIES, rng.randint(2, 3))
    records = []
    grouped = {c: [] for c in cats}
    for _ in range(rng.randint(10, 20)):
        c = rng.choice(cats)
        val = f"val-{rng.randint(100, 999)}"
        records.append(f"{c}:{val}")
        grouped[c].append(val)
    grouped = {c: v for c, v in grouped.items() if v}
    return dict(
        domain="file-ops", complexity=1, verifier="multi-file-state",
        n_steps="medium",
        instruction=(
            "records.txt has lines of the form CATEGORY:VALUE.\n"
            "For every distinct CATEGORY create a file named CATEGORY.txt "
            "containing that category's VALUEs (VALUE only, without the prefix), "
            "one per line, in the order they appear in records.txt."
        ),
        fixtures={"records.txt": "\n".join(records) + "\n"},
        solution=(
            "while IFS=: read -r cat val; do\n"
            "  [ -n \"$cat\" ] && printf '%s\\n' \"$val\" >> \"$cat.txt\"\n"
            "done < records.txt\n"
        ),
        test_body=(
            "def test_split():\n"
            f"    expected = {grouped!r}\n"
            "    for cat, vals in expected.items():\n"
            "        p = APP / (cat + '.txt')\n"
            "        assert p.exists(), cat + '.txt missing'\n"
            "        assert p.read_text().splitlines() == vals\n"
        ),
    )


def b_csv_groupsum(rng):
    regions = rng.sample(REGIONS, rng.randint(2, 4))
    rows, totals = [], {}
    for _ in range(rng.randint(8, 16)):
        r = rng.choice(regions)
        amt = rng.randint(1, 200)
        rows.append(f"{r},{amt}")
        totals[r] = totals.get(r, 0) + amt
    csv_text = "region,amount\n" + "\n".join(rows) + "\n"
    return dict(
        domain="data-transform", complexity=2, verifier="output-format-json",
        n_steps="medium",
        instruction=(
            "sales.csv has the header 'region,amount' followed by data rows.\n"
            "Write summary.json: a JSON object mapping each region to the INTEGER "
            "sum of its amounts."
        ),
        fixtures={"sales.csv": csv_text},
        solution=py_heredoc(
            "import csv, json\n"
            "totals = {}\n"
            "with open('sales.csv') as f:\n"
            "    for row in csv.DictReader(f):\n"
            "        totals[row['region']] = "
            "totals.get(row['region'], 0) + int(row['amount'])\n"
            "json.dump(totals, open('summary.json', 'w'))\n"
        ),
        test_body=(
            "def test_summary():\n"
            "    p = APP / 'summary.json'\n"
            "    assert p.exists(), 'summary.json missing'\n"
            f"    assert json.loads(p.read_text()) == {totals!r}\n"
        ),
    )


def b_mean(rng):
    xs = [round(rng.uniform(0, 100), 2) for _ in range(rng.randint(6, 15))]
    mean = sum(xs) / len(xs)
    return dict(
        domain="data-transform", complexity=2, verifier="metric-threshold",
        n_steps="short",
        instruction=(
            "numbers.txt has one number per line.\n"
            "Compute their arithmetic mean and write it to mean.txt. Your answer "
            "is accepted if it is within 0.01 of the true mean."
        ),
        fixtures={"numbers.txt": "\n".join(str(x) for x in xs) + "\n"},
        solution=py_heredoc(
            "xs = [float(l) for l in open('numbers.txt') if l.strip()]\n"
            "open('mean.txt', 'w').write(str(sum(xs) / len(xs)))\n"
        ),
        test_body=(
            "def test_mean():\n"
            "    p = APP / 'mean.txt'\n"
            "    assert p.exists(), 'mean.txt missing'\n"
            f"    assert abs(float(p.read_text().strip()) - {mean!r}) < 0.01\n"
        ),
    )


def b_ini_json(rng):
    keys = rng.sample(
        ["host", "port", "debug", "retries", "timeout", "user", "region"],
        rng.randint(3, 5),
    )
    conf = {k: str(rng.randint(1, 9999)) for k in keys}
    body_lines = ["# generated config", ""]
    for k, v in conf.items():
        body_lines.append(f"{k}={v}")
    rng.shuffle(body_lines)
    return dict(
        domain="config-parsing", complexity=2, verifier="output-format-json",
        n_steps="medium",
        instruction=(
            "settings.ini contains lines of the form KEY=VALUE, plus blank lines "
            "and comment lines starting with '#'.\n"
            "Write config.json: a JSON object mapping each KEY to its VALUE (as a "
            "string). Ignore blank and comment lines."
        ),
        fixtures={"settings.ini": "\n".join(body_lines) + "\n"},
        solution=py_heredoc(
            "import json\n"
            "d = {}\n"
            "for line in open('settings.ini'):\n"
            "    line = line.strip()\n"
            "    if not line or line.startswith('#'):\n"
            "        continue\n"
            "    k, _, v = line.partition('=')\n"
            "    d[k.strip()] = v.strip()\n"
            "json.dump(d, open('config.json', 'w'))\n"
        ),
        test_body=(
            "def test_config():\n"
            "    p = APP / 'config.json'\n"
            "    assert p.exists(), 'config.json missing'\n"
            f"    assert json.loads(p.read_text()) == {conf!r}\n"
        ),
    )


def b_json_filter(rng):
    items = [{"name": f"user{i}", "score": rng.randint(0, 100)}
             for i in range(rng.randint(6, 14))]
    expected = [o["name"] for o in items if o["score"] >= 60]
    return dict(
        domain="data-transform", complexity=2, verifier="output-format-json",
        n_steps="medium",
        instruction=(
            "input.json is a JSON array of objects, each with 'name' and "
            "'score'.\n"
            "Write passed.json: a JSON array of the names (strings) whose score "
            "is >= 60, preserving the input order."
        ),
        fixtures={"input.json": json.dumps(items, indent=2) + "\n"},
        solution=py_heredoc(
            "import json\n"
            "data = json.load(open('input.json'))\n"
            "out = [o['name'] for o in data if o['score'] >= 60]\n"
            "json.dump(out, open('passed.json', 'w'))\n"
        ),
        test_body=(
            "def test_passed():\n"
            "    p = APP / 'passed.json'\n"
            "    assert p.exists(), 'passed.json missing'\n"
            f"    assert json.loads(p.read_text()) == {expected!r}\n"
        ),
    )


def b_topwords(rng):
    words = [rng.choice(WORDS[:20]) for _ in range(rng.randint(30, 60))]
    counter = Counter(words)
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    top3 = [f"{w} {c}" for w, c in ranked[:3]]
    total = len(words)
    return dict(
        domain="text-processing", complexity=2, verifier="multi-file-state",
        n_steps="medium",
        instruction=(
            "prose.txt contains words separated by whitespace.\n"
            "Write total_words.txt with the total number of words.\n"
            "Write top3.txt listing the 3 most frequent words, one per line in "
            "the format 'WORD COUNT', ordered by descending count and then "
            "ascending word for ties."
        ),
        fixtures={"prose.txt": " ".join(words) + "\n"},
        solution=py_heredoc(
            "from collections import Counter\n"
            "words = open('prose.txt').read().split()\n"
            "c = Counter(words)\n"
            "ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))\n"
            "with open('top3.txt', 'w') as f:\n"
            "    for w, n in ranked[:3]:\n"
            "        f.write(f'{w} {n}\\n')\n"
            "open('total_words.txt', 'w').write(str(len(words)))\n"
        ),
        test_body=(
            "def test_topwords():\n"
            "    tp = APP / 'total_words.txt'\n"
            "    t3 = APP / 'top3.txt'\n"
            "    assert tp.exists() and t3.exists(), 'output files missing'\n"
            f"    assert tp.read_text().strip() == {str(total)!r}\n"
            f"    assert t3.read_text().splitlines() == {top3!r}\n"
        ),
    )


def b_awk_window(rng):
    base = rng.randint(1_600_000_000, 1_700_000_000)
    stamps = sorted(base + rng.randint(0, 5000) for _ in range(rng.randint(15, 30)))
    lo = base + 1000
    hi = base + 3500
    count = sum(1 for s in stamps if lo <= s <= hi)
    lines = [f"{s} {rng.choice(LEVELS)} event" for s in stamps]
    return dict(
        domain="log-analysis", complexity=1, verifier="metric-threshold",
        n_steps="medium",
        instruction=(
            "Each line of events.log begins with a unix timestamp.\n"
            f"Count how many events fall in the inclusive window [{lo}, {hi}] "
            "and write ONLY that count to window_count.txt."
        ),
        fixtures={"events.log": "\n".join(lines) + "\n"},
        solution=(
            f"awk -v a={lo} -v b={hi} "
            "'$1>=a && $1<=b {c++} END{print c+0}' events.log "
            "> window_count.txt\n"
        ),
        test_body=(
            "def test_window():\n"
            "    p = APP / 'window_count.txt'\n"
            "    assert p.exists(), 'window_count.txt missing'\n"
            f"    assert p.read_text().strip() == {str(count)!r}\n"
        ),
    )


def b_http_service(rng):
    port = rng.randint(20000, 60000)
    page = (
        "<html><body><h1>report</h1>"
        + "".join(f"<p>{rng.choice(WORDS)}</p>" for _ in range(rng.randint(3, 8)))
        + "</body></html>"
    )
    return dict(
        domain="service", complexity=3, verifier="exact-file-content",
        n_steps="medium",
        instruction=(
            "Serve the current directory over HTTP on a local port, request the "
            "file page.html from that server, and save the response body to "
            "fetched.html.\n"
            "(A simple background HTTP server plus a localhost fetch is enough; "
            "fetched.html must byte-match page.html.)"
        ),
        fixtures={"page.html": page},
        solution=(
            f"python3 -m http.server {port} --directory . >/dev/null 2>&1 &\n"
            "SRV=$!\n"
            "sleep 1\n"
            "python3 -c \"import urllib.request; "
            "open('fetched.html','wb').write("
            f"urllib.request.urlopen('http://localhost:{port}/page.html').read())\"\n"
            "kill $SRV 2>/dev/null || true\n"
        ),
        test_body=(
            "def test_fetched():\n"
            "    p = APP / 'fetched.html'\n"
            "    assert p.exists(), 'fetched.html missing'\n"
            f"    assert p.read_text() == {page!r}\n"
        ),
    )


BUILDERS = [
    b_linecount, b_grepcount, b_sortuniq, b_sha256, b_base64, b_split,
    b_awk_window, b_csv_groupsum, b_mean, b_ini_json, b_json_filter,
    b_topwords, b_http_service,
]


# --------------------------------------------------------------------------
# Difficulty + emission.
# --------------------------------------------------------------------------


def difficulty_of(complexity: int, n_steps: str) -> str:
    """easy | medium | hard from the axis combination (task.yaml enum)."""
    if complexity >= 3:
        return "hard"
    if complexity == 2:
        return "hard" if n_steps == "medium" else "medium"
    return "medium" if n_steps == "medium" else "easy"


def task_yaml(spec: dict, difficulty: str) -> str:
    instr = "\n".join("  " + l for l in spec["instruction"].splitlines())
    tags = [
        spec["domain"],
        f"complexity-{spec['complexity']}",
        spec["verifier"],
        f"steps-{spec['n_steps']}",
        "generated",
    ]
    tag_block = "".join(f"  - {t}\n" for t in tags)
    return (
        "instruction: |-\n" + instr + "\n"
        "author_name: arle-gen\n"
        "author_email: q1293822641@gmail.com\n"
        f"difficulty: {difficulty}\n"
        f"category: {spec['domain']}\n"
        "tags:\n" + tag_block
        + "max_agent_timeout_sec: 180.0\n"
        "max_test_timeout_sec: 30.0\n"
    )


def write_task(task_dir: Path, spec: dict, difficulty: str) -> None:
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "fixture").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.yaml").write_text(task_yaml(spec, difficulty))
    (task_dir / "Dockerfile").write_text(DOCKERFILE)
    (task_dir / "docker-compose.yaml").write_text(DOCKER_COMPOSE)
    (task_dir / "run-tests.sh").write_text(RUN_TESTS)
    (task_dir / "solution.sh").write_text("#!/bin/bash\n" + spec["solution"])
    (task_dir / "tests" / "test_outputs.py").write_text(
        TEST_HEADER + spec["test_body"]
    )
    for rel, content in spec["fixtures"].items():
        (task_dir / "fixture" / rel).write_text(content)


# --------------------------------------------------------------------------
# Self-check gate (Docker-free): unsolved fixture FAILS, solved fixture PASSES.
# --------------------------------------------------------------------------


def _run_tests(test_file: Path, appdir: Path) -> None:
    """Run every test_* fn in the module (what PytestParser does). Raises on fail."""
    import os
    os.environ["APP_DIR"] = str(appdir)
    sys.dont_write_bytecode = True  # keep __pycache__ out of the emitted task dir
    loader = importlib.util.spec_from_file_location(
        f"tb_test_{appdir.name}", test_file
    )
    mod = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    assert fns, "no test_* functions"
    for fn in fns:
        fn()


def self_check(task_dir: Path) -> str | None:
    """Return None if the task passes the gate, else a rejection reason."""
    test_file = task_dir / "tests" / "test_outputs.py"
    solution = task_dir / "solution.sh"
    with tempfile.TemporaryDirectory(prefix="gtask-") as tmp:
        app = Path(tmp) / "app"
        shutil.copytree(task_dir / "fixture", app)

        # Phase 1: unsolved fixture must FAIL the tests (else no gradient).
        try:
            _run_tests(test_file, app)
            return "unsolved fixture already PASSES (no signal)"
        except AssertionError:
            pass
        except Exception as e:  # noqa: BLE001 - any error is a fail signal
            _ = e

        # Phase 2: apply the oracle solution.
        r = subprocess.run(
            ["bash", str(solution)], cwd=app,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return f"solution.sh exited {r.returncode}: {r.stderr.strip()[:160]}"

        # Phase 3: solved fixture must PASS.
        try:
            _run_tests(test_file, app)
        except Exception as e:  # noqa: BLE001
            return f"solved fixture FAILS: {type(e).__name__}: {str(e)[:160]}"
    return None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--max-complexity", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--domains", type=str, default="",
                    help="comma-separated domains to restrict to (curriculum "
                         "targeting; empty = all)")
    args = ap.parse_args()

    want = {d.strip() for d in args.domains.split(",") if d.strip()}
    pool = [b for b in BUILDERS
            if b(random.Random(0))["complexity"] <= args.max_complexity
            and (not want or b(random.Random(0))["domain"] in want)]
    if not pool:
        sys.exit(f"no builders match domains={want} max_complexity={args.max_complexity}")
    # Round-robin over builders (shuffled by seed) so axes stay evenly spread.
    order = list(pool)
    random.Random(args.seed).shuffle(order)

    args.out.mkdir(parents=True, exist_ok=True)

    accepted, rejected = [], []
    axis_counts = {"domain": Counter(), "complexity": Counter(),
                   "verifier": Counter(), "difficulty": Counter()}

    for i in range(args.n):
        builder = order[i % len(order)]
        rng = random.Random(f"{args.seed}:{i}:{builder.__name__}")
        spec = builder(rng)
        difficulty = difficulty_of(spec["complexity"], spec["n_steps"])
        task_id = f"gen-{builder.__name__[2:]}-{i:03d}"
        task_dir = args.out / task_id
        write_task(task_dir, spec, difficulty)

        if args.self_check:
            reason = self_check(task_dir)
            if reason is not None:
                rejected.append((task_id, reason))
                shutil.rmtree(task_dir)
                continue

        accepted.append(task_id)
        axis_counts["domain"][spec["domain"]] += 1
        axis_counts["complexity"][f"c{spec['complexity']}"] += 1
        axis_counts["verifier"][spec["verifier"]] += 1
        axis_counts["difficulty"][difficulty] += 1

    print(f"\n=== gen_terminal_tasks: out={args.out} seed={args.seed} "
          f"n={args.n} max_complexity={args.max_complexity} ===")
    print(f"accepted: {len(accepted)}   rejected: {len(rejected)}")
    for tid, reason in rejected:
        print(f"  REJECT {tid}: {reason}")
    if args.self_check:
        print("self-check: base-FAIL / solution-PASS verified Docker-free "
              "(service tasks verified via localhost)")
    print("\naxis distribution (accepted):")
    for axis, counter in axis_counts.items():
        dist = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        print(f"  {axis:10s} {dist}")
    return 1 if (args.self_check and rejected) else 0


if __name__ == "__main__":
    sys.exit(main())
