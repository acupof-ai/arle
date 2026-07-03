#!/usr/bin/env python3
"""Generate the synthetic agent-OPD bug-fix corpus (SWE-bench-Pro schema).

36 distinct single-bug tasks over small self-contained Python packages —
12 train / 24 eval, split by FUNCTION (no eval function ever appears in
train), so the held-out pass-rate measures behavior-level generalization
(locate → minimal edit → hidden tests pass), not memorization.

Each task emits:
  - a staged repo tree under  <out>/staged/<instance_id>/   (plain files;
    `boot_workdir` git-inits the sandbox copy itself)
  - one JSONL row (train or eval) with the SWE-Pro fields the Rust loader
    (`crates/train/src/swe_dataset.rs`) reads, plus `gold_patch` /
    `archetype` extras the loader ignores.

`test_patch` is a git-appliable new-file diff adding tests/test_hidden.py;
`fail_to_pass` are its pytest node ids. Scoring runs `python3 -m pytest`
with cwd = repo root (cwd lands on sys.path, so flat modules import).

--self-check is the corpus correctness gate (mirrors
`sandbox.rs::score_workdir` semantics): for every task, the BASE tree must
FAIL the hidden tests and the gold-patched tree must PASS. Needs only
git + python3 + pytest; run it locally before shipping the corpus to a pod.

Plan: docs/plans/2026-07-03-agentic-opd-27b-capability-curve.md
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------
# Task pool. Each entry: slug, module, archetype, split, buggy, gold, extra
# (healthy sibling code kept identical across buggy/gold), test, statement.
# The statement describes the SYMPTOM with a concrete repro — never the fix.
# --------------------------------------------------------------------------

TASKS = [
    # ---------------- train (12) ----------------
    dict(
        slug="stock-restock",
        module="stock",
        archetype="inverted-comparison",
        split="train",
        buggy="def needs_restock(count, threshold):\n"
        '    """True when an item is low on stock and must be reordered."""\n'
        "    return count > threshold\n",
        gold="def needs_restock(count, threshold):\n"
        '    """True when an item is low on stock and must be reordered."""\n'
        "    return count < threshold\n",
        extra="def stock_value(count, unit_price):\n"
        '    """Total value of the units on hand."""\n'
        "    return count * unit_price\n",
        test="from stock import needs_restock\n\n\n"
        "def test_low_stock_needs_restock():\n"
        "    assert needs_restock(2, 10) is True\n\n\n"
        "def test_full_stock_does_not():\n"
        "    assert needs_restock(50, 10) is False\n",
        statement="Restock alerts fire for the wrong items: well-stocked SKUs "
        "get flagged and empty shelves never do.\n\nRepro:\n"
        ">>> needs_restock(50, 10)\nTrue   # expected: False\n"
        ">>> needs_restock(2, 10)\nFalse  # expected: True",
    ),
    dict(
        slug="pricing-discount",
        module="pricing",
        archetype="wrong-operator",
        split="train",
        buggy="def apply_discount(price, rate):\n"
        '    """Price after a fractional discount, e.g. rate=0.2 -> 20% off."""\n'
        "    return price * (1 + rate)\n",
        gold="def apply_discount(price, rate):\n"
        '    """Price after a fractional discount, e.g. rate=0.2 -> 20% off."""\n'
        "    return price * (1 - rate)\n",
        extra="def add_tax(price, rate):\n"
        '    """Price with sales tax added."""\n'
        "    return price * (1 + rate)\n",
        test="from pricing import apply_discount\n\n\n"
        "def test_twenty_percent_off():\n"
        "    assert apply_discount(100.0, 0.2) == 80.0\n\n\n"
        "def test_zero_rate_is_identity():\n"
        "    assert apply_discount(50.0, 0.0) == 50.0\n",
        statement="Discounted prices come out HIGHER than the list price.\n\n"
        "Repro:\n>>> apply_discount(100.0, 0.2)\n120.0   # expected: 80.0",
    ),
    dict(
        slug="dates-between",
        module="datecalc",
        archetype="off-by-one",
        split="train",
        buggy="from datetime import date\n\n\n"
        "def days_between(a, b):\n"
        '    """Whole days from date a to date b (b >= a)."""\n'
        "    return (b - a).days + 1\n",
        gold="from datetime import date\n\n\n"
        "def days_between(a, b):\n"
        '    """Whole days from date a to date b (b >= a)."""\n'
        "    return (b - a).days\n",
        extra="def is_same_month(a, b):\n"
        '    """True when both dates fall in the same calendar month."""\n'
        "    return (a.year, a.month) == (b.year, b.month)\n",
        test="from datetime import date\n\nfrom datecalc import days_between\n\n\n"
        "def test_adjacent_days():\n"
        "    assert days_between(date(2026, 1, 1), date(2026, 1, 2)) == 1\n\n\n"
        "def test_same_day_is_zero():\n"
        "    assert days_between(date(2026, 3, 5), date(2026, 3, 5)) == 0\n",
        statement="Every duration the date calculator reports is one day too "
        "long; identical dates report 1 day apart.\n\nRepro:\n"
        ">>> days_between(date(2026, 1, 1), date(2026, 1, 2))\n"
        "2   # expected: 1",
    ),
    dict(
        slug="config-getbool",
        module="conf",
        archetype="wrong-default",
        split="train",
        buggy="def get_bool(cfg, key):\n"
        '    """Read a boolean flag from a string-valued config dict."""\n'
        '    return cfg.get(key, "true").lower() == "true"\n',
        gold="def get_bool(cfg, key):\n"
        '    """Read a boolean flag from a string-valued config dict."""\n'
        '    return cfg.get(key, "false").lower() == "true"\n',
        extra="def get_int(cfg, key, default=0):\n"
        '    """Read an integer config value with a default."""\n'
        "    try:\n        return int(cfg.get(key, default))\n"
        "    except ValueError:\n        return default\n",
        test="from conf import get_bool\n\n\n"
        "def test_missing_flag_defaults_off():\n"
        "    assert get_bool({}, \"debug\") is False\n\n\n"
        "def test_explicit_true_is_on():\n"
        "    assert get_bool({\"debug\": \"true\"}, \"debug\") is True\n",
        statement="Feature flags that are absent from the config behave as if "
        "they were switched ON — every optional feature is enabled by "
        "default.\n\nRepro:\n>>> get_bool({}, \"debug\")\n"
        "True   # expected: False",
    ),
    dict(
        slug="stats-mean",
        module="stats_basic",
        archetype="off-by-one",
        split="train",
        buggy="def mean(xs):\n"
        '    """Arithmetic mean of a non-empty sequence."""\n'
        "    return sum(xs) / (len(xs) - 1)\n",
        gold="def mean(xs):\n"
        '    """Arithmetic mean of a non-empty sequence."""\n'
        "    return sum(xs) / len(xs)\n",
        extra="def total(xs):\n"
        '    """Sum of the sequence."""\n'
        "    return sum(xs)\n",
        test="from stats_basic import mean\n\n\n"
        "def test_mean_of_two():\n"
        "    assert mean([2, 4]) == 3\n\n\n"
        "def test_mean_of_constant():\n"
        "    assert mean([5, 5, 5]) == 5\n",
        statement="Reported averages are far too large, and a single-element "
        "average crashes with ZeroDivisionError.\n\nRepro:\n"
        ">>> mean([2, 4])\n6.0   # expected: 3.0",
    ),
    dict(
        slug="search-indexof",
        module="bsearch",
        archetype="boundary-condition",
        split="train",
        buggy="def index_of(sorted_xs, target):\n"
        '    """Index of target in a sorted list, -1 when absent."""\n'
        "    lo, hi = 1, len(sorted_xs) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if sorted_xs[mid] == target:\n            return mid\n"
        "        if sorted_xs[mid] < target:\n            lo = mid + 1\n"
        "        else:\n            hi = mid - 1\n"
        "    return -1\n",
        gold="def index_of(sorted_xs, target):\n"
        '    """Index of target in a sorted list, -1 when absent."""\n'
        "    lo, hi = 0, len(sorted_xs) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if sorted_xs[mid] == target:\n            return mid\n"
        "        if sorted_xs[mid] < target:\n            lo = mid + 1\n"
        "        else:\n            hi = mid - 1\n"
        "    return -1\n",
        extra="def contains(sorted_xs, target):\n"
        '    """Membership via index_of."""\n'
        "    return index_of(sorted_xs, target) != -1\n",
        test="from bsearch import index_of\n\n\n"
        "def test_finds_first_element():\n"
        "    assert index_of([3, 5, 7], 3) == 0\n\n\n"
        "def test_finds_middle_element():\n"
        "    assert index_of([3, 5, 7], 5) == 1\n",
        statement="The binary search never finds the FIRST element of the "
        "list; everything else is found fine.\n\nRepro:\n"
        ">>> index_of([3, 5, 7], 3)\n-1   # expected: 0",
    ),
    dict(
        slug="contact-email",
        module="contact",
        archetype="missing-none-guard",
        split="train",
        buggy="def normalize_email(email):\n"
        '    """Lower-cased, trimmed email; empty string for missing input."""\n'
        "    return email.strip().lower()\n",
        gold="def normalize_email(email):\n"
        '    """Lower-cased, trimmed email; empty string for missing input."""\n'
        "    if email is None:\n        return \"\"\n"
        "    return email.strip().lower()\n",
        extra="def domain_of(email):\n"
        '    """The part after @, or empty string."""\n'
        "    _, _, dom = email.partition(\"@\")\n    return dom\n",
        test="from contact import normalize_email\n\n\n"
        "def test_none_becomes_empty():\n"
        "    assert normalize_email(None) == \"\"\n\n\n"
        "def test_trims_and_lowers():\n"
        "    assert normalize_email(\" A@B.com \") == \"a@b.com\"\n",
        statement="Importing contacts crashes whenever a record has no email "
        "address.\n\nRepro:\n>>> normalize_email(None)\n"
        "AttributeError: 'NoneType' object has no attribute 'strip'\n"
        "# expected: \"\" (empty string)",
    ),
    dict(
        slug="registry-register",
        module="registry",
        archetype="mutable-default",
        split="train",
        buggy="def register(name, items=[]):\n"
        '    """Append name to items (a fresh list when omitted) and return it."""\n'
        "    items.append(name)\n    return items\n",
        gold="def register(name, items=None):\n"
        '    """Append name to items (a fresh list when omitted) and return it."""\n'
        "    if items is None:\n        items = []\n"
        "    items.append(name)\n    return items\n",
        extra="def unregister(name, items):\n"
        '    """Remove name from items when present."""\n'
        "    if name in items:\n        items.remove(name)\n    return items\n",
        test="from registry import register\n\n\n"
        "def test_fresh_list_per_call():\n"
        "    register(\"a\")\n"
        "    assert register(\"b\") == [\"b\"]\n\n\n"
        "def test_explicit_list_is_used():\n"
        "    assert register(\"c\", [\"x\"]) == [\"x\", \"c\"]\n",
        statement="Registrations LEAK across unrelated calls: registering a "
        "second plugin returns the first one too.\n\nRepro:\n"
        ">>> register(\"a\")\n['a']\n>>> register(\"b\")\n"
        "['a', 'b']   # expected: ['b']",
    ),
    dict(
        slug="batch-chunk",
        module="batching",
        archetype="boundary-condition",
        split="train",
        buggy="def chunk(xs, size):\n"
        '    """Split xs into consecutive chunks of at most `size` items."""\n'
        "    return [xs[i : i + size] for i in range(0, len(xs) - size, size)]\n",
        gold="def chunk(xs, size):\n"
        '    """Split xs into consecutive chunks of at most `size` items."""\n'
        "    return [xs[i : i + size] for i in range(0, len(xs), size)]\n",
        extra="def flatten(chunks):\n"
        '    """Inverse of chunk."""\n'
        "    return [x for c in chunks for x in c]\n",
        test="from batching import chunk\n\n\n"
        "def test_keeps_the_tail():\n"
        "    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n\n\n"
        "def test_exact_multiple():\n"
        "    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]\n",
        statement="Batching silently DROPS the last few records of every "
        "upload.\n\nRepro:\n>>> chunk([1, 2, 3, 4, 5], 2)\n"
        "[[1, 2], [3, 4]]   # expected: [[1, 2], [3, 4], [5]]",
    ),
    dict(
        slug="grades-letter",
        module="grades",
        archetype="boundary-condition",
        split="train",
        buggy="def letter_grade(score):\n"
        '    """A for 90+, B for 80+, C for 70+, else F."""\n'
        "    if score > 90:\n        return \"A\"\n"
        "    if score >= 80:\n        return \"B\"\n"
        "    if score >= 70:\n        return \"C\"\n"
        "    return \"F\"\n",
        gold="def letter_grade(score):\n"
        '    """A for 90+, B for 80+, C for 70+, else F."""\n'
        "    if score >= 90:\n        return \"A\"\n"
        "    if score >= 80:\n        return \"B\"\n"
        "    if score >= 70:\n        return \"C\"\n"
        "    return \"F\"\n",
        extra="def passed(score):\n"
        '    """True for any non-F grade."""\n'
        "    return letter_grade(score) != \"F\"\n",
        test="from grades import letter_grade\n\n\n"
        "def test_exactly_ninety_is_a():\n"
        "    assert letter_grade(90) == \"A\"\n\n\n"
        "def test_mid_b_band():\n"
        "    assert letter_grade(85) == \"B\"\n",
        statement="Students scoring exactly 90 are shown a B instead of an A; "
        "91 and up works.\n\nRepro:\n>>> letter_grade(90)\n"
        "'B'   # expected: 'A'",
    ),
    dict(
        slug="paging-count",
        module="paging",
        archetype="missing-ceil",
        split="train",
        buggy="def page_count(total, per_page):\n"
        '    """Number of pages needed to show `total` items."""\n'
        "    return total // per_page\n",
        gold="def page_count(total, per_page):\n"
        '    """Number of pages needed to show `total` items."""\n'
        "    return (total + per_page - 1) // per_page\n",
        extra="def page_slice(page, per_page):\n"
        '    """(start, end) item indices for a 0-based page."""\n'
        "    return page * per_page, (page + 1) * per_page\n",
        test="from paging import page_count\n\n\n"
        "def test_partial_last_page_counts():\n"
        "    assert page_count(10, 3) == 4\n\n\n"
        "def test_exact_fit():\n"
        "    assert page_count(9, 3) == 3\n",
        statement="The last, partially-filled page of results is "
        "unreachable — pagination shows one page too few.\n\nRepro:\n"
        ">>> page_count(10, 3)\n3   # expected: 4",
    ),
    dict(
        slug="jobs-fifo",
        module="jobs",
        archetype="wrong-end",
        split="train",
        buggy="def next_job(queue):\n"
        '    """Pop and return the OLDEST job (FIFO)."""\n'
        "    return queue.pop()\n",
        gold="def next_job(queue):\n"
        '    """Pop and return the OLDEST job (FIFO)."""\n'
        "    return queue.pop(0)\n",
        extra="def add_job(queue, job):\n"
        '    """Append a job to the queue."""\n'
        "    queue.append(job)\n    return queue\n",
        test="from jobs import next_job\n\n\n"
        "def test_fifo_order():\n"
        "    q = [\"first\", \"second\", \"third\"]\n"
        "    assert next_job(q) == \"first\"\n"
        "    assert next_job(q) == \"second\"\n",
        statement="Jobs run NEWEST-first: freshly queued work jumps ahead of "
        "jobs that have waited hours.\n\nRepro:\n"
        ">>> q = [\"first\", \"second\", \"third\"]\n>>> next_job(q)\n"
        "'third'   # expected: 'first'",
    ),
    # ---------------- eval (24) ----------------
    dict(
        slug="cart-subtotal",
        module="cart",
        archetype="wrong-operator",
        split="eval",
        buggy="def cart_subtotal(items):\n"
        '    """Sum of qty * price over line items ({\'qty\', \'price\'})."""\n'
        "    total = 0.0\n"
        "    for item in items:\n"
        "        total += item[\"qty\"] + item[\"price\"]\n"
        "    return total\n",
        gold="def cart_subtotal(items):\n"
        '    """Sum of qty * price over line items ({\'qty\', \'price\'})."""\n'
        "    total = 0.0\n"
        "    for item in items:\n"
        "        total += item[\"qty\"] * item[\"price\"]\n"
        "    return total\n",
        extra="def item_count(items):\n"
        '    """Total units across line items."""\n'
        "    return sum(item[\"qty\"] for item in items)\n",
        test="from cart import cart_subtotal\n\n\n"
        "def test_multiplies_qty_by_price():\n"
        "    assert cart_subtotal([{\"qty\": 2, \"price\": 5.0}]) == 10.0\n\n\n"
        "def test_multiple_lines():\n"
        "    items = [{\"qty\": 1, \"price\": 3.0}, {\"qty\": 3, \"price\": 2.0}]\n"
        "    assert cart_subtotal(items) == 9.0\n",
        statement="Cart totals are wildly wrong for multi-unit lines: 2 units "
        "at $5 shows $7.\n\nRepro:\n"
        ">>> cart_subtotal([{\"qty\": 2, \"price\": 5.0}])\n"
        "7.0   # expected: 10.0",
    ),
    dict(
        slug="sched-weekend",
        module="sched",
        archetype="boundary-condition",
        split="eval",
        buggy="def is_weekend(d):\n"
        '    """True for Saturday and Sunday."""\n'
        "    return d.weekday() > 5\n",
        gold="def is_weekend(d):\n"
        '    """True for Saturday and Sunday."""\n'
        "    return d.weekday() >= 5\n",
        extra="def is_month_start(d):\n"
        '    """True on the 1st."""\n'
        "    return d.day == 1\n",
        test="from datetime import date\n\nfrom sched import is_weekend\n\n\n"
        "def test_saturday_is_weekend():\n"
        "    assert is_weekend(date(2026, 1, 3)) is True\n\n\n"
        "def test_monday_is_not():\n"
        "    assert is_weekend(date(2026, 1, 5)) is False\n",
        statement="Saturday deliveries are being scheduled as if it were a "
        "weekday; Sunday is handled correctly.\n\nRepro:\n"
        ">>> is_weekend(date(2026, 1, 3))   # a Saturday\n"
        "False   # expected: True",
    ),
    dict(
        slug="text-slugify",
        module="slug",
        archetype="wrong-operator",
        split="eval",
        buggy="import re\n\n\n"
        "def slugify(s):\n"
        '    """Lower-case URL slug with words joined by hyphens."""\n'
        "    return re.sub(r\"[^a-z0-9]+\", \"\", s.lower()).strip(\"-\")\n",
        gold="import re\n\n\n"
        "def slugify(s):\n"
        '    """Lower-case URL slug with words joined by hyphens."""\n'
        "    return re.sub(r\"[^a-z0-9]+\", \"-\", s.lower()).strip(\"-\")\n",
        extra="def title_of(slug_text):\n"
        '    """Rough inverse: hyphens to spaces, title case."""\n'
        "    return slug_text.replace(\"-\", \" \").title()\n",
        test="from slug import slugify\n\n\n"
        "def test_words_are_hyphenated():\n"
        "    assert slugify(\"Hello World\") == \"hello-world\"\n\n\n"
        "def test_punctuation_collapses():\n"
        "    assert slugify(\"a  b!c\") == \"a-b-c\"\n",
        statement="Generated URL slugs mash all words together with no "
        "separator.\n\nRepro:\n>>> slugify(\"Hello World\")\n"
        "'helloworld'   # expected: 'hello-world'",
    ),
    dict(
        slug="text-truncate",
        module="trunc",
        archetype="missing-suffix",
        split="eval",
        buggy="def truncate(s, n):\n"
        '    """Cut s to n chars, appending \'...\' when it was cut."""\n'
        "    if len(s) > n:\n        return s[:n]\n"
        "    return s\n",
        gold="def truncate(s, n):\n"
        '    """Cut s to n chars, appending \'...\' when it was cut."""\n'
        "    if len(s) > n:\n        return s[:n] + \"...\"\n"
        "    return s\n",
        extra="def pad_to(s, n):\n"
        '    """Right-pad s with spaces to length n."""\n'
        "    return s.ljust(n)\n",
        test="from trunc import truncate\n\n\n"
        "def test_cut_text_gets_ellipsis():\n"
        "    assert truncate(\"abcdef\", 3) == \"abc...\"\n\n\n"
        "def test_short_text_untouched():\n"
        "    assert truncate(\"ab\", 3) == \"ab\"\n",
        statement="Truncated previews end abruptly with no '...' marker, so "
        "users can't tell the text was cut.\n\nRepro:\n"
        ">>> truncate(\"abcdef\", 3)\n'abc'   # expected: 'abc...'",
    ),
    dict(
        slug="csvlite-rows",
        module="csvlite",
        archetype="off-by-one",
        split="eval",
        buggy="def parse_rows(text):\n"
        '    """Data rows (header excluded) of comma-separated text."""\n'
        "    lines = [l for l in text.splitlines() if l.strip()]\n"
        "    return [l.split(\",\") for l in lines[2:]]\n",
        gold="def parse_rows(text):\n"
        '    """Data rows (header excluded) of comma-separated text."""\n'
        "    lines = [l for l in text.splitlines() if l.strip()]\n"
        "    return [l.split(\",\") for l in lines[1:]]\n",
        extra="def header_of(text):\n"
        '    """The header column names."""\n'
        "    lines = [l for l in text.splitlines() if l.strip()]\n"
        "    return lines[0].split(\",\") if lines else []\n",
        test="from csvlite import parse_rows\n\n\n"
        "def test_first_data_row_kept():\n"
        "    assert parse_rows(\"h1,h2\\na,b\\nc,d\") == [[\"a\", \"b\"], [\"c\", \"d\"]]\n",
        statement="Every import is missing exactly its FIRST data record; "
        "files with a single record import as empty.\n\nRepro:\n"
        ">>> parse_rows(\"h1,h2\\na,b\\nc,d\")\n"
        "[['c', 'd']]   # expected: [['a', 'b'], ['c', 'd']]",
    ),
    dict(
        slug="stats-stddev",
        module="spread",
        archetype="missing-step",
        split="eval",
        buggy="def stddev(xs):\n"
        '    """Population standard deviation."""\n'
        "    m = sum(xs) / len(xs)\n"
        "    return sum((x - m) ** 2 for x in xs) / len(xs)\n",
        gold="def stddev(xs):\n"
        '    """Population standard deviation."""\n'
        "    m = sum(xs) / len(xs)\n"
        "    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5\n",
        extra="def spread_range(xs):\n"
        '    """max - min of the sequence."""\n'
        "    return max(xs) - min(xs)\n",
        test="from spread import stddev\n\n\n"
        "def test_known_stddev():\n"
        "    assert stddev([1, 5]) == 2\n\n\n"
        "def test_constant_sequence():\n"
        "    assert stddev([4, 4, 4]) == 0\n",
        statement="Reported standard deviations are far too large for spread-"
        "out data — they grow with the SQUARE of the spread.\n\nRepro:\n"
        ">>> stddev([1, 5])\n4.0   # expected: 2.0",
    ),
    dict(
        slug="route-length",
        module="routes",
        archetype="off-by-one",
        split="eval",
        buggy="def route_length(stops):\n"
        '    """Number of LEGS travelled visiting stops in order."""\n'
        "    return len(stops)\n",
        gold="def route_length(stops):\n"
        '    """Number of LEGS travelled visiting stops in order."""\n'
        "    return max(len(stops) - 1, 0)\n",
        extra="def first_stop(stops):\n"
        '    """The first stop, or None for an empty route."""\n'
        "    return stops[0] if stops else None\n",
        test="from routes import route_length\n\n\n"
        "def test_three_stops_two_legs():\n"
        "    assert route_length([\"a\", \"b\", \"c\"]) == 2\n\n\n"
        "def test_empty_route():\n"
        "    assert route_length([]) == 0\n",
        statement="Billed mileage legs are one too many on every route: a "
        "3-stop route bills 3 legs.\n\nRepro:\n"
        ">>> route_length([\"a\", \"b\", \"c\"])\n3   # expected: 2",
    ),
    dict(
        slug="matrix-transpose",
        module="mat",
        archetype="swapped-indices",
        split="eval",
        buggy="def transpose(m):\n"
        '    """Transpose a rectangular row-major matrix."""\n'
        "    rows, cols = len(m), len(m[0])\n"
        "    return [[m[r][c] for c in range(cols)] for r in range(rows)]\n",
        gold="def transpose(m):\n"
        '    """Transpose a rectangular row-major matrix."""\n'
        "    rows, cols = len(m), len(m[0])\n"
        "    return [[m[r][c] for r in range(rows)] for c in range(cols)]\n",
        extra="def shape(m):\n"
        '    """(rows, cols) of a rectangular matrix."""\n'
        "    return len(m), len(m[0]) if m else 0\n",
        test="from mat import transpose\n\n\n"
        "def test_square_transpose():\n"
        "    assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n\n\n"
        "def test_rect_transpose():\n"
        "    assert transpose([[1, 2, 3]]) == [[1], [2], [3]]\n",
        statement="transpose() returns the matrix UNCHANGED.\n\nRepro:\n"
        ">>> transpose([[1, 2], [3, 4]])\n"
        "[[1, 2], [3, 4]]   # expected: [[1, 3], [2, 4]]",
    ),
    dict(
        slug="bank-withdraw",
        module="bank",
        archetype="boundary-condition",
        split="eval",
        buggy="def can_withdraw(amount, balance):\n"
        '    """True when the account covers the withdrawal exactly or more."""\n'
        "    return amount < balance\n",
        gold="def can_withdraw(amount, balance):\n"
        '    """True when the account covers the withdrawal exactly or more."""\n'
        "    return amount <= balance\n",
        extra="def fee_for(amount):\n"
        '    """Flat fee schedule."""\n'
        "    return 0.0 if amount < 1000 else 5.0\n",
        test="from bank import can_withdraw\n\n\n"
        "def test_exact_balance_allowed():\n"
        "    assert can_withdraw(100, 100) is True\n\n\n"
        "def test_overdraft_denied():\n"
        "    assert can_withdraw(101, 100) is False\n",
        statement="Customers cannot empty their own account: withdrawing the "
        "exact balance is rejected.\n\nRepro:\n"
        ">>> can_withdraw(100, 100)\nFalse   # expected: True",
    ),
    dict(
        slug="temp-c2f",
        module="thermo",
        archetype="wrong-constant",
        split="eval",
        buggy="def c_to_f(c):\n"
        '    """Celsius to Fahrenheit."""\n'
        "    return c * 5 / 9 + 32\n",
        gold="def c_to_f(c):\n"
        '    """Celsius to Fahrenheit."""\n'
        "    return c * 9 / 5 + 32\n",
        extra="def f_to_c(f):\n"
        '    """Fahrenheit to Celsius."""\n'
        "    return (f - 32) * 5 / 9\n",
        test="from thermo import c_to_f\n\n\n"
        "def test_boiling_point():\n"
        "    assert c_to_f(100) == 212\n\n\n"
        "def test_freezing_point():\n"
        "    assert c_to_f(0) == 32\n",
        statement="Celsius→Fahrenheit conversions are badly wrong at high "
        "temperatures (freezing point converts fine).\n\nRepro:\n"
        ">>> c_to_f(100)\n87.55...   # expected: 212",
    ),
    dict(
        slug="geo-circle",
        module="geo",
        archetype="wrong-formula",
        split="eval",
        buggy="import math\n\n\n"
        "def circle_area(r):\n"
        '    """Area of a circle of radius r."""\n'
        "    return 2 * math.pi * r\n",
        gold="import math\n\n\n"
        "def circle_area(r):\n"
        '    """Area of a circle of radius r."""\n'
        "    return math.pi * r * r\n",
        extra="def circle_perimeter(r):\n"
        '    """Circumference of a circle of radius r."""\n'
        "    return 2 * math.pi * r\n",
        test="import math\n\nfrom geo import circle_area\n\n\n"
        "def test_area_radius_three():\n"
        "    assert abs(circle_area(3) - 9 * math.pi) < 1e-9\n\n\n"
        "def test_area_grows_quadratically():\n"
        "    assert circle_area(2) == 4 * circle_area(1)\n",
        statement="Circle areas scale LINEARLY with radius — doubling the "
        "radius only doubles the area. circle_area(3) equals "
        "circle_perimeter(3).\n\nRepro:\n>>> circle_area(3)\n"
        "18.84...   # expected: 28.27... (9*pi)",
    ),
    dict(
        slug="words-palindrome",
        module="words",
        archetype="missing-normalize",
        split="eval",
        buggy="def is_palindrome(s):\n"
        '    """Case-insensitive palindrome check."""\n'
        "    return s == s[::-1]\n",
        gold="def is_palindrome(s):\n"
        '    """Case-insensitive palindrome check."""\n'
        "    t = s.lower()\n    return t == t[::-1]\n",
        extra="def reverse_words(s):\n"
        '    """Words in reverse order."""\n'
        "    return \" \".join(reversed(s.split()))\n",
        test="from words import is_palindrome\n\n\n"
        "def test_mixed_case_palindrome():\n"
        "    assert is_palindrome(\"Level\") is True\n\n\n"
        "def test_non_palindrome():\n"
        "    assert is_palindrome(\"world\") is False\n",
        statement="The palindrome checker claims to be case-insensitive but "
        "rejects capitalized palindromes.\n\nRepro:\n"
        ">>> is_palindrome(\"Level\")\nFalse   # expected: True",
    ),
    dict(
        slug="billing-due",
        module="billing",
        archetype="early-return",
        split="eval",
        buggy="def balance_due(total, payments):\n"
        '    """Amount still owed after partial payments (never negative)."""\n'
        "    if payments:\n        return 0.0\n"
        "    return max(total - sum(payments), 0.0)\n",
        gold="def balance_due(total, payments):\n"
        '    """Amount still owed after partial payments (never negative)."""\n'
        "    return max(total - sum(payments), 0.0)\n",
        extra="def paid_in_full(total, payments):\n"
        '    """True when nothing is owed."""\n'
        "    return balance_due(total, payments) == 0.0\n",
        test="from billing import balance_due\n\n\n"
        "def test_partial_payment_still_owes():\n"
        "    assert balance_due(100.0, [40.0]) == 60.0\n\n\n"
        "def test_no_payments_owes_total():\n"
        "    assert balance_due(100.0, []) == 100.0\n",
        statement="Any partial payment — even $1 — marks the whole invoice "
        "as settled.\n\nRepro:\n>>> balance_due(100.0, [40.0])\n"
        "0.0   # expected: 60.0",
    ),
    dict(
        slug="sku-format",
        module="sku",
        archetype="swapped-args",
        split="eval",
        buggy="def format_sku(prefix, num):\n"
        '    """SKU label: PREFIX-NNNN with the number zero-padded to 4."""\n'
        "    return f\"{num:04d}-{prefix}\"\n",
        gold="def format_sku(prefix, num):\n"
        '    """SKU label: PREFIX-NNNN with the number zero-padded to 4."""\n'
        "    return f\"{prefix}-{num:04d}\"\n",
        extra="def parse_sku(label):\n"
        '    """(prefix, num) from a PREFIX-NNNN label."""\n'
        "    prefix, _, num = label.rpartition(\"-\")\n"
        "    return prefix, int(num)\n",
        test="from sku import format_sku\n\n\n"
        "def test_prefix_first():\n"
        "    assert format_sku(\"AB\", 7) == \"AB-0007\"\n",
        statement="Printed SKU labels have the number and warehouse prefix "
        "swapped, breaking every barcode scanner.\n\nRepro:\n"
        ">>> format_sku(\"AB\", 7)\n'0007-AB'   # expected: 'AB-0007'",
    ),
    dict(
        slug="duration-parse",
        module="dur",
        archetype="ignored-component",
        split="eval",
        buggy="import re\n\n\n"
        "def parse_duration(text):\n"
        '    """Total minutes from \'2h\', \'45m\' or \'1h30m\'."""\n'
        "    h = re.search(r\"(\\d+)h\", text)\n"
        "    return int(h.group(1)) * 60 if h else 0\n",
        gold="import re\n\n\n"
        "def parse_duration(text):\n"
        '    """Total minutes from \'2h\', \'45m\' or \'1h30m\'."""\n'
        "    h = re.search(r\"(\\d+)h\", text)\n"
        "    m = re.search(r\"(\\d+)m\", text)\n"
        "    total = int(h.group(1)) * 60 if h else 0\n"
        "    return total + (int(m.group(1)) if m else 0)\n",
        extra="def format_minutes(minutes):\n"
        '    """Inverse-ish: minutes to \'XhYYm\'."""\n'
        "    return f\"{minutes // 60}h{minutes % 60:02d}m\"\n",
        test="from dur import parse_duration\n\n\n"
        "def test_hours_and_minutes():\n"
        "    assert parse_duration(\"1h30m\") == 90\n\n\n"
        "def test_minutes_only():\n"
        "    assert parse_duration(\"45m\") == 45\n",
        statement="Meeting durations lose their minutes: '1h30m' books 60 "
        "minutes and '45m' books ZERO.\n\nRepro:\n"
        ">>> parse_duration(\"1h30m\")\n60   # expected: 90\n"
        ">>> parse_duration(\"45m\")\n0    # expected: 45",
    ),
    dict(
        slug="retry-backoff",
        module="retrying",
        archetype="wrong-formula",
        split="eval",
        buggy="def backoff_delay(base, attempt):\n"
        '    """Exponential backoff: base doubles each attempt (attempt 0 = base)."""\n'
        "    return base * attempt\n",
        gold="def backoff_delay(base, attempt):\n"
        '    """Exponential backoff: base doubles each attempt (attempt 0 = base)."""\n'
        "    return base * 2 ** attempt\n",
        extra="def max_attempts(budget, base):\n"
        '    """How many attempts fit in a linear time budget (approx)."""\n'
        "    n = 0\n    spent = 0.0\n"
        "    while spent + base <= budget:\n"
        "        spent += base\n        n += 1\n"
        "    return n\n",
        test="from retrying import backoff_delay\n\n\n"
        "def test_doubles_per_attempt():\n"
        "    assert backoff_delay(1.0, 3) == 8.0\n\n\n"
        "def test_attempt_zero_is_base():\n"
        "    assert backoff_delay(1.0, 0) == 1.0\n",
        statement="Retry delays grow LINEARLY, hammering the upstream "
        "service; attempt 0 waits zero seconds.\n\nRepro:\n"
        ">>> backoff_delay(1.0, 3)\n3.0   # expected: 8.0 (1 * 2**3)",
    ),
    dict(
        slug="dedupe-unique",
        module="dedupe",
        archetype="order-lost",
        split="eval",
        buggy="def unique(xs):\n"
        '    """De-duplicate, PRESERVING first-seen order."""\n'
        "    return list(set(xs))\n",
        gold="def unique(xs):\n"
        '    """De-duplicate, PRESERVING first-seen order."""\n'
        "    seen = set()\n    out = []\n"
        "    for x in xs:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n            out.append(x)\n"
        "    return out\n",
        extra="def count_distinct(xs):\n"
        '    """Number of distinct values."""\n'
        "    return len(set(xs))\n",
        test="from dedupe import unique\n\n\n"
        "def test_first_seen_order_kept():\n"
        "    assert unique([3, 1, 3, 2]) == [3, 1, 2]\n",
        statement="De-duplicating a recently-viewed list scrambles its "
        "order.\n\nRepro:\n>>> unique([3, 1, 3, 2])\n"
        "[1, 2, 3]   # expected: [3, 1, 2]",
    ),
    dict(
        slug="urls-join",
        module="urls",
        archetype="missing-normalize",
        split="eval",
        buggy="def join_url(base, path):\n"
        '    """Join a base URL and a path with exactly one slash."""\n'
        "    return base + \"/\" + path\n",
        gold="def join_url(base, path):\n"
        '    """Join a base URL and a path with exactly one slash."""\n'
        "    return base.rstrip(\"/\") + \"/\" + path.lstrip(\"/\")\n",
        extra="def is_absolute(url):\n"
        '    """True for http(s) URLs."""\n'
        "    return url.startswith(\"http://\") or url.startswith(\"https://\")\n",
        test="from urls import join_url\n\n\n"
        "def test_trailing_slash_base():\n"
        "    assert join_url(\"http://x.com/\", \"api\") == \"http://x.com/api\"\n\n\n"
        "def test_bare_base():\n"
        "    assert join_url(\"http://x.com\", \"api\") == \"http://x.com/api\"\n",
        statement="API calls 404 whenever the configured base URL ends with "
        "a slash — requests go to a '//' path.\n\nRepro:\n"
        ">>> join_url(\"http://x.com/\", \"api\")\n"
        "'http://x.com//api'   # expected: 'http://x.com/api'",
    ),
    dict(
        slug="filters-active",
        module="filters",
        archetype="inverted-comparison",
        split="eval",
        buggy="def active_items(items):\n"
        '    """Only the items whose \'active\' flag is truthy."""\n'
        "    return [i for i in items if not i.get(\"active\")]\n",
        gold="def active_items(items):\n"
        '    """Only the items whose \'active\' flag is truthy."""\n'
        "    return [i for i in items if i.get(\"active\")]\n",
        extra="def by_name(items, name):\n"
        '    """Items matching a name."""\n'
        "    return [i for i in items if i.get(\"name\") == name]\n",
        test="from filters import active_items\n\n\n"
        "def test_keeps_only_active():\n"
        "    items = [{\"id\": 1, \"active\": True}, {\"id\": 2, \"active\": False}]\n"
        "    assert active_items(items) == [{\"id\": 1, \"active\": True}]\n",
        statement="The 'active only' view shows exactly the DISABLED items "
        "and hides the enabled ones.\n\nRepro:\n"
        ">>> active_items([{\"id\": 1, \"active\": True}, {\"id\": 2, \"active\": False}])\n"
        "[{'id': 2, 'active': False}]   # expected: [{'id': 1, 'active': True}]",
    ),
    dict(
        slug="tally-common",
        module="tally",
        archetype="inverted-comparison",
        split="eval",
        buggy="def most_common(xs):\n"
        '    """The value occurring most often (any tie-winner)."""\n'
        "    counts = {}\n"
        "    for x in xs:\n"
        "        counts[x] = counts.get(x, 0) + 1\n"
        "    return min(counts, key=counts.get)\n",
        gold="def most_common(xs):\n"
        '    """The value occurring most often (any tie-winner)."""\n'
        "    counts = {}\n"
        "    for x in xs:\n"
        "        counts[x] = counts.get(x, 0) + 1\n"
        "    return max(counts, key=counts.get)\n",
        extra="def frequency(xs, x):\n"
        '    """How many times x occurs."""\n'
        "    return sum(1 for v in xs if v == x)\n",
        test="from tally import most_common\n\n\n"
        "def test_returns_the_mode():\n"
        "    assert most_common([\"a\", \"b\", \"a\", \"a\", \"b\"]) == \"a\"\n",
        statement="'Top search term' reports the RAREST term instead of the "
        "most frequent.\n\nRepro:\n"
        ">>> most_common([\"a\", \"b\", \"a\", \"a\", \"b\"])\n"
        "'b'   # expected: 'a'",
    ),
    dict(
        slug="intervals-overlap",
        module="intervals",
        archetype="boundary-condition",
        split="eval",
        buggy="def overlaps(a, b):\n"
        '    """True when closed intervals a=(lo,hi), b=(lo,hi) share a point."""\n'
        "    return a[0] < b[1] and b[0] < a[1]\n",
        gold="def overlaps(a, b):\n"
        '    """True when closed intervals a=(lo,hi), b=(lo,hi) share a point."""\n'
        "    return a[0] <= b[1] and b[0] <= a[1]\n",
        extra="def length(a):\n"
        '    """Length of interval a."""\n'
        "    return a[1] - a[0]\n",
        test="from intervals import overlaps\n\n\n"
        "def test_touching_endpoints_overlap():\n"
        "    assert overlaps((0, 5), (5, 8)) is True\n\n\n"
        "def test_disjoint_do_not():\n"
        "    assert overlaps((0, 1), (2, 3)) is False\n",
        statement="Back-to-back bookings that share an endpoint (2pm-3pm and "
        "3pm-4pm) are treated as NON-conflicting, double-booking the room "
        "boundary.\n\nRepro:\n>>> overlaps((0, 5), (5, 8))\n"
        "False   # expected: True (closed intervals share the point 5)",
    ),
    dict(
        slug="money-cents",
        module="money",
        archetype="truncation",
        split="eval",
        buggy="def to_cents(dollars):\n"
        '    """Whole cents for a dollar amount (nearest cent)."""\n'
        "    return int(dollars * 100)\n",
        gold="def to_cents(dollars):\n"
        '    """Whole cents for a dollar amount (nearest cent)."""\n'
        "    return round(dollars * 100)\n",
        extra="def from_cents(cents):\n"
        '    """Dollar amount for whole cents."""\n'
        "    return cents / 100\n",
        test="from money import to_cents\n\n\n"
        "def test_float_edge_rounds_up():\n"
        "    assert to_cents(0.29) == 29\n\n\n"
        "def test_whole_dollars():\n"
        "    assert to_cents(3.0) == 300\n",
        statement="Some charges are exactly one cent short: $0.29 charges 28 "
        "cents.\n\nRepro:\n>>> to_cents(0.29)\n28   # expected: 29",
    ),
    dict(
        slug="semver-newer",
        module="semver",
        archetype="string-compare",
        split="eval",
        buggy="def is_newer(a, b):\n"
        '    """True when version a is strictly newer than b (\'1.10.0\' > \'1.9.0\')."""\n'
        "    return a > b\n",
        gold="def is_newer(a, b):\n"
        '    """True when version a is strictly newer than b (\'1.10.0\' > \'1.9.0\')."""\n'
        "    pa = tuple(int(x) for x in a.split(\".\"))\n"
        "    pb = tuple(int(x) for x in b.split(\".\"))\n"
        "    return pa > pb\n",
        extra="def major_of(version):\n"
        '    """The major component."""\n'
        "    return int(version.split(\".\")[0])\n",
        test="from semver import is_newer\n\n\n"
        "def test_double_digit_minor():\n"
        "    assert is_newer(\"1.10.0\", \"1.9.0\") is True\n\n\n"
        "def test_equal_versions():\n"
        "    assert is_newer(\"1.2.3\", \"1.2.3\") is False\n",
        statement="The updater refuses to upgrade from 1.9.0 to 1.10.0 — it "
        "thinks 1.10.0 is OLDER.\n\nRepro:\n"
        ">>> is_newer(\"1.10.0\", \"1.9.0\")\nFalse   # expected: True",
    ),
    dict(
        slug="md-bold",
        module="md",
        archetype="first-only",
        split="eval",
        buggy="import re\n\n\n"
        "def bold(s):\n"
        '    """Render every **span** as <b>span</b>."""\n'
        "    return re.sub(r\"\\*\\*(.+?)\\*\\*\", r\"<b>\\1</b>\", s, count=1)\n",
        gold="import re\n\n\n"
        "def bold(s):\n"
        '    """Render every **span** as <b>span</b>."""\n'
        "    return re.sub(r\"\\*\\*(.+?)\\*\\*\", r\"<b>\\1</b>\", s)\n",
        extra="def strip_markup(s):\n"
        '    """Remove <b> tags."""\n'
        "    return s.replace(\"<b>\", \"\").replace(\"</b>\", \"\")\n",
        test="from md import bold\n\n\n"
        "def test_all_spans_render():\n"
        "    assert bold(\"**a** and **b**\") == \"<b>a</b> and <b>b</b>\"\n",
        statement="Only the FIRST bold span in a document renders; the rest "
        "show raw asterisks.\n\nRepro:\n>>> bold(\"**a** and **b**\")\n"
        "'<b>a</b> and **b**'   # expected: '<b>a</b> and <b>b</b>'",
    ),
]


HELPERS_PY = '''"""Shared helpers used across the package."""


def safe_int(value, default=0):
    """int(value), or default when it does not parse."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value, lo, hi):
    """value bounded to [lo, hi]."""
    return max(lo, min(hi, value))
'''

# Generic-but-plausible scenery modules. medium samples 2, hard samples 8 —
# the buggy module hides in a wider tree, so locating it is a real search
# (the 27B solved 8/8 held-out at easy on the 2026-07-03 smoke: no dynamic
# range for a curve without this).
DISTRACTOR_POOL = {
    "validators.py": '''"""Input validation helpers."""


def non_empty(s):
    """True for a non-blank string."""
    return isinstance(s, str) and bool(s.strip())


def in_range(x, lo, hi):
    """True when lo <= x <= hi."""
    return lo <= x <= hi
''',
    "formatting.py": '''"""Display formatting."""


def humanize_bytes(n):
    """1536 -> '1.5 KB' (decimal steps)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000
    return f"{n:.1f} TB"


def ellipsize_middle(s, n):
    """Shorten long ids keeping both ends."""
    return s if len(s) <= n else s[: n // 2] + "…" + s[-(n - n // 2 - 1):]
''',
    "cachebox.py": '''"""A tiny bounded cache."""


class CacheBox:
    def __init__(self, cap=128):
        self.cap = cap
        self._d = {}

    def get(self, k, default=None):
        return self._d.get(k, default)

    def put(self, k, v):
        if len(self._d) >= self.cap:
            self._d.pop(next(iter(self._d)))
        self._d[k] = v
''',
    "textutils.py": '''"""Small text helpers."""


def indent(text, n=2):
    """Indent every line by n spaces."""
    pad = " " * n
    return "\\n".join(pad + line for line in text.splitlines())


def word_count(text):
    """Whitespace-separated word count."""
    return len(text.split())
''',
    "mathutils.py": '''"""Numeric helpers."""


def lerp(a, b, t):
    """Linear interpolation at t in [0, 1]."""
    return a + (b - a) * t


def sign(x):
    """-1, 0 or 1."""
    return (x > 0) - (x < 0)
''',
    "ids.py": '''"""Opaque id helpers."""

import hashlib


def short_id(text, n=8):
    """Stable short hex id for a string."""
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def is_hex_id(s):
    """True for a lowercase hex string."""
    return bool(s) and all(c in "0123456789abcdef" for c in s)
''',
    "timers.py": '''"""Wall-clock helpers (no scheduling)."""

import time


class Stopwatch:
    def __init__(self):
        self._start = time.monotonic()

    def elapsed(self):
        return time.monotonic() - self._start

    def restart(self):
        self._start = time.monotonic()
''',
    "sorting.py": '''"""Ordering helpers."""


def by_key(items, key, reverse=False):
    """Sort dicts by a key, missing values last."""
    return sorted(items, key=lambda d: (key not in d, d.get(key)), reverse=reverse)


def top_n(xs, n):
    """Largest n values, descending."""
    return sorted(xs, reverse=True)[:n]
''',
    "encoding.py": '''"""Serialization helpers."""

import base64
import json


def to_b64(obj):
    """JSON -> urlsafe base64."""
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()


def from_b64(s):
    """Inverse of to_b64."""
    return json.loads(base64.urlsafe_b64decode(s.encode()))
''',
    "tableview.py": '''"""Column-aligned plain-text tables."""


def render(rows, headers=None):
    """Rows (+ optional headers) as aligned text."""
    data = ([headers] if headers else []) + [[str(c) for c in r] for r in rows]
    if not data:
        return ""
    widths = [max(len(r[i]) for r in data) for i in range(len(data[0]))]
    return "\\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in data)
''',
}

DISTRACTOR_COUNT = {"easy": 0, "medium": 2, "hard": 8}


def distractors_for(task, difficulty, seed):
    k = DISTRACTOR_COUNT[difficulty]
    names = random.Random(f"{seed}:{task['slug']}").sample(
        sorted(DISTRACTOR_POOL), k
    )
    return {name: DISTRACTOR_POOL[name] for name in names}


def readme_for(task, files):
    listed = "\n".join(f"- `{f}`" for f in sorted(files) if f.endswith(".py"))
    return (
        f"# {task['slug']}\n\n"
        f"Small utility package.\n\nModules:\n{listed}\n\n"
        "Run the test suite with `python3 -m pytest`.\n"
    )


def module_source(task, fixed):
    fn = task["gold"] if fixed else task["buggy"]
    return (
        f'"""{task["slug"].replace("-", " ").title()} utilities."""\n\n\n'
        + fn
        + "\n\n"
        + task["extra"]
    )


def repo_files(task, difficulty, seed, fixed=False):
    files = {
        f"{task['module']}.py": module_source(task, fixed),
        "helpers.py": HELPERS_PY,
    }
    files.update(distractors_for(task, difficulty, seed))
    files["README.md"] = readme_for(task, files)
    return files


# ---------------------------------------------------------------- diffs ---


def new_file_diff(relpath, content):
    """git-appliable unified diff creating `relpath` with `content`."""
    lines = content.splitlines()
    body = "\n".join("+" + l for l in lines)
    return (
        f"diff --git a/{relpath} b/{relpath}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{relpath}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n" + body + "\n"
    )


def edit_diff(relpath, old, new):
    """git-appliable unified diff rewriting `relpath` from old to new."""
    hunks = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{relpath}",
        tofile=f"b/{relpath}",
    )
    return f"diff --git a/{relpath} b/{relpath}\n" + "".join(hunks)


def hidden_test_path(task):
    return f"tests/test_hidden_{task['module']}.py"


def fail_to_pass_ids(task):
    path = hidden_test_path(task)
    names = [
        line.split("(")[0].removeprefix("def ").strip()
        for line in task["test"].splitlines()
        if line.startswith("def test_")
    ]
    return [f"{path}::{name}" for name in names]


def jsonl_row(task, difficulty):
    module_file = f"{task['module']}.py"
    statement = task["statement"]
    if difficulty == "easy":
        statement += f"\n\nThe bug is somewhere in `{module_file}`."
    elif difficulty == "hard":
        # Prose symptom only — no repro snippet naming the call, so locating
        # the function is a real search through the (wider) tree.
        statement = statement.split("\n\nRepro:")[0]
    return {
        "instance_id": f"arle__{task['slug']}",
        "problem_statement": statement,
        "repo": f"arle-tasks/{task['slug']}",
        "base_commit": "synthetic-base",
        "test_patch": new_file_diff(hidden_test_path(task), task["test"]),
        "fail_to_pass": fail_to_pass_ids(task),
        "selected_test_files_to_run": [hidden_test_path(task)],
        # Extras the Rust loader ignores (corpus provenance + self-check):
        "gold_patch": edit_diff(
            module_file,
            module_source(task, fixed=False),
            module_source(task, fixed=True),
        ),
        "archetype": task["archetype"],
        "split": task["split"],
    }


# ------------------------------------------------------------ self-check ---


def run(cmd, cwd):
    # No bytecode: a stale __pycache__ (same mtime-second + size after the
    # gold patch) runs pre-patch bytecode — flaked on-pod for the two
    # byte-length-preserving bugs (stock-restock, pricing-discount).
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False, env=env
    )


def self_check_task(task, staged_dir):
    """Mirror sandbox.rs::score_workdir: base FAILS, gold-patched PASSES."""
    with tempfile.TemporaryDirectory(prefix="aopd-check-") as tmp:
        repo = Path(tmp) / "repo"
        shutil.copytree(staged_dir, repo)
        for cmd in (
            ["git", "init", "-q"],
            ["git", "add", "-A"],
            ["git", "-c", "user.email=a@b.c", "-c", "user.name=arle",
             "commit", "-qm", "base"],
        ):
            r = run(cmd, repo)
            if r.returncode != 0:
                return f"git setup failed: {r.stderr.strip()}"

        row = jsonl_row(task, "easy")
        for name, patch in (("test_patch", row["test_patch"]),
                            ("gold_patch", None)):
            if patch is not None:
                (repo / ".p.diff").write_text(patch)
                r = run(["git", "apply", ".p.diff"], repo)
                if r.returncode != 0:
                    return f"{name} does not apply: {r.stderr.strip()}"

            pytest_cmd = ["python3", "-m", "pytest", "-q",
                          "-p", "no:cacheprovider", *row["fail_to_pass"]]
            r = run(pytest_cmd, repo)
            if name == "test_patch" and r.returncode == 0:
                return "BASE tree unexpectedly PASSES the hidden tests"
            if name == "gold_patch" and r.returncode != 0:
                tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
                return "gold-patched tree FAILS: " + " | ".join(tail)

            if name == "test_patch":
                (repo / ".p.diff").write_text(row["gold_patch"])
                r = run(["git", "apply", ".p.diff"], repo)
                if r.returncode != 0:
                    return f"gold_patch does not apply: {r.stderr.strip()}"
    return None


# ------------------------------------------------------------------ main ---


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("agent_opd_tasks"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                    default="easy")
    ap.add_argument("--self-check", action="store_true",
                    help="verify base-fails / gold-passes for every task")
    args = ap.parse_args()

    slugs = [t["slug"] for t in TASKS]
    assert len(set(slugs)) == len(slugs), "duplicate slug"
    modules = [t["module"] for t in TASKS]
    assert len(set(modules)) == len(modules), "duplicate module name"

    staged_root = args.out / "staged"
    staged_root.mkdir(parents=True, exist_ok=True)

    rows = {"train": [], "eval": []}
    for task in TASKS:
        instance_dir = staged_root / f"arle__{task['slug']}"
        if instance_dir.exists():
            shutil.rmtree(instance_dir)
        for rel, content in repo_files(task, args.difficulty, args.seed).items():
            path = instance_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        rows[task["split"]].append(jsonl_row(task, args.difficulty))

    rng = random.Random(args.seed)
    for split, split_rows in rows.items():
        rng.shuffle(split_rows)
        out = args.out / f"tasks_{split}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in split_rows))
        print(f"{out}: {len(split_rows)} tasks")
    print(f"{staged_root}: {len(TASKS)} staged trees "
          f"(difficulty={args.difficulty}, seed={args.seed})")

    if args.self_check:
        failures = []
        for task in TASKS:
            err = self_check_task(task, staged_root / f"arle__{task['slug']}")
            status = "ok" if err is None else f"FAIL — {err}"
            print(f"  self-check {task['slug']}: {status}")
            if err:
                failures.append(task["slug"])
        if failures:
            print(f"SELF-CHECK FAILED for {len(failures)}: {failures}")
            return 1
        print(f"self-check: all {len(TASKS)} tasks base-FAIL / gold-PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
