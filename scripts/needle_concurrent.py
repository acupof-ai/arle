#!/usr/bin/env python3
"""Concurrent needle gate: N in-flight requests, a DIFFERENT needle per row.

The batched linear core reaches each row's conv ring and recurrent state
through a pointer table. Its failure mode is row i advancing row j's state,
which a single-request ladder cannot see and a degeneracy check cannot see
either — every row would still emit fluent text, just the wrong secret. One
distinct needle per row makes that failure a miss.

`depth_pct` places the needle that far into the filler instead of at the front.
A front needle sits inside the KV-recall sink window (`n_init`), which is pinned
forever — so the default gate passes under `--kv-recall` without ever exercising
recall's retrieval. Gate recall at depth 50.

Usage: needle_concurrent.py [port] [concurrency] [prompt_tokens] [rounds] [depth_pct]

Exit codes carry the verdict distinctly:
  0  every row retrieved its own needle
  1  a row got a valid response that was not its needle (cross-row state mix-up
     or a retrieval miss — the batched-decode defect this gate exists for)
  2  one or more rows hit a request ERROR (dead serve, connection refused,
     malformed response). That is an invalid run, NOT a needle miss, so the
     summary reports total_error separately and infra failure is never written
     into the cross-row miss count.
"""
import json, sys, threading, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18189
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TARGET = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 0
BASE = "http://127.0.0.1:%d" % PORT

TOPICS = [
    "The river flowed gently past the old stone bridge.",
    "Mountains rose sharply against the pale morning sky.",
    "She opened the wooden door and stepped into the hall.",
    "The market was full of fruit, spices, and fresh bread.",
    "A long train crossed the wide green valley at dawn.",
    "Children played near the fountain in the city square.",
    "The library held thousands of dusty leather books.",
    "Rain fell softly on the roof throughout the night.",
]


def prompt_for(row, needle):
    n = max(4, TARGET // 16)
    # Row-unique filler: identical prefixes would let a prefix-cache hit mask a
    # state mix-up, which is the thing being tested.
    sents = ["Row %d note %d: %s" % (row, i + 1, TOPICS[i % len(TOPICS)]) for i in range(n)]
    secret = "Important: the secret access code is %s. Keep it in mind." % needle
    tail = "\n\nRecall the secret access code stated earlier. The secret access code is"
    if DEPTH <= 0:
        return secret + "\n\n" + " ".join(sents) + tail
    sents.insert(max(2, min(n - 2, n * DEPTH // 100)), secret)
    return "Read the following notes carefully.\n\n" + " ".join(sents) + tail


def ask(row, needle, out, errors):
    body = json.dumps({
        "model": "x", "prompt": prompt_for(row, needle),
        "max_tokens": 16, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        out[row] = d["choices"][0]["text"].strip()
    except Exception as e:
        # Recorded as a distinct error, not folded into a needle miss: a dead
        # serve is an invalid run, never evidence that batched decode mixed up
        # cross-row state. The row still can't contain its needle, so it is
        # excluded from the miss verdict and fails the run separately (rc 2).
        errors.add(row)
        out[row] = "ERROR: %s" % e


miss_total = 0
error_total = 0
for rnd in range(ROUNDS):
    needles = ["%06d" % (100000 + rnd * CONC + i) for i in range(CONC)]
    out = {}
    errors = set()
    ts = [threading.Thread(target=ask, args=(i, needles[i], out, errors)) for i in range(CONC)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    error_total += len(errors)
    miss = []
    for i in range(CONC):
        if i in errors:
            continue  # request failed; judged via the error path, not as a miss
        got = out.get(i, "")
        if needles[i] not in got:
            miss.append((i, needles[i], got[:40], [j for j in range(CONC)
                                                   if j != i and needles[j] in got]))
    miss_total += len(miss)
    print("round=%d conc=%d pt~%d depth=%d%% exact=%d miss=%d errors=%d"
          % (rnd, CONC, TARGET, DEPTH, CONC - len(miss) - len(errors),
             len(miss), len(errors)))
    for i in sorted(errors):
        print("  row=%d ERROR '%s'" % (i, out.get(i, "")[:60]))
    for i, want, got, cross in miss:
        tag = " CROSS_ROW=%s" % cross if cross else ""
        print("  row=%d want=%s got='%s'%s" % (i, want, got, tag))

# A request error is checked before the miss verdict and is fatal in its own
# code (2), so a transcript of a broken run can never be quoted as "N rows
# missed the needle" — the record distinguishes infra failure from a wrong
# secret. Cross-row/retrieval misses stay exit 1; clean pass exit 0.
if error_total:
    print("CONCURRENT_NEEDLE ERROR total_error=%d total_miss=%d"
          % (error_total, miss_total))
    sys.exit(2)
print("CONCURRENT_NEEDLE %s total_miss=%d total_error=0"
      % ("PASS" if miss_total == 0 else "FAIL", miss_total))
sys.exit(1 if miss_total else 0)
