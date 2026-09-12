"""Model/backend-neutral correctness gate: needle ladder × same-config-repeat
control (generalized from dsv4_needle_gate.py per #68).

Usage: python3 needle_gate.py [lengths_csv] [runs] [depth]
  lengths_csv  approx prompt-token targets (default spans the 241 boundary)
  runs         same-config repeats per length (default 3)
  depth        needle depth 0.0=start .. 1.0=end (default 0.0)

  --check      standalone threshold gate (default behavior; kept for explicit
               callers): PASS (exit 0) only if every length has >= --min-exact
               exact hits; a model miss exits 1.
  --report     report-only: print summaries and exit 0 on a model miss so an
               external comparator can apply its own verdict (lever_gate.sh
               does this against a baseline envelope).
  --min-exact  minimum exact hits per length for the gate (default 1).

  Exit codes are the verdict, distinct so a caller never confuses an infra
  failure with a model miss:
    0  PASS (gate) / model miss tolerated (--report)
    1  model miss in the gate (a length under the exact-hit threshold)
    2  request ERROR in ANY mode — a run that could not fetch every
       completion (dead serve, connection refused, malformed transcript)
       is invalid and must never be read as a miss.

Routing (the gate is model-neutral via the checkpoint Jinja chat template, #66):
  default      /v1/chat/completions — correct for any model, no per-model shim
  RAW=1        raw /v1/completions (+ TEMPLATE=qwen3_nonthink for Qwen3.x
               non-thinking ChatML)

Env: PORT (default 18189), MODEL (default "x"), KV_DTYPE (label only, e.g.
bf16/int8/fp8 — printed on every line so a precision-matrix run can be
grep-split; the caller restarts serve with the matching --kv-cache-dtype).

Prints one line per run with the raw decoded completion, then a per-length
summary line: exact/partial/miss counts + deterministic? (all runs identical).
Each run also prints loc=, where the needle surfaced across content and
reasoning_content (content/both/reasoning_only/neither); reasoning_only is a
diagnostic and never changes the verdict — see
docs/plans/2026-09-12-reasoning-content-criterion.md.
"""
import os, sys, json, urllib.request, time

REPORT = "--report" in sys.argv
# Default (no --report) is the gate: exit 1 on any length under the exact-hit
# threshold or on any request error. --check is accepted as a no-op alias for
# that default (existing explicit callers). --report inverts the miss verdict
# for external comparators but request errors stay fatal (see --report).
if "--report" in sys.argv:
    sys.argv.remove("--report")
if "--check" in sys.argv:
    sys.argv.remove("--check")
MIN_EXACT = 1
for i, a in enumerate(sys.argv):
    if a == "--min-exact" and i + 1 < len(sys.argv):
        MIN_EXACT = int(sys.argv[i + 1])
        sys.argv[i : i + 2] = []
        break

BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18189")
NEEDLE = "738291"
PRE = "Important: the secret access code is " + NEEDLE + ". Keep it in mind.\n\n"
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
CUE = "\n\nRecall the secret access code stated earlier. The secret access code is"

TEMP_ARM = len(sys.argv) > 1 and sys.argv[1] == "temp"
lengths = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 and not TEMP_ARM else
           "115,180,241,300,446,1000,2000,4000,8000").split(",")]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0


def build_prompt(target, depth):
    # Unique per-sentence prefix kills exact-repetition degeneracy (a looping
    # filler is itself a degenerate prompt and confounds retrieval verdicts).
    n = max(1, target // 16)
    sents = ["Note %d: %s" % (i + 1, TOPICS[i % len(TOPICS)]) for i in range(n)]
    k = int(len(sents) * depth)
    filler_a = " ".join(sents[:k])
    filler_b = " ".join(sents[k:])
    mid = (filler_a + ("\n\n" if filler_a else "")) + PRE + filler_b
    return mid + CUE


def wrap_template(prompt):
    if os.environ.get("TEMPLATE") == "qwen3_nonthink":
        return ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return prompt


def one_completion(prompt):
    body = {"model": os.environ.get("MODEL", "x"), "prompt": wrap_template(prompt),
            "max_tokens": int(os.environ.get("NEEDLE_MAX_TOKENS", 16)), "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    dt = time.time() - t0
    # Raw completions have no reasoning channel; judge the whole text.
    return d["choices"][0]["text"], "", d.get("usage", {}).get("prompt_tokens"), dt


def one_chat(prompt):
    body = {"model": os.environ.get("MODEL", "x"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(os.environ.get("NEEDLE_MAX_TOKENS", 16)), "temperature": 0}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    # CRITERION (deliberately unchanged — see reasoning-location note below):
    # the greedy verdict reads message.content ONLY. A real API caller receives
    # content; a needle that surfaces only in reasoning_content was not returned
    # to the caller. reasoning is captured separately for diagnostics, never
    # folded into the judged text.
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content, reasoning, d.get("usage", {}).get("prompt_tokens"), dt


one = one_completion if os.environ.get("RAW") == "1" else one_chat

KV_DTYPE = os.environ.get("KV_DTYPE", "")


def glued_repeat(out):
    # Flattened-logits salad glues fragments back-to-back ("memoizatmemoizat");
    # an order-preserving distortion passes every greedy probe, so this is the
    # signature the temp arm keys on (errors/2026-07-20-hd256-fp8-temp-...).
    for k in range(5, 17):
        for i in range(len(out) - 2 * k + 1):
            frag = out[i : i + k]
            if frag == out[i + k : i + 2 * k] and frag.strip() and " " not in frag:
                return out[i : i + 2 * k]
    return None


def temp_arm():
    """temp=1.0 coherence arm: the greedy-only gate misses any distortion that
    preserves argmax ordering. One sampled generation must run long and clean."""
    want = int(os.environ.get("TEMP_TOKENS", 200))
    prompt = "Explain, in plain prose, how a hash map works and when to use one."
    body = {"model": os.environ.get("MODEL", "x"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": want, "temperature": 1.0, "seed": 7}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    msg = d["choices"][0]["message"]
    # Thinking models put text in reasoning_content with content empty — an
    # empty string would make the glued check vacuous (observed 2026-07-24).
    out = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    got = d.get("usage", {}).get("completion_tokens", 0)
    rep = glued_repeat(out)
    if not out.strip():
        print("TEMP-ARM FAIL empty output (tokens=%d)" % got)
        sys.exit(1)
    early = got < want // 2
    verdict = "FAIL" if early or rep else "PASS"
    print("TEMP-ARM %s tokens=%d/%d glued=%r out=%r" % (verdict, got, want, rep, out[:200]))
    sys.exit(1 if verdict == "FAIL" else 0)


if TEMP_ARM:
    temp_arm()


def classify(out):
    if NEEDLE in out:
        return "exact"
    if "738" in out:
        return "partial"
    return "miss"


def needle_location(content, reasoning):
    # Diagnostic label for where the needle surfaced; never enters the verdict.
    if NEEDLE in content:
        return "content" if NEEDLE not in reasoning else "both"
    return "reasoning_only" if NEEDLE in reasoning else "neither"


exact_per_length = {}
errors_per_length = {}
for target in lengths:
    prompt = build_prompt(target, depth)
    outs = []
    n_errors = 0
    n_reasoning_only = 0
    for r in range(runs):
        try:
            out, reasoning, pt, dt = one(prompt)
        except Exception as e:  # noqa: BLE001 - surface and count; fatal at the end
            print("len=%d depth=%.2f run=%d ERROR %r" % (target, depth, r, e))
            n_errors += 1
            continue
        loc = needle_location(out, reasoning)
        if loc == "reasoning_only":
            # Diagnostic only — does NOT change cls/verdict/exit. This is the
            # open thinking-model criterion question (see PR/wins note): the
            # caller-facing content missed the needle even though reasoning had
            # it, so the greedy arm still scores a miss while flagging where
            # the fact actually surfaced.
            n_reasoning_only += 1
            print("NEEDLE_REASONING_ONLY len=%d depth=%.2f run=%d "
                  "(needle in reasoning_content, content=%r)"
                  % (target, depth, r, out[:60]))
        outs.append(out)
        print("len=%d depth=%.2f run=%d pt=%s cls=%s loc=%s wall=%.1fs kv=%s out=%r"
              % (target, depth, r, pt, classify(out), loc, dt, KV_DTYPE, out))
    errors_per_length[target] = n_errors
    ok = outs
    cls = [classify(o) for o in ok]
    n_exact = cls.count("exact")
    exact_per_length[target] = n_exact
    det = "DET" if len(set(ok)) <= 1 and len(ok) == runs else "NONDET"
    print("SUMMARY len=%d depth=%.2f exact=%d partial=%d miss=%d reasoning_only=%d %s kv=%s"
          % (target, depth, n_exact, cls.count("partial"),
             cls.count("miss"), n_reasoning_only, det, KV_DTYPE))
    sys.stdout.flush()

# A request error is never a miss: the run could not retrieve, so the gate has
# no result to judge. This is fatal in every mode, including --report, so a dead
# serve or malformed transcript can't be read as "the model forgot the needle".
errored = [t for t in lengths if errors_per_length.get(t, 0) > 0]
if errored:
    print("GATE ERROR: %d request ERROR(s) at lengths %s (run failed, not a miss)"
          % (sum(errors_per_length.values()), errored))
    sys.exit(2)

if not REPORT:
    bad = [t for t in lengths if exact_per_length.get(t, 0) < MIN_EXACT]
    if bad:
        print("CHECK FAIL: lengths %s have < %d exact hits" % (bad, MIN_EXACT))
        sys.exit(1)
    print("CHECK PASS: all %d lengths have >= %d exact hits" % (len(lengths), MIN_EXACT))
sys.exit(0)
