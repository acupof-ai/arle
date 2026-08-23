"""Model/backend-neutral correctness gate: needle ladder × same-config-repeat
control (generalized from dsv4_needle_gate.py per #68).

Usage: python3 needle_gate.py [lengths_csv] [runs] [depth]
  lengths_csv  approx prompt-token targets (default spans the 241 boundary)
  runs         same-config repeats per length (default 3)
  depth        needle depth 0.0=start .. 1.0=end (default 0.0)

  --check      exit 0/1: PASS if every length has >= --min-exact exact hits,
               FAIL otherwise. Standalone gate without a baseline log.
  --min-exact  minimum exact hits per length for --check (default 1).

Routing (the gate is model-neutral via the checkpoint Jinja chat template, #66):
  default      /v1/chat/completions — correct for any model, no per-model shim
  RAW=1        raw /v1/completions (+ TEMPLATE=qwen3_nonthink for Qwen3.x
               non-thinking ChatML)

Env: PORT (default 18189), MODEL (default "x"), KV_DTYPE (label only, e.g.
bf16/int8/fp8 — printed on every line so a precision-matrix run can be
grep-split; the caller restarts serve with the matching --kv-cache-dtype).

Prints one line per run with the raw decoded completion, then a per-length
summary line: exact/partial/miss counts + deterministic? (all runs identical).
"""
import os, sys, json, urllib.request, time

CHECK = "--check" in sys.argv
MIN_EXACT = 1
if CHECK:
    sys.argv.remove("--check")
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
    out = d["choices"][0]["text"]
    pt = d.get("usage", {}).get("prompt_tokens")
    return out, pt, dt


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
    out = d["choices"][0]["message"]["content"]
    pt = d.get("usage", {}).get("prompt_tokens")
    return out, pt, dt


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


exact_per_length = {}
for target in lengths:
    prompt = build_prompt(target, depth)
    outs = []
    for r in range(runs):
        try:
            out, pt, dt = one(prompt)
        except Exception as e:  # noqa: BLE001 - surface and continue the matrix
            print("len=%d depth=%.2f run=%d ERROR %r" % (target, depth, r, e))
            outs.append(None)
            continue
        outs.append(out)
        print("len=%d depth=%.2f run=%d pt=%s cls=%s wall=%.1fs kv=%s out=%r"
              % (target, depth, r, pt, classify(out), dt, KV_DTYPE, out))
    ok = [o for o in outs if o is not None]
    cls = [classify(o) for o in ok]
    n_exact = cls.count("exact")
    exact_per_length[target] = n_exact
    det = "DET" if len(set(ok)) <= 1 and len(ok) == runs else "NONDET"
    print("SUMMARY len=%d depth=%.2f exact=%d partial=%d miss=%d %s kv=%s"
          % (target, depth, n_exact, cls.count("partial"),
             cls.count("miss"), det, KV_DTYPE))
    sys.stdout.flush()

if CHECK:
    bad = [t for t in lengths if exact_per_length.get(t, 0) < MIN_EXACT]
    if bad:
        print("CHECK FAIL: lengths %s have < %d exact hits" % (bad, MIN_EXACT))
        sys.exit(1)
    print("CHECK PASS: all %d lengths have >= %d exact hits" % (len(lengths), MIN_EXACT))
    sys.exit(0)
