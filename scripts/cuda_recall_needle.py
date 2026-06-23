#!/usr/bin/env python3
"""Long-memory (KV-recall) needle grid against a running arle CUDA server.

Plants a unique passkey at several depths inside a long filler context that far
exceeds the recall working set (sink+local+top_k ≈ 544 tok), asks for it, and
checks retrieval — the case-level test of whether recall actually preserves the
needle while the middle is evicted. Run with the server up on --kv-recall, and
again with it off (control / ceiling). Pair with `nvidia-smi` for flat-VRAM.

Usage (on the pod, server on 127.0.0.1:8077):
  python3 cuda_recall_needle.py --model Qwen3.6-27B-FP8 \
      --ctx 2000,8000,16000 --depths 0,0.25,0.5,0.75,1.0
"""
import argparse, json, urllib.request, re, time

FILLER = ("The grass is green and the sky is blue. Birds fly south in winter. "
          "The river flows to the sea. Mountains stand tall under the sun. ")

def build_prompt(ctx_tokens, depth, passkey):
    # ~0.75 words/token for English filler; pad to roughly ctx_tokens.
    reps = max(1, int(ctx_tokens / 12))
    body = (FILLER * reps).split()
    needle = f"REMEMBER THIS: the secret passkey is {passkey}. ".split()
    at = int(len(body) * depth)
    merged = body[:at] + needle + body[at:]
    text = " ".join(merged)
    return (f"{text}\n\nQuestion: What is the secret passkey mentioned in the "
            f"text above? Answer with only the number.")

def ask(base, model, prompt, max_tokens=24):
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({"model": model,
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8077")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", default="2000,8000,16000")
    ap.add_argument("--depths", default="0,0.25,0.5,0.75,1.0")
    args = ap.parse_args()
    ctxs = [int(x) for x in args.ctx.split(",")]
    depths = [float(x) for x in args.depths.split(",")]

    print(f"model={args.model}  ctx={ctxs}  depths={depths}")
    grid = {}
    ok = total = 0
    for ctx in ctxs:
        for d in depths:
            passkey = str(31000 + ctx % 7000 + int(d * 137))  # unique-ish per case
            prompt = build_prompt(ctx, d, passkey)
            t0 = time.time()
            try:
                ans = ask(args.base, args.model, prompt)
                hit = passkey in re.sub(r"[^0-9]", "", ans)
            except Exception as e:
                ans, hit = f"ERR {e}", False
            grid[(ctx, d)] = hit
            total += 1; ok += hit
            print(f"  ctx~{ctx:6} depth={d:<4} key={passkey} "
                  f"{'OK ' if hit else 'MISS'} ({time.time()-t0:.1f}s) out={ans[:40]!r}")
    print(f"\n=== grid (retrieved?) ===")
    print("depth\\ctx  " + "  ".join(f"{c:>6}" for c in ctxs))
    for d in depths:
        print(f"  {d:<6}  " + "  ".join(("  OK  " if grid[(c, d)] else " MISS ") for c in ctxs))
    print(f"\nretrieval: {ok}/{total} = {ok/total:.2f}")

if __name__ == "__main__":
    main()
