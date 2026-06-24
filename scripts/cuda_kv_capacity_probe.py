#!/usr/bin/env python3
"""Push context length up until the KV pool breaks — find the max + the failure mode.

Sends increasingly long prompts with minimal decode (max_tokens=4) so the test is
dominated by prefill capacity. Reports the actual prompt_tokens the server saw, the
time, VRAM, and whether it succeeded / was rejected gracefully / crashed.
"""
import argparse, json, urllib.request, time, subprocess

FILLER = ("The grass is green and the sky is blue. Birds fly south in winter. "
          "The river flows to the sea. Mountains stand tall under the sun. ")

def gpu_mem(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
            timeout=8).decode().strip()
        return f"{out}MiB"
    except Exception:
        return "?"

def probe(base, model, approx_ctx, gpu):
    reps = max(1, int(approx_ctx / 28))   # ~28 tok per filler block
    prompt = (FILLER * reps) + "\n\nReply with just: OK"
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({"model": model,
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": 4, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
        pt = out.get("usage", {}).get("prompt_tokens", "?")
        return ("OK", time.time() - t0, f"prompt_tokens={pt}", gpu_mem(gpu))
    except urllib.error.HTTPError as e:
        body = e.read()[:120].decode("utf-8", "replace")
        return (f"REJECT({e.code})", time.time() - t0, body, gpu_mem(gpu))
    except Exception as e:
        return ("FAIL", time.time() - t0, str(e)[:100], gpu_mem(gpu))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8077")
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ctx", default="128000,256000,512000,768000,950000,1100000")
    args = ap.parse_args()
    print(f"model={args.model}  pushing context until it breaks  (idle GPU{args.gpu}={gpu_mem(args.gpu)})")
    for c in [int(x) for x in args.ctx.split(",")]:
        status, dt, info, mem = probe(args.base, args.model, c, args.gpu)
        print(f"  ctx~{c:>8}: {status:<12} ({dt:6.1f}s)  GPU={mem:<10}  {info}")
        if status not in ("OK",):
            print(f"  --> broke at ~{c}; failure mode above. Server alive after?")
            up = subprocess.run(["bash","-c",
                f"curl -s {args.base}/v1/models >/dev/null 2>&1 && echo ALIVE || echo DEAD"],
                capture_output=True, text=True).stdout.strip()
            print(f"      server: {up}")
            break

if __name__ == "__main__":
    main()
