#!/usr/bin/env python3
"""HumanEval pass@1 evaluation for arle serve.

Filters out security-related problems (encrypt, decrypt, md5, sha, etc.).
Uses numeric/string answer extraction from the model's completion.
"""
import argparse
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


SECURITY_PATTERNS = [
    r"\bencrypt\b", r"\bdecrypt\b", r"\bcipher\b",
    r"\bmd5\b", r"\bsha-?\d+\b", r"\bhash\b",
    r"\bpassword\b", r"\bpasswd\b",
    r"\bcrack\b", r"\bbrute[_\s]?force\b",
    r"\bsocket\b", r"\bport[_\s]?scan\b",
    r"\bbackdoor\b", r"\bexploit\b",
    r"\bmalware\b", r"\bvirus\b", r"\btrojan\b", r"\bworm\b",
    r"\bransomware\b",
    r"\bssh\b", r"\btelnet\b", r"\bftp\b",
]


def is_security(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in SECURITY_PATTERNS)


def extract_code(response: str, entry_point: str) -> str:
    """Extract the function body from the model's response."""
    # Try to find the function definition
    lines = response.split("\n")
    code_lines = []
    in_func = False
    func_indent = None

    for line in lines:
        if not in_func:
            if f"def {entry_point}" in line:
                in_func = True
                func_indent = len(line) - len(line.lstrip())
                code_lines.append(line)
            continue
        # Inside function: collect lines that are indented more than the def
        stripped = line.lstrip()
        if not stripped:
            code_lines.append("")
            continue
        indent = len(line) - len(stripped)
        if indent > func_indent:
            code_lines.append(line)
        elif stripped.startswith("def ") or stripped.startswith("class "):
            break
        else:
            break

    if not code_lines:
        # Fallback: return the whole response
        return response

    return "\n".join(code_lines)


def run_test(prompt: str, code: str, test: str, entry_point: str, timeout: float = 5.0) -> bool:
    """Run the test cases against the generated code."""
    full_code = prompt + code + "\n\n" + test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        fname = f.name
    try:
        result = subprocess.run(
            ["python3", fname],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        import os
        os.unlink(fname)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--input", required=True, help="HumanEval.jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.input) as f:
        problems = [json.loads(l) for l in f if l.strip()]

    # Filter security-related
    safe = [p for p in problems if not is_security(p["prompt"] + p["canonical_solution"])]
    print(f"[humaneval] {len(problems)} total, {len(safe)} safe (filtered {len(problems)-len(safe)} security)", flush=True)

    session = requests.Session()

    def work(p):
        prompt = p["prompt"]
        entry = p["entry_point"]
        test = p["test"]
        try:
            resp = session.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                },
                timeout=300,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return p, None, str(e)

        code = extract_code(content, entry)
        passed = run_test(prompt, code, test, entry)
        return p, code, passed

    results = []
    passed = 0
    failed = 0
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(work, p): p for p in safe}
        for i, fut in enumerate(as_completed(futures), 1):
            p, code, status = fut.result()
            if isinstance(status, bool):
                if status:
                    passed += 1
                else:
                    failed += 1
                results.append({"task_id": p["task_id"], "passed": status, "code": code})
            else:
                errors += 1
                results.append({"task_id": p["task_id"], "passed": False, "error": status})

            if i % 20 == 0 or i == len(safe):
                acc = passed / max(1, passed + failed)
                print(f"[humaneval] {i}/{len(safe)} pass@1={acc:.3f} passed={passed} failed={failed} errors={errors}", flush=True)

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    acc = passed / max(1, passed + failed)
    print(f"[humaneval] done: pass@1={acc:.4f} ({passed}/{passed+failed}) errors={errors} in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
