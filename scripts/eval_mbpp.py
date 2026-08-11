#!/usr/bin/env python3
"""MBPP pass@1 evaluation for arle serve."""
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


def extract_code(response: str, expected_func: str = None) -> str:
    """Extract Python code from the model's response."""
    # Try to find code in markdown blocks
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.find("```", start)
        if end > start:
            code = response[start:end].strip()
        else:
            code = response.strip()
    elif "```" in response:
        start = response.index("```") + 3
        end = response.find("```", start)
        if end > start:
            code = response[start:end].strip()
        else:
            code = response.strip()
    else:
        code = response.strip()

    # Rename function to match expected name
    if expected_func:
        code = re.sub(r'^def\s+\w+\s*\(', f'def {expected_func}(', code, count=1)

    return code


def run_test(code: str, test_setup: str, test_list: list, timeout: float = 5.0) -> bool:
    """Run the test cases against the generated code."""
    full_code = code + "\n\n" + test_setup + "\n" + "\n".join(test_list)
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


def extract_func_name(test_list):
    """Extract function name from the first test assertion."""
    for test in test_list:
        m = re.match(r'assert\s+(\w+)\(', test)
        if m:
            return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--input", required=True, help="MBPP jsonl")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.input) as f:
        problems = [json.loads(l) for l in f if l.strip()]

    safe = [p for p in problems if not is_security(p["text"] + " " + p["code"])]
    print(f"[mbpp] {len(problems)} total, {len(safe)} safe", flush=True)

    session = requests.Session()

    def work(p):
        text = p["text"]
        test_setup = p.get("test_setup_code", "")
        test_list = p.get("test_list", [])
        func_name = extract_func_name(test_list)
        prompt = text
        if func_name:
            prompt += f"\n\nThe function must be named `{func_name}`."
        prompt += "\n\nWrite the Python function. Return only the code."
        try:
            resp = session.post(
                f"{args.base_url}/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": text + "\n\nWrite the Python function. Return only the code."}],
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                },
                timeout=300,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return p, None, str(e)

        code = extract_code(content, func_name)
        passed = run_test(code, test_setup, test_list)
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

            if i % 50 == 0 or i == len(safe):
                acc = passed / max(1, passed + failed)
                print(f"[mbpp] {i}/{len(safe)} pass@1={acc:.3f} passed={passed} failed={failed} errors={errors}", flush=True)

    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    acc = passed / max(1, passed + failed)
    print(f"[mbpp] done: pass@1={acc:.4f} ({passed}/{passed+failed}) errors={errors} in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
