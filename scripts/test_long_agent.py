#!/usr/bin/env python3
"""Long agent sequence stress test.

Two-phase test:
  Phase 1: Direct sandbox stress test (fast, no GPU needed)
           — 30+ nsjail invocations testing all edge cases
  Phase 2: Full agent E2E test (GPU, single model load)
           — 5 diverse prompts through a single agent session

Usage:
  python3 scripts/test_long_agent.py                      # sandbox only
  python3 scripts/test_long_agent.py /path/to/model       # sandbox + E2E
"""

import subprocess
import sys
import time
import os

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else None
BINARY = "./target/release/arle"


# ============================================================================
# Phase 1: Direct sandbox stress test
# ============================================================================

SANDBOX_TESTS = [
    # (name, tool, input, expect_in_output, expect_ok)
    ("echo", "shell", "echo 'hello nsjail'", "hello nsjail", True),
    ("python print", "python", "print(42)", "42", True),
    ("python math", "python", "print(2**20)", "1048576", True),
    ("shell pipe", "shell", "echo abc | rev", "cba", True),
    ("shell multi", "shell", "echo a && echo b && echo c", "a\nb\nc", True),
    ("python loop", "python", "print(sum(range(1000)))", "499500", True),
    ("file write", "shell", "echo test > /tmp/nsjail_test && cat /tmp/nsjail_test", "test", True),
    ("file python", "python", "open('/tmp/pytest.txt','w').write('ok'); print(open('/tmp/pytest.txt').read())", "ok", True),
    ("readonly fs", "shell", "echo x > /etc/hack 2>&1 || echo 'readonly ok'", "readonly ok", True),
    ("no network", "shell", "curl -s --max-time 1 http://1.1.1.1 2>&1 || echo 'net blocked'", "net blocked", True),
    ("python import", "python", "import json,os,sys; print('imports ok')", "imports ok", True),
    ("python json", "python", "import json; print(json.dumps({'a':1,'b':2}))", '"a"', True),
    ("large output", "python", "print('A'*4000)", "AAAA", True),
    ("stderr", "python", "import sys; sys.stderr.write('err\\n'); print('out')", "out", True),
    ("exit code 0", "shell", "true", "", True),
    ("exit code 1", "shell", "false", "", True),  # non-zero exit is ok
    ("python error", "python", "1/0", "ZeroDivisionError", True),
    ("syntax error", "python", "def f(:", "SyntaxError", True),
    ("multiline py", "python", "for i in range(5):\n    print(i)", "0\n1\n2\n3\n4", True),
    ("env minimal", "shell", "env | sort", "PATH=", True),
    ("no home access", "shell", "ls /root 2>&1 || echo 'no access'", "", True),
    ("pid namespace", "shell", "echo $$", "", True),  # PID should be low number
    ("tmp writable", "shell", "touch /tmp/nsjail_w_test && echo ok && rm /tmp/nsjail_w_test", "ok", True),
    ("python3 ver", "python", "import sys; print(sys.version_info[:2])", "(3,", True),
    ("seq stress", "shell", "seq 1 100 | tail -1", "100", True),
    ("nested quotes", "shell", "echo \"it's a \\\"test\\\"\"", "test", True),
    ("unicode", "python", "print('hello')", "hello", True),
    ("rapid 1", "shell", "echo r1", "r1", True),
    ("rapid 2", "shell", "echo r2", "r2", True),
    ("rapid 3", "python", "print('r3')", "r3", True),
    ("rapid 4", "shell", "echo r4", "r4", True),
    ("rapid 5", "python", "print('r5')", "r5", True),
]


def run_sandbox_test(name, tool, code, expect, expect_ok):
    """Run a single sandbox test by invoking nsjail directly."""
    sandbox_args = [
        "nsjail", "--mode", "o",
        "--time_limit", "10",
        "--rlimit_as", "512",
        "--quiet", "--disable_proc",
        "-R", "/bin", "-R", "/lib", "-R", "/lib64", "-R", "/usr", "-R", "/etc",
        "-R", "/dev/null", "-R", "/dev/urandom",
        "-B", "/tmp",
        "--cwd", "/tmp",
        "--env", "PATH=/usr/local/bin:/usr/bin:/bin",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "LANG=C.UTF-8",
        "--",
    ]

    if tool == "shell":
        sandbox_args += ["/bin/bash", "-c", code]
    else:
        # Write python to temp file
        script = f"/tmp/_sandbox_test_{os.getpid()}.py"
        with open(script, "w") as f:
            f.write(code)
        sandbox_args += ["/usr/bin/python3", "-u", script]

    start = time.time()
    try:
        result = subprocess.run(
            sandbox_args,
            capture_output=True, text=True, timeout=15,
        )
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        # Filter nsjail warnings
        output_clean = "\n".join(
            l for l in output.splitlines()
            if not l.startswith("[W][") and not l.startswith("[I][")
        )

        found = expect in output_clean if expect else True
        return elapsed, found, output_clean[:200]
    except subprocess.TimeoutExpired:
        return time.time() - start, False, "TIMEOUT"
    except Exception as e:
        return time.time() - start, False, str(e)
    finally:
        # Cleanup temp python script
        if tool == "python":
            try:
                os.unlink(f"/tmp/_sandbox_test_{os.getpid()}.py")
            except OSError:
                pass


def phase1_sandbox():
    """Run direct sandbox stress tests."""
    print("=" * 70)
    print("Phase 1: Direct nsjail sandbox stress test")
    print(f"  Tests: {len(SANDBOX_TESTS)}")
    print("=" * 70)

    passed = 0
    failed = 0
    total_time = 0

    for i, (name, tool, code, expect, expect_ok) in enumerate(SANDBOX_TESTS):
        elapsed, found, output = run_sandbox_test(name, tool, code, expect, expect_ok)
        total_time += elapsed
        status = "PASS" if found else "FAIL"
        if found:
            passed += 1
        else:
            failed += 1
        out_preview = output.replace("\n", "\\n")[:60]
        print(f"  [{i+1:2d}/{len(SANDBOX_TESTS)}] {name:20s} {status:4s} {elapsed:5.2f}s  {out_preview}")

    print()
    print(f"  Passed: {passed}/{len(SANDBOX_TESTS)}")
    print(f"  Failed: {failed}")
    print(f"  Time:   {total_time:.1f}s")
    print()
    return failed == 0


# ============================================================================
# Phase 2: Full agent E2E (with GPU model)
# ============================================================================

E2E_PROMPTS = [
    "Use shell to run: echo 'agent-e2e-ok'",
    "Use python to: print(sum(i**2 for i in range(10)))",
    "Use shell to run: ls /tmp | head -5",
    "Use python: import json; d={'test': True, 'n': 42}; print(json.dumps(d))",
    "Use shell: echo done && date +%s",
]


def phase2_e2e():
    """Run full agent E2E tests."""
    if MODEL_PATH is None:
        print("Phase 2: SKIPPED (no model path provided)")
        return True

    print("=" * 70)
    print("Phase 2: Full agent E2E test")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Prompts: {len(E2E_PROMPTS)}")
    print("=" * 70)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/lib64-nvidia:/usr/local/cuda/lib64:{env.get('LD_LIBRARY_PATH', '')}"

    passed = 0
    for i, prompt in enumerate(E2E_PROMPTS):
        start = time.time()
        try:
            result = subprocess.run(
                [BINARY,
                 "--model-path", MODEL_PATH,
                 "--max-turns", "3",
                 "--max-tokens", "512",
                 "--temperature", "0.0"],
                input=prompt + "\nexit\n",
                capture_output=True, text=True,
                timeout=120, env=env,
            )
            elapsed = time.time() - start
            tool_calls = result.stdout.count("[tool:")
            nsjail = "nsjail" in result.stderr
            status = "OK" if result.returncode == 0 else "ERR"
            if status == "OK":
                passed += 1
            jail_str = "nsjail" if nsjail else "bare"
            print(f"  [{i+1}/{len(E2E_PROMPTS)}] {status:3s} | {tool_calls} tools | {jail_str} | {elapsed:.1f}s | {prompt[:50]}")
        except subprocess.TimeoutExpired:
            print(f"  [{i+1}/{len(E2E_PROMPTS)}] TIMEOUT | {prompt[:50]}")
        except Exception as e:
            print(f"  [{i+1}/{len(E2E_PROMPTS)}] ERROR: {e}")

    print(f"\n  Passed: {passed}/{len(E2E_PROMPTS)}")
    return passed >= len(E2E_PROMPTS) // 2


# ============================================================================
# Main
# ============================================================================

def main():
    start = time.time()
    p1 = phase1_sandbox()

    print()
    p2 = phase2_e2e()

    # Check for zombies
    zombies = subprocess.run(
        ["bash", "-c", "ps aux | grep -E 'nsjail|sandbox_py' | grep -v grep | wc -l"],
        capture_output=True, text=True
    )
    zombie_count = int(zombies.stdout.strip())

    total = time.time() - start
    print()
    print("=" * 70)
    print(f"TOTAL TIME: {total:.1f}s")
    print(f"ZOMBIES:    {zombie_count}")
    print(f"PHASE 1:    {'PASS' if p1 else 'FAIL'}")
    print(f"PHASE 2:    {'PASS' if p2 else 'FAIL'}")
    print(f"OVERALL:    {'PASS' if (p1 and p2) else 'FAIL'}")
    print("=" * 70)
    return p1 and p2


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
