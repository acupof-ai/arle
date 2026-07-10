"""Regression gate for the GuideLLM /v1/stats evidence chain.

Run with: python -m unittest scripts.tests.test_bench_guidellm_stats
"""

import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "scripts" / "bench_guidellm.sh"


def shell_function(source: str, name: str) -> str:
    lines = source.splitlines()
    start = lines.index(f"{name}() {{")
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    return "\n".join(lines[start : end + 1])


def summary_program(source: str) -> str:
    marker = 'python3 - "$trace_file" "$before_file" "$after_file" "$summary_file" "$interval_ms" <<\'PY\''
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == marker) + 1
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "PY")
    return "\n".join(lines[start:end])


class GuideLlmStatsEvidence(unittest.TestCase):
    def test_nested_json_deltas_and_invalid_json_fail_closed(self):
        self.assertIsNotNone(shutil.which("jq"), "bench_guidellm.sh requires jq")
        source = BENCH_SCRIPT.read_text()
        emit = shell_function(source, "emit_service_stats_record")

        before = self.stats(hits=1, fallback=0)
        during = self.stats(hits=3, fallback=0)
        after = self.stats(hits=5, fallback=1)
        trace_line = self.emit(emit, during)
        trace = json.loads(trace_line)
        self.assertIsInstance(trace["stats"], dict)

        invalid = json.loads(self.emit(emit, "not-json"))
        self.assertFalse(invalid["ok"])
        self.assertNotIn("stats", invalid)
        self.assertIn("invalid /v1/stats JSON", invalid["error"])

        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            trace_path = directory / "trace.jsonl"
            before_path = directory / "before.json"
            after_path = directory / "after.json"
            summary_path = directory / "summary.md"
            trace_path.write_text(trace_line)
            before_path.write_text(before)
            after_path.write_text(after)
            run = subprocess.run(
                [
                    "python3",
                    "-",
                    str(trace_path),
                    str(before_path),
                    str(after_path),
                    str(summary_path),
                    "1000",
                ],
                input=summary_program(source),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = summary_path.read_text()

        self.assertIn("Operator policy hash: `policy-1`", summary)
        self.assertIn("Operator fallback count: `1` (delta: `1`)", summary)
        self.assertIn("| `cuda.test` | 5 | 4 |", summary)

    @staticmethod
    def emit(function: str, stats: str) -> str:
        run = subprocess.run(
            ["bash", "-c", f'{function}\nemit_service_stats_record "$TS" 0 "$STATS"'],
            env={"TS": "2026-07-10T00:00:00Z", "STATS": stats},
            text=True,
            capture_output=True,
            check=False,
        )
        if run.returncode != 0:
            raise AssertionError(run.stderr)
        return run.stdout

    @staticmethod
    def stats(hits: int, fallback: int) -> str:
        return json.dumps(
            {
                "scheduler": {"active_requests": 1, "queue_depth": 0, "kv_free_pages": 8},
                "throughput": {
                    "steps": hits,
                    "prefill_tokens": 4,
                    "generated_tokens": 2,
                    "requests_completed": 1,
                },
                "prefix_cache": {"hit_rate": 0.5},
                "operator_dispatch": {
                    "policy_hash": "policy-1",
                    "product_id": "arle-1",
                    "bundle_digest": "sha256:abc",
                    "implementation_hits": [
                        {"implementation_id": "cuda.test", "hits": hits}
                    ],
                    "fallback_count": fallback,
                },
            }
        )


if __name__ == "__main__":
    unittest.main()
