import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class Handler(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *_args):
        pass

    def do_GET(self):
        body = json.dumps({"scheduler": {"active_requests": 0}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        body = b"".join((
            b'data: {"choices":[{"text":"Hello "}]}\n\n',
            b'data: {"choices":[{"text":"world","finish_reason":"stop"}],',
            b'"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n',
            b"data: [DONE]\n\n",
        ))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class BenchThroughputTest(unittest.TestCase):
    def test_streaming_report(self):
        Handler.requests.clear()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "bench"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/bench_throughput.py"),
                        "--url", f"http://127.0.0.1:{server.server_port}",
                        "--model", "test",
                        "--concurrency-grid", "1",
                        "--requests-per-concurrency", "2",
                        "--max-tokens", "2",
                        "--output", str(output),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(output.with_suffix(".json").read_text())
                summary = report["points"][0]["summary"]
                self.assertEqual(report["schema"], "arle.bench_throughput.v1")
                self.assertTrue(report["config"]["ignore_eos"])
                self.assertTrue(all(request["ignore_eos"] for request in Handler.requests))
                self.assertEqual((summary["complete"], summary["error"]), (2, 0))
                self.assertEqual((summary["prompt_tokens"], summary["output_tokens"]), (2, 4))
                self.assertTrue(all(result["output_events"] == 2 for result in report["points"][0]["results"]))
                self.assertTrue(output.with_suffix(".csv").is_file())
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
