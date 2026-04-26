"""
OpenAI API compatibility tests — no GPU required for mock-server tests.

Tests the OpenAI-compatible /v1/completions and /v1/chat/completions
API shapes using a mock HTTP server. Integration tests against a real
running infer server are skipped when no server is available.

Run:
    pytest tests/python/test_openai_api.py -v                   # mock only
    INFER_URL=http://localhost:8000 pytest tests/python/test_openai_api.py -v  # + integration
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import pytest

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

INFER_URL = os.environ.get("INFER_URL", "")
HAS_SERVER = bool(INFER_URL)

skip_without_server = pytest.mark.skipif(
    not HAS_SERVER,
    reason="INFER_URL not set — skipping integration tests",
)

DEFAULT_MODEL = os.environ.get("INFER_MODEL", "Qwen3-8B")
COMPLETIONS_PATH = "/v1/completions"
CHAT_PATH = "/v1/chat/completions"


# ---------------------------------------------------------------------------
# Mock HTTP server
# ---------------------------------------------------------------------------

class MockHandler(BaseHTTPRequestHandler):
    """Minimal mock HTTP server that validates request shape and returns canned responses."""

    def log_message(self, format, *args):
        pass  # suppress access logs during tests

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode())
        self.server.last_request = body
        self.server.last_path = self.path

        # Route to the appropriate handler
        if self.path == COMPLETIONS_PATH:
            self._handle_completions(body)
        elif self.path == CHAT_PATH:
            self._handle_chat(body)
        else:
            self._json_response(404, {"error": {"message": "Not Found", "type": "not_found"}})

    def _handle_completions(self, body: dict):
        stream = body.get("stream", False)
        if stream:
            self._sse_response([
                {"choices": [{"text": "Hello", "finish_reason": None, "index": 0}]},
                {"choices": [{"text": " world", "finish_reason": "stop", "index": 0}]},
                {"choices": [{"text": "", "finish_reason": "stop", "index": 0}]},
            ])
        else:
            self._json_response(200, {
                "id": "cmpl-test",
                "object": "text_completion",
                "created": 1700000000,
                "model": body.get("model", DEFAULT_MODEL),
                "choices": [
                    {
                        "text": "Hello world",
                        "index": 0,
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            })

    def _handle_chat(self, body: dict):
        stream = body.get("stream", False)
        if stream:
            self._sse_response([
                {"choices": [{"delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None, "index": 0}]},
                {"choices": [{"delta": {"content": " there"}, "finish_reason": "stop", "index": 0}]},
            ])
        else:
            self._json_response(200, {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": body.get("model", DEFAULT_MODEL),
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi there"},
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            })

    def _json_response(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse_response(self, events: list[dict]):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for event in events:
            line = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(line)
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def start_mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    server.last_request = None
    server.last_path = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def post_json(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(data)),
    })
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        # urllib raises HTTPError for 4xx/5xx — read body for error details
        try:
            body_bytes = e.read()
            return e.code, json.loads(body_bytes)
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 500, {"error": str(e)}


def stream_sse(url: str, body: dict) -> list[dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, method="POST", headers={
        "Content-Type": "application/json",
        "Content-Length": str(len(data)),
    })
    events = []
    with urlopen(req, timeout=10) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_server():
    server = start_mock_server()
    yield server
    server.shutdown()


def completions_url(server: HTTPServer) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{COMPLETIONS_PATH}"


def chat_url(server: HTTPServer) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{CHAT_PATH}"


# ---------------------------------------------------------------------------
# Request shape helpers
# ---------------------------------------------------------------------------

def make_completions_request(
    prompt="The capital of France is",
    model=DEFAULT_MODEL,
    max_tokens=32,
    **kwargs,
) -> dict:
    return {"model": model, "prompt": prompt, "max_tokens": max_tokens, **kwargs}


def make_chat_request(
    messages=None,
    model=DEFAULT_MODEL,
    max_tokens=32,
    **kwargs,
) -> dict:
    if messages is None:
        messages = [{"role": "user", "content": "Hello"}]
    return {"model": model, "messages": messages, "max_tokens": max_tokens, **kwargs}


# ---------------------------------------------------------------------------
# Tests: /v1/completions — response shape
# ---------------------------------------------------------------------------

class TestCompletionsResponseShape:

    def test_basic_completion(self, mock_server):
        status, resp = post_json(completions_url(mock_server), make_completions_request())
        assert status == 200
        assert resp["object"] == "text_completion"

    def test_response_has_id(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        assert "id" in resp
        assert isinstance(resp["id"], str)
        assert len(resp["id"]) > 0

    def test_response_has_created(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        assert "created" in resp
        assert isinstance(resp["created"], int)

    def test_response_has_model(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        assert "model" in resp
        assert resp["model"] == DEFAULT_MODEL

    def test_response_has_choices(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        assert "choices" in resp
        assert isinstance(resp["choices"], list)
        assert len(resp["choices"]) >= 1

    def test_choice_has_text(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        choice = resp["choices"][0]
        assert "text" in choice
        assert isinstance(choice["text"], str)

    def test_choice_has_finish_reason(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        choice = resp["choices"][0]
        assert "finish_reason" in choice
        assert choice["finish_reason"] in ("stop", "length", None)

    def test_choice_has_index(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        assert resp["choices"][0]["index"] == 0

    def test_response_has_usage(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        assert "usage" in resp
        usage = resp["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_usage_total_tokens(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        u = resp["usage"]
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_usage_tokens_positive(self, mock_server):
        _, resp = post_json(completions_url(mock_server), make_completions_request())
        u = resp["usage"]
        assert u["prompt_tokens"] > 0
        assert u["completion_tokens"] > 0
        assert u["total_tokens"] > 0

    def test_model_echoed_in_response(self, mock_server):
        """Model field in response should match request."""
        req = make_completions_request(model=DEFAULT_MODEL)
        _, resp = post_json(completions_url(mock_server), req)
        assert resp["model"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Tests: /v1/completions — request fields forwarded
# ---------------------------------------------------------------------------

class TestCompletionsRequestFields:

    def test_prompt_required_field(self, mock_server):
        req = make_completions_request(prompt="Test prompt")
        _, _ = post_json(completions_url(mock_server), req)
        last = mock_server.last_request
        assert last["prompt"] == "Test prompt"

    def test_temperature_forwarded(self, mock_server):
        req = make_completions_request(temperature=0.7)
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["temperature"] == 0.7

    def test_top_p_forwarded(self, mock_server):
        req = make_completions_request(top_p=0.9)
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["top_p"] == 0.9

    def test_max_tokens_forwarded(self, mock_server):
        req = make_completions_request(max_tokens=128)
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["max_tokens"] == 128

    def test_seed_forwarded(self, mock_server):
        req = make_completions_request(seed=12345)
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["seed"] == 12345

    def test_stop_string_forwarded(self, mock_server):
        req = make_completions_request(stop="<|end|>")
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["stop"] == "<|end|>"

    def test_stop_list_forwarded(self, mock_server):
        req = make_completions_request(stop=["<|end|>", "\n\n"])
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["stop"] == ["<|end|>", "\n\n"]

    def test_frequency_penalty_forwarded(self, mock_server):
        req = make_completions_request(frequency_penalty=0.5)
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["frequency_penalty"] == 0.5

    def test_presence_penalty_forwarded(self, mock_server):
        req = make_completions_request(presence_penalty=0.3)
        _, _ = post_json(completions_url(mock_server), req)
        assert mock_server.last_request["presence_penalty"] == 0.3


# ---------------------------------------------------------------------------
# Tests: /v1/completions — streaming
# ---------------------------------------------------------------------------

class TestCompletionsStreaming:

    def test_streaming_returns_events(self, mock_server):
        req = make_completions_request(stream=True)
        events = stream_sse(completions_url(mock_server), req)
        assert len(events) > 0

    def test_each_event_has_choices(self, mock_server):
        req = make_completions_request(stream=True)
        events = stream_sse(completions_url(mock_server), req)
        for event in events:
            assert "choices" in event
            assert isinstance(event["choices"], list)

    def test_event_choices_have_text(self, mock_server):
        req = make_completions_request(stream=True)
        events = stream_sse(completions_url(mock_server), req)
        for event in events:
            choice = event["choices"][0]
            assert "text" in choice

    def test_streamed_text_concatenates(self, mock_server):
        req = make_completions_request(stream=True)
        events = stream_sse(completions_url(mock_server), req)
        full_text = "".join(e["choices"][0]["text"] for e in events)
        assert len(full_text) > 0

    def test_last_event_has_finish_reason_stop(self, mock_server):
        req = make_completions_request(stream=True)
        events = stream_sse(completions_url(mock_server), req)
        last = events[-1]
        assert last["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# Tests: /v1/chat/completions — response shape
# ---------------------------------------------------------------------------

class TestChatCompletionsResponseShape:

    def test_basic_chat(self, mock_server):
        status, resp = post_json(chat_url(mock_server), make_chat_request())
        assert status == 200
        assert resp["object"] == "chat.completion"

    def test_response_has_choices(self, mock_server):
        _, resp = post_json(chat_url(mock_server), make_chat_request())
        assert "choices" in resp
        assert len(resp["choices"]) >= 1

    def test_choice_has_message(self, mock_server):
        _, resp = post_json(chat_url(mock_server), make_chat_request())
        choice = resp["choices"][0]
        assert "message" in choice
        msg = choice["message"]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], str)

    def test_choice_has_finish_reason(self, mock_server):
        _, resp = post_json(chat_url(mock_server), make_chat_request())
        assert resp["choices"][0]["finish_reason"] in ("stop", "length", None)

    def test_response_has_usage(self, mock_server):
        _, resp = post_json(chat_url(mock_server), make_chat_request())
        u = resp["usage"]
        assert u["prompt_tokens"] > 0
        assert u["completion_tokens"] > 0

    def test_chat_streaming(self, mock_server):
        req = make_chat_request(stream=True)
        events = stream_sse(chat_url(mock_server), req)
        assert len(events) > 0
        for event in events:
            assert "choices" in event


# ---------------------------------------------------------------------------
# Tests: Request body validation (expected shapes)
# ---------------------------------------------------------------------------

class TestRequestBodyShapes:

    def test_completions_path(self, mock_server):
        post_json(completions_url(mock_server), make_completions_request())
        assert mock_server.last_path == COMPLETIONS_PATH

    def test_chat_path(self, mock_server):
        post_json(chat_url(mock_server), make_chat_request())
        assert mock_server.last_path == CHAT_PATH

    def test_sampling_params_structure(self, mock_server):
        """All standard sampling params should be accepted."""
        req = make_completions_request(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            min_p=0.05,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            seed=42,
        )
        status, resp = post_json(completions_url(mock_server), req)
        assert status == 200

    def test_chat_messages_structure(self, mock_server):
        """Chat API accepts system + user messages."""
        req = make_chat_request(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ])
        status, _ = post_json(chat_url(mock_server), req)
        assert status == 200


# ---------------------------------------------------------------------------
# Tests: error cases
# ---------------------------------------------------------------------------

class TestErrorCases:

    def test_unknown_path_returns_404(self, mock_server):
        host, port = mock_server.server_address
        url = f"http://{host}:{port}/v1/unknown"
        status, resp = post_json(url, {"prompt": "test"})
        assert status == 404
        assert "error" in resp

    def test_completions_body_passed_as_json(self, mock_server):
        """Even unusual extra fields should be forwarded without crashing mock."""
        req = make_completions_request(unknown_field="foo")
        status, _ = post_json(completions_url(mock_server), req)
        assert status == 200


# ---------------------------------------------------------------------------
# Integration tests (requires running infer server)
# ---------------------------------------------------------------------------

@skip_without_server
class TestIntegrationCompletions:

    @pytest.fixture(autouse=True)
    def server_url(self):
        self.base_url = INFER_URL.rstrip("/")

    def test_simple_completion(self):
        req = make_completions_request(
            prompt="The capital of France is",
            max_tokens=5,
            temperature=0.0,
        )
        status, resp = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        assert status == 200
        text = resp["choices"][0]["text"]
        assert len(text) > 0

    def test_greedy_determinism(self):
        """Same seed + temperature=0 should give same output."""
        req = make_completions_request(
            prompt="The sky is",
            max_tokens=10,
            temperature=0.0,
            seed=42,
        )
        _, resp1 = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        _, resp2 = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        assert resp1["choices"][0]["text"] == resp2["choices"][0]["text"]

    def test_stop_sequence_respected(self):
        req = make_completions_request(
            prompt="Repeat: ABC ABC ABC",
            max_tokens=50,
            temperature=0.0,
            stop=["ABC"],
        )
        _, resp = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        text = resp["choices"][0]["text"]
        assert "ABC" not in text

    def test_max_tokens_limit(self):
        req = make_completions_request(
            prompt="Count to one hundred: 1, 2, 3,",
            max_tokens=5,
            temperature=0.0,
        )
        _, resp = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        tokens = resp["usage"]["completion_tokens"]
        assert tokens <= 5

    def test_usage_counts(self):
        req = make_completions_request(
            prompt="Hello world",
            max_tokens=10,
            temperature=0.0,
        )
        _, resp = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        u = resp["usage"]
        assert u["prompt_tokens"] > 0
        assert u["completion_tokens"] > 0
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]

    def test_streaming_completions(self):
        req = make_completions_request(
            prompt="The sky is",
            max_tokens=10,
            temperature=0.0,
            stream=True,
        )
        events = stream_sse(f"{self.base_url}{COMPLETIONS_PATH}", req)
        assert len(events) > 0
        full_text = "".join(e["choices"][0]["text"] for e in events)
        assert len(full_text) > 0

    def test_high_temperature_produces_output(self):
        req = make_completions_request(
            prompt="Tell me a joke:",
            max_tokens=20,
            temperature=1.0,
            seed=0,
        )
        status, resp = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
        assert status == 200
        assert len(resp["choices"][0]["text"]) > 0


@skip_without_server
class TestIntegrationConcurrency:

    @pytest.fixture(autouse=True)
    def server_url(self):
        self.base_url = INFER_URL.rstrip("/")

    def test_two_concurrent_requests(self):
        import threading

        results = {}

        def run(i):
            req = make_completions_request(
                prompt=f"Request {i}:",
                max_tokens=5,
                temperature=0.0,
            )
            status, resp = post_json(f"{self.base_url}{COMPLETIONS_PATH}", req)
            results[i] = (status, resp)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, (status, resp) in results.items():
            assert status == 200, f"Request {i} failed: {resp}"
            assert len(resp["choices"][0]["text"]) > 0
