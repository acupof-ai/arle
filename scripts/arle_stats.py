#!/usr/bin/env python3
"""Shared /v1/stats and chat-message parsing helpers (#203).

Two recurring traps: probe records wrap the response as
{"status_code", "body"} so the counters live under "body", and thinking
models put text in `reasoning_content` while leaving `content` empty.
"""

import json
import urllib.request


def unwrap_body(stats):
    """Return the stats counters dict, unwrapping a {status_code, body} record."""
    if isinstance(stats, dict) and "body" in stats and "status_code" in stats:
        stats = stats["body"]
    if isinstance(stats, str):
        try:
            stats = json.loads(stats)
        except ValueError:
            return {}
    return stats if isinstance(stats, dict) else {}


def fetch_stats(port, host="127.0.0.1", timeout=5.0):
    """GET /v1/stats and return the unwrapped JSON body.

    `port` is an int, or a full base URL string such as "http://10.0.0.1:8000".
    """
    base = port if isinstance(port, str) else f"http://{host}:{port}"
    url = f"{base.rstrip('/')}/v1/stats"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return unwrap_body(json.loads(r.read().decode()))


def spec_decode(body):
    """Return the spec_decode counters dict from a stats body ({} when absent)."""
    return unwrap_body(body).get("spec_decode") or {}


def message_text(choice_message):
    """Merge reasoning_content + content of a chat message or stream delta."""
    if not isinstance(choice_message, dict):
        return ""
    return (choice_message.get("reasoning_content") or "") + (
        choice_message.get("content") or ""
    )
