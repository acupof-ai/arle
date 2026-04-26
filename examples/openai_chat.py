#!/usr/bin/env python3
"""Minimal OpenAI-SDK example against an `arle serve` instance.

    pip install openai
    python examples/openai_chat.py
"""
import os

from openai import OpenAI

base_url = os.environ.get("ARLE_BASE_URL", "http://127.0.0.1:8000") + "/v1"
model = os.environ.get("ARLE_MODEL", "qwen3-4b")

client = OpenAI(base_url=base_url, api_key="not-needed")
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Hello from ARLE"}],
    max_tokens=64,
)
print(response.choices[0].message.content)
