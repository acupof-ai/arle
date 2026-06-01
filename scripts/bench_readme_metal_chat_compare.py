#!/usr/bin/env python3
"""Generate the README Metal ARLE-vs-mlx-lm chart from one measurement path."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import psutil

MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 12288]
ARLE_CACHE_DIR = Path.home() / ".cache" / "arle" / "metal_kv"


def gib(n: float) -> float:
    return n / 1024 / 1024 / 1024


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def process_rss_gib(pid: int) -> float | None:
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                pass
        return gib(float(rss))
    except psutil.Error:
        return None


def disk_usage_gib(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return gib(float(total))


class RssSampler:
    def __init__(self, pid: int, interval_s: float) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "RssSampler":
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.sample()
        self._stop.set()
        self._thread.join(timeout=2.0)

    def sample(self) -> None:
        rss = process_rss_gib(self.pid)
        vm = psutil.virtual_memory()
        if rss is None:
            return
        self.samples.append(
            {
                "t_s": time.time(),
                "rss_gib": rss,
                "system_used_gib": gib(float(vm.used)),
                "system_available_gib": gib(float(vm.available)),
            }
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def summary(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {
                "n": 0,
                "rss_start_gib": None,
                "rss_peak_gib": None,
                "rss_end_gib": None,
                "system_used_peak_gib": None,
            }
        rss = [s["rss_gib"] for s in self.samples]
        used = [s["system_used_gib"] for s in self.samples]
        return {
            "n": len(self.samples),
            "rss_start_gib": rss[0],
            "rss_peak_gib": max(rss),
            "rss_end_gib": rss[-1],
            "system_used_peak_gib": max(used),
        }


def wait_ready(port: int, proc: subprocess.Popen[bytes], timeout_s: float) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during boot with status {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"server not ready after {timeout_s}s: {last_error!r}")


def terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def make_prompt(target_tokens: int, backend: str, repeat: int, attempt: int) -> str:
    marker = uuid.uuid4().hex
    terms = (
        "metal runtime scheduler residency prefix cache tokenizer latency "
        "throughput allocator qwen moe apple silicon measurement "
    ).split()
    body_words = max(16, int(target_tokens * 0.93) - 70)
    body = " ".join(terms[i % len(terms)] for i in range(body_words))
    return (
        f"Case marker {backend}-{target_tokens}-{repeat}-{attempt}-{marker}. "
        "Write a long technical essay from these notes. Keep expanding the "
        "analysis until the hard generation limit; do not conclude early. "
        "Use paragraphs, avoid lists, and avoid repeated single-word loops. "
        f"Notes: {body}"
    )


def chat_body(prompt: str, max_tokens: int) -> bytes:
    return json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "enable_thinking": False,
            "chat_template_kwargs": {
                "enable_thinking": False,
                "thinking": False,
            },
        }
    ).encode()


def extract_content(payload: dict[str, Any]) -> str:
    choice = payload.get("choices", [{}])[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        return delta.get("content") or delta.get("reasoning_content") or ""
    message = choice.get("message") or {}
    if isinstance(message, dict):
        return message.get("content") or ""
    return ""


def stream_chat(port: int, prompt: str, max_tokens: int) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=chat_body(prompt, max_tokens),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first: float | None = None
    token_times: list[float] = []
    text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if payload.get("usage"):
                usage = payload["usage"]
            content = extract_content(payload)
            if not content:
                continue
            t = time.time()
            if first is None:
                first = t
            token_times.append(t)
            text_parts.append(content)
    elapsed = time.time() - t0
    if first is None:
        raise RuntimeError("stream produced no content chunks")
    intervals_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0
        for i in range(1, len(token_times))
    ]
    steady = intervals_ms[1:]
    return {
        "elapsed_s": elapsed,
        "ttft_s": first - t0,
        "tpot_ms": statistics.mean(steady) if steady else None,
        "first_interval_ms": intervals_ms[0] if intervals_ms else None,
        "chunks": len(token_times),
        "usage": usage,
        "text_chars": sum(len(p) for p in text_parts),
        "first_text": "".join(text_parts)[:120],
    }


def stat(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "min": min(clean),
        "max": max(clean),
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ttft_s": stat([s["ttft_s"] for s in samples]),
        "tpot_ms": stat([s["tpot_ms"] for s in samples]),
        "chunks": stat([float(s["chunks"]) for s in samples]),
        "rss_request_peak_gib": stat([s["rss"]["rss_peak_gib"] for s in samples]),
        "rss_end_gib": stat([s["rss"]["rss_end_gib"] for s in samples]),
        "system_used_peak_gib": stat(
            [s["rss"]["system_used_peak_gib"] for s in samples]
        ),
    }


def launch_arle(port: int, log_path: Path) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.setdefault("RUST_LOG", "info")
    return subprocess.Popen(
        [
            "target/release/metal_serve",
            "--model-path",
            MODEL,
            "--port",
            str(port),
            "--max-running-requests",
            "1",
            "--max-batch-tokens",
            "4096",
            "--warmup",
            "0",
        ],
        stdout=log_path.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def launch_mlx(port: int, log_path: Path) -> subprocess.Popen[bytes]:
    py = os.environ.get("MLX_PYTHON", "/opt/homebrew/opt/python@3.11/bin/python3.11")
    if not Path(py).exists():
        py = sys.executable
    return subprocess.Popen(
        [
            py,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            MODEL,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
        ],
        stdout=log_path.open("wb"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=dict(os.environ),
    )


def run_one_request(
    port: int,
    pid: int,
    backend: str,
    target_len: int,
    repeat: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    last_sample: dict[str, Any] | None = None
    for attempt in range(1, args.max_attempts + 1):
        prompt = make_prompt(target_len, backend, repeat, attempt)
        with RssSampler(pid, args.sample_interval_s) as sampler:
            result = stream_chat(port, prompt, args.max_tokens)
        sample = {
            "repeat": repeat + 1,
            "attempt": attempt,
            **result,
            "rss": sampler.summary(),
        }
        last_sample = sample
        if sample["chunks"] >= args.min_chunks:
            return sample
        print(
            f"[{now()}] {backend} len={target_len} rep={repeat + 1}: "
            f"retry after short output chunks={sample['chunks']}",
            flush=True,
        )
    assert last_sample is not None
    return last_sample


def run_backend(args: argparse.Namespace, backend: str) -> dict[str, Any]:
    port = args.port + (0 if backend == "arle" else 1)
    log_path = Path(args.log_dir) / f"{backend}.log"
    launcher = launch_arle if backend == "arle" else launch_mlx
    print(f"[{now()}] {backend}: launch port={port}", flush=True)
    proc = launcher(port, log_path)
    try:
        wait_ready(port, proc, args.ready_timeout_s)
        ready_rss = process_rss_gib(proc.pid)
        print(f"[{now()}] {backend}: ready rss={ready_rss:.2f} GiB", flush=True)
        warm_prompt = make_prompt(128, backend, -1, 1)
        stream_chat(port, warm_prompt, args.max_tokens)
        warm_rss = process_rss_gib(proc.pid)
        print(f"[{now()}] {backend}: warm rss={warm_rss:.2f} GiB", flush=True)
        rows: dict[str, Any] = {}
        for target_len in args.lens:
            samples = []
            for rep in range(args.repeats):
                sample = run_one_request(port, proc.pid, backend, target_len, rep, args)
                samples.append(sample)
                rss = sample["rss"]
                print(
                    f"[{now()}] {backend} len={target_len:5d} rep={rep + 1}: "
                    f"ttft={sample['ttft_s']:.3f}s "
                    f"tpot={sample['tpot_ms']:.2f}ms "
                    f"rss_peak={rss['rss_peak_gib']:.2f}GiB "
                    f"rss_end={rss['rss_end_gib']:.2f}GiB "
                    f"chunks={sample['chunks']}",
                    flush=True,
                )
            rows[str(target_len)] = {"samples": samples, "summary": summarize(samples)}
        return {
            "label": "ARLE" if backend == "arle" else "mlx-lm",
            "port": port,
            "log": str(log_path),
            "ready_rss_gib": ready_rss,
            "warm_rss_gib": warm_rss,
            "final_rss_gib": process_rss_gib(proc.pid),
            "arle_cache_gib": disk_usage_gib(ARLE_CACHE_DIR) if backend == "arle" else None,
            "rows": rows,
        }
    finally:
        print(f"[{now()}] {backend}: terminate", flush=True)
        terminate(proc)


def plot_results(json_path: Path, png_path: Path, kind: str = "full") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(json_path.read_text())
    lens = [int(n) for n in data["lens"]]
    xs = list(range(len(lens)))
    def fmt_len(n: int) -> str:
        return f"{n // 1024}K" if n >= 1024 and n % 1024 == 0 else str(n)

    labels = [fmt_len(n) for n in lens]
    colors = {"arle": "#E24A0A", "mlx_lm": "#1F6FB2"}
    names = {"arle": "ARLE", "mlx_lm": "mlx-lm"}
    short_note = (
        f"n={data['shape']['repeats']}; streaming chat; "
        "steady TPOT excludes first decode interval; RSS is process high-water"
    )

    def row(series: str, n: int) -> dict[str, Any]:
        return data["series"][series]["rows"][str(n)]["summary"]

    def means(series: str, metric: str) -> list[float]:
        return [row(series, n)[metric]["mean"] for n in lens]

    def stds(series: str, metric: str) -> list[float]:
        return [row(series, n)[metric]["std"] for n in lens]

    def high_water(series: str, metric: str) -> list[float]:
        peak = 0.0
        out = []
        for n in lens:
            value = row(series, n)[metric]["max"]
            if value is not None:
                peak = max(peak, value)
            out.append(peak)
        return out

    if kind == "ttft":
        fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.4), dpi=200)
        for series in ("arle", "mlx_lm"):
            ax.errorbar(
                xs,
                means(series, "ttft_s"),
                yerr=stds(series, "ttft_s"),
                marker="o" if series == "arle" else "s",
                linewidth=2.0,
                markersize=4.0,
                capsize=3,
                color=colors[series],
                label=names[series],
            )
        ax.set_title("TTFT vs input length", fontsize=12)
        ax.set_ylabel("TTFT (seconds)", fontsize=10)
        ax.set_xlabel("Target input tokens", fontsize=10)
        ax.set_xticks(xs, labels)
        ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        fig.text(
            0.5,
            0.035,
            f"n={data['shape']['repeats']}; streaming chat; Qwen3.6 35B-A3B 4-bit",
            ha="center",
            fontsize=8,
            color="#666666",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path)
        plt.close(fig)
        return

    if kind == "tpot-rss":
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=200)
        panels = [
            ("TPOT vs input length", "TPOT (ms/token)", "tpot_ms", True, False),
            (
                "Process RSS high-water",
                "RSS high-water (GiB)",
                "rss_request_peak_gib",
                False,
                True,
            ),
        ]
        for ax, (title, ylabel, metric, yerr, cumulative) in zip(
            axes, panels, strict=True
        ):
            for series in ("arle", "mlx_lm"):
                vals = high_water(series, metric) if cumulative else means(series, metric)
                err = stds(series, metric) if yerr else None
                ax.errorbar(
                    xs,
                    vals,
                    yerr=err,
                    marker="o" if series == "arle" else "s",
                    linewidth=2.0,
                    markersize=4.0,
                    capsize=3 if yerr else 0,
                    color=colors[series],
                    label=names[series],
                )
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xlabel("Target input tokens", fontsize=9)
            ax.set_xticks(xs, labels)
            ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
            ax.tick_params(labelsize=8)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.text(
            0.5,
            0.035,
            short_note,
            ha="center",
            fontsize=8,
            color="#666666",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.25), dpi=200)
    panels = [
        ("TTFT vs input length", "TTFT (seconds)", "ttft_s", True, False),
        ("TPOT vs input length", "TPOT (ms/token)", "tpot_ms", True, False),
        (
            "Process RSS high-water",
            "RSS high-water (GiB)",
            "rss_request_peak_gib",
            False,
            True,
        ),
    ]
    for ax, (title, ylabel, metric, yerr, cumulative) in zip(
        axes, panels, strict=True
    ):
        for series in ("arle", "mlx_lm"):
            vals = high_water(series, metric) if cumulative else means(series, metric)
            err = stds(series, metric) if yerr else None
            ax.errorbar(
                xs,
                vals,
                yerr=err,
                marker="o" if series == "arle" else "s",
                linewidth=2.0,
                markersize=4.0,
                capsize=3 if yerr else 0,
                color=colors[series],
                label=names[series],
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Target input tokens", fontsize=9)
        ax.set_xticks(xs, labels)
        ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.8)
        ax.tick_params(labelsize=8)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.5, 0.035, short_note, ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/bench_readme_metal_chat_compare.json")
    parser.add_argument("--plot", default="")
    parser.add_argument("--plot-input", default="")
    parser.add_argument("--plot-kind", choices=("full", "ttft", "tpot-rss"), default="full")
    parser.add_argument("--log-dir", default="/tmp/bench_readme_metal_chat_compare_logs")
    parser.add_argument("--backends", default="arle,mlx_lm")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-chunks", type=int, default=220)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--port", type=int, default=9071)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument("--clear-arle-cache", action="store_true")
    parser.add_argument("--lens", type=int, nargs="*", default=LENS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_input:
        if not args.plot:
            raise SystemExit("--plot is required with --plot-input")
        plot_results(Path(args.plot_input), Path(args.plot), args.plot_kind)
        print(f"[{now()}] plotted -> {args.plot}", flush=True)
        return
    if args.clear_arle_cache and ARLE_CACHE_DIR.exists():
        shutil.rmtree(ARLE_CACHE_DIR)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "generated_at": now(),
        "model": MODEL,
        "shape": {
            "endpoint": "/v1/chat/completions",
            "stream": True,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "rss": "process RSS sampled during each request; chart uses cumulative high-water by target length",
            "note": f"n={args.repeats}; steady TPOT drops token1->token2 interval; RSS is cumulative process high-water; ARLE uses bounded 20 GiB SSD KV default",
        },
        "lens": args.lens,
        "series": {},
    }
    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        out["series"][backend] = run_backend(args, backend)
        Path(args.output).write_text(json.dumps(out, indent=2))
    Path(args.output).write_text(json.dumps(out, indent=2))
    if args.plot:
        plot_results(Path(args.output), Path(args.plot), args.plot_kind)
    print(f"[{now()}] done -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
