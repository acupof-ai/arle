#!/usr/bin/env python3
"""
Direct HTTP benchmark for ARLE Metal baseline
Makes direct HTTP requests to measure throughput and latency
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional
import sys


@dataclass
class RequestResult:
    """Result of a single completion request"""
    ttft_ms: float  # Time to first token
    total_time_ms: float  # Total request time
    tokens: int  # Number of tokens generated
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_s: float

    # Latency metrics (ms)
    ttft_p50: float
    ttft_p99: float
    ttft_mean: float

    total_time_p50: float
    total_time_p99: float
    total_time_mean: float

    # Throughput metrics
    tokens_per_second: float
    requests_per_second: float

    # Token generation metrics
    total_tokens: int
    avg_tokens_per_request: float


class DirectHttpBenchmark:
    """Direct HTTP benchmark client"""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
        timeout = aiohttp.ClientTimeout(total=300)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def single_request(self, prompt: str = "Write a short story.", max_tokens: int = 100) -> RequestResult:
        """Make a single completion request and measure latency"""
        if not self.session:
            raise RuntimeError("Session not initialized")

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "model": self.model
        }

        start_time = time.time()
        ttft_time = None
        total_tokens = 0

        try:
            async with self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        ttft_ms=0,
                        total_time_ms=(time.time() - start_time) * 1000,
                        tokens=0,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )

                result = await response.json()
                end_time = time.time()

                # Extract metrics
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0].get("text", "")
                    total_tokens = len(text.split())  # Rough token estimate

                # For streaming responses, TTFT would be time to first chunk
                # For non-streaming, we'll use total time as approximation
                ttft_ms = (end_time - start_time) * 1000
                total_time_ms = (end_time - start_time) * 1000

                return RequestResult(
                    ttft_ms=ttft_ms,
                    total_time_ms=total_time_ms,
                    tokens=total_tokens,
                    success=True
                )

        except Exception as e:
            return RequestResult(
                ttft_ms=0,
                total_time_ms=(time.time() - start_time) * 1000,
                tokens=0,
                success=False,
                error=str(e)
            )

    async def concurrent_benchmark(
        self,
        concurrency: int,
        duration_s: int,
        prompt: str = "Write a short story about artificial intelligence.",
        max_tokens: int = 100
    ) -> BenchmarkResults:
        """Run concurrent benchmark for specified duration"""

        results: List[RequestResult] = []
        start_time = time.time()
        end_time = start_time + duration_s

        async def worker():
            """Worker coroutine that makes requests until time limit"""
            while time.time() < end_time:
                result = await self.single_request(prompt, max_tokens)
                results.append(result)

        # Start concurrent workers
        print(f"Starting {concurrency} concurrent workers for {duration_s}s...")
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]

        # Wait for all workers to complete or timeout
        await asyncio.gather(*workers, return_exceptions=True)

        # Calculate metrics
        actual_duration = time.time() - start_time

        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        if not successful_results:
            raise RuntimeError(f"All requests failed. Errors: {[r.error for r in failed_results[:5]]}")

        # Extract metrics
        ttft_times = [r.ttft_ms for r in successful_results]
        total_times = [r.total_time_ms for r in successful_results]
        token_counts = [r.tokens for r in successful_results]

        total_tokens = sum(token_counts)

        return BenchmarkResults(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            total_time_s=actual_duration,

            ttft_p50=statistics.median(ttft_times) if ttft_times else 0,
            ttft_p99=statistics.quantiles(ttft_times, n=100)[98] if len(ttft_times) > 1 else ttft_times[0] if ttft_times else 0,
            ttft_mean=statistics.mean(ttft_times) if ttft_times else 0,

            total_time_p50=statistics.median(total_times) if total_times else 0,
            total_time_p99=statistics.quantiles(total_times, n=100)[98] if len(total_times) > 1 else total_times[0] if total_times else 0,
            total_time_mean=statistics.mean(total_times) if total_times else 0,

            tokens_per_second=total_tokens / actual_duration if actual_duration > 0 else 0,
            requests_per_second=len(successful_results) / actual_duration if actual_duration > 0 else 0,

            total_tokens=total_tokens,
            avg_tokens_per_request=statistics.mean(token_counts) if token_counts else 0
        )


async def main():
    parser = argparse.ArgumentParser(description="Direct HTTP benchmark for ARLE")
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--model", default="Qwen3.5-0.8B", help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup duration")

    args = parser.parse_args()

    async with DirectHttpBenchmark(args.url, args.model) as benchmark:
        # Warmup
        if args.warmup > 0:
            print(f"Warmup for {args.warmup}s...")
            try:
                await benchmark.concurrent_benchmark(2, args.warmup)
                print("Warmup completed")
            except Exception as e:
                print(f"Warmup failed: {e}")

        # Main benchmark
        print(f"Running benchmark: {args.concurrency} concurrent for {args.duration}s")
        results = await benchmark.concurrent_benchmark(
            args.concurrency,
            args.duration
        )

        print("\nBenchmark Results:")
        print(f"Total requests: {results.total_requests}")
        print(f"Successful: {results.successful_requests}")
        print(f"Failed: {results.failed_requests}")
        print(f"Duration: {results.total_time_s:.1f}s")
        print(f"TTFT p50: {results.ttft_p50:.1f}ms")
        print(f"TTFT p99: {results.ttft_p99:.1f}ms")
        print(f"Total time p50: {results.total_time_p50:.1f}ms")
        print(f"Throughput: {results.tokens_per_second:.1f} tokens/s")
        print(f"Request rate: {results.requests_per_second:.1f} req/s")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(results), f, indent=2)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())