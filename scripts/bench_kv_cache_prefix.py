#!/usr/bin/env python3
"""KV Cache Prefix Benchmark - Long Prefix (~2000+ tokens) to show real speedup."""

import subprocess
import os
import time
import httpx
import sys
import statistics

# --- Build a LONG realistic prefix (~2000-3000 tokens) ---

TOOL_DEFINITIONS = """
## Available Tools

### Tool: web_search
Description: Search the web for current information on any topic.
Parameters:
  - query (string, required): The search query to execute. Should be specific and descriptive.
  - num_results (integer, optional, default=10): Number of results to return, between 1 and 50.
  - language (string, optional, default="en"): ISO 639-1 language code for results.
  - safe_search (boolean, optional, default=true): Whether to enable safe search filtering.
  - time_range (string, optional): Filter by time range. One of: "day", "week", "month", "year".
Returns: A JSON object with fields: results (array of {title, url, snippet, date}), total_results (integer).

### Tool: read_file
Description: Read the contents of a file from the local filesystem. Supports text files, PDFs, images, and notebooks.
Parameters:
  - file_path (string, required): Absolute path to the file to read.
  - offset (integer, optional): Line number to start reading from (1-indexed).
  - limit (integer, optional): Maximum number of lines to read.
  - encoding (string, optional, default="utf-8"): File encoding to use.
Returns: The file contents as a string with line numbers prefixed.

### Tool: write_file
Description: Write content to a file on the local filesystem. Creates parent directories if needed.
Parameters:
  - file_path (string, required): Absolute path to the file to write.
  - content (string, required): The content to write to the file.
  - mode (string, optional, default="overwrite"): Write mode, one of "overwrite" or "append".
  - encoding (string, optional, default="utf-8"): File encoding to use.
  - create_dirs (boolean, optional, default=true): Whether to create parent directories.
Returns: Confirmation message with bytes written.

### Tool: execute_command
Description: Execute a shell command in the user's environment. Use for running scripts, installing packages, etc.
Parameters:
  - command (string, required): The shell command to execute.
  - working_dir (string, optional): Working directory for the command.
  - timeout (integer, optional, default=120): Timeout in seconds.
  - env (object, optional): Additional environment variables as key-value pairs.
Returns: JSON with stdout, stderr, exit_code, and duration_ms.

### Tool: code_search
Description: Search for code patterns across a repository or directory using regex.
Parameters:
  - pattern (string, required): Regular expression pattern to search for.
  - path (string, optional, default="."): Directory to search in.
  - file_glob (string, optional): Glob pattern to filter files (e.g., "*.py").
  - context_lines (integer, optional, default=2): Lines of context around matches.
  - max_results (integer, optional, default=100): Maximum results to return.
Returns: Array of matches with file, line_number, content, and context.

### Tool: create_pull_request
Description: Create a GitHub pull request from the current branch.
Parameters:
  - title (string, required): PR title, should be concise (under 72 chars).
  - body (string, required): PR description in markdown format.
  - base (string, optional, default="main"): Base branch to merge into.
  - draft (boolean, optional, default=false): Whether to create as draft PR.
  - reviewers (array of strings, optional): GitHub usernames to request review from.
  - labels (array of strings, optional): Labels to apply to the PR.
Returns: JSON with pr_number, url, and status.

### Tool: database_query
Description: Execute a read-only SQL query against the configured database.
Parameters:
  - query (string, required): SQL query to execute. Must be SELECT only.
  - database (string, optional, default="primary"): Database connection name.
  - limit (integer, optional, default=100): Maximum rows to return.
  - format (string, optional, default="table"): Output format: "table", "json", or "csv".
Returns: Query results in the specified format with row count and execution time.
"""

INSTRUCTIONS = """
## General Instructions

You are an advanced AI coding assistant with access to the tools defined above. Follow these guidelines:

1. Always analyze the user's request carefully before taking action. Break down complex tasks.
2. When modifying code, first read the existing file to understand context and style.
3. Use code_search to find relevant code before making changes across a codebase.
4. When running commands, check for errors and handle them appropriately.
5. Write clean, well-documented code that follows the project's existing conventions.
6. For database queries, always use parameterized queries to prevent SQL injection.
7. When creating PRs, write clear descriptions that explain the what, why, and how.
8. If a task requires multiple steps, plan them out and execute sequentially.
9. Always verify your changes work by running relevant tests after modifications.
10. Handle edge cases and error conditions in any code you write or modify.

## Response Format

When responding to the user:
- Be concise but thorough in explanations
- Show relevant code snippets when discussing changes
- Provide step-by-step breakdowns for complex operations
- Suggest follow-up actions when appropriate
- Flag any potential risks or concerns with proposed changes

## Code Style Guidelines

Follow these coding standards:
- Python: PEP 8 with type hints, docstrings for public functions
- JavaScript/TypeScript: ESLint standard, JSDoc comments
- Rust: rustfmt defaults, clippy clean
- Go: gofmt, effective Go guidelines
- Use meaningful variable and function names
- Keep functions focused and under 50 lines when possible
- Write unit tests for new functionality
- Document public APIs and complex algorithms

## Error Handling

When errors occur:
- Read error messages carefully and diagnose root cause
- Check logs, stack traces, and related files
- Propose fixes with explanations
- Verify fixes resolve the issue
- Document any workarounds needed

## Security Considerations

Always keep security in mind:
- Never commit secrets, API keys, or credentials
- Validate and sanitize all user inputs
- Use HTTPS for external requests
- Follow principle of least privilege
- Review dependencies for known vulnerabilities
"""

EXAMPLES = """
## Example Interactions

### Example 1: Bug Fix Workflow
User: "The login page throws a 500 error when the email contains a plus sign"
Steps:
1. Search for login-related code: code_search("login.*email|email.*valid")
2. Read the relevant handler file
3. Identify the regex or validation that fails on "+"
4. Fix the validation logic
5. Add a test case for emails with "+"
6. Run tests to verify
7. Create a PR with the fix

### Example 2: Feature Addition
User: "Add pagination to the /api/users endpoint"
Steps:
1. Read the current endpoint implementation
2. Check the database schema for the users table
3. Add page/limit query parameters
4. Implement cursor-based pagination in the query
5. Update the response format to include pagination metadata
6. Add tests for pagination edge cases
7. Update API documentation

### Example 3: Performance Investigation
User: "The dashboard page loads slowly, taking over 5 seconds"
Steps:
1. Check the dashboard endpoint for N+1 queries
2. Review database query execution plans
3. Look for missing indexes
4. Check for unnecessary data fetching
5. Implement query optimization (joins, indexes, caching)
6. Measure improvement with before/after benchmarks
"""

# Build the full prefix - system message with tools, instructions, examples
PREFIX = (
    "<|im_start|>system\n"
    "You are Claude, an advanced AI assistant created by Anthropic. You have access to a comprehensive "
    "set of tools for software development, code review, debugging, and system administration. "
    "You should use these tools proactively to help users accomplish their tasks efficiently and correctly.\n\n"
    + TOOL_DEFINITIONS
    + INSTRUCTIONS
    + EXAMPLES
    + "\n<|im_end|>\n"
    "<|im_start|>user\n"
)

# Different user queries that share the same prefix
USER_QUERIES = [
    "How do I fix a segfault in my C program that happens when processing large arrays?",
    "Write a Python function to merge two sorted linked lists efficiently.",
    "Explain the difference between TCP and UDP protocols with examples.",
    "What are the best practices for designing a REST API?",
    "How can I optimize a slow PostgreSQL query that joins three tables?",
    "Write a Rust function to parse a CSV file and return structured data.",
    "Explain how garbage collection works in Java and Go.",
    "What is the CAP theorem and how does it apply to distributed databases?",
]

# Completely different eviction prompts (short, unrelated)
EVICTION_PROMPTS = [
    "<|im_start|>system\nYou are a pirate.\n<|im_end|>\n<|im_start|>user\nTell me about treasure.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a chef.\n<|im_end|>\n<|im_start|>user\nHow do I make pasta?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a poet.\n<|im_end|>\n<|im_start|>user\nWrite a haiku about rain.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a historian.\n<|im_end|>\n<|im_start|>user\nTell me about Rome.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a musician.\n<|im_end|>\n<|im_start|>user\nExplain chord progressions.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a botanist.\n<|im_end|>\n<|im_start|>user\nHow do plants photosynthesize?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are an astronaut.\n<|im_end|>\n<|im_start|>user\nDescribe life on the ISS.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a detective.\n<|im_end|>\n<|im_start|>user\nHow do you solve a mystery?<|im_end|>\n<|im_start|>assistant\n",
]

BASE_URL = "http://localhost:8200/v1/completions"


def send_completion(client: httpx.Client, prompt: str, max_tokens: int = 32) -> tuple[str, float]:
    """Send a completion request and return (response_text, elapsed_seconds)."""
    t0 = time.perf_counter()
    resp = client.post(
        BASE_URL,
        json={
            "model": "Qwen3-8B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=120.0,
    )
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["text"] if data.get("choices") else ""
    return text, elapsed


def main():
    print("=" * 70)
    print("KV CACHE PREFIX BENCHMARK")
    print("=" * 70)

    # Estimate prefix length
    prefix_chars = len(PREFIX)
    est_tokens = prefix_chars // 4  # rough estimate
    print(f"\nPrefix length: {prefix_chars} chars (~{est_tokens} estimated tokens)")
    print(f"Number of queries: {len(USER_QUERIES)}")
    print(f"Max tokens per response: 32")
    print()

    # Start the server
    print("Starting infer server...")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64-nvidia:/usr/local/cuda/lib64"
    server = subprocess.Popen(
        [
            "./target/release/infer",
            "--model-path", "models/Qwen3-8B",
            "--port", "8200",
            "--cuda-graph=false",
        ],
        env=env,
        stdout=open("/tmp/infer_bench.log", "w"),
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    client = httpx.Client()
    print("Waiting for server to start (model loading takes ~10s)...")
    time.sleep(10)  # Model loading takes ~5-10s
    for attempt in range(30):
        time.sleep(2)
        try:
            # Use a simple completion as health check (no /v1/models endpoint)
            resp = client.post(
                BASE_URL,
                json={"model": "Qwen3-8B", "prompt": "Hi", "max_tokens": 1, "temperature": 0},
                timeout=10.0,
            )
            if resp.status_code == 200:
                print(f"Server ready after {10 + (attempt + 1) * 2}s")
                break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            pass
    else:
        print("ERROR: Server failed to start within 120s")
        server.terminate()
        sys.exit(1)

    # Warmup: send one request to warm up the model
    print("\nWarming up...")
    warmup_prompt = PREFIX + "Say hello.<|im_end|>\n<|im_start|>assistant\n"
    _, warmup_time = send_completion(client, warmup_prompt)
    print(f"Warmup complete ({warmup_time:.3f}s)")

    # ================================================================
    # MODE A: EVICTION MODE - evict KV cache between each query
    # ================================================================
    print("\n" + "=" * 70)
    print("MODE A: EVICTION (force full re-prefill each time)")
    print("=" * 70)

    eviction_times = []
    for i, query in enumerate(USER_QUERIES):
        # First, evict the cache by sending a completely different prompt
        evict_prompt = EVICTION_PROMPTS[i % len(EVICTION_PROMPTS)]
        send_completion(client, evict_prompt, max_tokens=16)

        # Now send the actual query with our long prefix
        full_prompt = PREFIX + query + "<|im_end|>\n<|im_start|>assistant\n"
        text, elapsed = send_completion(client, full_prompt)
        eviction_times.append(elapsed)
        print(f"  Query {i+1}: {elapsed*1000:7.1f}ms | {query[:60]}...")

    # ================================================================
    # MODE B: KV CACHE MODE - reuse prefix cache
    # ================================================================
    print("\n" + "=" * 70)
    print("MODE B: KV CACHE (reuse prefix, sequential queries)")
    print("=" * 70)

    # Send an eviction prompt first to clear any cached state, so the
    # first cached query starts fresh
    send_completion(client, EVICTION_PROMPTS[0], max_tokens=16)

    cache_times = []
    for i, query in enumerate(USER_QUERIES):
        full_prompt = PREFIX + query + "<|im_end|>\n<|im_start|>assistant\n"
        text, elapsed = send_completion(client, full_prompt)
        cache_times.append(elapsed)
        print(f"  Query {i+1}: {elapsed*1000:7.1f}ms | {query[:60]}...")

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    evict_total = sum(eviction_times)
    cache_total = sum(cache_times)
    evict_mean = statistics.mean(eviction_times)
    cache_mean = statistics.mean(cache_times)

    # For cache mode, first query has no cache benefit, so compare from query 2 onward
    evict_after_first = eviction_times[1:]
    cache_after_first = cache_times[1:]
    evict_mean_af = statistics.mean(evict_after_first)
    cache_mean_af = statistics.mean(cache_after_first)

    print(f"\n{'Metric':<35} {'Eviction':>12} {'KV Cache':>12} {'Speedup':>10}")
    print("-" * 70)
    print(f"{'Total time (all queries)':<35} {evict_total*1000:>10.1f}ms {cache_total*1000:>10.1f}ms {evict_total/cache_total:>9.2f}x")
    print(f"{'Mean time per query (all)':<35} {evict_mean*1000:>10.1f}ms {cache_mean*1000:>10.1f}ms {evict_mean/cache_mean:>9.2f}x")
    print(f"{'Mean time (queries 2-8, steady)':<35} {evict_mean_af*1000:>10.1f}ms {cache_mean_af*1000:>10.1f}ms {evict_mean_af/cache_mean_af:>9.2f}x")
    print(f"{'First query time':<35} {eviction_times[0]*1000:>10.1f}ms {cache_times[0]*1000:>10.1f}ms {'(cold)':>10}")

    print(f"\nPer-query comparison (queries 2-8):")
    print(f"  {'Query':<6} {'Eviction':>12} {'KV Cache':>12} {'Savings':>12} {'Speedup':>10}")
    for i in range(1, len(USER_QUERIES)):
        savings = eviction_times[i] - cache_times[i]
        speedup = eviction_times[i] / cache_times[i] if cache_times[i] > 0 else float('inf')
        print(f"  {i+1:<6} {eviction_times[i]*1000:>10.1f}ms {cache_times[i]*1000:>10.1f}ms {savings*1000:>10.1f}ms {speedup:>9.2f}x")

    print()

    # Cleanup
    print("Shutting down server...")
    server.terminate()
    server.wait(timeout=10)
    print("Done.")


if __name__ == "__main__":
    main()
