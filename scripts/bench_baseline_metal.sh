#!/bin/bash
set -euo pipefail

# Baseline Metal runtime benchmark script
# Creates single-threaded baseline for comparison with multi-threaded runtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default parameters
LABEL="${1:-baseline-$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-models/Qwen3.5-0.8B}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION="${DURATION:-300}"
WARMUP="${WARMUP:-30}"

# Benchmark configuration
OUTPUT_DIR="${PROJECT_ROOT}/docs/experience/wins"
REPORT_FILE="${OUTPUT_DIR}/$(date +%Y-%m-%d)-bench-baseline-${LABEL}.md"

echo "===================================================="
echo "ARLE Metal Baseline Benchmark"
echo "===================================================="
echo "Label: $LABEL"
echo "Model: $MODEL"
echo "Concurrency: $CONCURRENCY"
echo "Duration: ${DURATION}s"
echo "Report: $REPORT_FILE"
echo "===================================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Create benchmark report header
cat > "$REPORT_FILE" << EOF
# Metal Baseline Benchmark: $LABEL

**Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Model:** $MODEL
**Concurrency:** $CONCURRENCY
**Duration:** ${DURATION}s

## Goal

Establish single-threaded Metal runtime performance baseline for comparison with multi-threaded implementation.

## Hypothesis

Single-threaded Metal runtime provides baseline performance that multi-threaded implementation should improve upon by 1.5-2x at high concurrency.

## Environment

\`\`\`
$(uname -a)
$(rustc --version)
$(system_profiler SPHardwareDataType 2>/dev/null | grep "Chip\|Memory" || echo "Hardware info unavailable")
\`\`\`

## Parameters

- **Model:** $MODEL
- **Backend:** Metal (single-threaded)
- **Concurrency:** $CONCURRENCY requests
- **Duration:** ${DURATION}s
- **Warmup:** ${WARMUP}s

## Results

EOF

# Function to run baseline benchmark
run_baseline_benchmark() {
    echo "Starting Metal baseline benchmark"

    # Start the server
    local server_log="/tmp/arle-baseline-bench.log"
    echo "Starting metal_serve with model: $MODEL"

    ${PROJECT_ROOT}/target/release/metal_serve \
        --model-path "$MODEL" \
        --port 8080 \
        --bind 0.0.0.0 \
        --warmup 1 > "$server_log" 2>&1 &

    local server_pid=$!
    echo "Server PID: $server_pid"

    # Wait for server to start
    echo "Waiting for server startup..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "Server failed to start within 30s"
            kill $server_pid 2>/dev/null || true
            return 1
        fi
        sleep 1
    done

    # Warmup phase
    echo "Warmup phase (${WARMUP}s)..."
    python -m guidellm benchmark run \
        --target http://localhost:8080 \
        --model "$MODEL" \
        --profile concurrent \
        --rate 4 \
        --max-seconds "${WARMUP}" \
        --outputs json \
        --output-path /tmp/warmup-baseline.json \
        --disable-console \
        --request-format /v1/completions > /dev/null 2>&1 || true

    # Main benchmark
    echo "Running baseline benchmark (${DURATION}s)..."
    local bench_output="/tmp/bench-baseline.json"

    python -m guidellm benchmark run \
        --target http://localhost:8080 \
        --model "$MODEL" \
        --profile concurrent \
        --rate "$CONCURRENCY" \
        --max-seconds "${DURATION}" \
        --outputs json \
        --output-path "$bench_output" \
        --random-seed 42 \
        --request-format /v1/completions

    # Collect system metrics during benchmark
    local metrics_file="/tmp/metrics-baseline.txt"
    {
        echo "=== CPU Usage ==="
        top -l 1 -n 0 | grep "CPU usage" || echo "CPU metrics unavailable"

        echo "=== Memory Usage ==="
        vm_stat | head -10 || echo "Memory metrics unavailable"

        echo "=== GPU Usage ==="
        if command -v powermetrics &> /dev/null; then
            sudo powermetrics -n 1 --samplers gpu_power,cpu_power | grep -E "GPU|CPU" | head -10 || true
        else
            echo "Metal GPU metrics require powermetrics"
        fi

        echo "=== Server Resource Usage ==="
        ps -p $server_pid -o pid,ppid,pcpu,pmem,time,comm 2>/dev/null || echo "Process not found"

    } > "$metrics_file"

    # Stop server
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    wait $server_pid 2>/dev/null || true

    # Process results
    if [ -f "$bench_output" ]; then
        echo "Benchmark completed successfully"

        # Extract key metrics
        local ttft_p50=$(jq -r '.metrics.ttft_ms.p50' "$bench_output" 2>/dev/null || echo "N/A")
        local ttft_p99=$(jq -r '.metrics.ttft_ms.p99' "$bench_output" 2>/dev/null || echo "N/A")
        local itl_p50=$(jq -r '.metrics.itl_ms.p50' "$bench_output" 2>/dev/null || echo "N/A")
        local itl_p99=$(jq -r '.metrics.itl_ms.p99' "$bench_output" 2>/dev/null || echo "N/A")
        local throughput=$(jq -r '.metrics.throughput_token_per_s' "$bench_output" 2>/dev/null || echo "N/A")
        local req_per_s=$(jq -r '.metrics.request_throughput_req_per_s' "$bench_output" 2>/dev/null || echo "N/A")

        # Append results to report
        cat >> "$REPORT_FILE" << EOF

### Metal Single-Threaded Baseline

| Metric | Value |
|--------|-------|
| TTFT p50 | ${ttft_p50} ms |
| TTFT p99 | ${ttft_p99} ms |
| ITL p50 | ${itl_p50} ms |
| ITL p99 | ${itl_p99} ms |
| Throughput | ${throughput} tok/s |
| Request Rate | ${req_per_s} req/s |

**System Metrics:**
\`\`\`
$(cat "$metrics_file")
\`\`\`

**Raw GuideLLM Output:**
\`\`\`json
$(cat "$bench_output")
\`\`\`

EOF

        # Store raw data for analysis
        cp "$bench_output" "${OUTPUT_DIR}/raw-baseline-${LABEL}.json"

        echo "Results saved: Metal baseline"
        echo "  TTFT p50: ${ttft_p50} ms"
        echo "  ITL p50: ${itl_p50} ms"
        echo "  Throughput: ${throughput} tok/s"
        echo "  Request Rate: ${req_per_s} req/s"

    else
        echo "Benchmark failed"
        cat >> "$REPORT_FILE" << EOF

### Metal Single-Threaded Baseline

❌ **Benchmark Failed**

Server log:
\`\`\`
$(tail -50 "$server_log" 2>/dev/null || echo "No server log available")
\`\`\`

EOF
    fi

    # Cleanup
    rm -f "/tmp/bench-baseline.json" "/tmp/warmup-baseline.json" "$metrics_file" "$server_log"
}

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required for JSON parsing"
    exit 1
fi

if ! python -c "import guidellm" 2>/dev/null; then
    echo "Installing GuideLLM..."
    pip install guidellm
fi

# Run baseline benchmark
run_baseline_benchmark

# Generate analysis
echo "Generating baseline analysis..."

cat >> "$REPORT_FILE" << EOF

## Analysis

### Baseline Metrics

This establishes the single-threaded Metal runtime performance baseline. Key observations:

EOF

# Extract metrics for analysis
baseline_json="${OUTPUT_DIR}/raw-baseline-${LABEL}.json"
if [ -f "$baseline_json" ]; then
    cat >> "$REPORT_FILE" << EOF

$(python3 << 'EOF_PYTHON'
import json
import os

output_dir = os.environ.get('OUTPUT_DIR', './docs/experience/wins')
label = os.environ.get('LABEL', 'test')

baseline_file = f"{output_dir}/raw-baseline-{label}.json"
if os.path.exists(baseline_file):
    try:
        with open(baseline_file, 'r') as f:
            data = json.load(f)

        throughput = float(data['metrics']['throughput_token_per_s'])
        ttft_p50 = float(data['metrics']['ttft_ms']['p50'])
        itl_p50 = float(data['metrics']['itl_ms']['p50'])
        req_rate = float(data['metrics']['request_throughput_req_per_s'])

        print(f"1. **Throughput:** {throughput:.1f} tokens/second")
        print(f"2. **Latency:** TTFT p50 = {ttft_p50:.1f}ms, ITL p50 = {itl_p50:.1f}ms")
        print(f"3. **Request Rate:** {req_rate:.1f} requests/second")
        print(f"4. **Target for Multi-threading:** ≥{throughput * 1.5:.1f} tokens/second (1.5x improvement)")

    except (KeyError, ValueError, TypeError) as e:
        print(f"Unable to analyze baseline metrics: {e}")
else:
    print("Baseline data file not found")
EOF_PYTHON
)

EOF
fi

cat >> "$REPORT_FILE" << EOF

### Next Steps

1. **Multi-threaded Implementation:** Compare against this baseline once multi-threaded runtime is ready
2. **Target Performance:** Multi-threaded runtime should achieve 1.5-2x throughput improvement
3. **Latency Maintenance:** TTFT/ITL should not regress significantly
4. **Resource Utilization:** Multi-threaded version should better utilize available CPU cores

---

*Generated by ARLE Metal baseline benchmark suite*
*Report: $(basename "$REPORT_FILE")*
EOF

echo
echo "===================================================="
echo "Baseline Benchmark Complete!"
echo "===================================================="
echo "Report: $REPORT_FILE"
echo "Raw data: ${OUTPUT_DIR}/raw-baseline-${LABEL}.json"
echo
echo "Use this baseline to compare multi-threaded runtime performance."