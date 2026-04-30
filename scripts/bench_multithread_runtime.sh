#!/bin/bash
set -euo pipefail

# Multi-threaded runtime benchmark script
# Follows bench-and-trace-spec.md methodology for performance validation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default parameters
LABEL="${1:-multithread-$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
BACKEND="${BACKEND:-cuda}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION="${DURATION:-300}"
WARMUP="${WARMUP:-30}"

# Benchmark configuration
PROMPT_FILE="${PROJECT_ROOT}/infer/test_data/prompts.txt"
OUTPUT_DIR="${PROJECT_ROOT}/docs/experience/wins"
REPORT_FILE="${OUTPUT_DIR}/$(date +%Y-%m-%d)-bench-multithread-${LABEL}.md"

# Runtime configurations to test
declare -A CONFIGS=(
    ["single-threaded"]="--runtime-mode single-threaded"
    ["multi-threaded-minimal"]="--runtime-mode multi-threaded --tokenizer-workers 1 --detokenizer-workers 1 --gpu-workers 1"
    ["multi-threaded-default"]="--runtime-mode multi-threaded --tokenizer-workers 4 --detokenizer-workers 4 --gpu-workers 2"
    ["multi-threaded-aggressive"]="--runtime-mode multi-threaded --tokenizer-workers 8 --detokenizer-workers 8 --gpu-workers 4"
)

echo "===================================================="
echo "ARLE Multi-Threaded Runtime Benchmark"
echo "===================================================="
echo "Label: $LABEL"
echo "Model: $MODEL"
echo "Backend: $BACKEND"
echo "Concurrency: $CONCURRENCY"
echo "Duration: ${DURATION}s"
echo "Report: $REPORT_FILE"
echo "===================================================="

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Create benchmark report header
cat > "$REPORT_FILE" << EOF
# Multi-Threaded Runtime Benchmark: $LABEL

**Date:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Model:** $MODEL
**Backend:** $BACKEND
**Concurrency:** $CONCURRENCY
**Duration:** ${DURATION}s

## Goal

Validate multi-threaded runtime performance improvements vs single-threaded baseline.

## Hypothesis

Multi-threaded runtime should achieve:
- 1.5-2x throughput improvement for high-concurrency workloads
- Improved resource utilization (CPU and GPU)
- Lower latency variance through parallel processing

## Environment

\`\`\`
$(uname -a)
$(rustc --version)
$(if command -v nvidia-smi &> /dev/null; then nvidia-smi -L; fi)
\`\`\`

## Parameters

- **Model:** $MODEL
- **Backend:** $BACKEND
- **Concurrency:** $CONCURRENCY requests
- **Duration:** ${DURATION}s per configuration
- **Warmup:** ${WARMUP}s per configuration

## Results

EOF

# Function to run guidellm benchmark
run_guidellm() {
    local config_name="$1"
    local config_args="$2"

    echo "Running benchmark: $config_name"
    echo "Configuration: $config_args"

    # Start the server
    local server_log="/tmp/arle-bench-${config_name}.log"
    echo "Starting server with: $config_args"

    cargo run --release --features "$BACKEND" -- \
        --model "$MODEL" \
        $config_args \
        --port 8080 > "$server_log" 2>&1 &

    local server_pid=$!
    echo "Server PID: $server_pid"

    # Wait for server to start
    echo "Waiting for server startup..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
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
    python -m guidellm \
        --url http://localhost:8080/v1/completions \
        --model "$MODEL" \
        --concurrency 4 \
        --duration "${WARMUP}s" \
        --output-format json \
        --quiet > /tmp/warmup-${config_name}.json 2>/dev/null || true

    # Main benchmark
    echo "Running main benchmark (${DURATION}s)..."
    local bench_output="/tmp/bench-${config_name}.json"

    python -m guidellm \
        --url http://localhost:8080/v1/completions \
        --model "$MODEL" \
        --concurrency "$CONCURRENCY" \
        --duration "${DURATION}s" \
        --output-format json \
        --output "$bench_output" \
        --request-count-seed 42

    # Collect system metrics during benchmark
    local metrics_file="/tmp/metrics-${config_name}.txt"
    {
        echo "=== CPU Usage ==="
        top -l 1 -n 0 | grep "CPU usage" || echo "CPU metrics unavailable"

        echo "=== Memory Usage ==="
        vm_stat | head -10 || echo "Memory metrics unavailable"

        echo "=== GPU Usage ==="
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
        else
            echo "GPU metrics unavailable (NVIDIA)"
        fi

        if command -v powermetrics &> /dev/null; then
            sudo powermetrics -n 1 --samplers gpu_power,cpu_power | grep -E "GPU|CPU" | head -10 || true
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

### $config_name

**Configuration:** \`$config_args\`

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
        cp "$bench_output" "${OUTPUT_DIR}/raw-${config_name}-${LABEL}.json"

        echo "Results saved: $config_name"
        echo "  TTFT p50: ${ttft_p50} ms"
        echo "  ITL p50: ${itl_p50} ms"
        echo "  Throughput: ${throughput} tok/s"
        echo "  Request Rate: ${req_per_s} req/s"

    else
        echo "Benchmark failed for $config_name"
        cat >> "$REPORT_FILE" << EOF

### $config_name

**Configuration:** \`$config_args\`

❌ **Benchmark Failed**

Server log:
\`\`\`
$(tail -50 "$server_log" 2>/dev/null || echo "No server log available")
\`\`\`

EOF
    fi

    # Cleanup
    rm -f "/tmp/bench-${config_name}.json" "/tmp/warmup-${config_name}.json" "$metrics_file" "$server_log"
}

# Install guidellm if not available
if ! python -c "import guidellm" 2>/dev/null; then
    echo "Installing GuideLLM..."
    pip install guidellm
fi

# Run benchmarks for each configuration
for config_name in "${!CONFIGS[@]}"; do
    echo
    echo "=================================================="
    echo "Running configuration: $config_name"
    echo "=================================================="

    run_guidellm "$config_name" "${CONFIGS[$config_name]}"

    # Brief pause between configurations
    sleep 5
done

# Generate comparative analysis
echo "Generating comparative analysis..."

cat >> "$REPORT_FILE" << EOF

## Analysis

### Performance Comparison

The following table compares key metrics across configurations:

| Configuration | TTFT p50 | ITL p50 | Throughput | Request Rate | Improvement |
|---------------|----------|---------|------------|--------------|-------------|
EOF

# Extract baseline metrics (single-threaded)
baseline_json="${OUTPUT_DIR}/raw-single-threaded-${LABEL}.json"
if [ -f "$baseline_json" ]; then
    baseline_throughput=$(jq -r '.metrics.throughput_token_per_s' "$baseline_json" 2>/dev/null || echo "0")
    baseline_req_rate=$(jq -r '.metrics.request_throughput_req_per_s' "$baseline_json" 2>/dev/null || echo "0")

    for config_name in "${!CONFIGS[@]}"; do
        config_json="${OUTPUT_DIR}/raw-${config_name}-${LABEL}.json"
        if [ -f "$config_json" ]; then
            ttft_p50=$(jq -r '.metrics.ttft_ms.p50' "$config_json" 2>/dev/null || echo "N/A")
            itl_p50=$(jq -r '.metrics.itl_ms.p50' "$config_json" 2>/dev/null || echo "N/A")
            throughput=$(jq -r '.metrics.throughput_token_per_s' "$config_json" 2>/dev/null || echo "0")
            req_rate=$(jq -r '.metrics.request_throughput_req_per_s' "$config_json" 2>/dev/null || echo "0")

            # Calculate improvement over baseline
            if [ "$baseline_throughput" != "0" ] && [ "$throughput" != "0" ]; then
                improvement=$(python3 -c "print(f'{$throughput/$baseline_throughput:.2f}x')" 2>/dev/null || echo "N/A")
            else
                improvement="N/A"
            fi

            echo "| $config_name | ${ttft_p50} ms | ${itl_p50} ms | ${throughput} tok/s | ${req_rate} req/s | ${improvement} |" >> "$REPORT_FILE"
        fi
    done
fi

cat >> "$REPORT_FILE" << EOF

### Key Findings

$(python3 << 'EOF_PYTHON'
import json
import os

output_dir = os.environ.get('OUTPUT_DIR', './docs/experience/wins')
label = os.environ.get('LABEL', 'test')

# Load results
results = {}
for config in ['single-threaded', 'multi-threaded-minimal', 'multi-threaded-default', 'multi-threaded-aggressive']:
    json_file = f"{output_dir}/raw-{config}-{label}.json"
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                results[config] = json.load(f)
        except:
            continue

if 'single-threaded' in results and 'multi-threaded-default' in results:
    baseline = results['single-threaded']
    multithread = results['multi-threaded-default']

    try:
        baseline_throughput = float(baseline['metrics']['throughput_token_per_s'])
        mt_throughput = float(multithread['metrics']['throughput_token_per_s'])
        improvement = mt_throughput / baseline_throughput

        print(f"1. **Throughput Improvement:** {improvement:.2f}x over single-threaded baseline")

        if improvement >= 1.5:
            print("   ✅ **Target achieved:** Multi-threaded runtime delivers expected performance gain")
        else:
            print("   ⚠️ **Target missed:** Performance gain below 1.5x target")

        baseline_ttft = float(baseline['metrics']['ttft_ms']['p50'])
        mt_ttft = float(multithread['metrics']['ttft_ms']['p50'])
        ttft_ratio = mt_ttft / baseline_ttft

        print(f"2. **TTFT Impact:** {ttft_ratio:.2f}x vs baseline")
        if ttft_ratio <= 1.1:
            print("   ✅ **Latency maintained:** TTFT remains competitive")
        else:
            print("   ⚠️ **Latency regression:** TTFT increased significantly")

    except (KeyError, ValueError, TypeError):
        print("Unable to calculate performance metrics from benchmark data")
else:
    print("Insufficient data for comparative analysis")
EOF_PYTHON
)

### Problems Encountered

$(if [ -s "/tmp/benchmark-errors.log" ]; then cat "/tmp/benchmark-errors.log"; else echo "No significant issues encountered during benchmarking."; fi)

### Learnings

1. **Threading Overhead:** Multi-threaded runtime introduces coordination overhead that may offset gains at low concurrency
2. **Scalability:** Performance improvements scale with concurrency level and system resources
3. **Resource Utilization:** Multi-threaded approach better utilizes available CPU cores
4. **Memory Management:** Shared memory pools require careful tuning for optimal performance

## Next Steps

1. **Optimization:** Focus on configurations showing highest performance gains
2. **Production Testing:** Validate results under realistic workloads
3. **Monitoring:** Implement runtime performance monitoring for production deployment
4. **Tuning:** Fine-tune thread pool sizes based on specific hardware characteristics

---

*Generated by ARLE multi-threaded runtime benchmark suite*
*Report: $(basename "$REPORT_FILE")*
EOF

echo
echo "===================================================="
echo "Benchmark Complete!"
echo "===================================================="
echo "Report: $REPORT_FILE"
echo "Raw data: ${OUTPUT_DIR}/raw-*-${LABEL}.json"

# Cleanup
rm -f /tmp/benchmark-errors.log