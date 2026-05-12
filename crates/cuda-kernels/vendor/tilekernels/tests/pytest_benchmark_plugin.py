"""Benchmark & GPU memory profiling pytest plugin for tile_kernels.

CLI options, markers, fixtures, and regression reporting for kernel
benchmarks and GPU memory profiling.

This file is deliberately NOT named ``conftest.py`` — it is loaded via
``pytest_plugins`` in the root ``conftest.py``.  A non-conftest name
prevents pluggy's duplicate-registration error.
"""

import json
import math
import os
import threading

import pytest
import torch

from tile_kernels.testing.bench import make_param_key

# Baseline file, co-located with this plugin
_BASELINES_PATH = os.path.join(os.path.dirname(__file__), 'benchmark_baselines.jsonl')


# Prefix stripped from pytest node IDs to form stable, short keys
_TILE_KERNELS_PREFIX = os.path.join('tests', '')


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        '--run-benchmark',
        action='store_true',
        default=False,
        help='Run benchmark tests (skipped by default)',
    )
    parser.addoption(
        '--benchmark-output',
        default=None,
        help='Path to write benchmark results as JSONL (one JSON object per line)',
    )
    parser.addoption(
        '--benchmark-regression-threshold',
        default=0.15,
        type=float,
        help='Fraction of slowdown that triggers a regression warning (default: 0.15 = 15%%)',
    )
    parser.addoption(
        '--benchmark-verbose',
        action='store_true',
        default=False,
        help='Show extras columns (e.g., speedup, …) in the benchmark regression report',
    )


# ---------------------------------------------------------------------------
# Marker registration & GPU binding
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line('markers', 'benchmark: mark test as benchmark (skip by default)')
    # Bind each xdist worker to a GPU via CUDA_VISIBLE_DEVICES and restrict
    # per-process GPU memory so that concurrent workers don't OOM.
    worker_id = os.environ.get('PYTEST_XDIST_WORKER', None)
    if worker_id is not None:
        gpu_id = int(worker_id.replace('gw', ''))
        num_gpus = torch.cuda.device_count()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id % num_gpus)

        # Restrict each worker's GPU memory to (total - 10 GB) / workers_per_gpu.
        # PYTEST_XDIST_WORKER_COUNT is set by pytest-xdist automatically.
        total_workers = int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
        workers_per_gpu = math.ceil(total_workers / num_gpus)
        _reserve_bytes = 10 * (1024 ** 3)  # 10 GB reserved for system / frameworks
        total_mem = torch.cuda.mem_get_info(0)[1]
        usable_mem = max(total_mem - _reserve_bytes, 0)
        mem_per_worker = usable_mem / workers_per_gpu
        fraction = mem_per_worker / total_mem
        fraction = max(min(fraction, 1.0), 0.0)
        torch.cuda.set_per_process_memory_fraction(fraction)

    # Shared state for collecting benchmark results across this session
    config._benchmark_results = []
    config._benchmark_results_lock = threading.Lock()

    # Disable warnings during benchmark setting
    if config.getoption('--run-benchmark', default=None):
        config.option.disable_warnings = True


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-benchmark'):
        # Without --run-benchmark, skip all benchmark tests
        skip_bench = pytest.mark.skip(reason='need --run-benchmark to run')
        for item in items:
            if 'benchmark' in item.keywords:
                item.add_marker(skip_bench)
    # With --run-benchmark, benchmark tests run alongside correctness tests
    # (e.g. `pytest kernel.py --run-benchmark`).
    # Use `-m benchmark` explicitly if you want ONLY benchmarks.



# ---------------------------------------------------------------------------
# Regression detection & exit code
# ---------------------------------------------------------------------------


def _detect_regressions(config):
    """Check benchmark results against baselines and return regressions.

    Returns:
        A tuple ``(results, baselines, regressions, improvements, missing)``
        or ``None`` if no results were collected.
    """
    results = getattr(config, '_benchmark_results', [])
    if not results:
        output_path = config.getoption('--benchmark-output', default=None)
        if output_path and os.path.exists(output_path):
            with open(output_path) as f:
                results = [json.loads(line) for line in f if line.strip()]
        return None

    threshold = config.getoption('--benchmark-regression-threshold')
    baselines = _load_baselines()

    regressions = []
    improvements = []
    missing = []

    for rec in results:
        key = _make_key(rec)
        if key not in baselines:
            missing.append((key, rec['time_us']))
            continue
        baseline_us = baselines[key]['time_us']
        current_us = rec['time_us']
        ratio = current_us / baseline_us
        if ratio > 1.0 + threshold:
            regressions.append((key, baseline_us, current_us, ratio))
        elif ratio < 1.0 - threshold:
            improvements.append((key, baseline_us, current_us, ratio))

    return results, baselines, regressions, improvements, missing


def pytest_sessionfinish(session, exitstatus):
    """Set non-zero exit code when benchmark regressions are detected.

    Runs before ``pytest_terminal_summary``, so regression detection is
    performed here and stashed on ``config`` for the terminal report.
    """
    result = _detect_regressions(session.config)
    if result is None:
        return
    results, baselines, regressions, improvements, missing = result
    # Stash for pytest_terminal_summary
    session.config._benchmark_detection = result
    if (regressions or missing) and exitstatus == 0:
        session.exitstatus = 1


# ---------------------------------------------------------------------------
# Terminal summary: regression report
# ---------------------------------------------------------------------------

def pytest_terminal_summary(terminalreporter, config):
    """Print a benchmark regression report at the end of the pytest session."""
    # Use pre-computed results from pytest_sessionfinish if available,
    # otherwise compute now
    detection = getattr(config, '_benchmark_detection', None)
    if detection is None:
        detection = _detect_regressions(config)
    if detection is None:
        # No benchmark results — nothing to report
        return

    results, baselines, regressions, improvements, missing = detection
    threshold = config.getoption('--benchmark-regression-threshold')
    verbose = config.getoption('--benchmark-verbose')

    tr = terminalreporter
    tr.section('Benchmark Regression Report')

    if baselines:
        # Collect extras column names when verbose
        extras_keys = []
        if verbose:
            extras_keys = _collect_extras_keys(results, baselines)

        # Compute dynamic Kernel column width
        matched_keys = [
            _make_key(r) for r in results if _make_key(r) in baselines
        ]
        kw = max((len(k) for k in matched_keys), default=20) + 2

        # Extras column widths: fit header label or widest value
        ek_widths = {}
        for ek in extras_keys:
            cur_label = ek + '(cur)'
            ref_label = ek + '(ref)'
            w = max(len(cur_label), len(ref_label), 8)
            for rec in results:
                rk = _make_key(rec)
                if rk not in baselines:
                    continue
                for src in (rec, baselines[rk]):
                    v = (src.get('extras') or {}).get(ek)
                    w = max(w, len(_fmt_extra(v)))
            ek_widths[ek] = w

        # Header
        hdr = (
            f"{'Kernel':<{kw}} {'Latency':>11} {'Bandwidth':>11} {'Ratio':>8} {'Stat':>4}"
        )
        for ek in extras_keys:
            w = ek_widths[ek]
            hdr += f"  {(ek + '(cur)'):>{w}}  {(ek + '(ref)'):>{w}}"
        tr.write_line(hdr)
        tr.write_line('-' * len(hdr))

        for rec in results:
            key = _make_key(rec)
            if key not in baselines:
                continue
            baseline_rec = baselines[key]
            baseline_us = baseline_rec['time_us']
            current_us = rec['time_us']
            ratio = current_us / baseline_us
            if ratio > 1.0 + threshold:
                status = '--'
            elif ratio < 1.0 - threshold:
                status = '++'
            else:
                status = '='
            cur_bw = rec['bandwidth_gbs']
            line = (
                f'{key:<{kw}} {current_us:>8.1f} us {_fmt_bw(cur_bw):>11} '
                f'{ratio:>7.2f}x {status:>4}'
            )
            for ek in extras_keys:
                w = ek_widths[ek]
                cur_v = (rec.get('extras') or {}).get(ek)
                ref_v = (baseline_rec.get('extras') or {}).get(ek)
                line += f'  {_fmt_extra(cur_v):>{w}}  {_fmt_extra(ref_v):>{w}}'
            tr.write_line(line)
    else:
        tr.write_line('No baseline file found — skipping regression comparison.')
        tr.write_line(f'  (looked at: {_BASELINES_PATH})')

    # New benchmarks without baselines
    if missing:
        new_recs = [r for r in results if _make_key(r) not in baselines]
        tr.write_line('')

        # Dynamic column widths
        new_keys = [_make_key(r) for r in new_recs]
        nkw = max((len(k) for k in new_keys), default=20) + 2

        # Bandwidth column width for new-benchmarks table
        new_bw_col_w = 9
        for r in new_recs:
            v = r.get('bandwidth_gbs', None)
            new_bw_col_w = max(new_bw_col_w, len(_fmt_bw(v)))

        new_extras_keys = []
        new_ek_widths = {}
        if verbose:
            ek_set = set()
            for r in new_recs:
                ek_set.update((r.get('extras') or {}).keys())
            new_extras_keys = sorted(ek_set)
            for ek in new_extras_keys:
                w = len(ek)
                for r in new_recs:
                    v = (r.get('extras') or {}).get(ek)
                    w = max(w, len(_fmt_extra(v)))
                new_ek_widths[ek] = max(w, 8)

        # Header
        nhdr = f"{'Kernel':<{nkw}} {'Current':>11}  {'Bandwidth':>{new_bw_col_w}}"
        for ek in new_extras_keys:
            nhdr += f'  {ek:>{new_ek_widths[ek]}}'
        tr.write_line(nhdr)
        tr.write_line('-' * len(nhdr))

        for r in new_recs:
            key = _make_key(r)
            bw = r.get('bandwidth_gbs', None)
            line = f"{key:<{nkw}} {r['time_us']:>8.1f} us  {_fmt_bw(bw):>{new_bw_col_w}}"
            for ek in new_extras_keys:
                w = new_ek_widths[ek]
                v = (r.get('extras') or {}).get(ek)
                line += f'  {_fmt_extra(v):>{w}}'
            tr.write_line(line)

    # Summary
    matched = sum(1 for r in results if baselines and _make_key(r) in baselines)
    tr.write_line('')
    tr.write_line(
        f'Total: {len(results)} benchmarks, {matched} with baselines, '
        f'{len(missing)} missing, '
        f'{len(regressions)} regressions, {len(improvements)} improvements '
        f'(threshold: {threshold:.0%})'
    )

    if regressions:
        tr.write_line('')
        tr.write_line('!! REGRESSIONS DETECTED !!')
        for key, baseline_us, current_us, ratio in regressions:
            tr.write_line(
                f'  {key}: {current_us:.1f} us vs baseline {baseline_us:.1f} us '
                f'({ratio:.2f}x slower)'
            )



def _fmt_extra(v):
    """Format an extras value for display."""
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.2f}'
    return str(v)


def _fmt_bw(v):
    """Format a bandwidth_gbs value for display (e.g. '1234.56 GB/s')."""
    if v is None:
        return '-'
    return f'{v:6.1f} GB/s'


def _collect_extras_keys(results, baselines):
    """Return a sorted list of extras keys across results and baselines,
    excluding bandwidth_gbs (reported as a dedicated column)."""
    keys = set()
    for rec in results:
        key = _make_key(rec)
        if key not in baselines:
            continue
        for e in (rec.get('extras') or {}, (baselines[key].get('extras') or {})):
            keys.update(e.keys())
    return sorted(keys)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Lock for concurrent JSONL writes from xdist workers
_jsonl_write_lock = threading.Lock()


@pytest.fixture
def benchmark_record(request):
    """Record a benchmark result for regression tracking.

    Prints a human-readable summary, appends a JSONL record to
    ``--benchmark-output`` (if given), collects the result for the terminal
    regression report, and emits a pytest warning on regressions.

    JSONL schema::

        {
            "kernel":         str,
            "operation":      str,
            "params":         dict,
            "time_us":        float,
            "bandwidth_gbs":  float | None,
            "extras":         dict | None,
        }
    """
    output_path = request.config.getoption('--benchmark-output')

    def _record(*, kernel, operation, params, time_us, bandwidth_gbs=None, extras=None):
        # Build a unique key: kernel/operation[k1=v1,k2=v2]
        # Keys are sorted for deterministic ordering
        if params:
            param_str = make_param_key(params)
            key = f'{kernel}/{operation}[{param_str}]'
        else:
            key = f'{kernel}/{operation}'

        # Human-readable print
        parts = [f'  BENCH {key}: {time_us:.1f} us']
        if bandwidth_gbs is not None:
            parts.append(f', bandwidth_gbs={bandwidth_gbs:.2f}')
        if extras:
            for ek, ev in extras.items():
                if isinstance(ev, float):
                    parts.append(f', {ek}={ev:.2f}')
                else:
                    parts.append(f', {ek}={ev}')
        print(''.join(parts))

        # Write JSONL
        record = {
            'kernel': kernel,
            'operation': operation,
            'params': dict(sorted(params.items())) if params else params,
            'time_us': round(time_us, 2),
        }
        if bandwidth_gbs is not None:
            record['bandwidth_gbs'] = round(bandwidth_gbs, 4)
        if extras:
            record['extras'] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in extras.items()
            }
        if output_path:
            line = json.dumps(record, ensure_ascii=False)
            with _jsonl_write_lock:
                with open(output_path, 'a') as f:
                    f.write(line + '\n')

        # Collect for terminal summary
        with request.config._benchmark_results_lock:
            request.config._benchmark_results.append(record)


    return _record


@pytest.fixture
def benchmark_timer():
    """Return a callable that measures kernel execution time in microseconds.

    Wraps ``tilelang.profiler.bench.do_bench`` with CUPTI backend by default.
    Keyword arguments are forwarded to ``do_bench``, allowing per-test
    overrides (e.g. ``benchmark_timer(fn, rep=30)``).

    Returns:
        A callable ``(fn, **overrides) -> float`` returning time in
        microseconds.
    """
    from tilelang.profiler.bench import do_bench

    def _timer(fn, **overrides):
        kwargs = dict(backend='cupti', warmup=0, rep=30)
        kwargs.update(overrides)
        return do_bench(fn, **kwargs) * 1e3  # ms → us

    return _timer


def _make_key(rec):
    """Build a baseline-compatible key from a benchmark record."""
    kernel, operation = rec['kernel'], rec['operation']
    params = rec.get('params')
    if params:
        param_str = make_param_key(params)
        return f'{kernel}/{operation}[{param_str}]'
    return f'{kernel}/{operation}'


def _load_baselines():
    """Load the baseline JSONL file into a ``{key: record}`` dict.

    Returns ``None`` if the file does not exist.
    """
    if not os.path.exists(_BASELINES_PATH):
        return {}
    baselines = {}
    with open(_BASELINES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            baselines[_make_key(rec)] = rec
    return baselines


