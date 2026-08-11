#!/usr/bin/env python3
"""ARLE real-time dashboard with per-metric charts + GPU monitoring."""

import argparse
import json
import subprocess
import threading
import time
import urllib.request
from collections import deque
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

ARLE_BASE = "http://10.37.2.27:8000"
GPU_HOST = "v100"
_cache = {"stats": {}, "gpu": {}}
_lock = threading.Lock()
_history = deque(maxlen=300)


def _fetch(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def _fetch_stats():
    return json.loads(_fetch(f"{ARLE_BASE}/v1/stats"))


def _fetch_gpu():
    try:
        out = subprocess.run(
            ["ssh", GPU_HOST,
             "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw "
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        line = out.stdout.strip().splitlines()
        if not line:
            return {}
        parts = [p.strip() for p in line[0].split(",")]
        if len(parts) < 4:
            return {}
        return {
            "gpu_util": float(parts[0]),
            "mem_used": float(parts[1]),
            "mem_total": float(parts[2]),
            "power": float(parts[3]),
        }
    except Exception:
        pass
    return {}


def _poll_loop():
    while True:
        try:
            stats = _fetch_stats()
            gpu = _fetch_gpu()
            tp = stats.get("throughput", {})
            sc = stats.get("scheduler", {})
            kv = stats.get("kv_system", {})
            free = sc.get("kv_free_pages", 0)
            total = free + kv.get("resident_pages", 0) + kv.get("host_demoted_pages", 0)
            kv_used = (1 - free / total) * 100 if total else 0
            dec_tokens = tp.get("generated_tokens", 0)
            dec_us = tp.get("decode_forward_busy_micros", 0)
            decode_rate = dec_tokens / (dec_us / 1e6) if dec_us else 0
            spec = stats.get("spec_decode", {})
            with _lock:
                _cache["stats"] = stats
                _cache["gpu"] = gpu
                _history.append({
                    "decode_rate": decode_rate,
                    "active": sc.get("active_requests", 0),
                    "kv_used": kv_used,
                    "accept_rate": spec.get("accept_rate", 0) or 0,
                    "gpu_util": gpu.get("gpu_util", 0),
                    "mem_used": gpu.get("mem_used", 0),
                    "power": gpu.get("power", 0),
                })
        except Exception:
            pass
        time.sleep(1)


@app.route("/api/data")
def api_data():
    with _lock:
        return jsonify({
            "stats": _cache["stats"],
            "gpu": _cache["gpu"],
            "history": list(_history),
        })


PAGE = """
<!doctype html>
<html><head><title>ARLE Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0d0d0d; color: #eee; margin: 20px; }
h1 { font-size: 18px; margin-bottom: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 16px; }
.card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 12px; }
.card .label { color: #888; font-size: 12px; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 2px; }
.green { color: #7ec699; }
.red { color: #f07178; }
.blue { color: #82aaff; }
.purple { color: #c792ea; }
.yellow { color: #ffcb6b; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }
.chart-box { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 14px; }
.chart-box h3 { font-size: 13px; color: #888; font-weight: 500; margin-bottom: 8px; }
.chart-wrap { position: relative; height: 180px; }
table { width: 100%; margin-top: 16px; border-collapse: collapse; }
td { padding: 5px 10px; border-bottom: 1px solid #222; font-size: 13px; }
td:first-child { color: #888; }
</style></head>
<body>
<h1>ARLE Serve Dashboard</h1>
<div class="grid">
  <div class="card"><div class="label">Active Requests</div><div class="value" id="active">—</div></div>
  <div class="card"><div class="label">Queue Depth</div><div class="value" id="queue">—</div></div>
  <div class="card"><div class="label">KV Used</div><div class="value" id="kvused">—</div></div>
  <div class="card"><div class="label">Decode Rate</div><div class="value blue" id="rate">—</div><div style="font-size:11px;color:#888">tok/s</div></div>
  <div class="card"><div class="label">GPU Util</div><div class="value purple" id="gpuutil">—</div><div style="font-size:11px;color:#888">%</div></div>
  <div class="card"><div class="label">GPU Memory</div><div class="value yellow" id="gpumem">—</div></div>
  <div class="card"><div class="label">Power</div><div class="value" id="power">—</div><div style="font-size:11px;color:#888">W</div></div>
  <div class="card"><div class="label">DSpark</div><div class="value" id="dspark">—</div></div>
</div>
<div class="charts">
  <div class="chart-box"><h3>Decode Rate (tok/s)</h3><div class="chart-wrap"><canvas id="cRate"></canvas></div></div>
  <div class="chart-box"><h3>Active Requests</h3><div class="chart-wrap"><canvas id="cActive"></canvas></div></div>
  <div class="chart-box"><h3>KV Used (%)</h3><div class="chart-wrap"><canvas id="cKv"></canvas></div></div>
  <div class="chart-box"><h3>DSpark Accept Rate (%)</h3><div class="chart-wrap"><canvas id="cAccept"></canvas></div></div>
  <div class="chart-box"><h3>GPU Utilization (%)</h3><div class="chart-wrap"><canvas id="cGpu"></canvas></div></div>
  <div class="chart-box"><h3>GPU Memory (MiB)</h3><div class="chart-wrap"><canvas id="cMem"></canvas></div></div>
  <div class="chart-box"><h3>GPU Power (W)</h3><div class="chart-wrap"><canvas id="cPower"></canvas></div></div>
</div>
<table id="kv"></table>
<script>
function mk(id, color, ymax) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: color, backgroundColor: color+'20', borderWidth: 2, tension: 0.3, pointRadius: 0, fill: true }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { grid: { color: '#222' }, ticks: { color: '#888' }, suggestedMin: 0, ...(ymax ? { suggestedMax: ymax } : {}) } } }
  });
}
const charts = {
  rate: mk('cRate', '#82aaff'),
  active: mk('cActive', '#c792ea'),
  kv: mk('cKv', '#ffcb6b', 100),
  accept: mk('cAccept', '#7ec699', 100),
  gpu: mk('cGpu', '#c792ea', 100),
  mem: mk('cMem', '#ffcb6b'),
  power: mk('cPower', '#f07178'),
};
async function tick() {
  const r = await fetch('/api/data');
  const d = await r.json();
  const s = d.stats || {};
  const sc = s.scheduler || {};
  const tp = s.throughput || {};
  const kv = s.kv_system || {};
  const spec = s.spec_decode || {};
  const g = d.gpu || {};
  document.getElementById('active').textContent = sc.active_requests ?? '—';
  document.getElementById('queue').textContent = sc.queue_depth ?? '—';
  const free = sc.kv_free_pages ?? 0;
  const total = free + (kv.resident_pages ?? 0) + (kv.host_demoted_pages ?? 0);
  document.getElementById('kvused').textContent = total ? ((1-free/total)*100).toFixed(1)+'%' : '—';
  const dec_us = tp.decode_forward_busy_micros ?? 0;
  const gen = tp.generated_tokens ?? 0;
  document.getElementById('rate').textContent = dec_us ? (gen/(dec_us/1e6)).toFixed(1) : '—';
  document.getElementById('gpuutil').textContent = g.gpu_util != null ? g.gpu_util.toFixed(0) : '—';
  document.getElementById('gpumem').textContent = g.mem_used != null ? (g.mem_used/1024).toFixed(1)+'GB' : '—';
  document.getElementById('power').textContent = g.power != null ? g.power.toFixed(0) : '—';
  const ds = spec.available ? 'ON' : 'off';
  const dsEl = document.getElementById('dspark');
  dsEl.textContent = ds;
  dsEl.className = 'value ' + (spec.available ? 'green' : 'red');
  if (spec.available && spec.accept_rate != null)
    dsEl.textContent += ' (' + (spec.accept_rate*100).toFixed(0) + '%)';
  const h = d.history || [];
  if (h.length) {
    const setData = (c, mapper) => {
      c.data.labels = h.map(() => '');
      c.data.datasets[0].data = h.map(mapper);
      c.update('none');
    };
    setData(charts.rate, p => p.decode_rate || 0);
    setData(charts.active, p => p.active || 0);
    setData(charts.kv, p => p.kv_used || 0);
    setData(charts.accept, p => (p.accept_rate || 0) * 100);
    setData(charts.gpu, p => p.gpu_util || 0);
    setData(charts.mem, p => p.mem_used || 0);
    setData(charts.power, p => p.power || 0);
  }
  const rows = [
    ['KV resident (GPU)', kv.resident_pages ?? 0],
    ['KV host_demoted (L2)', kv.host_demoted_pages ?? 0],
    ['KV disk (L3)', kv.disk_pages ?? 0],
    ['Prefill tokens', tp.prefill_tokens ?? 0],
    ['Generated tokens', tp.generated_tokens ?? 0],
    ['Requests completed', tp.requests_completed ?? 0],
    ['L2 tier', s.kv_tier?.available ? 'yes' : 'no'],
    ['L3 SSD', s.ssd_recall?.available ? 'yes' : 'no'],
  ];
  document.getElementById('kv').innerHTML = rows.map(r => '<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
}
setInterval(tick, 2000); tick();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=ARLE_BASE)
    ap.add_argument("--gpu-host", default=GPU_HOST)
    ap.add_argument("--port", type=int, default=3000)
    args = ap.parse_args()
    ARLE_BASE = args.base_url.rstrip("/")
    GPU_HOST = args.gpu_host
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=args.port, debug=False)
