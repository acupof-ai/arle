#!/usr/bin/env python3
"""ARLE real-time dashboard.

Serves a single-page dashboard that polls the ARLE /metrics and /v1/stats
endpoints and renders live load + throughput. Defaults to the V100 serve.
"""

import json
import threading
import time
import urllib.request
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

ARLE_BASE = "http://10.37.2.27:8000"
_cache = {"stats": {}, "metrics": {}, "ts": 0}
_lock = threading.Lock()


def _fetch(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read().decode()


def _fetch_stats():
    return json.loads(_fetch(f"{ARLE_BASE}/v1/stats"))


def _fetch_metrics():
    out = {}
    for line in _fetch(f"{ARLE_BASE}/metrics").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            out[name] = float(parts[1])
        except ValueError:
            pass
    return out


def _poll_loop():
    while True:
        try:
            stats = _fetch_stats()
            metrics = _fetch_metrics()
            with _lock:
                _cache["stats"] = stats
                _cache["metrics"] = metrics
                _cache["ts"] = time.time()
        except Exception as e:
            with _lock:
                _cache["error"] = str(e)
        time.sleep(2)


@app.route("/api/data")
def api_data():
    with _lock:
        return jsonify(_cache)


PAGE = """
<!doctype html>
<html><head><title>ARLE Dashboard</title>
<style>
body { font-family: monospace; background: #111; color: #eee; margin: 20px; }
h1 { font-size: 18px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
.card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 14px; }
.card .label { color: #888; font-size: 12px; }
.card .value { font-size: 24px; margin-top: 4px; }
.green { color: #4c4; }
.red { color: #c44; }
.yellow { color: #cc4; }
table { width: 100%; margin-top: 16px; border-collapse: collapse; }
td { padding: 4px 8px; border-bottom: 1px solid #222; }
td:first-child { color: #888; }
</style></head>
<body>
<h1>ARLE Serve Dashboard — <span id="model"></span></h1>
<div class="grid">
  <div class="card"><div class="label">Active Requests</div><div class="value" id="active">—</div></div>
  <div class="card"><div class="label">Queue Depth</div><div class="value" id="queue">—</div></div>
  <div class="card"><div class="label">KV Used</div><div class="value" id="kvused">—</div></div>
  <div class="card"><div class="label">Decode Rate (tok/s)</div><div class="value" id="rate">—</div></div>
  <div class="card"><div class="label">Total Generated</div><div class="value" id="gentok">—</div></div>
  <div class="card"><div class="label">DSpark</div><div class="value" id="dspark">—</div></div>
</div>
<table id="kv"></table>
<script>
let prev = null;
async function tick() {
  const r = await fetch('/api/data');
  const d = await r.json();
  const s = d.stats || {};
  const sc = s.scheduler || {};
  const tp = s.throughput || {};
  const kv = s.kv_system || {};
  const spec = s.spec_decode || {};
  document.getElementById('active').textContent = sc.active_requests ?? '—';
  document.getElementById('queue').textContent = sc.queue_depth ?? '—';
  const free = sc.kv_free_pages ?? 0;
  const total = free + (kv.resident_pages ?? 0) + (kv.host_demoted_pages ?? 0);
  const used = total ? ((1 - free/total)*100).toFixed(1) + '%' : '—';
  document.getElementById('kvused').textContent = used;
  document.getElementById('gentok').textContent = tp.generated_tokens ?? '—';
  const decode_us = tp.decode_forward_busy_micros ?? 0;
  const gen = tp.generated_tokens ?? 0;
  const avg = decode_us ? (gen / (decode_us/1e6)).toFixed(1) : '—';
  document.getElementById('rate').textContent = avg;
  const ds = spec.available ? 'ON' : 'off';
  document.getElementById('dspark').textContent = ds;
  document.getElementById('dspark').className = 'value ' + (spec.available ? 'green' : 'red');
  if (spec.available) {
    document.getElementById('dspark').textContent += ' (accept=' + (spec.accept_rate ?? 'n/a') + ')';
  }
  const rows = [
    ['KV resident (GPU)', kv.resident_pages ?? 0],
    ['KV host_demoted (L2)', kv.host_demoted_pages ?? 0],
    ['KV disk (L3)', kv.disk_pages ?? 0],
    ['Prefill tokens', tp.prefill_tokens ?? 0],
    ['Requests completed', tp.requests_completed ?? 0],
    ['L2 tier available', s.kv_tier?.available ? 'yes' : 'no'],
    ['L3 SSD recall', s.ssd_recall?.available ? 'yes' : 'no'],
  ];
  document.getElementById('kv').innerHTML = rows.map(r =>
    '<tr><td>'+r[0]+'</td><td>'+r[1]+'</td></tr>').join('');
}
setInterval(tick, 2000);
tick();
</script></body></html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    t = threading.Thread(target=_poll_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=3000, debug=False)
