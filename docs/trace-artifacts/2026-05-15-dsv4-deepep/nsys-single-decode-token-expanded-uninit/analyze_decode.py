import csv
import json
import sqlite3
from pathlib import Path

out = Path(__file__).resolve().parent
conn = sqlite3.connect(out / "trace.sqlite")
cur = conn.cursor()
tables = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if "NVTX_EVENTS" not in tables:
    raise SystemExit("NVTX_EVENTS table missing")

columns = {row[1] for row in cur.execute("PRAGMA table_info(NVTX_EVENTS)")}
if "text" in columns:
    ranges = cur.execute(
        """
        SELECT start, end FROM NVTX_EVENTS
        WHERE text = 'step_decode_kernel_launch' AND end IS NOT NULL
        ORDER BY start
        """
    ).fetchall()
else:
    ranges = []
if not ranges:
    ranges = [
        (start, end)
        for _name, start, end in cur.execute(
            """
            SELECT n.value, e.start, e.end FROM NVTX_EVENTS e
            JOIN StringIds n ON e.textId = n.id
            WHERE n.value = 'step_decode_kernel_launch' AND e.end IS NOT NULL
            ORDER BY e.start
            """
        ).fetchall()
    ]
if not ranges:
    raise SystemExit("decode NVTX ranges missing")

cur.execute("CREATE TEMP TABLE decode_ranges_tmp(start INTEGER, end INTEGER)")
cur.executemany("INSERT INTO decode_ranges_tmp VALUES (?, ?)", ranges)

runtime_rows = cur.execute(
    """
    WITH hits AS (
        SELECT DISTINCT r.rowid AS rid,
               COALESCE(s.value, printf('%d', r.nameId)) AS name,
               (r.end-r.start)/1e6 AS time_ms
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        LEFT JOIN StringIds s ON r.nameId = s.id
        JOIN decode_ranges_tmp d ON r.start >= d.start AND r.end <= d.end
    )
    SELECT name,
           COUNT(*) AS calls,
           SUM(time_ms) AS total_ms,
           AVG(time_ms) AS avg_ms
    FROM hits
    GROUP BY 1 ORDER BY total_ms DESC LIMIT 40
    """
).fetchall()

kernel_rows = cur.execute(
    """
    WITH hits AS (
        SELECT DISTINCT k.rowid AS kid,
               COALESCE(s.value, k.demangledName, k.shortName) AS name,
               (k.end-k.start)/1e6 AS time_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
        JOIN decode_ranges_tmp d ON k.start >= d.start AND k.end <= d.end
    )
    SELECT name,
           COUNT(*) AS calls,
           SUM(time_ms) AS total_ms,
           AVG(time_ms) AS avg_ms
    FROM hits
    GROUP BY 1 ORDER BY total_ms DESC LIMIT 50
    """
).fetchall()

memcpy_rows = []
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    memcpy_rows = cur.execute(
        """
        WITH hits AS (
            SELECT DISTINCT m.rowid AS mid,
                   m.copyKind AS copy_kind,
                   m.bytes AS bytes,
                   (m.end-m.start)/1e6 AS time_ms
            FROM CUPTI_ACTIVITY_KIND_MEMCPY m
            JOIN decode_ranges_tmp d ON m.start >= d.start AND m.end <= d.end
        )
        SELECT COALESCE(e.label, e.name, printf('%d', h.copy_kind)) AS kind,
               COUNT(*) AS calls,
               SUM(bytes) AS bytes,
               SUM(time_ms) AS total_ms,
               AVG(time_ms) AS avg_ms
        FROM hits h
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON h.copy_kind = e.id
        GROUP BY 1 ORDER BY total_ms DESC
        """
    ).fetchall()

range_ms = [(end - start) / 1e6 for start, end in ranges]
wave_ranges = [ranges]
if len(ranges) % 8 == 0:
    wave_ranges = [ranges[i : i + 8] for i in range(0, len(ranges), 8)]
wave_wall_ms = [
    (max(end for _start, end in wave) - min(start for start, _end in wave)) / 1e6
    for wave in wave_ranges
]
range_count = len(ranges)
summary = {
    "capture": "single profile request, filtered to step_decode_kernel_launch NVTX ranges",
    "decode_ranges": len(ranges),
    "decode_waves": (len(ranges) // 8) if len(ranges) % 8 == 0 else None,
    "decode_wave_wall_ms": wave_wall_ms,
    "decode_wave_wall_ms_max": max(wave_wall_ms),
    "decode_range_ms_min": min(range_ms),
    "decode_range_ms_p50": sorted(range_ms)[len(range_ms) // 2],
    "decode_range_ms_max": max(range_ms),
    "top_runtime_apis": [
        {
            "name": name,
            "time_ms_per_rank_range": total / range_count,
            "total_time_ms_all_ranges": total,
            "calls": calls,
            "avg_ms": avg,
        }
        for name, calls, total, avg in runtime_rows[:15]
    ],
    "top_kernels": [
        {
            "name": name,
            "time_ms_per_rank_range": total / range_count,
            "total_time_ms_all_ranges": total,
            "calls": calls,
            "avg_ms": avg,
        }
        for name, calls, total, avg in kernel_rows[:20]
    ],
    "memcpy_activity": [
        {
            "kind": kind,
            "calls": calls,
            "bytes": bytes_,
            "time_ms_per_rank_range": total / range_count,
            "total_time_ms_all_ranges": total,
            "avg_ms": avg,
        }
        for kind, calls, bytes_, total, avg in memcpy_rows
    ],
}

(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with (out / "decode-only-runtime-api-top.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["name", "calls", "total_ms", "avg_ms"])
    writer.writerows(runtime_rows)
with (out / "decode-only-kernel-top.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["name", "calls", "total_ms", "avg_ms"])
    writer.writerows(kernel_rows)
with (out / "decode-only-memcpy-summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(["kind", "calls", "bytes", "total_ms", "avg_ms"])
    writer.writerows(memcpy_rows)

print(json.dumps(summary, indent=2))
