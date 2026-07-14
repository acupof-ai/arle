#!/usr/bin/env python3

import argparse
import bisect
import json
import sqlite3


GLOBAL_ID_SHIFT = 24


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else 0


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def clip(intervals, starts, start, end):
    index = max(0, bisect.bisect_right(starts, start) - 1)
    clipped = []
    while index < len(intervals) and intervals[index][0] < end:
        left, right = intervals[index]
        if right > start:
            clipped.append((max(left, start), min(right, end)))
        index += 1
    return clipped


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(left, right):
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def summarize(rows):
    fields = ("wall", "gpu_busy", "host_launch", "overlap", "outside_gpu")
    result = {"ticks": len(rows)}
    for field in fields:
        values = [row[field] / 1e6 for row in rows]
        result[f"{field}_ms_p50"] = round(percentile(values, 0.5), 3)
        result[f"{field}_ms_p90"] = round(percentile(values, 0.9), 3)
    wall = sum(row["wall"] for row in rows)
    outside_gpu = sum(row["outside_gpu"] for row in rows)
    launch = sum(row["host_launch"] for row in rows)
    result["outside_gpu_wall_pct"] = round(100 * outside_gpu / wall, 3)
    result["launch_covered_pct"] = round(100 * (launch - outside_gpu) / launch, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description="Measure host/GPU overlap in a DSv4 NVTX nsys SQLite export.")
    parser.add_argument("sqlite")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--max-gap-ms", type=float, default=150)
    args = parser.parse_args()

    connection = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    tids = [
        row[0]
        for row in connection.execute(
            "SELECT globalTid FROM NVTX_EVENTS WHERE text = ? GROUP BY globalTid ORDER BY globalTid",
            ("dsv4/lm_head_sample_batched",),
        )
    ]
    tid = tids[args.rank]
    global_pid = tid >> GLOBAL_ID_SHIFT << GLOBAL_ID_SHIFT

    embeds = [
        row[0]
        for row in connection.execute(
            "SELECT start FROM NVTX_EVENTS WHERE globalTid = ? AND text = ? ORDER BY start",
            (tid, "dsv4/embed"),
        )
    ]
    samples = [
        row[0]
        for row in connection.execute(
            "SELECT start FROM NVTX_EVENTS WHERE globalTid = ? AND text = ? ORDER BY start",
            (tid, "dsv4/lm_head_sample"),
        )
    ]
    heads = list(
        connection.execute(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE globalTid = ? AND text = ? AND start >= ? ORDER BY start",
            (tid, "dsv4/lm_head_sample_batched", embeds[0]),
        )
    )
    ticks = []
    for head_start, head_end in heads:
        start = embeds[bisect.bisect_right(embeds, head_start) - 1]
        batch = bisect.bisect_right(samples, head_end) - bisect.bisect_left(samples, head_start)
        ticks.append({"start": start, "forward_end": head_end, "batch": batch})

    trace_start = ticks[0]["start"]
    trace_end = ticks[-1]["forward_end"]
    process_end = global_pid + (1 << GLOBAL_ID_SHIFT)
    launches = {}
    for start, end, correlation in connection.execute(
        "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
        "WHERE globalTid >= ? AND globalTid < ? AND start < ? AND end > ? AND correlationId IS NOT NULL",
        (global_pid, process_end, trace_end, trace_start),
    ):
        launches[correlation] = (start, end)

    kernels = []
    delays = []
    matched = set()
    for start, end, correlation in connection.execute(
        "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE globalPid = ? AND start < ? AND end > ? ORDER BY start",
        (global_pid, trace_end, trace_start),
    ):
        kernels.append((start, end))
        if correlation in launches:
            matched.add(correlation)
            delays.append(start - launches[correlation][1])
    connection.close()

    gpu = union(kernels)
    host = union(launches[correlation] for correlation in matched)
    gpu_starts = [item[0] for item in gpu]
    host_starts = [item[0] for item in host]
    rows = []
    for tick, next_tick in zip(ticks, ticks[1:]):
        start, end = tick["start"], next_tick["start"]
        if end - start > args.max_gap_ms * 1e6:
            continue
        gpu_tick = clip(gpu, gpu_starts, start, end)
        host_tick = clip(host, host_starts, start, end)
        overlap = intersection(host_tick, gpu_tick)
        rows.append(
            {
                "batch": tick["batch"],
                "wall": end - start,
                "gpu_busy": duration(gpu_tick),
                "host_launch": duration(host_tick),
                "overlap": overlap,
                "outside_gpu": duration(host_tick) - overlap,
            }
        )

    batches = {
        batch: summarize([row for row in rows if row["batch"] == batch])
        for batch in sorted({row["batch"] for row in rows})
    }
    positive = sum(delay >= 0 for delay in delays)
    output = {
        "sqlite": args.sqlite,
        "rank": args.rank,
        "global_tid": tid,
        "matched_kernels": len(delays),
        "positive_queue_delay_pct": round(100 * positive / len(delays), 4),
        "queue_delay_ms": {
            "p50": round(percentile(delays, 0.5) / 1e6, 3),
            "p90": round(percentile(delays, 0.9) / 1e6, 3),
            "p99": round(percentile(delays, 0.99) / 1e6, 3),
        },
        "all": summarize(rows),
        "by_batch": batches,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
