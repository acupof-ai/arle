#!/usr/bin/env python3
"""Reduce qualified exact-cell evidence into Qwen FP8 dense policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path

OPERATOR = "qwen.fp8_dense_projection"
CANDIDATE = "cuda.qwen.fp8_pack_deepgemm"
REFERENCE = "cuda.qwen.fp8_gemv"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _qualified(run: dict, measurement: dict, config: dict) -> bool:
    _require(run.get("schema_version") == "arle.operator-evidence/v1", "unsupported schema")
    _require(run.get("operator_id") == OPERATOR, "wrong operator")
    _require(measurement["candidate"]["id"] == CANDIDATE, "wrong candidate")
    _require(measurement["reference"]["id"] == REFERENCE, "wrong reference")
    for route in (measurement["candidate"], measurement["reference"]):
        samples = route["cuda_us"]
        _require(
            len(samples) >= 3 and all(math.isfinite(value) and value > 0 for value in samples),
            "invalid samples",
        )
    _require(measurement["dtype"] == "bf16xfp8_e4m3_block128->bf16", "wrong dtype")
    _require(measurement["layout"] == "row_major_activation+nt_weight", "wrong layout")
    timing = run["timing"]
    _require(timing["method"] == "cuda_event_batched", "wrong timing method")
    _require(
        all(len(route["cuda_us"]) == timing["samples"] for route in (measurement["candidate"], measurement["reference"])),
        "timing sample count mismatch",
    )
    components = measurement["candidate"]["components"]
    pack = components["pack"]
    gemm = components["gemm"]
    total = measurement["candidate"]["cuda_us"]
    _require(len(pack) == len(total) and len(gemm) == len(total), "component sample count mismatch")
    _require(
        all(abs(candidate - (pack_us + gemm_us)) <= 1e-9 for candidate, pack_us, gemm_us in zip(total, pack, gemm)),
        "candidate total does not equal pack+gemm",
    )
    expected_launches = timing["warmup"] + timing["iterations_per_sample"] * timing["samples"]
    engagement = measurement["engagement"]
    _require(
        engagement["reference_launches"] == expected_launches
        and engagement["candidate_pack_launches"] == expected_launches
        and engagement["candidate_gemm_launches"] == expected_launches,
        "engagement count mismatch",
    )
    numeric = measurement["numeric"]
    _require(
        numeric["tolerance_mode"] == config["numeric_tolerance_mode"],
        "tolerance mode differs from registry",
    )
    abs_tolerance = numeric["max_abs_tolerance"]
    rel_tolerance = numeric["max_rel_tolerance"]
    _require(
        abs_tolerance == config["numeric_abs_tolerance"],
        "absolute tolerance differs from registry",
    )
    _require(
        rel_tolerance == config["numeric_rel_tolerance"],
        "relative tolerance differs from registry",
    )
    numeric_passed = (
        numeric["max_tolerance_ratio"] <= 1.0
        and numeric["violations"] == 0
    )
    _require(numeric["passed"] is numeric_passed, "inconsistent numeric verdict")
    source = run["source"]
    product = run["product"]
    return (
        not source["dirty"]
        and source["commit"] != "unreported"
        and product["binary_id"] != "unreported"
        and product["bundle_id"] != "unreported"
        and run["model_revision"] != "unreported"
        and run["hardware"]["gpu"] != "unreported"
        and all(value != "unreported" for value in run["software"].values())
        and run["e2e_gate"]["passed"] is True
        and len(run["e2e_gate"]["artifact_sha256"] or "") == 64
        and numeric["passed"] is True
    )


def reduce_runs(runs: list[tuple[dict, str]], config: dict) -> dict:
    cells: dict[tuple[int, ...], dict] = {}
    for run, digest in runs:
        hardware = run["hardware"]
        for measurement in run["measurements"]:
            if not _qualified(run, measurement, config):
                continue
            key = (
                hardware["sm_major"],
                hardware["sm_minor"],
                hardware["sm_count"],
                measurement["m"],
                measurement["n"],
                measurement["k"],
            )
            _require(key not in cells, f"duplicate qualified cell: {key}")
            candidate_us = statistics.median(measurement["candidate"]["cuda_us"])
            reference_us = statistics.median(measurement["reference"]["cuda_us"])
            winner = CANDIDATE if candidate_us < reference_us else REFERENCE
            cells[key] = {
                "sm_major": key[0],
                "sm_minor": key[1],
                "sm_count": key[2],
                "m": key[3],
                "n": key[4],
                "k": key[5],
                "winner": winner,
                "candidate_median_us": candidate_us,
                "reference_median_us": reference_us,
                "evidence_sha256": digest,
            }
    policy = {
        "schema_version": "arle.operator-policy/v1",
        "operator_id": OPERATOR,
        "fallback": {
            "candidate": CANDIDATE,
            "min_m": config["fallback_min_m"],
            "reference": REFERENCE,
            "status": config["fallback_status"],
        },
        "numeric_gate": {
            "abs_tolerance": config["numeric_abs_tolerance"],
            "rel_tolerance": config["numeric_rel_tolerance"],
            "tolerance_mode": config["numeric_tolerance_mode"],
        },
        "exact_cells": [cells[key] for key in sorted(cells)],
    }
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    policy["policy_id"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return policy


def render_json(policy: dict) -> str:
    return json.dumps(policy, indent=2, sort_keys=True) + "\n"


def render_rust(policy: dict) -> str:
    cells = policy["exact_cells"]
    arms = []
    for cell in cells:
        route = "PackDeepGemm" if cell["winner"] == CANDIDATE else "Gemv"
        key = ", ".join(str(cell[name]) for name in ("sm_major", "sm_minor", "sm_count", "m", "n", "k"))
        arms.append(f"        ({key}) => Some(Route::{route}),")
    arms.append("        _ => None,")
    min_m = policy["fallback"]["min_m"]
    if cells:
        args = "    sm_major: i32,\n    sm_minor: i32,\n    sm_count: usize,\n    m: usize,\n    n: usize,\n    k: usize,"
        body = f"    match (sm_major, sm_minor, sm_count, m, n, k) {{\n{chr(10).join(arms)}\n    }}"
    else:
        args = "    _sm_major: i32,\n    _sm_minor: i32,\n    _sm_count: usize,\n    _m: usize,\n    _n: usize,\n    _k: usize,"
        body = "    None"
    return f'''// @generated by scripts/reduce_operator_evidence.py; do not edit.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Route {{
    Gemv,
    PackDeepGemm,
}}

pub(super) const HAS_EXACT_CELLS: bool = {str(bool(cells)).lower()};
pub(super) const POLICY_ID: &str =
    "{policy['policy_id']}";

pub(super) fn select_exact(
{args}
) -> Option<Route> {{
{body}
}}

pub(super) const fn fallback(m: usize) -> Route {{
    if m >= {min_m} {{
        Route::PackDeepGemm
    }} else {{
        Route::Gemv
    }}
}}
'''


def _load_run(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        _require(path.read_text() == content, f"stale generated file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _self_test() -> None:
    run = {
        "schema_version": "arle.operator-evidence/v1",
        "operator_id": OPERATOR,
        "source": {"commit": "a" * 40, "dirty": False},
        "product": {"binary_id": "binary", "bundle_id": "bundle"},
        "hardware": {"gpu": "H20", "sm_major": 9, "sm_minor": 0, "sm_count": 78},
        "model_revision": "model@revision",
        "software": {"driver": "12000", "toolkit": "12.8", "provider": "deepgemm"},
        "timing": {"method": "cuda_event_batched", "warmup": 1, "iterations_per_sample": 2, "samples": 3},
        "e2e_gate": {"passed": True, "artifact_sha256": "b" * 64},
        "measurements": [
            {
                "m": 2,
                "n": 8,
                "k": 128,
                "dtype": "bf16xfp8_e4m3_block128->bf16",
                "layout": "row_major_activation+nt_weight",
                "candidate": {
                    "id": CANDIDATE,
                    "cuda_us": [2.0, 1.0, 1.5],
                    "components": {"pack": [0.5, 0.25, 0.5], "gemm": [1.5, 0.75, 1.0]},
                },
                "reference": {"id": REFERENCE, "cuda_us": [3.0, 2.5, 4.0]},
                "numeric": {
                    "max_abs": 0.01,
                    "max_rel": 0.02,
                    "max_abs_tolerance": 1.0,
                    "max_rel_tolerance": 0.02,
                    "tolerance_mode": "abs+rel*abs(reference)",
                    "max_tolerance_ratio": 0.2,
                    "violations": 0,
                    "passed": True,
                },
                "engagement": {
                    "reference_launches": 7,
                    "candidate_pack_launches": 7,
                    "candidate_gemm_launches": 7,
                },
            }
        ],
    }
    config = {
        "fallback_min_m": 2,
        "fallback_status": "legacy-default-unqualified",
        "numeric_abs_tolerance": 1.0,
        "numeric_rel_tolerance": 0.02,
        "numeric_tolerance_mode": "abs+rel*abs(reference)",
    }
    first = reduce_runs([(run, "digest")], config)
    second = reduce_runs([(json.loads(json.dumps(run, sort_keys=True)), "digest")], config)
    _require(render_json(first) == render_json(second), "reducer is not deterministic")
    _require(first["exact_cells"][0]["winner"] == CANDIDATE, "wrong winner")
    _require("(9, 0, 78, 2, 8, 128)" in render_rust(first), "exact cell missing")
    try:
        reduce_runs([(run, "first"), (run, "second")], config)
    except ValueError as error:
        _require("duplicate qualified cell" in str(error), "wrong duplicate error")
    else:
        raise ValueError("duplicate qualified cell accepted")
    run["source"]["dirty"] = True
    _require(not reduce_runs([(run, "digest")], config)["exact_cells"], "dirty run qualified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--registry", type=Path, default=Path("operators/registry.toml"))
    parser.add_argument("--output-json", type=Path, default=Path("benchmarks/operators/optimal.json"))
    parser.add_argument(
        "--output-rust",
        type=Path,
        default=Path("crates/infer-cuda/src/ops/generated/qwen_fp8_dense_projection.rs"),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    registry = tomllib.loads(args.registry.read_text())
    config = registry["policy"][OPERATOR]
    policy = reduce_runs([_load_run(path) for path in args.runs], config)
    _write_or_check(args.output_json, render_json(policy), args.check)
    _write_or_check(args.output_rust, render_rust(policy), args.check)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"operator evidence reduction failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
