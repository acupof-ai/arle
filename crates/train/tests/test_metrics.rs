//! Tests for the training metrics sinks.

use std::fs;
use std::io::BufRead;
use std::path::PathBuf;

use tempfile::tempdir;
use train::metrics::{JsonlSink, MetricSample, MetricSink, MultiSink, NullSink, open_sink};

fn read_lines(path: &PathBuf) -> Vec<String> {
    let file = fs::File::open(path).expect("open jsonl");
    std::io::BufReader::new(file)
        .lines()
        .map(|l| l.expect("read line"))
        .collect()
}

#[test]
fn null_sink_emit_does_not_panic() {
    let mut sink = NullSink;
    let fields = [("loss", 1.5f64), ("lr", 1e-4f64)];
    sink.emit(&MetricSample {
        step: 0,
        fields: &fields,
    });
    sink.flush();
}

#[test]
fn jsonl_sink_roundtrip_three_samples() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("metrics.jsonl");

    {
        let mut sink = JsonlSink::create(&path).expect("create jsonl");
        let f1 = [("loss", 2.5f64), ("lr", 3e-4f64)];
        sink.emit(&MetricSample {
            step: 1,
            fields: &f1,
        });
        let f2 = [("loss", 1.25f64), ("grad_norm", 0.875f64)];
        sink.emit(&MetricSample {
            step: 2,
            fields: &f2,
        });
        let f3 = [("tokens_per_s", 1234.5f64)];
        sink.emit(&MetricSample {
            step: 3,
            fields: &f3,
        });
        // drop flushes
    }

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 3, "expected 3 lines, got {:?}", lines);

    let v1: serde_json::Value = serde_json::from_str(&lines[0]).expect("line 1 parses");
    assert_eq!(v1["step"], serde_json::json!(1));
    assert_eq!(v1["loss"].as_f64().unwrap(), 2.5);
    assert_eq!(v1["lr"].as_f64().unwrap(), 3e-4);

    let v2: serde_json::Value = serde_json::from_str(&lines[1]).expect("line 2 parses");
    assert_eq!(v2["step"], serde_json::json!(2));
    assert_eq!(v2["loss"].as_f64().unwrap(), 1.25);
    assert_eq!(v2["grad_norm"].as_f64().unwrap(), 0.875);

    let v3: serde_json::Value = serde_json::from_str(&lines[2]).expect("line 3 parses");
    assert_eq!(v3["step"], serde_json::json!(3));
    assert_eq!(v3["tokens_per_s"].as_f64().unwrap(), 1234.5);
}

#[test]
fn multi_sink_fans_out_to_two_files() {
    let dir = tempdir().expect("tempdir");
    let path_a = dir.path().join("a.jsonl");
    let path_b = dir.path().join("b.jsonl");

    {
        let a = JsonlSink::create(&path_a).expect("create a");
        let b = JsonlSink::create(&path_b).expect("create b");
        let mut multi = MultiSink::new(vec![Box::new(a), Box::new(b)]);
        let fields = [("loss", 0.5f64)];
        multi.emit(&MetricSample {
            step: 7,
            fields: &fields,
        });
        multi.flush();
    }

    let lines_a = read_lines(&path_a);
    let lines_b = read_lines(&path_b);
    assert_eq!(lines_a.len(), 1);
    assert_eq!(lines_b.len(), 1);

    let va: serde_json::Value = serde_json::from_str(&lines_a[0]).unwrap();
    let vb: serde_json::Value = serde_json::from_str(&lines_b[0]).unwrap();
    assert_eq!(va["step"], serde_json::json!(7));
    assert_eq!(vb["step"], serde_json::json!(7));
    assert_eq!(va["loss"].as_f64().unwrap(), 0.5);
    assert_eq!(vb["loss"].as_f64().unwrap(), 0.5);
}

#[test]
fn open_sink_none_no_stdout_returns_null_like() {
    let mut sink = open_sink(None, false).expect("open null sink");
    let fields = [("loss", 1.0f64)];
    // Just assert no panic.
    sink.emit(&MetricSample {
        step: 0,
        fields: &fields,
    });
    sink.flush();
}

#[test]
fn open_sink_jsonl_plus_stdout_emits_without_panic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("metrics.jsonl");

    {
        let mut sink = open_sink(Some(&path), true).expect("open multi sink");
        let fields = [("loss", 0.25f64), ("lr", 1e-3f64), ("step_ms", 12.345f64)];
        sink.emit(&MetricSample {
            step: 42,
            fields: &fields,
        });
        sink.flush();
    }

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 1, "expected one line in {:?}", path);
    let v: serde_json::Value = serde_json::from_str(&lines[0]).expect("parse");
    assert_eq!(v["step"], serde_json::json!(42));
    assert_eq!(v["loss"].as_f64().unwrap(), 0.25);
}

#[test]
fn jsonl_sink_missing_parent_dir_errors() {
    // Sibling of tempdir that does not exist: parent dir must exist.
    let dir = tempdir().expect("tempdir");
    let bogus = dir.path().join("no_such_subdir").join("m.jsonl");
    let res = JsonlSink::create(&bogus);
    assert!(res.is_err(), "expected missing-parent-dir error");
}

// M-8 — guard against manual-JSON-string-building drift: every line written
// by JsonlSink must parse cleanly with serde_json and round-trip the fields
// emitted. If someone "optimises" emit() into hand-rolled string concat, this
// catches dropped quoting / numeric formatting regressions.
#[test]
fn jsonl_line_is_parseable_by_serde_json() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("parseable.jsonl");

    {
        let mut sink = JsonlSink::create(&path).expect("create jsonl");
        // Mix of float magnitudes and a negative value — exercises the
        // default number formatter.
        let f1 = [("loss", 2.5f64), ("lr", 3e-4f64)];
        sink.emit(&MetricSample {
            step: 1,
            fields: &f1,
        });
        let f2 = [("loss", -0.125f64), ("grad_norm", 1.5e-12f64)];
        sink.emit(&MetricSample {
            step: 2,
            fields: &f2,
        });
        // NaN + Inf serialise as JSON null per M-4/M-5; the line must still
        // parse as a well-formed JSON object (regression would be "NaN" leaking
        // into the stream and breaking serde_json).
        let f3 = [
            ("loss", f64::NAN),
            ("tokens_per_s", 1234.5f64),
            ("grad_norm", f64::INFINITY),
        ];
        sink.emit(&MetricSample {
            step: 3,
            fields: &f3,
        });
        sink.flush();
    }

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 3, "expected 3 lines, got {:?}", lines);

    // Every line parses as a JSON object and carries the emitted fields.
    let v1: serde_json::Value = serde_json::from_str(&lines[0]).expect("line 1 parses");
    assert!(v1.is_object(), "line 1 must be a JSON object");
    assert_eq!(v1["step"], serde_json::json!(1));
    assert_eq!(v1["loss"].as_f64().unwrap(), 2.5);
    assert_eq!(v1["lr"].as_f64().unwrap(), 3e-4);

    let v2: serde_json::Value = serde_json::from_str(&lines[1]).expect("line 2 parses");
    assert!(v2.is_object());
    assert_eq!(v2["step"], serde_json::json!(2));
    assert_eq!(v2["loss"].as_f64().unwrap(), -0.125);
    assert_eq!(v2["grad_norm"].as_f64().unwrap(), 1.5e-12);

    let v3: serde_json::Value = serde_json::from_str(&lines[2]).expect("line 3 parses");
    assert!(v3.is_object());
    assert_eq!(v3["step"], serde_json::json!(3));
    // NaN / Inf must have been substituted with JSON null per the sink contract.
    assert!(v3["loss"].is_null(), "NaN should serialise as null");
    assert_eq!(v3["tokens_per_s"].as_f64().unwrap(), 1234.5);
    assert!(v3["grad_norm"].is_null(), "Inf should serialise as null");
}
