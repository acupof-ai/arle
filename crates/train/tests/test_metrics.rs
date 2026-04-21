//! Tests for the training metrics sinks.

use std::fs;
use std::io::BufRead;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tempfile::tempdir;
use train::metrics::{
    JsonlSink, MetricSample, MetricSink, MlflowConfig, MlflowSink, MultiSink, NullSink,
    OtlpLogConfig, OtlpLogSink, TrainEvent, WandbConfig, WandbProcessSink, open_shared_sink,
    open_shared_sink_append, open_sink, open_sink_append,
};

type MockMlflowArtifactRequest = (String, String, Vec<u8>);
type MockOtlpRequest = (String, String, Vec<u8>);

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
        phase: "train",
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
            phase: "train",
            fields: &f1,
        });
        let f2 = [("loss", 1.25f64), ("grad_norm", 0.875f64)];
        sink.emit(&MetricSample {
            step: 2,
            phase: "train",
            fields: &f2,
        });
        let f3 = [("tokens_per_s", 1234.5f64)];
        sink.emit(&MetricSample {
            step: 3,
            phase: "train",
            fields: &f3,
        });
        // drop flushes
    }

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 3, "expected 3 lines, got {:?}", lines);

    let v1: serde_json::Value = serde_json::from_str(&lines[0]).expect("line 1 parses");
    assert_eq!(v1["step"], serde_json::json!(1));
    assert_eq!(v1["kind"], serde_json::json!("metric"));
    assert_eq!(v1["phase"], serde_json::json!("train"));
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
            phase: "train",
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
        phase: "train",
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
            phase: "train",
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
            phase: "train",
            fields: &f1,
        });
        let f2 = [("loss", -0.125f64), ("grad_norm", 1.5e-12f64)];
        sink.emit(&MetricSample {
            step: 2,
            phase: "train",
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
            phase: "train",
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

/// Phase 4 follow-up (commit 60f7183): `JsonlSink::open_append` is the
/// multi-phase-binary sibling of `create`. `train_grpo` uses it for the
/// GRPO phase so JSONL output from the SFT-phase Trainer doesn't get
/// clobbered. Pins the truncate-vs-append contract so a future
/// "simplify" refactor can't silently swap append for truncate.
#[test]
fn jsonl_sink_open_append_extends_existing_file() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_phase.jsonl");

    // Phase 1: create (truncate) + two samples.
    {
        let mut sink = JsonlSink::create(&path).expect("create jsonl");
        let f1 = [("loss", 1.0f64)];
        sink.emit(&MetricSample {
            step: 1,
            phase: "train",
            fields: &f1,
        });
        let f2 = [("loss", 0.5f64)];
        sink.emit(&MetricSample {
            step: 2,
            phase: "train",
            fields: &f2,
        });
    }

    // Phase 2: open_append + one sample. Must NOT truncate.
    {
        let mut sink = JsonlSink::open_append(&path).expect("open_append jsonl");
        let f3 = [("reward", 0.125f64)];
        sink.emit(&MetricSample {
            step: 3,
            phase: "grpo",
            fields: &f3,
        });
    }

    let lines = read_lines(&path);
    assert_eq!(
        lines.len(),
        3,
        "open_append must extend, not truncate — got {:?}",
        lines
    );
    let v1: serde_json::Value = serde_json::from_str(&lines[0]).expect("parse 1");
    let v2: serde_json::Value = serde_json::from_str(&lines[1]).expect("parse 2");
    let v3: serde_json::Value = serde_json::from_str(&lines[2]).expect("parse 3");
    assert_eq!(v1["step"], serde_json::json!(1));
    assert_eq!(v2["step"], serde_json::json!(2));
    assert_eq!(v3["step"], serde_json::json!(3));
    assert_eq!(v3["reward"].as_f64().unwrap(), 0.125);
}

/// Factory-level variant of the above: `open_sink_append` must yield a
/// sink that extends rather than truncates, matching what `train_grpo`
/// actually calls. Also verifies `open_sink_append` creates the file
/// when absent (i.e. single-phase binaries wouldn't break if they
/// accidentally used the append variant).
#[test]
fn open_sink_append_factory_extends_and_creates() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("factory.jsonl");

    // First call: file does not exist — append factory must create it.
    {
        let mut sink = open_sink_append(Some(&path), false).expect("open_sink_append create");
        let f = [("loss", 0.75f64)];
        sink.emit(&MetricSample {
            step: 10,
            phase: "train",
            fields: &f,
        });
    }
    assert_eq!(read_lines(&path).len(), 1);

    // Second call: file exists — append factory must extend.
    {
        let mut sink = open_sink_append(Some(&path), false).expect("open_sink_append extend");
        let f = [("loss", 0.25f64)];
        sink.emit(&MetricSample {
            step: 11,
            phase: "train",
            fields: &f,
        });
    }
    let lines = read_lines(&path);
    assert_eq!(lines.len(), 2, "factory must append, got {:?}", lines);
    let v10: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
    let v11: serde_json::Value = serde_json::from_str(&lines[1]).unwrap();
    assert_eq!(v10["step"], serde_json::json!(10));
    assert_eq!(v11["step"], serde_json::json!(11));
}

#[test]
fn jsonl_sink_serializes_lifecycle_events() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("events.jsonl");

    {
        let mut sink = JsonlSink::create(&path).expect("create jsonl");
        let strings = [
            ("run_id", "run-123"),
            ("job", "train_sft"),
            ("artifact_model", "model.safetensors"),
        ];
        let scalars = [("total_steps", 5.0), ("best_reward", f64::NAN)];
        let bools = [("resumed", true)];
        sink.event(&TrainEvent {
            kind: "run_start",
            step: Some(3),
            strings: &strings,
            scalars: &scalars,
            bools: &bools,
        });
        sink.flush();
    }

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 1);
    let value: serde_json::Value = serde_json::from_str(&lines[0]).expect("parse event");
    assert_eq!(value["kind"], serde_json::json!("run_start"));
    assert_eq!(value["step"], serde_json::json!(3));
    assert_eq!(value["run_id"], serde_json::json!("run-123"));
    assert_eq!(value["job"], serde_json::json!("train_sft"));
    assert_eq!(
        value["artifact_model"],
        serde_json::json!("model.safetensors")
    );
    assert_eq!(value["total_steps"].as_f64().unwrap(), 5.0);
    assert!(
        value["best_reward"].is_null(),
        "NaN should serialize as null"
    );
    assert_eq!(value["resumed"], serde_json::json!(true));
}

#[test]
fn shared_sink_flushes_metrics_and_events() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shared.jsonl");

    let sink = open_shared_sink(Some(&path), false).expect("open shared sink");
    let fields = [("loss", 0.125f64)];
    sink.emit_metric(&MetricSample {
        step: 1,
        phase: "train",
        fields: &fields,
    });
    sink.emit_event(&TrainEvent {
        kind: "run_end",
        step: Some(1),
        strings: &[("status", "completed")],
        scalars: &[],
        bools: &[],
    });
    sink.flush_blocking();

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 2, "shared sink flush should drain worker");
    let metric: serde_json::Value = serde_json::from_str(&lines[0]).expect("parse metric");
    let event: serde_json::Value = serde_json::from_str(&lines[1]).expect("parse event");
    assert_eq!(metric["kind"], serde_json::json!("metric"));
    assert_eq!(metric["phase"], serde_json::json!("train"));
    assert_eq!(event["kind"], serde_json::json!("run_end"));
    assert_eq!(event["status"], serde_json::json!("completed"));
}

#[test]
fn open_shared_sink_append_extends_existing_file() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shared_append.jsonl");

    let sink = open_shared_sink(Some(&path), false).expect("create shared sink");
    sink.emit_metric(&MetricSample {
        step: 1,
        phase: "train",
        fields: &[("loss", 1.0)],
    });
    sink.flush_blocking();
    drop(sink);

    let sink = open_shared_sink_append(Some(&path), false).expect("append shared sink");
    sink.emit_metric(&MetricSample {
        step: 2,
        phase: "grpo",
        fields: &[("mean_reward", 0.5)],
    });
    sink.flush_blocking();

    let lines = read_lines(&path);
    assert_eq!(lines.len(), 2);
    let first: serde_json::Value = serde_json::from_str(&lines[0]).expect("parse first");
    let second: serde_json::Value = serde_json::from_str(&lines[1]).expect("parse second");
    assert_eq!(first["step"], serde_json::json!(1));
    assert_eq!(second["step"], serde_json::json!(2));
    assert_eq!(second["phase"], serde_json::json!("grpo"));
}

#[test]
fn otlp_log_sink_posts_metric_and_event_records() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind otlp mock");
    let addr = listener.local_addr().expect("mock addr");
    let requests: Arc<Mutex<Vec<MockOtlpRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let requests_thread = Arc::clone(&requests);
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk).expect("read");
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let header_end = buf
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .expect("header terminator");
            let header = String::from_utf8(buf[..header_end].to_vec()).expect("utf8 header");
            let content_length = header
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length:")
                        .map(str::trim)
                        .and_then(|value| value.parse::<usize>().ok())
                })
                .unwrap_or(0);
            let mut body = buf[header_end + 4..].to_vec();
            while body.len() < content_length {
                let n = stream.read(&mut chunk).expect("read body");
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..n]);
            }
            let request_line = header.lines().next().expect("request line");
            let path = request_line
                .split_whitespace()
                .nth(1)
                .expect("request path")
                .to_string();
            requests_thread.lock().expect("requests lock").push((
                path,
                header.to_ascii_lowercase(),
                body,
            ));
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .expect("write response");
        }
    });

    let mut sink = OtlpLogSink::new(OtlpLogConfig {
        endpoint: format!("http://{}", addr),
        service_name: "agent-infer.train.test".into(),
        timeout: Some(Duration::from_secs(2)),
        headers: vec![("x-test-token".into(), "abc123".into())],
    })
    .expect("otlp sink");
    sink.emit(&MetricSample {
        step: 7,
        phase: "train",
        fields: &[("loss", 0.25), ("tok_per_sec", 123.0)],
    });
    sink.event(&TrainEvent {
        kind: "run_end",
        step: Some(7),
        strings: &[("status", "completed"), ("run_id", "run-otel")],
        scalars: &[("completed_steps", 7.0)],
        bools: &[],
    });
    sink.flush();

    server.join().expect("server joined");
    let requests = requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 2);
    for (path, headers, body) in requests.iter() {
        assert_eq!(path, "/v1/logs");
        assert!(
            headers.contains("content-type: application/x-protobuf"),
            "expected protobuf content-type, got {headers}"
        );
        assert!(
            headers.contains("x-test-token: abc123"),
            "expected custom header, got {headers}"
        );
        assert!(!body.is_empty(), "otlp request body should not be empty");
    }
}

#[test]
fn wandb_process_sink_forwards_messages_to_helper() {
    let dir = tempdir().expect("tempdir");
    let capture_path = dir.path().join("wandb-capture.jsonl");
    let helper_path = dir.path().join("wandb-helper.py");
    let helper_body = format!(
        r#"#!/usr/bin/env python3
import json
import os
from pathlib import Path

capture = Path(r"{capture}")
with capture.open("w", encoding="utf-8") as out:
    out.write(json.dumps({{
        "type": "env",
        "project": os.environ.get("WANDB_PROJECT"),
        "mode": os.environ.get("WANDB_MODE"),
        "entity": os.environ.get("WANDB_ENTITY"),
        "group": os.environ.get("WANDB_RUN_GROUP"),
        "tags": os.environ.get("TRAIN_WANDB_TAGS"),
        "disable_code": os.environ.get("WANDB_DISABLE_CODE"),
        "silent": os.environ.get("WANDB_SILENT"),
    }}) + "\n")
    out.flush()
    for raw in os.sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        message = json.loads(raw)
        out.write(json.dumps(message) + "\n")
        out.flush()
        if message.get("type") == "finish":
            break
"#,
        capture = capture_path.display()
    );
    fs::write(&helper_path, helper_body).expect("write helper");

    {
        let mut sink = WandbProcessSink::new(WandbConfig {
            project: "agent-infer-tests".into(),
            entity: Some("ci".into()),
            name: Some("wandb-process-sink".into()),
            notes: Some("unit-test".into()),
            group: Some("train".into()),
            job_type: Some("grpo".into()),
            run_id: Some("run-123".into()),
            resume: Some("allow".into()),
            mode: "offline".into(),
            dir: Some(dir.path().join("wandb")),
            base_url: Some("http://localhost:8080".into()),
            tags: vec!["rust".into(), "rl".into()],
            helper_program: "python3".into(),
            helper_script: helper_path.clone(),
            log_checkpoints: true,
            disable_code: true,
            silent: true,
        })
        .expect("wandb sink");
        sink.emit(&MetricSample {
            step: 4,
            phase: "grpo",
            fields: &[("loss", 0.25), ("mean_reward", 0.5)],
        });
        sink.event(&TrainEvent {
            kind: "checkpoint",
            step: Some(4),
            strings: &[
                ("path", "/tmp/run"),
                ("artifact_model", "model.safetensors"),
            ],
            scalars: &[("dropped_metrics", 0.0)],
            bools: &[("save_requested", true)],
        });
    }

    let lines = read_lines(&capture_path);
    assert_eq!(lines.len(), 4, "env + metric + event + finish");
    let env: serde_json::Value = serde_json::from_str(&lines[0]).expect("env json");
    assert_eq!(env["project"], serde_json::json!("agent-infer-tests"));
    assert_eq!(env["mode"], serde_json::json!("offline"));
    assert_eq!(env["entity"], serde_json::json!("ci"));
    assert_eq!(env["group"], serde_json::json!("train"));
    assert_eq!(env["tags"], serde_json::json!("rust,rl"));
    assert_eq!(env["disable_code"], serde_json::json!("true"));
    assert_eq!(env["silent"], serde_json::json!("true"));

    let metric: serde_json::Value = serde_json::from_str(&lines[1]).expect("metric json");
    assert_eq!(metric["type"], serde_json::json!("metric"));
    assert_eq!(metric["phase"], serde_json::json!("grpo"));
    assert_eq!(metric["fields"]["loss"], serde_json::json!(0.25));

    let event: serde_json::Value = serde_json::from_str(&lines[2]).expect("event json");
    assert_eq!(event["type"], serde_json::json!("event"));
    assert_eq!(event["kind"], serde_json::json!("checkpoint"));
    assert_eq!(
        event["strings"]["artifact_model"],
        serde_json::json!("model.safetensors")
    );
    assert_eq!(event["bools"]["save_requested"], serde_json::json!(true));

    let finish: serde_json::Value = serde_json::from_str(&lines[3]).expect("finish json");
    assert_eq!(finish["type"], serde_json::json!("finish"));
}

#[test]
fn mlflow_sink_posts_run_metrics_and_run_end() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mlflow mock");
    let addr = listener.local_addr().expect("mock addr");
    let requests: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let requests_thread = Arc::clone(&requests);
    let server = thread::spawn(move || {
        for _ in 0..5 {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk).expect("read");
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8(buf).expect("utf8 request");
            let header_end = request.find("\r\n\r\n").expect("header terminator");
            let header = &request[..header_end];
            let content_length = header
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length:")
                        .map(str::trim)
                        .and_then(|value| value.parse::<usize>().ok())
                })
                .unwrap_or(0);
            let mut body = request.as_bytes()[header_end + 4..].to_vec();
            while body.len() < content_length {
                let n = stream.read(&mut chunk).expect("read body");
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..n]);
            }
            let path = header
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .expect("request path")
                .to_string();
            requests_thread
                .lock()
                .expect("requests lock")
                .push((path, String::from_utf8(body).expect("utf8 body")));
            let response_body = if requests_thread.lock().expect("requests lock").len() == 1 {
                r#"{"run":{"info":{"run_id":"run-123"}}}"#
            } else {
                "{}"
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            )
            .expect("write response");
        }
    });

    let mut sink = MlflowSink::new(MlflowConfig {
        tracking_uri: format!("http://{}", addr),
        experiment_id: "0".into(),
        run_name: Some("unit-test".into()),
        auth_token: None,
        upload_artifacts: false,
        artifact_path_prefix: "checkpoints".into(),
    });
    sink.event(&TrainEvent {
        kind: "run_start",
        step: Some(0),
        strings: &[("run_id", "local-run"), ("job", "train_sft")],
        scalars: &[("total_steps", 2.0)],
        bools: &[("resumed", false)],
    });
    sink.emit(&MetricSample {
        step: 1,
        phase: "train",
        fields: &[("loss", 0.5), ("tok_per_sec", 123.0)],
    });
    sink.event(&TrainEvent {
        kind: "run_end",
        step: Some(2),
        strings: &[("status", "completed")],
        scalars: &[("completed_steps", 2.0)],
        bools: &[],
    });

    server.join().expect("server joined");
    let requests = requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[0].0, "/api/2.0/mlflow/runs/create");
    assert_eq!(requests[1].0, "/api/2.0/mlflow/runs/log-batch");
    assert_eq!(requests[2].0, "/api/2.0/mlflow/runs/log-batch");
    assert_eq!(requests[3].0, "/api/2.0/mlflow/runs/log-batch");
    assert_eq!(requests[4].0, "/api/2.0/mlflow/runs/update");
    assert!(requests[0].1.contains("\"experiment_id\":\"0\""));
    assert!(requests[1].1.contains("\"train.total_steps\""));
    assert!(requests[2].1.contains("\"train.loss\""));
    assert!(requests[3].1.contains("\"event.run_end.completed_steps\""));
    assert!(requests[4].1.contains("\"status\":\"FINISHED\""));
}

#[test]
fn mlflow_sink_uploads_checkpoint_artifacts_when_enabled() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind mlflow mock");
    let addr = listener.local_addr().expect("mock addr");
    let requests: Arc<Mutex<Vec<MockMlflowArtifactRequest>>> = Arc::new(Mutex::new(Vec::new()));
    let requests_thread = Arc::clone(&requests);
    let server = thread::spawn(move || {
        for _ in 0..5 {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let n = stream.read(&mut chunk).expect("read");
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8(buf).expect("utf8 request");
            let header_end = request.find("\r\n\r\n").expect("header terminator");
            let header = &request[..header_end];
            let content_length = header
                .lines()
                .find_map(|line| {
                    line.strip_prefix("Content-Length:")
                        .map(str::trim)
                        .and_then(|value| value.parse::<usize>().ok())
                })
                .unwrap_or(0);
            let mut body = request.as_bytes()[header_end + 4..].to_vec();
            while body.len() < content_length {
                let n = stream.read(&mut chunk).expect("read body");
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..n]);
            }
            let mut parts = header
                .lines()
                .next()
                .expect("request line")
                .split_whitespace();
            let method = parts.next().expect("method").to_string();
            let path = parts.next().expect("path").to_string();
            requests_thread
                .lock()
                .expect("requests lock")
                .push((method, path, body));
            let response_body = if requests_thread.lock().expect("requests lock").len() == 1 {
                r#"{"run":{"info":{"run_id":"run-456"}}}"#
            } else {
                "{}"
            };
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            )
            .expect("write response");
        }
    });

    let dir = tempdir().expect("tempdir");
    let ckpt = dir.path().join("step_000007");
    fs::create_dir_all(&ckpt).expect("checkpoint dir");
    fs::write(ckpt.join("model.safetensors"), b"model-bytes").expect("model bytes");
    fs::write(ckpt.join("trainer_state.json"), b"{\"step\":7}").expect("state bytes");

    let mut sink = MlflowSink::new(MlflowConfig {
        tracking_uri: format!("http://{}", addr),
        experiment_id: "0".into(),
        run_name: Some("artifact-test".into()),
        auth_token: None,
        upload_artifacts: true,
        artifact_path_prefix: "checkpoints".into(),
    });
    sink.event(&TrainEvent {
        kind: "run_start",
        step: Some(0),
        strings: &[("run_id", "artifact-run"), ("job", "train_sft")],
        scalars: &[("total_steps", 2.0)],
        bools: &[],
    });
    let ckpt_path = ckpt.display().to_string();
    sink.event(&TrainEvent {
        kind: "checkpoint",
        step: Some(7),
        strings: &[
            ("path", ckpt_path.as_str()),
            ("artifact_model", "model.safetensors"),
            ("artifact_state", "trainer_state.json"),
        ],
        scalars: &[],
        bools: &[],
    });

    server.join().expect("server joined");
    let requests = requests.lock().expect("requests lock");
    assert_eq!(requests.len(), 5);
    assert_eq!(requests[0].0, "POST");
    assert_eq!(requests[0].1, "/api/2.0/mlflow/runs/create");
    assert_eq!(requests[1].0, "POST");
    assert_eq!(requests[1].1, "/api/2.0/mlflow/runs/log-batch");
    assert_eq!(requests[2].0, "POST");
    assert_eq!(requests[2].1, "/api/2.0/mlflow/runs/log-batch");
    assert_eq!(requests[3].0, "PUT");
    assert_eq!(
        requests[3].1,
        "/api/2.0/mlflow-artifacts/artifacts/checkpoints/step_000007/model.safetensors?run_id=run-456"
    );
    assert_eq!(requests[3].2, b"model-bytes");
    assert_eq!(requests[4].0, "PUT");
    assert_eq!(
        requests[4].1,
        "/api/2.0/mlflow-artifacts/artifacts/checkpoints/step_000007/trainer_state.json?run_id=run-456"
    );
    assert_eq!(requests[4].2, b"{\"step\":7}");
}
