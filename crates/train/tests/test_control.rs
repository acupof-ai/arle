//! Covers `TrainingController` state transitions plus the hand-rolled
//! JSON renderer + HTTP surface in `train::server`. No network — we
//! bind to an ephemeral port, curl via raw `TcpStream`, and assert
//! exact bytes.

use std::{
    io::{Read, Write},
    net::TcpStream,
    sync::Arc,
    thread,
    time::Duration,
};

use train::{
    control::TrainingController,
    server::{bind_and_serve_on_thread, render_status_json},
};

#[test]
fn update_and_snapshot_roundtrip() {
    let ctrl = TrainingController::new();
    ctrl.update(|s| {
        s.iter = 3;
        s.total_iters = 10;
        s.mean_reward = 0.25;
        s.best_reward = 0.5;
        s.last_kl = 0.01;
        s.last_loss = 1.125;
        s.wall_secs = 4.5;
        s.started = true;
    });
    let snap = ctrl.snapshot();
    assert_eq!(snap.iter, 3);
    assert_eq!(snap.total_iters, 10);
    assert!((snap.mean_reward - 0.25).abs() < 1e-6);
    assert!((snap.best_reward - 0.5).abs() < 1e-6);
    assert!(snap.started);
    assert!(!snap.finished);
}

#[test]
fn stop_is_sticky() {
    let ctrl = TrainingController::new();
    assert!(!ctrl.should_stop());
    ctrl.request_stop();
    assert!(ctrl.should_stop());
    assert!(ctrl.should_stop(), "stop stays true across reads");
}

#[test]
fn save_is_edge_triggered() {
    let ctrl = TrainingController::new();
    assert!(!ctrl.take_save_request());
    ctrl.request_save();
    assert!(ctrl.take_save_request());
    assert!(
        !ctrl.take_save_request(),
        "second take should observe the cleared flag"
    );
}

#[test]
fn render_status_json_shape() {
    let ctrl = TrainingController::new();
    ctrl.update(|s| {
        s.iter = 7;
        s.total_iters = 20;
        s.mean_reward = 0.125;
        s.best_reward = 0.25;
        s.last_kl = 0.01;
        s.last_loss = 1.5;
        s.wall_secs = 12.0;
        s.started = true;
    });
    let body = render_status_json(&ctrl.snapshot());
    assert!(body.starts_with('{') && body.ends_with('}'));
    assert!(body.contains("\"iter\":7"));
    assert!(body.contains("\"total_iters\":20"));
    assert!(body.contains("\"started\":true"));
    assert!(body.contains("\"finished\":false"));
    assert!(body.contains("\"mean_reward\":0.125"));
}

fn send_request(port: u16, request: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .expect("set timeout");
    stream.write_all(request.as_bytes()).expect("write request");
    let mut buf = String::new();
    stream.read_to_string(&mut buf).expect("read response");
    let status: u16 = buf
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status code");
    let body = buf.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

#[test]
fn http_status_stop_save_404() {
    // Use port 0 trick: bind, discover port, then start the server on that addr.
    // The server helper binds internally, so we pick a port deterministically
    // from a port-range and retry if flaky. Simpler: bind TcpListener once,
    // drop it, grab the port. On macOS this rarely TOCTOUs in a unit test.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let ctrl = TrainingController::new();
    ctrl.update(|s| {
        s.iter = 1;
        s.total_iters = 2;
        s.started = true;
    });

    let _handle = bind_and_serve_on_thread(Arc::clone(&ctrl), format!("127.0.0.1:{port}"))
        .expect("bind server");
    // Give the listener thread a moment to arm accept().
    thread::sleep(Duration::from_millis(50));

    let (status, body) = send_request(
        port,
        "GET /v1/train/status HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(status, 200);
    assert!(body.contains("\"iter\":1"), "body was {body:?}");
    assert!(body.contains("\"started\":true"));

    let (status, body) = send_request(
        port,
        "POST /v1/train/save HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(status, 200);
    assert!(body.contains("\"save_requested\":true"));
    assert!(ctrl.take_save_request(), "save flag should have been set");

    let (status, body) = send_request(
        port,
        "POST /v1/train/stop HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(status, 200);
    assert!(body.contains("\"stop_requested\":true"));
    assert!(ctrl.should_stop());

    let (status, body) = send_request(
        port,
        "GET /nope HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(status, 404);
    assert!(body.contains("not found"));
}
