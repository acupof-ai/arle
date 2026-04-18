//! Minimal HTTP control plane for `TrainingController`.
//!
//! Zero framework dependencies — std `TcpListener` + line-at-a-time
//! request parsing. Bodies are ignored; responses are tiny JSON blobs
//! built by hand. This is deliberately spartan: the trainer is the
//! process that matters, and the control-plane endpoints only need to
//! carry start/stop/status/save intent.
//!
//! ## Routes
//! - `GET /v1/train/status` → 200 JSON snapshot of `TrainingStatus`
//! - `POST /v1/train/stop`  → 200 `{"stop_requested":true}`; trainer
//!   honours at next iteration boundary.
//! - `POST /v1/train/save`  → 200 `{"save_requested":true}`; trainer
//!   flushes a checkpoint at next iteration boundary.
//! - Anything else          → 404 `{"error":"not found"}`
//!
//! ## Design notes
//! The listener runs on a dedicated thread. Each accept is handled
//! synchronously — the expected call rate is human-scale (operator
//! hitting curl), not fan-out. Requests longer than 8 KiB are
//! rejected, guarding against pathological input while keeping the
//! parser under 60 lines.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;

use crate::control::{TrainingController, TrainingStatus};

const MAX_REQUEST_BYTES: usize = 8 * 1024;

pub fn serve(controller: Arc<TrainingController>, addr: impl ToSocketAddrs) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr)?;
    loop {
        let (stream, _) = listener.accept()?;
        let ctrl = Arc::clone(&controller);
        if let Err(err) = handle_connection(ctrl, stream) {
            eprintln!("[train_server] connection error: {err}");
        }
        if controller.should_stop() && controller.snapshot().finished {
            break;
        }
    }
    Ok(())
}

pub fn bind_and_serve_on_thread(
    controller: Arc<TrainingController>,
    addr: String,
) -> std::io::Result<std::thread::JoinHandle<()>> {
    let listener = TcpListener::bind(&addr)?;
    listener.set_nonblocking(false)?;
    let handle = std::thread::spawn(move || {
        for incoming in listener.incoming() {
            match incoming {
                Ok(stream) => {
                    let ctrl = Arc::clone(&controller);
                    if let Err(err) = handle_connection(ctrl, stream) {
                        eprintln!("[train_server] connection error: {err}");
                    }
                    if controller.snapshot().finished {
                        break;
                    }
                }
                Err(err) => {
                    eprintln!("[train_server] accept error: {err}");
                    break;
                }
            }
        }
    });
    Ok(handle)
}

fn handle_connection(
    controller: Arc<TrainingController>,
    mut stream: TcpStream,
) -> std::io::Result<()> {
    let mut reader = BufReader::new(
        stream
            .try_clone()
            .expect("tcp stream must be cloneable for buffered read"),
    );
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    // Read headers until blank line; capture Content-Length.
    let mut content_length = 0usize;
    let mut total_read = request_line.len();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        total_read += bytes;
        if total_read > MAX_REQUEST_BYTES {
            return write_response(&mut stream, 413, "{\"error\":\"payload too large\"}");
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some(value) = line
            .to_ascii_lowercase()
            .strip_prefix("content-length:")
            .map(str::trim)
            .and_then(|s| s.parse::<usize>().ok())
        {
            content_length = value.min(MAX_REQUEST_BYTES);
        }
    }

    // Drain body if present (we don't use it, but leaving bytes
    // buffered means the next connection on a keep-alive channel
    // would see garbage — safer to consume).
    if content_length > 0 {
        let mut sink = vec![0u8; content_length];
        reader.read_exact(&mut sink)?;
    }

    let (method, path) = parse_request_line(&request_line);
    match (method.as_str(), path.as_str()) {
        ("GET", "/v1/train/status") => {
            let body = render_status_json(&controller.snapshot());
            write_response(&mut stream, 200, &body)
        }
        ("POST", "/v1/train/stop") => {
            controller.request_stop();
            write_response(&mut stream, 200, "{\"stop_requested\":true}")
        }
        ("POST", "/v1/train/save") => {
            controller.request_save();
            write_response(&mut stream, 200, "{\"save_requested\":true}")
        }
        _ => write_response(&mut stream, 404, "{\"error\":\"not found\"}"),
    }
}

fn parse_request_line(line: &str) -> (String, String) {
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_ascii_uppercase();
    let path = parts.next().unwrap_or("").to_string();
    (method, path)
}

fn write_response<W: Write>(writer: &mut W, status: u16, body: &str) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        413 => "Payload Too Large",
        _ => "Other",
    };
    write!(
        writer,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body}",
        len = body.len(),
    )
}

pub fn render_status_json(status: &TrainingStatus) -> String {
    format!(
        "{{\"iter\":{iter},\"total_iters\":{total},\"mean_reward\":{mean:.6},\"best_reward\":{best:.6},\"last_kl\":{kl:.6},\"last_loss\":{loss:.6},\"wall_secs\":{wall:.3},\"started\":{started},\"finished\":{finished}}}",
        iter = status.iter,
        total = status.total_iters,
        mean = status.mean_reward,
        best = status.best_reward,
        kl = status.last_kl,
        loss = status.last_loss,
        wall = status.wall_secs,
        started = status.started,
        finished = status.finished,
    )
}
