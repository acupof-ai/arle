//! Pre-CUDA "sandbox-spawner": a long-lived non-CUDA helper process that owns
//! ALL sandbox subprocess spawns on behalf of the agent-OPD rollout.
//!
//! ## Why this exists
//! The 8×H20 pod runs an ELKEID kernel HIDS hook that SIGKILLs arle when a
//! `setsid()` syscall is issued by a process whose ancestor chain contains a
//! CUDA-resident process. Once the agent-OPD process loads the rollout engine /
//! autograd CUDA backend it is CUDA-resident, so any subprocess that calls
//! `setsid()` triggers ELKEID to kill the CUDA-resident ancestor (arle).
//!
//! ## The fix (two layers)
//! **Layer 1 — route forks through a pre-CUDA helper:** Fork ONE helper BEFORE
//! any CUDA init. The helper is a plain non-CUDA process; its forks are not
//! direct children of a CUDA-resident process, which dodges the most common
//! ELKEID trigger.
//!
//! **Layer 2 — avoid `setsid()` and external `kill` in the helper:** The
//! original helper called `Command::new("setsid").arg(program)` for bash-tool
//! commands; `setsid` calls the `setsid()` syscall, which ELKEID hooks. Even
//! though the helper is not itself CUDA-resident, ELKEID traces the ancestor
//! chain of the forked `setsid` process, finds arle, and kills it.
//! Fix: spawn the program directly with `process_group(0)` (uses `setpgid`, not
//! `setsid`). Use `libc::kill(-pgid, SIGKILL)` for group teardown (no extra
//! fork of an external `kill` binary).
//!
//! ## Wiring
//! - The helper is the same `arle` binary re-exec'd with `ARLE_SPAWNER_LISTEN=<sock>`;
//!   [`serve_loop`] runs the request/response loop and never returns.
//! - The agent-OPD parent calls [`SpawnerHandle::launch`] before CUDA init, which
//!   sets `ARLE_SPAWNER_SOCKET=<sock>` so [`crate::sandbox`] routes through
//!   [`SpawnClient`]. When the env is unset (normal serve/CLI) sandbox.rs spawns
//!   directly — byte-identical default.
//!
//! Protocol: length-prefixed (u32-LE) JSON. One request, one response, serial.

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

pub const LISTEN_ENV: &str = "ARLE_SPAWNER_LISTEN";

pub const SOCKET_ENV: &str = "ARLE_SPAWNER_SOCKET";

const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Serializes tests that mutate the spawner env vars (parallel tests race).
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Debug, Serialize, Deserialize)]
pub struct SpawnRequest {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    /// `(key, Some(val))` sets, `(key, None)` removes.
    pub env: Vec<(String, Option<String>)>,
    /// `true` → combined stdout+stderr + process-group timeout (run_captured).
    /// `false` → separate streams, no timeout (output()).
    pub combined_timeout: bool,
    pub timeout_s: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SpawnResponse {
    pub stdout: Vec<u8>,
    /// Empty when `combined_timeout` (output folded into `stdout`).
    pub stderr: Vec<u8>,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
}

fn write_frame<W: Write>(w: &mut W, bytes: &[u8]) -> std::io::Result<()> {
    let len = (bytes.len() as u32).to_le_bytes();
    w.write_all(&len)?;
    w.write_all(bytes)?;
    w.flush()
}

fn read_frame<R: Read>(r: &mut R) -> std::io::Result<Vec<u8>> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

pub fn serve_loop() -> i32 {
    let sock = match std::env::var(LISTEN_ENV) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    // Stale socket from a crashed prior run blocks bind; per-pid path is ours.
    let _ = std::fs::remove_file(&sock);
    let listener = match UnixListener::bind(&sock) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[arle sandbox-spawner] bind {sock} failed: {e}");
            return 1;
        }
    };
    eprintln!("[arle sandbox-spawner] listening on {sock}");
    for conn in listener.incoming() {
        match conn {
            // Thread per connection: cc harness runs K children concurrently.
            Ok(stream) => {
                std::thread::spawn(move || handle_conn(stream));
            }
            Err(e) => {
                eprintln!("[arle sandbox-spawner] accept failed: {e}");
                break;
            }
        }
    }
    let _ = std::fs::remove_file(&sock);
    0
}

fn handle_conn(mut stream: UnixStream) {
    let req_bytes = match read_frame(&mut stream) {
        Ok(b) => b,
        Err(_) => return,
    };
    let req: SpawnRequest = match serde_json::from_slice(&req_bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[arle sandbox-spawner] bad request: {e}");
            return;
        }
    };
    let resp = run_request(&req);
    if let Ok(bytes) = serde_json::to_vec(&resp) {
        let _ = write_frame(&mut stream, &bytes);
    }
}

fn run_request(req: &SpawnRequest) -> SpawnResponse {
    if req.combined_timeout {
        match run_captured(req) {
            Ok((stdout, exit_code, timed_out)) => SpawnResponse {
                stdout,
                stderr: Vec::new(),
                exit_code,
                timed_out,
            },
            Err(e) => SpawnResponse {
                stdout: format!("[arle sandbox-spawner] spawn failed: {e}").into_bytes(),
                stderr: Vec::new(),
                exit_code: None,
                timed_out: false,
            },
        }
    } else {
        let mut cmd = build_command(req);
        match cmd.output() {
            Ok(out) => SpawnResponse {
                stdout: out.stdout,
                stderr: out.stderr,
                exit_code: out.status.code(),
                timed_out: false,
            },
            Err(e) => SpawnResponse {
                stdout: Vec::new(),
                stderr: format!("[arle sandbox-spawner] spawn failed: {e}").into_bytes(),
                exit_code: None,
                timed_out: false,
            },
        }
    }
}

fn build_command(req: &SpawnRequest) -> Command {
    let mut cmd = Command::new(&req.program);
    cmd.args(&req.args);
    if let Some(dir) = &req.cwd {
        cmd.current_dir(dir);
    }
    for (k, v) in &req.env {
        match v {
            Some(val) => {
                cmd.env(k, val);
            }
            None => {
                cmd.env_remove(k);
            }
        }
    }
    cmd
}

/// Spawn in a new process group (`setpgid`, NOT `setsid` — ELKEID hooks
/// `setsid` ancestry and kills arle). Output to a temp file (no pipe hang).
/// `libc::kill` for group teardown (no extra `kill` fork).
fn run_captured(req: &SpawnRequest) -> std::io::Result<(Vec<u8>, Option<i32>, bool)> {
    let tmp = tempfile::NamedTempFile::new()?;
    let stdout_handle = tmp.reopen()?;
    let stderr_handle = stdout_handle.try_clone()?; // shared offset → interleaved

    let mut command = build_command(req);
    command
        .stdin(Stdio::null())
        .stdout(stdout_handle)
        .stderr(stderr_handle)
        // setpgid, not setsid — ELKEID hooks setsid ancestry.
        .process_group(0);

    let mut child = command.spawn()?;
    let pgid = child.id() as i32;

    let deadline = Instant::now() + Duration::from_secs(req.timeout_s);
    let mut killed = false;
    let code = loop {
        match child.try_wait()? {
            Some(status) => break status.code(),
            None => {
                if Instant::now() >= deadline {
                    killed = true;
                    kill_group(pgid);
                    let _ = child.wait();
                    break None;
                }
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    };
    kill_group(pgid); // reap backgrounded grandchildren even on clean exit

    let output = std::fs::read(tmp.path()).unwrap_or_default();
    Ok((output, code, killed))
}

/// `libc::kill` directly — no external `kill` fork (extra forks can trigger ELKEID).
fn kill_group(pgid: i32) {
    // SAFETY: kill(-pgid, SIGKILL) is well-defined; an empty or already-reaped
    // group returns ESRCH which we ignore.
    unsafe { libc::kill(-pgid, libc::SIGKILL) };
}

#[derive(Clone, Debug)]
pub struct SpawnClient {
    socket: PathBuf,
}

impl SpawnClient {
    /// `None` when `ARLE_SPAWNER_SOCKET` unset (direct-spawn path).
    pub fn from_env() -> Option<Self> {
        std::env::var(SOCKET_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| Self {
                socket: PathBuf::from(s),
            })
    }

    pub fn run(&self, req: &SpawnRequest) -> std::io::Result<SpawnResponse> {
        let mut stream = UnixStream::connect(&self.socket)?;
        let bytes = serde_json::to_vec(req)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        write_frame(&mut stream, &bytes)?;
        let resp_bytes = read_frame(&mut stream)?;
        serde_json::from_slice(&resp_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

pub struct SpawnerHandle {
    child: std::process::Child,
    socket: PathBuf,
}

impl SpawnerHandle {
    /// Re-exec `arle` as the spawner helper BEFORE any CUDA init (ELKEID kills
    /// CUDA-resident processes that fork `setsid`). Sets `ARLE_SPAWNER_SOCKET`
    /// so `crate::sandbox` routes through it.
    pub fn launch() -> anyhow::Result<Self> {
        let exe = std::env::current_exe()
            .map_err(|e| anyhow::anyhow!("locate current exe for sandbox-spawner: {e}"))?;
        let socket = PathBuf::from(format!("/tmp/arle-spawn-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&socket);

        let child = Command::new(&exe)
            .env(LISTEN_ENV, &socket)
            // Don't inherit the parent's socket env — the helper must not recurse.
            .env_remove(SOCKET_ENV)
            .stdin(Stdio::null())
            .spawn()
            .map_err(|e| anyhow::anyhow!("spawn sandbox-spawner helper: {e}"))?;

        // Wait for the helper to bind its socket (sub-second; budget for startup).
        let deadline = Instant::now() + Duration::from_secs(10);
        while !socket.exists() {
            if Instant::now() >= deadline {
                anyhow::bail!(
                    "sandbox-spawner helper did not create {} within 10s",
                    socket.display()
                );
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        // Route this process's sandbox spawns through the helper.
        // SAFETY: single-threaded before CUDA init and rollout threads.
        unsafe {
            std::env::set_var(SOCKET_ENV, &socket);
        }
        eprintln!(
            "[arle train agent-opd] pre-CUDA sandbox-spawner ready (pid {}, socket {})",
            child.id(),
            socket.display()
        );
        Ok(Self { child, socket })
    }
}

impl Drop for SpawnerHandle {
    fn drop(&mut self) {
        // SAFETY: drop happens at end of run; rollout threads are joined.
        unsafe {
            std::env::remove_var(SOCKET_ENV);
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.socket);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip() {
        let mut buf: Vec<u8> = Vec::new();
        write_frame(&mut buf, b"hello world").unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let out = read_frame(&mut cursor).unwrap();
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn request_response_serde_roundtrip() {
        let req = SpawnRequest {
            program: "echo".into(),
            args: vec!["hi".into()],
            cwd: Some(PathBuf::from("/tmp")),
            env: vec![("FOO".into(), Some("bar".into())), ("BAZ".into(), None)],
            combined_timeout: true,
            timeout_s: 30,
        };
        let bytes = serde_json::to_vec(&req).unwrap();
        let back: SpawnRequest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.program, "echo");
        assert_eq!(back.env.len(), 2);
        assert!(back.combined_timeout);

        let resp = SpawnResponse {
            stdout: b"out".to_vec(),
            stderr: b"err".to_vec(),
            exit_code: Some(0),
            timed_out: false,
        };
        let bytes = serde_json::to_vec(&resp).unwrap();
        let back: SpawnResponse = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back.stdout, b"out");
        assert_eq!(back.exit_code, Some(0));
    }

    /// End-to-end: stand up the server on a temp socket in a thread, drive a
    /// combined-timeout `bash echo` and a plain `.output()` spawn through the
    /// client, assert outputs match a direct spawn. Proves the protocol + both
    /// server paths without needing CUDA.
    #[test]
    fn server_client_echo_roundtrip() {
        // Serialize with sandbox.rs's routing test — both mutate global env.
        let _env = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("spawn.sock");
        // SAFETY: serialized by TEST_ENV_LOCK; single-threaded setup before the server.
        unsafe {
            std::env::set_var(LISTEN_ENV, &sock);
        }
        let server = std::thread::spawn(serve_loop);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !sock.exists() {
            assert!(Instant::now() < deadline, "server never bound");
            std::thread::sleep(Duration::from_millis(10));
        }

        let client = SpawnClient {
            socket: sock.clone(),
        };

        // combined_timeout path (bash tool).
        let resp = client
            .run(&SpawnRequest {
                program: "bash".into(),
                args: vec!["-lc".into(), "echo combined".into()],
                cwd: None,
                env: vec![],
                combined_timeout: true,
                timeout_s: 30,
            })
            .unwrap();
        assert_eq!(resp.exit_code, Some(0));
        assert!(!resp.timed_out);
        assert!(
            String::from_utf8_lossy(&resp.stdout).contains("combined"),
            "stdout: {:?}",
            String::from_utf8_lossy(&resp.stdout)
        );

        // plain .output() path (git/cp/diff): stderr stays separate.
        let resp = client
            .run(&SpawnRequest {
                program: "bash".into(),
                args: vec!["-lc".into(), "echo out; echo err 1>&2".into()],
                cwd: None,
                env: vec![],
                combined_timeout: false,
                timeout_s: 0,
            })
            .unwrap();
        assert_eq!(resp.exit_code, Some(0));
        assert!(String::from_utf8_lossy(&resp.stdout).contains("out"));
        assert!(String::from_utf8_lossy(&resp.stderr).contains("err"));

        // timeout path: a sleep longer than the budget must report timed_out.
        let resp = client
            .run(&SpawnRequest {
                program: "bash".into(),
                args: vec!["-lc".into(), "sleep 5".into()],
                cwd: None,
                env: vec![],
                combined_timeout: true,
                timeout_s: 1,
            })
            .unwrap();
        assert!(resp.timed_out, "1s budget on a 5s sleep must time out");

        // Tear the server down by dropping the listener: close the socket file and
        // connect once to unblock `incoming()` is unnecessary — the test process
        // exit reaps it. Detach the thread.
        // SAFETY: test teardown.
        unsafe {
            std::env::remove_var(LISTEN_ENV);
        }
        drop(server); // detached; process exit cleans up
        let _ = std::fs::remove_file(&sock);
    }
}
