//! Glue that lets the no-args `arle` front door pick a model, serve it
//! locally, and hand the session off to the **Eli** agent framework
//! (`../eli`, a sibling Rust binary) pointed at that local OpenAI-compatible
//! endpoint.
//!
//! ## Routing (the load-bearing facts, proven empirically 2026-06-22)
//!
//! Eli's provider layer (`nexil`) has a dedicated keyless **`local`** provider
//! that collapses the `arle` / `ollama` / `vllm` / `lmstudio` /
//! `llama.cpp` aliases onto one OpenAI-Chat-Completions endpoint
//! (`crates/nexil/src/core/provider_registry.rs`). To route Eli at a local
//! `arle serve` we set, env-only (NEVER touching `~/.eli/config.toml`):
//!
//! | env | value |
//! |-----|-------|
//! | `ELI_API_BASE`   | `http://127.0.0.1:<port>/v1` |
//! | `ELI_API_KEY`    | `local` (any non-empty; both KEY+BASE set → single override for all providers) |
//! | `ELI_MODEL`      | `local:<served-id>` |
//! | `ELI_API_FORMAT` | `completion` (chat-completions; `messages` is Anthropic-only) |
//! | `ELI_FALLBACK_MODELS` | *empty* — **disables** Eli's auto cross-provider rollover |
//!
//! `ELI_FALLBACK_MODELS=""` is load-bearing: without it, an Eli call that fails
//! against the local endpoint silently rolls over to the user's deepseek /
//! anthropic profiles (confirmed: a dead local port drifted to
//! `deepseek:deepseek-v4-pro`). The empty list pins Eli to the local model.
//!
//! ## Lifecycle
//!
//! [`launch_eli`] starts `arle serve` (in-process server) as a **child**, polls
//! `/v1/models` until ready, reads the served id, then runs `eli chat` (or
//! `eli gateway`) with the env above and inherited stdio. A [`ServeGuard`]
//! kills the serve child on Eli exit OR SIGINT, so no orphaned server survives.

use std::io::{BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use console::style;

/// Base port for the free-port scan. Stays clear of the `arle serve` default
/// (8000) so a manually-launched serve does not collide with our auto pick.
const PORT_SCAN_BASE: u16 = 8000;
/// How many consecutive ports to probe before giving up.
const PORT_SCAN_SPAN: u16 = 200;
/// Max time to wait for the spawned serve to answer `/v1/models`.
const SERVE_READY_TIMEOUT: Duration = Duration::from_secs(180);

/// ARLE's own agent preference, persisted at
/// `${XDG_CONFIG_HOME:-$HOME/.config}/arle/agent.toml`.
///
/// Distinct from `~/.eli/config.toml` (Eli's own config, which we never
/// touch). This file only records *which* agent `arle` should launch and the
/// model + mode to launch it with, so a second no-args run goes straight to
/// the agent ("下次默认启动 eli").
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct AgentConfig {
    /// `"eli"` when Eli is the default agent. Other values (or absent) mean
    /// fall back to the built-in REPL.
    pub(crate) agent: Option<String>,
    /// Model source (local dir path or HF id) to serve.
    pub(crate) model: Option<String>,
    /// Launch mode: `"chat"` (interactive REPL) or `"gateway"` (serve mode).
    pub(crate) mode: Option<String>,
}

impl AgentConfig {
    /// `true` when this config selects Eli as the default agent AND carries a
    /// model to serve — i.e. a subsequent run can launch directly.
    pub(crate) fn selects_eli(&self) -> bool {
        self.agent.as_deref() == Some("eli")
            && self.model.as_deref().is_some_and(|m| !m.trim().is_empty())
    }

    /// Resolve the launch mode, defaulting to `Chat`.
    pub(crate) fn launch_mode(&self) -> EliMode {
        match self.mode.as_deref() {
            Some("gateway") => EliMode::Gateway,
            _ => EliMode::Chat,
        }
    }

    /// Serialize to the minimal flat TOML this module reads back. Only the
    /// three string keys are emitted; absent keys are omitted.
    pub(crate) fn to_toml(&self) -> String {
        let mut out = String::new();
        let mut push = |k: &str, v: &Option<String>| {
            if let Some(val) = v {
                // Values here are agent names / model ids / mode words — no
                // embedded quotes or newlines in practice. Escape defensively.
                let escaped = val.replace('\\', "\\\\").replace('"', "\\\"");
                out.push_str(&format!("{k} = \"{escaped}\"\n"));
            }
        };
        push("agent", &self.agent);
        push("model", &self.model);
        push("mode", &self.mode);
        out
    }

    /// Parse the minimal flat TOML produced by [`to_toml`](Self::to_toml).
    /// Tolerant of comments (`#`), blank lines, and unknown keys.
    pub(crate) fn from_toml(text: &str) -> Self {
        let mut cfg = AgentConfig::default();
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let val = unquote_toml(val.trim());
            match key {
                "agent" => cfg.agent = Some(val),
                "model" => cfg.model = Some(val),
                "mode" => cfg.mode = Some(val),
                _ => {}
            }
        }
        cfg
    }

    /// Load from the default config path. Returns the default (all-`None`)
    /// config when the file is absent or unreadable — a missing config is not
    /// an error, it just means "no default agent yet".
    pub(crate) fn load() -> Self {
        match agent_config_path().and_then(|p| std::fs::read_to_string(p).ok()) {
            Some(text) => Self::from_toml(&text),
            None => AgentConfig::default(),
        }
    }

    /// Persist to the default config path, creating parent dirs as needed.
    pub(crate) fn save(&self) -> Result<()> {
        let path = agent_config_path()
            .ok_or_else(|| anyhow!("cannot resolve arle config dir ($HOME / $XDG_CONFIG_HOME)"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating config dir {}", parent.display()))?;
        }
        std::fs::write(&path, self.to_toml())
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

/// Strip surrounding double quotes and unescape `\"` / `\\` from a TOML scalar.
fn unquote_toml(s: &str) -> String {
    let inner = s
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(s);
    inner.replace("\\\"", "\"").replace("\\\\", "\\")
}

/// `${XDG_CONFIG_HOME:-$HOME/.config}/arle/agent.toml`. Mirrors
/// `welcome::config_home()` so the agent.toml lives beside the `seen` marker.
fn agent_config_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(x) if !x.is_empty() => PathBuf::from(x),
        _ => PathBuf::from(std::env::var_os("HOME")?).join(".config"),
    };
    Some(base.join("arle").join("agent.toml"))
}

/// How to run Eli once the local model is serving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EliMode {
    /// Interactive REPL (`eli chat`). Stdio inherited.
    Chat,
    /// Gateway / serve mode (`eli gateway`) — message-listener daemon.
    Gateway,
}

impl EliMode {
    fn subcommand(self) -> &'static str {
        match self {
            EliMode::Chat => "chat",
            EliMode::Gateway => "gateway",
        }
    }

    /// The string persisted into `agent.toml`'s `mode` key.
    pub(crate) fn config_value(self) -> &'static str {
        match self {
            EliMode::Chat => "chat",
            EliMode::Gateway => "gateway",
        }
    }
}

/// Scan `[base, base+span)` for a TCP port that is currently free to bind on
/// `127.0.0.1`. Returns the first free port, so "端口被占用" is handled by
/// simply skipping to the next candidate.
pub(crate) fn find_free_port(base: u16, span: u16) -> Result<u16> {
    for offset in 0..span {
        let Some(port) = base.checked_add(offset) else {
            break;
        };
        if port_is_free(port) {
            return Ok(port);
        }
    }
    bail!(
        "no free TCP port found in [{base}, {}); is something holding the whole range?",
        base.saturating_add(span)
    )
}

/// `true` when a fresh `TcpListener` can bind `127.0.0.1:port`. The listener is
/// dropped immediately, freeing the port for the child `arle serve` to claim.
/// There is an inherent TOCTOU window between this probe and the child's bind,
/// but the scan base is chosen to avoid the common defaults, so a collision is
/// rare and surfaces as a clean child-exit error rather than a silent hang.
fn port_is_free(port: u16) -> bool {
    TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).is_ok()
}

/// Locate the `eli` executable. Resolution order:
/// 1. `$ELI_BIN` (explicit override),
/// 2. `eli` on `$PATH`,
/// 3. the sibling build at `../eli/target/release/eli` relative to the arle
///    binary's grandparent workspace, and a couple of common dev layouts.
///
/// Returns a friendly buildable-hint error when none resolve, so a missing Eli
/// does not hard-fail the whole `arle` process.
pub(crate) fn find_eli_binary() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("ELI_BIN") {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Ok(p);
        }
    }

    if let Some(p) = which_on_path("eli") {
        return Ok(p);
    }

    for cand in sibling_eli_candidates() {
        if cand.is_file() {
            return Ok(cand);
        }
    }

    bail!(
        "Eli not found — build it: cd ../eli && cargo build --release\n\
         (looked on $PATH, $ELI_BIN, and ../eli/target/release/eli)"
    )
}

/// Probe `$PATH` for an executable named `name`.
fn which_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

/// Candidate sibling locations for a locally-built Eli, relative to the running
/// `arle` binary. The arle binary normally lives at
/// `<ws>/target/release/arle`, so the sibling eli is `<ws>/../eli/target/...`.
fn sibling_eli_candidates() -> Vec<PathBuf> {
    // exe = <ws>/target/<profile>/arle → climb to <ws> then to <ws>/..
    let mut out = std::env::current_exe()
        .ok()
        .and_then(|exe| {
            let ws = exe.parent()?.parent()?.parent()?;
            let parent = ws.parent()?;
            Some(vec![
                parent.join("eli/target/release/eli"),
                parent.join("eli/target/debug/eli"),
            ])
        })
        .unwrap_or_default();
    out.push(PathBuf::from("../eli/target/release/eli")); // CWD-relative dev fallback
    out
}

/// Owns the spawned `arle serve` child and guarantees it is killed on drop
/// (Eli exit) or on SIGINT (the installed Ctrl-C handler flips `interrupted`).
struct ServeGuard {
    child: Child,
    port: u16,
}

impl ServeGuard {
    /// Best-effort terminate. Idempotent; safe to call from `drop`.
    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for ServeGuard {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Spawn `arle serve --backend <backend> --model-path <model> --port <port>`
/// as a background child, using the **current** executable so the child is the
/// exact same binary (in-process serve; there is no standalone serve binary).
fn spawn_serve(model: &str, port: u16, backend: &str) -> Result<Child> {
    let exe = std::env::current_exe().context("resolving current arle executable")?;
    let child = Command::new(exe)
        .arg("serve")
        .arg("--backend")
        .arg(backend)
        .arg("--model-path")
        .arg(model)
        .arg("--port")
        .arg(port.to_string())
        // Bind loopback explicitly; the served endpoint is local-only.
        .arg("--bind")
        .arg("127.0.0.1")
        .stdin(Stdio::null())
        // Inherit so serve diagnostics/errors are visible while we wait.
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning `arle serve` on port {port}"))?;
    Ok(child)
}

/// Poll `http://127.0.0.1:<port>/v1/models` until it returns a non-empty model
/// id, the child exits, the interrupt flag flips, or the timeout elapses.
///
/// Returns the served model id on success. If the child exits first (e.g. the
/// memory guard rejected the load) we surface a clear, actionable error instead
/// of spinning until timeout.
fn wait_for_serve(guard: &mut ServeGuard, interrupted: &AtomicBool) -> Result<String> {
    let deadline = Instant::now() + SERVE_READY_TIMEOUT;
    loop {
        if interrupted.load(Ordering::SeqCst) {
            bail!("interrupted while waiting for the local model to load");
        }
        // Child died before becoming ready → load failed.
        match guard.child.try_wait() {
            Ok(Some(status)) => {
                bail!(
                    "model failed to load: `arle serve` exited ({status}) before becoming ready; \
                     try a smaller model (see the serve output above for the reason)"
                );
            }
            Ok(None) => {}
            Err(e) => bail!("failed to poll serve child status: {e}"),
        }

        if let Some(id) = query_served_model(guard.port)
            && !id.trim().is_empty()
        {
            return Ok(id);
        }

        if Instant::now() >= deadline {
            bail!(
                "local model did not become ready within {}s on port {}",
                SERVE_READY_TIMEOUT.as_secs(),
                guard.port
            );
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Minimal blocking GET of `/v1/models`, extracting the first `"id"` value.
/// Returns `None` on any connection / parse miss (server not up yet). Kept
/// dependency-free (raw `TcpStream`) so the readiness poll has no async runtime
/// and no extra HTTP client allocation per attempt.
fn query_served_model(port: u16) -> Option<String> {
    let addr = SocketAddrV4::new(Ipv4Addr::LOCALHOST, port);
    let mut stream = TcpStream::connect_timeout(&addr.into(), Duration::from_millis(500)).ok()?;
    stream
        .set_read_timeout(Some(Duration::from_millis(800)))
        .ok()?;
    let req = format!(
        "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n\r\n"
    );
    stream.write_all(req.as_bytes()).ok()?;
    let mut body = String::new();
    BufReader::new(stream).read_to_string(&mut body).ok()?;
    parse_first_model_id(&body)
}

/// Pull the first `"id":"..."` out of an OpenAI `/v1/models` JSON body. A tiny
/// hand parser avoids a serde_json round-trip on the raw HTTP response (which
/// includes headers before the JSON). Returns `None` when no id is present.
fn parse_first_model_id(http_response: &str) -> Option<String> {
    // Skip HTTP headers: JSON starts at the first '{'.
    let json_start = http_response.find('{')?;
    let json = &http_response[json_start..];
    let key = "\"id\"";
    let idx = json.find(key)?;
    let after = &json[idx + key.len()..];
    let colon = after.find(':')?;
    let after_colon = after[colon + 1..].trim_start();
    let rest = after_colon.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// The env var set that routes Eli at the local OpenAI-compatible endpoint.
/// Pure function (no process env mutation) so it is unit-testable.
pub(crate) fn eli_env(port: u16, served_id: &str) -> Vec<(String, String)> {
    vec![
        ("ELI_API_BASE".into(), format!("http://127.0.0.1:{port}/v1")),
        ("ELI_API_KEY".into(), "local".into()),
        ("ELI_MODEL".into(), format!("local:{served_id}")),
        ("ELI_API_FORMAT".into(), "completion".into()),
        // Empty → disable Eli's auto cross-provider fallback (otherwise a
        // local failure silently rolls over to deepseek / anthropic).
        ("ELI_FALLBACK_MODELS".into(), String::new()),
    ]
}

/// Serve `model` locally on a free port and hand the session to Eli.
///
/// Steps: locate eli → pick free port → spawn `arle serve` child → wait for
/// `/v1/models` → run `eli <chat|gateway>` with [`eli_env`] (stdio inherited
/// for chat) → kill the serve child on Eli exit or SIGINT.
///
/// `model` is a local dir path or HF id (whatever the picker resolved).
pub(crate) fn launch_eli(model: &str, mode: EliMode, backend: &str) -> Result<()> {
    // Resolve eli up front so a missing binary fails *before* we spawn a serve.
    let eli_bin = find_eli_binary()?;

    let port = find_free_port(PORT_SCAN_BASE, PORT_SCAN_SPAN)?;
    eprintln!(
        "  {} {} on 127.0.0.1:{}",
        style("serving").green().bold(),
        style(model).bold(),
        port
    );

    let child = spawn_serve(model, port, backend)?;
    let mut guard = ServeGuard { child, port };

    // Ctrl-C handler: flip the flag so the readiness loop and Eli wait can both
    // unwind, and `guard` drops → serve child is killed. We do NOT call any
    // pkill — cleanup is scoped to *our* child only.
    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&interrupted);
        // ctrlc::set_handler errors only if a handler is already installed
        // (e.g. the REPL path set one earlier in this process). That is
        // non-fatal here: the existing handler plus our Drop still clean up.
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        });
    }

    let served_id = match wait_for_serve(&mut guard, &interrupted) {
        Ok(id) => id,
        Err(e) => {
            // guard drops at function return → serve child killed.
            return Err(e);
        }
    };

    eprintln!(
        "  {} model `{}` via Eli ({})",
        style("ready").green().bold(),
        style(&served_id).bold(),
        mode.subcommand()
    );

    let env = eli_env(port, &served_id);
    let mut cmd = Command::new(&eli_bin);
    cmd.arg(mode.subcommand());
    for (k, v) in &env {
        cmd.env(k, v);
    }
    // Inherit stdio: chat is interactive; gateway streams its own logs.
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd.status().with_context(|| {
        format!(
            "launching eli ({}) at {}",
            mode.subcommand(),
            eli_bin.display()
        )
    });

    // Whether eli succeeded, errored, or we were interrupted, kill the serve
    // child now (explicit, in addition to the Drop, so the order is obvious).
    guard.kill();

    let status = status?;
    if !status.success() && !interrupted.load(Ordering::SeqCst) {
        bail!("eli exited with status {status}");
    }
    Ok(())
}
