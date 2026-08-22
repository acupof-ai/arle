//! Glue that lets the no-args `arle` front door pick a model, serve it
//! locally, and hand the session off to the **Eli** agent framework
//! (`../eli`, a sibling Rust binary) pointed at that local OpenAI-compatible
//! endpoint.
//!
//! Routing is env-only (NEVER touch `~/.eli/config.toml`); see [`eli_env`].
//! `ELI_FALLBACK_MODELS=""` is load-bearing: without it, an Eli call that
//! fails against the local endpoint silently rolls over to the user's
//! deepseek / anthropic profiles (confirmed: a dead local port drifted to
//! `deepseek:deepseek-v4-pro`). The empty list pins Eli to the local model.

use std::io::{BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use console::style;

/// Stays clear of the `arle serve` default (8000) so a manually-launched
/// serve does not collide with our auto pick.
const PORT_SCAN_BASE: u16 = 8000;
const PORT_SCAN_SPAN: u16 = 200;
const SERVE_READY_TIMEOUT: Duration = Duration::from_secs(180);

/// ARLE's own agent preference, persisted at
/// `${XDG_CONFIG_HOME:-$HOME/.config}/arle/agent.toml` — distinct from
/// `~/.eli/config.toml` (Eli's own config, which we never touch).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct AgentConfig {
    /// `"eli"` when Eli is the default agent. Other values (or absent) mean
    /// fall back to the built-in REPL.
    pub(crate) agent: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) mode: Option<String>,
}

impl AgentConfig {
    pub(crate) fn selects_eli(&self) -> bool {
        self.agent.as_deref() == Some("eli")
            && self.model.as_deref().is_some_and(|m| !m.trim().is_empty())
    }

    pub(crate) fn launch_mode(&self) -> EliMode {
        match self.mode.as_deref() {
            Some("gateway") => EliMode::Gateway,
            _ => EliMode::Chat,
        }
    }

    pub(crate) fn to_toml(&self) -> String {
        let mut out = String::new();
        let mut push = |k: &str, v: &Option<String>| {
            if let Some(val) = v {
                // No embedded quotes/newlines in practice; escape defensively.
                let escaped = val.replace('\\', "\\\\").replace('"', "\\\"");
                out.push_str(&format!("{k} = \"{escaped}\"\n"));
            }
        };
        push("agent", &self.agent);
        push("model", &self.model);
        push("mode", &self.mode);
        out
    }

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

    /// A missing config is not an error — it just means "no default agent yet".
    pub(crate) fn load() -> Self {
        match agent_config_path().and_then(|p| std::fs::read_to_string(p).ok()) {
            Some(text) => Self::from_toml(&text),
            None => AgentConfig::default(),
        }
    }

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

fn unquote_toml(s: &str) -> String {
    let inner = s
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(s);
    inner.replace("\\\"", "\"").replace("\\\\", "\\")
}

/// Mirrors `welcome::config_home()` so agent.toml lives beside the `seen` marker.
fn agent_config_path() -> Option<PathBuf> {
    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(x) if !x.is_empty() => PathBuf::from(x),
        _ => PathBuf::from(std::env::var_os("HOME")?).join(".config"),
    };
    Some(base.join("arle").join("agent.toml"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EliMode {
    Chat,
    Gateway,
}

impl EliMode {
    fn subcommand(self) -> &'static str {
        match self {
            EliMode::Chat => "chat",
            EliMode::Gateway => "gateway",
        }
    }

    pub(crate) fn config_value(self) -> &'static str {
        match self {
            EliMode::Chat => "chat",
            EliMode::Gateway => "gateway",
        }
    }
}

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

/// The listener is dropped immediately, freeing the port for the child's bind.
/// The probe→bind TOCTOU window is acceptable: the scan base avoids common
/// defaults, so a collision is rare and surfaces as a clean child-exit error
/// rather than a silent hang.
fn port_is_free(port: u16) -> bool {
    TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).is_ok()
}

/// Resolution order: `$ELI_BIN`, then `eli` on `$PATH`, then sibling dev builds.
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

/// Uses the current executable so the child is the exact same binary (there is
/// no standalone serve binary).
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

/// Returns the served model id. A child exit before ready surfaces as a clear
/// load error instead of a spin until timeout.
fn wait_for_serve(guard: &mut ServeGuard, interrupted: &AtomicBool) -> Result<String> {
    let deadline = Instant::now() + SERVE_READY_TIMEOUT;
    loop {
        if interrupted.load(Ordering::SeqCst) {
            bail!("interrupted while waiting for the local model to load");
        }
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

/// Dependency-free (raw `TcpStream`): the readiness poll must not pull in an
/// async runtime or allocate an HTTP client per attempt.
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

/// Hand parser: the raw response includes HTTP headers before the JSON, and a
/// serde_json round-trip per poll attempt is not worth it.
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
        Err(e) => return Err(e),
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
