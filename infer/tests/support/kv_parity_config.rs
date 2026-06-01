use anyhow::{Context, Result};

pub(crate) const FULL_DEFAULT_MAX_TOKENS: usize = 64;
pub(crate) const SMOKE_DEFAULT_MAX_TOKENS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KvParityProfile {
    Full,
    Smoke,
}

fn parse_kv_parity_profile(value: Option<&str>) -> Result<KvParityProfile> {
    match value.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(KvParityProfile::Full),
        Some(profile) if profile.eq_ignore_ascii_case("full") => Ok(KvParityProfile::Full),
        Some(profile) if profile.eq_ignore_ascii_case("smoke") => Ok(KvParityProfile::Smoke),
        Some(other) => {
            anyhow::bail!("unsupported KV_PARITY_PROFILE={other:?}; expected 'full' or 'smoke'")
        }
    }
}

pub(crate) fn max_tokens_for_config(
    max_tokens_env: Option<&str>,
    profile_env: Option<&str>,
) -> Result<usize> {
    if let Some(raw) = max_tokens_env {
        let parsed = raw
            .parse::<usize>()
            .with_context(|| format!("parse KV_PARITY_MAX_TOKENS={raw:?}"))?;
        anyhow::ensure!(parsed > 0, "KV_PARITY_MAX_TOKENS must be > 0");
        return Ok(parsed);
    }
    Ok(match parse_kv_parity_profile(profile_env)? {
        KvParityProfile::Full => FULL_DEFAULT_MAX_TOKENS,
        KvParityProfile::Smoke => SMOKE_DEFAULT_MAX_TOKENS,
    })
}

#[allow(dead_code)]
pub(crate) fn max_tokens_from_env() -> Result<usize> {
    let max_tokens_env = std::env::var("KV_PARITY_MAX_TOKENS").ok();
    let profile_env = std::env::var("KV_PARITY_PROFILE").ok();
    max_tokens_for_config(max_tokens_env.as_deref(), profile_env.as_deref())
}
