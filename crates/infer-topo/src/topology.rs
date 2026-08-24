//! Multi-axis parallelism topology (TP/PP + attention-DP/CP) and rank-group
//! math.
//!
//! Port of SGLang `parallel_state.py` / `dp_attention.py` (per-function line
//! refs kept on the comments below).

use crate::error::{Result, bail};
use crate::sharding::parse_parallel_env_usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiAxisConfig {
    pub tp_size: usize,
    pub pp_size: usize,
    pub attn_dp_size: usize,
    pub attn_cp_size: usize,
}

impl MultiAxisConfig {
    #[must_use]
    pub fn single() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
        }
    }

    /// This does not by itself enable DP/CP execution. It is the runtime
    /// contract input used by DSv4 startup diagnostics and fail-closed guards so
    /// a run cannot silently claim a SGLang-equivalent layout while only wiring
    /// global TP.
    pub fn current_route_from_env_with_defaults(tp_world_size: usize) -> Result<Self> {
        Self::from_lookup_with_defaults(tp_world_size, 1, |key| std::env::var(key).ok())
    }

    fn from_lookup_with_defaults(
        tp_default: usize,
        pp_default: usize,
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self> {
        let cfg = Self {
            tp_size: parse_parallel_env_usize("INFER_TP_SIZE", None, tp_default, &mut lookup)?,
            pp_size: parse_parallel_env_usize(
                "INFER_PP_SIZE",
                Some("ARLE_PP_SIZE"),
                pp_default,
                &mut lookup,
            )?,
            attn_dp_size: parse_parallel_env_usize(
                "INFER_ATTN_DP_SIZE",
                Some("ARLE_ATTN_DP_SIZE"),
                1,
                &mut lookup,
            )?,
            attn_cp_size: parse_parallel_env_usize("INFER_ATTN_CP_SIZE", None, 1, &mut lookup)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "tp={} pp={} attn_dp={} attn_cp={} attn_tp={} world={}",
            self.tp_size,
            self.pp_size,
            self.attn_dp_size,
            self.attn_cp_size,
            self.attn_tp_size(),
            self.world_size(),
        )
    }

    /// SGLang `parallel_state.py:1781,1827-1829`.
    pub fn validate(&self) -> Result<()> {
        if self.tp_size == 0
            || self.pp_size == 0
            || self.attn_dp_size == 0
            || self.attn_cp_size == 0
        {
            bail!(
                "all axis sizes must be >= 1 (tp={}, pp={}, attn_dp={}, attn_cp={})",
                self.tp_size,
                self.pp_size,
                self.attn_dp_size,
                self.attn_cp_size,
            );
        }
        let attn_div = self.attn_dp_size * self.attn_cp_size;
        if !self.tp_size.is_multiple_of(attn_div) {
            bail!(
                "assert tp_size % (attn_dp_size * attn_cp_size) == 0 failed: tp={}, attn_dp={}, attn_cp={}",
                self.tp_size,
                self.attn_dp_size,
                self.attn_cp_size,
            );
        }
        Ok(())
    }

    /// SGLang `parallel_state.py:1781`.
    #[must_use]
    pub fn world_size(&self) -> usize {
        self.tp_size * self.pp_size
    }

    /// SGLang `parallel_state.py:1829`.
    #[must_use]
    pub fn attn_tp_size(&self) -> usize {
        self.tp_size / self.attn_dp_size / self.attn_cp_size
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankCoord {
    pub attn_tp_rank: usize,
    pub attn_dp_rank: usize,
    pub attn_cp_rank: usize,
}

impl RankCoord {
    /// SGLang `dp_attention.py:240-254` + `parallel_state.py:1789,1981`.
    pub fn from_world_rank(cfg: MultiAxisConfig, world_rank: usize) -> Result<Self> {
        cfg.validate()?;
        let world = cfg.world_size();
        if world_rank >= world {
            bail!("world_rank ({world_rank}) must be < world_size ({world})");
        }
        let tp_rank = world_rank % cfg.tp_size;
        let attn_tp = cfg.attn_tp_size();
        let attn_tp_rank = tp_rank % attn_tp;
        let attn_cp_rank = (tp_rank / attn_tp) % cfg.attn_cp_size;
        let attn_dp_rank = tp_rank / (attn_tp * cfg.attn_cp_size);
        Ok(Self {
            attn_tp_rank,
            attn_dp_rank,
            attn_cp_rank,
        })
    }
}

/// SGLang `parallel_state.py:1789-1800`.
#[must_use]
pub fn build_tp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let n = cfg.world_size() / cfg.tp_size;
    (0..n)
        .map(|i| (i * cfg.tp_size..(i + 1) * cfg.tp_size).collect())
        .collect()
}

/// SGLang `parallel_state.py:1838-1853`.
#[must_use]
pub fn build_attn_cp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.attn_cp_size == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let n = cfg.world_size() / cfg.tp_size;
    let attn_tp = cfg.attn_tp_size();
    (0..n)
        .flat_map(|g| {
            (0..cfg.attn_dp_size).flat_map(move |d| {
                (0..attn_tp).map(move |t| {
                    let st = g * cfg.tp_size + d * attn_tp * cfg.attn_cp_size + t;
                    let en = g * cfg.tp_size + (d + 1) * attn_tp * cfg.attn_cp_size + t;
                    (st..en).step_by(attn_tp).collect()
                })
            })
        })
        .collect()
}

/// SGLang `parallel_state.py:1871-1883`.
#[must_use]
pub fn build_attn_tp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let attn_tp = cfg.attn_tp_size();
    if attn_tp == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let n = cfg.world_size() / cfg.tp_size;
    (0..n)
        .flat_map(|g| {
            (0..cfg.attn_cp_size * cfg.attn_dp_size).map(move |i| {
                let st = g * cfg.tp_size + i * attn_tp;
                (st..st + attn_tp).collect()
            })
        })
        .collect()
}
