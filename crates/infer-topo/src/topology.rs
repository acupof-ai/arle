//! Multi-axis parallelism topology (TP/PP/EP + attention-DP/CP + MoE-DP) and
//! rank-group math.
//!
//! Port of SGLang `parallel_state.py` / `dp_attention.py` (per-function line
//! refs kept on the comments below). Device-coupled NCCL collectives land in a
//! later H20-gated phase.

use crate::error::{Result, bail};
use crate::sharding::parse_parallel_env_usize;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MultiAxisConfig {
    pub tp_size: usize,
    pub pp_size: usize,
    pub ep_size: usize,
    pub attn_dp_size: usize,
    pub attn_cp_size: usize,
    pub moe_dp_size: usize,
}

impl MultiAxisConfig {
    #[must_use]
    pub fn single() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        }
    }

    /// The only multi-rank DSv4 execution shape wired today: global TP and EP,
    /// no attention-DP/CP or MoE-DP subgroups.
    ///
    /// # Errors
    /// Errors if the resulting config fails [`Self::validate`].
    pub fn global_tp_ep(tp_size: usize, ep_size: usize) -> Result<Self> {
        let cfg = Self {
            tp_size,
            pp_size: 1,
            ep_size,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// This does not by itself enable DP/CP/MoE-DP execution. It is the runtime
    /// contract input used by DSv4 startup diagnostics and fail-closed guards so
    /// a run cannot silently claim a SGLang-equivalent layout while only wiring
    /// global TP/EP.
    ///
    /// # Errors
    /// Errors on a non-`usize` env value or a config that fails [`Self::validate`].
    pub fn from_env() -> Result<Self> {
        Self::from_env_with_defaults(1, 1, 1)
    }

    /// # Errors
    /// Errors on a non-`usize` env value or a config that fails [`Self::validate`].
    pub fn from_env_with_defaults(
        tp_default: usize,
        pp_default: usize,
        ep_default: usize,
    ) -> Result<Self> {
        Self::from_lookup_with_defaults(tp_default, pp_default, ep_default, |key| {
            std::env::var(key).ok()
        })
    }

    /// # Errors
    /// Errors on a non-`usize` env value or a config that fails [`Self::validate`].
    pub fn current_route_from_env_with_defaults(
        tp_world_size: usize,
        ep_world_size: usize,
    ) -> Result<Self> {
        Self::current_route_from_lookup_with_defaults(tp_world_size, ep_world_size, |key| {
            std::env::var(key).ok()
        })
    }

    #[cfg(test)]
    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Result<Self> {
        Self::from_lookup_with_defaults(1, 1, 1, &mut lookup)
    }

    fn current_route_from_lookup_with_defaults(
        tp_world_size: usize,
        ep_world_size: usize,
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self> {
        let axis_tp_default = tp_world_size.max(ep_world_size);
        match Self::from_lookup_with_defaults(axis_tp_default, 1, ep_world_size, &mut lookup) {
            Ok(cfg) => Ok(cfg),
            Err(err)
                if tp_world_size == 1
                    && ep_world_size > 1
                    && !subgroup_axis_env_present(&mut lookup) =>
            {
                // Current ARLE still supports a legacy EP-only override
                // (`tp=1, ep=world`). SGLang's axis math represents EP inside
                // the TP axis, so use `tp=world, ep=world` for diagnostics
                // while keeping the actual runtime TP config unchanged.
                Self::global_tp_ep(ep_world_size, ep_world_size).map_err(|fallback_err| {
                    crate::error::TopoError::new(format!(
                        "failed to build legacy EP-only multi-axis diagnostic fallback: {fallback_err}; original parse error: {err}"
                    ))
                })
            }
            Err(err) => Err(err),
        }
    }

    fn from_lookup_with_defaults(
        tp_default: usize,
        pp_default: usize,
        ep_default: usize,
        mut lookup: impl FnMut(&str) -> Option<String>,
    ) -> Result<Self> {
        let cfg = Self {
            tp_size: parse_parallel_env_usize(
                "INFER_TP_SIZE",
                "ARLE_TP_SIZE",
                tp_default,
                &mut lookup,
            )?,
            pp_size: parse_parallel_env_usize(
                "INFER_PP_SIZE",
                "ARLE_PP_SIZE",
                pp_default,
                &mut lookup,
            )?,
            ep_size: parse_parallel_env_usize(
                "INFER_EP_SIZE",
                "ARLE_EP_SIZE",
                ep_default,
                &mut lookup,
            )?,
            attn_dp_size: parse_parallel_env_usize(
                "INFER_ATTN_DP_SIZE",
                "ARLE_ATTN_DP_SIZE",
                1,
                &mut lookup,
            )?,
            attn_cp_size: parse_parallel_env_usize(
                "INFER_ATTN_CP_SIZE",
                "ARLE_ATTN_CP_SIZE",
                1,
                &mut lookup,
            )?,
            moe_dp_size: parse_parallel_env_usize(
                "INFER_MOE_DP_SIZE",
                "ARLE_MOE_DP_SIZE",
                1,
                &mut lookup,
            )?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "tp={} pp={} ep={} attn_dp={} attn_cp={} attn_tp={} moe_dp={} moe_tp={} world={}",
            self.tp_size,
            self.pp_size,
            self.ep_size,
            self.attn_dp_size,
            self.attn_cp_size,
            self.attn_tp_size(),
            self.moe_dp_size,
            self.moe_tp_size(),
            self.world_size(),
        )
    }

    /// The only multi-rank DSv4 execution shape wired today: global TP and
    /// global EP, with no attention-DP/CP or MoE-DP subgroups.
    #[must_use]
    pub fn is_global_tp_ep_only(&self, tp_size: usize, ep_size: usize) -> bool {
        self.tp_size == tp_size
            && self.pp_size == 1
            && self.ep_size == ep_size
            && self.attn_dp_size == 1
            && self.attn_cp_size == 1
            && self.moe_dp_size == 1
    }

    /// SGLang `parallel_state.py:1781,1827-1829,1897-1899`.
    ///
    /// # Errors
    /// Errors if any axis size is 0, if `tp_size % (attn_dp_size * attn_cp_size) != 0`,
    /// or if `tp_size % (ep_size * moe_dp_size) != 0`.
    pub fn validate(&self) -> Result<()> {
        if self.tp_size == 0
            || self.pp_size == 0
            || self.ep_size == 0
            || self.attn_dp_size == 0
            || self.attn_cp_size == 0
            || self.moe_dp_size == 0
        {
            bail!(
                "all axis sizes must be >= 1 (tp={}, pp={}, ep={}, attn_dp={}, attn_cp={}, moe_dp={})",
                self.tp_size,
                self.pp_size,
                self.ep_size,
                self.attn_dp_size,
                self.attn_cp_size,
                self.moe_dp_size,
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
        let moe_div = self.ep_size * self.moe_dp_size;
        if !self.tp_size.is_multiple_of(moe_div) {
            bail!(
                "assert tp_size % (ep_size * moe_dp_size) == 0 failed: tp={}, ep={}, moe_dp={}",
                self.tp_size,
                self.ep_size,
                self.moe_dp_size,
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

    /// SGLang `parallel_state.py:1899`.
    #[must_use]
    pub fn moe_tp_size(&self) -> usize {
        self.tp_size / self.ep_size / self.moe_dp_size
    }
}

fn subgroup_axis_env_present(lookup: &mut impl FnMut(&str) -> Option<String>) -> bool {
    [
        "INFER_PP_SIZE",
        "ARLE_PP_SIZE",
        "INFER_ATTN_DP_SIZE",
        "ARLE_ATTN_DP_SIZE",
        "INFER_ATTN_CP_SIZE",
        "ARLE_ATTN_CP_SIZE",
        "INFER_MOE_DP_SIZE",
        "ARLE_MOE_DP_SIZE",
    ]
    .into_iter()
    .any(|key| lookup(key).is_some())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankCoord {
    pub world_rank: usize,
    pub tp_rank: usize,
    pub pp_rank: usize,
    pub attn_tp_rank: usize,
    pub attn_dp_rank: usize,
    pub attn_cp_rank: usize,
    pub moe_tp_rank: usize,
    pub moe_ep_rank: usize,
    pub moe_dp_rank: usize,
}

impl RankCoord {
    /// SGLang `dp_attention.py:240-254` + `parallel_state.py:1789,1981`.
    ///
    /// # Errors
    /// Errors if `cfg` fails [`MultiAxisConfig::validate`] or
    /// `world_rank >= cfg.world_size()`.
    pub fn from_world_rank(cfg: MultiAxisConfig, world_rank: usize) -> Result<Self> {
        cfg.validate()?;
        let world = cfg.world_size();
        if world_rank >= world {
            bail!("world_rank ({world_rank}) must be < world_size ({world})");
        }
        let tp_rank = world_rank % cfg.tp_size;
        let pp_rank = world_rank / cfg.tp_size;
        let attn_tp = cfg.attn_tp_size();
        let attn_tp_rank = tp_rank % attn_tp;
        let attn_cp_rank = (tp_rank / attn_tp) % cfg.attn_cp_size;
        let attn_dp_rank = tp_rank / (attn_tp * cfg.attn_cp_size);
        let moe_tp = cfg.moe_tp_size();
        let moe_tp_rank = tp_rank % moe_tp;
        let moe_ep_rank = (tp_rank / moe_tp) % cfg.ep_size;
        let moe_dp_rank = tp_rank / (moe_tp * cfg.ep_size);
        Ok(Self {
            world_rank,
            tp_rank,
            pp_rank,
            attn_tp_rank,
            attn_dp_rank,
            attn_cp_rank,
            moe_tp_rank,
            moe_ep_rank,
            moe_dp_rank,
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

/// SGLang `parallel_state.py:1981-1989`.
#[must_use]
pub fn build_pp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let world = cfg.world_size();
    let n = world / cfg.pp_size;
    (0..n).map(|i| (i..world).step_by(n).collect()).collect()
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

/// SGLang `parallel_state.py:1838-1853` (attn_dp uses same outer layout as
/// attn_cp; each attn_dp group is a stride across `attn_cp_size * attn_tp_size`).
#[must_use]
pub fn build_attn_dp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.attn_dp_size == 1 {
        return (0..cfg.world_size()).map(|r| vec![r]).collect();
    }
    let n = cfg.world_size() / cfg.tp_size;
    let attn_tp = cfg.attn_tp_size();
    let stride = attn_tp * cfg.attn_cp_size;
    (0..n)
        .flat_map(|g| {
            (0..cfg.attn_cp_size).flat_map(move |c| {
                (0..attn_tp).map(move |t| {
                    let st = g * cfg.tp_size + c * attn_tp + t;
                    (st..g * cfg.tp_size + cfg.tp_size)
                        .step_by(stride)
                        .collect()
                })
            })
        })
        .collect()
}

/// SGLang `DataParallelController` request-ownership groups.
///
/// Work requests are routed to one attention-DP slice. The slice leader
/// receives from the controller, then broadcasts within ATTN_TP and ATTN_CP.
/// This is intentionally different from [`build_attn_dp_groups`]: that function
/// builds cross-DP communication groups for gather/scatter, while this function
/// builds per-DP compute-owner groups.
#[must_use]
pub fn build_attn_owner_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let n = cfg.world_size() / cfg.tp_size;
    let sz = cfg.attn_tp_size() * cfg.attn_cp_size;
    (0..n)
        .flat_map(|g| {
            (0..cfg.attn_dp_size).map(move |d| {
                let st = g * cfg.tp_size + d * sz;
                (st..st + sz).collect()
            })
        })
        .collect()
}

/// SGLang `parallel_state.py:1903-1919`.
#[must_use]
pub fn build_moe_dp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.attn_cp_size > cfg.moe_dp_size {
        return build_attn_cp_groups(cfg);
    }
    if cfg.moe_dp_size == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let n = cfg.world_size() / cfg.tp_size;
    let moe_tp = cfg.moe_tp_size();
    let stride = moe_tp * cfg.ep_size;
    (0..n)
        .flat_map(|g| {
            (0..moe_tp * cfg.ep_size).map(move |i| {
                let st = g * cfg.tp_size + i;
                (st..(g + 1) * cfg.tp_size + i).step_by(stride).collect()
            })
        })
        .collect()
}

/// SGLang `parallel_state.py:1929-1943`.
#[must_use]
pub fn build_moe_ep_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    if cfg.ep_size == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let n = cfg.world_size() / cfg.tp_size;
    let moe_tp = cfg.moe_tp_size();
    (0..n)
        .flat_map(|g| {
            (0..cfg.moe_dp_size).flat_map(move |d| {
                (0..moe_tp).map(move |t| {
                    let st = g * cfg.tp_size + d * cfg.ep_size * moe_tp + t;
                    (st..st + cfg.ep_size * moe_tp).step_by(moe_tp).collect()
                })
            })
        })
        .collect()
}

/// SGLang `parallel_state.py:1955-1970`.
#[must_use]
pub fn build_moe_tp_groups(cfg: MultiAxisConfig) -> Vec<Vec<usize>> {
    let moe_tp = cfg.moe_tp_size();
    if moe_tp == cfg.tp_size {
        return build_tp_groups(cfg);
    }
    let n = cfg.world_size() / cfg.tp_size;
    (0..n)
        .flat_map(|g| {
            (0..cfg.ep_size * cfg.moe_dp_size).map(move |i| {
                let st = g * cfg.tp_size + i * moe_tp;
                (st..st + moe_tp).collect()
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_config_world_size_1() {
        let cfg = MultiAxisConfig::single();
        assert_eq!(cfg.world_size(), 1);
        assert_eq!(cfg.attn_tp_size(), 1);
        assert_eq!(cfg.moe_tp_size(), 1);
        cfg.validate().unwrap();
        let coord = RankCoord::from_world_rank(cfg, 0).unwrap();
        assert_eq!(
            coord,
            RankCoord {
                world_rank: 0,
                tp_rank: 0,
                pp_rank: 0,
                attn_tp_rank: 0,
                attn_dp_rank: 0,
                attn_cp_rank: 0,
                moe_tp_rank: 0,
                moe_ep_rank: 0,
                moe_dp_rank: 0,
            }
        );
    }

    // tp=8, pp=1, ep=1, dp=1 over 8 ranks.
    #[test]
    fn tp8_pp1_ep1_dp1_topology() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.world_size(), 8);
        for r in 0..8 {
            let coord = RankCoord::from_world_rank(cfg, r).unwrap();
            assert_eq!(coord.tp_rank, r);
            assert_eq!(coord.pp_rank, 0);
        }
        assert_eq!(build_tp_groups(cfg), vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]);
        assert_eq!(
            build_pp_groups(cfg),
            vec![
                vec![0],
                vec![1],
                vec![2],
                vec![3],
                vec![4],
                vec![5],
                vec![6],
                vec![7]
            ]
        );
    }

    // SGLang parallel_state.py:1749-1756
    #[test]
    fn tp2_pp4_groups_sglang_docstring_1749_1756() {
        let cfg = MultiAxisConfig {
            tp_size: 2,
            pp_size: 4,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.world_size(), 8);
        assert_eq!(
            build_tp_groups(cfg),
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
        );
        assert_eq!(
            build_pp_groups(cfg),
            vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]]
        );
    }

    // tp=2, pp=2 over 4 ranks.
    #[test]
    fn tp2_pp2_four_ranks_coords_and_groups() {
        let cfg = MultiAxisConfig {
            tp_size: 2,
            pp_size: 2,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.world_size(), 4);
        let expect = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)];
        for (r, (tp, pp)) in expect.iter().enumerate() {
            let coord = RankCoord::from_world_rank(cfg, r).unwrap();
            assert_eq!(coord.tp_rank, *tp, "rank {r} tp_rank");
            assert_eq!(coord.pp_rank, *pp, "rank {r} pp_rank");
            assert_eq!(coord.pp_rank * cfg.tp_size + coord.tp_rank, r);
        }
        assert_eq!(build_tp_groups(cfg), vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(build_pp_groups(cfg), vec![vec![0, 2], vec![1, 3]]);
    }

    // SGLang parallel_state.py:1758-1769
    #[test]
    fn attn_cp2_attn_tp4_moe_dp2_moe_ep4_groups_sglang_docstring_1758_1769() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 4,
            attn_dp_size: 1,
            attn_cp_size: 2,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), 4);
        assert_eq!(cfg.moe_tp_size(), 1);
        assert_eq!(
            build_attn_tp_groups(cfg),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        );
        assert_eq!(
            build_attn_cp_groups(cfg),
            vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        );
        assert_eq!(
            build_moe_ep_groups(cfg),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
        );
        assert_eq!(
            build_moe_dp_groups(cfg),
            vec![vec![0, 4], vec![1, 5], vec![2, 6], vec![3, 7]],
        );
    }

    #[test]
    fn validate_rejects_world_size_mismatch() {
        // tp=3 not divisible by ep*moe_dp=4
        let cfg = MultiAxisConfig {
            tp_size: 3,
            pp_size: 1,
            ep_size: 2,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("ep_size * moe_dp_size"), "got: {err}");
    }

    #[test]
    fn validate_rejects_attn_div_mismatch() {
        // tp=4 not divisible by attn_dp*attn_cp = 3
        let cfg = MultiAxisConfig {
            tp_size: 4,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 3,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("attn_dp_size * attn_cp_size"), "got: {err}");
    }

    #[test]
    fn rank_coord_decomposition_round_trip() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 2,
            ep_size: 4,
            attn_dp_size: 1,
            attn_cp_size: 2,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        for world_rank in 0..cfg.world_size() {
            let coord = RankCoord::from_world_rank(cfg, world_rank).unwrap();
            assert_eq!(coord.world_rank, world_rank);
            let attn_tp = cfg.attn_tp_size();
            let reassembled_tp = (coord.attn_dp_rank * cfg.attn_cp_size + coord.attn_cp_rank)
                * attn_tp
                + coord.attn_tp_rank;
            assert_eq!(reassembled_tp, coord.tp_rank);
            let moe_tp = cfg.moe_tp_size();
            let reassembled_tp_moe =
                (coord.moe_dp_rank * cfg.ep_size + coord.moe_ep_rank) * moe_tp + coord.moe_tp_rank;
            assert_eq!(reassembled_tp_moe, coord.tp_rank);
            assert_eq!(coord.pp_rank * cfg.tp_size + coord.tp_rank, world_rank);
        }
    }

    // SGLang dp_attention.py:240-254
    #[test]
    fn dp_attention_math_matches_sglang() {
        // tp=8, dp=2, attn_cp=2 => attn_tp=2.
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 2,
            attn_cp_size: 2,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), 2);
        for tp_rank in 0..cfg.tp_size {
            let coord = RankCoord::from_world_rank(cfg, tp_rank).unwrap();
            assert_eq!(coord.attn_tp_rank, tp_rank % 2);
            assert_eq!(coord.attn_cp_rank, (tp_rank / 2) % 2);
            assert_eq!(coord.attn_dp_rank, tp_rank / (2 * 2));
        }
    }

    #[test]
    fn attn_owner_groups_match_sglang_dp_request_slices() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 4,
            attn_dp_size: 4,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), 2);
        assert_eq!(
            build_attn_owner_groups(cfg),
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]],
        );
        assert_eq!(
            build_attn_dp_groups(cfg),
            vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]],
        );
    }

    #[test]
    fn attn_owner_groups_collapse_to_whole_tp_when_dp_disabled() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 8,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(
            build_attn_owner_groups(cfg),
            vec![vec![0, 1, 2, 3, 4, 5, 6, 7]]
        );
    }

    #[test]
    fn multi_axis_from_lookup_uses_runtime_defaults() {
        let cfg = MultiAxisConfig::from_lookup_with_defaults(8, 1, 8, |_| None).unwrap();
        assert_eq!(
            cfg,
            MultiAxisConfig {
                tp_size: 8,
                pp_size: 1,
                ep_size: 8,
                attn_dp_size: 1,
                attn_cp_size: 1,
                moe_dp_size: 1,
            }
        );
        assert_eq!(
            cfg.summary(),
            "tp=8 pp=1 ep=8 attn_dp=1 attn_cp=1 attn_tp=8 moe_dp=1 moe_tp=1 world=8"
        );
        assert!(cfg.is_global_tp_ep_only(8, 8));
    }

    #[test]
    fn multi_axis_current_route_preserves_legacy_ep_only_override() {
        let cfg = MultiAxisConfig::current_route_from_lookup_with_defaults(1, 8, |_| None).unwrap();
        assert_eq!(
            cfg,
            MultiAxisConfig {
                tp_size: 8,
                pp_size: 1,
                ep_size: 8,
                attn_dp_size: 1,
                attn_cp_size: 1,
                moe_dp_size: 1,
            }
        );
        assert!(cfg.is_global_tp_ep_only(8, 8));
    }

    #[test]
    fn multi_axis_current_route_does_not_hide_advanced_ep_only_axes() {
        let cfg = MultiAxisConfig::current_route_from_lookup_with_defaults(1, 8, |key| match key {
            "INFER_ATTN_DP_SIZE" => Some("2".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(cfg.attn_dp_size, 2);
        assert!(!cfg.is_global_tp_ep_only(8, 8));
    }

    #[test]
    fn multi_axis_from_lookup_reads_sglang_axes() {
        let cfg = MultiAxisConfig::from_lookup(|key| match key {
            "INFER_TP_SIZE" => Some("8".to_string()),
            "INFER_EP_SIZE" => Some("4".to_string()),
            "INFER_ATTN_DP_SIZE" => Some("2".to_string()),
            "INFER_ATTN_CP_SIZE" => Some("1".to_string()),
            "INFER_MOE_DP_SIZE" => Some("2".to_string()),
            _ => None,
        })
        .unwrap();
        assert_eq!(cfg.attn_tp_size(), 4);
        assert_eq!(cfg.moe_tp_size(), 1);
        assert!(!cfg.is_global_tp_ep_only(8, 4));
    }

    #[test]
    fn attn_tp_size_when_dp_off() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.attn_tp_size(), cfg.tp_size);
    }

    #[test]
    fn moe_tp_size_division() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 2,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.moe_tp_size(), 2);
    }

    #[test]
    fn from_world_rank_rejects_out_of_range() {
        let cfg = MultiAxisConfig::single();
        assert!(RankCoord::from_world_rank(cfg, 1).is_err());
    }

    #[test]
    fn build_pp_groups_single_tp() {
        let cfg = MultiAxisConfig {
            tp_size: 1,
            pp_size: 4,
            ep_size: 1,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 1,
        };
        cfg.validate().unwrap();
        assert_eq!(build_pp_groups(cfg), vec![vec![0, 1, 2, 3]]);
        assert_eq!(
            build_tp_groups(cfg),
            vec![vec![0], vec![1], vec![2], vec![3]]
        );
    }

    #[test]
    fn moe_tp_groups_tp8_ep4_moedp2() {
        let cfg = MultiAxisConfig {
            tp_size: 8,
            pp_size: 1,
            ep_size: 4,
            attn_dp_size: 1,
            attn_cp_size: 1,
            moe_dp_size: 2,
        };
        cfg.validate().unwrap();
        assert_eq!(cfg.moe_tp_size(), 1);
        assert_eq!(
            build_moe_tp_groups(cfg),
            vec![
                vec![0],
                vec![1],
                vec![2],
                vec![3],
                vec![4],
                vec![5],
                vec![6],
                vec![7]
            ],
        );
    }
}
