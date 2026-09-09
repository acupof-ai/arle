//! Two-phase commit across TP ranks for rank-locked operations (KV tier
//! demote/promote today). Each rank attempts locally; a scalar min-reduce is
//! the verdict. The collective runs on EVERY path — a rank whose local
//! attempt failed must still enter the reduce, or the lockstep deadlocks.
//!
//! `agree` is injected: production passes the backend's min-reduce, tests
//! pass a fake that constructs partial-rank failure — a scenario that is
//! nearly impossible to set up on real hardware.

/// One agree-or-abort round. `local` is this rank's attempt result. When the
/// min-reduce returns 0 some rank failed: return the local error, or a
/// peer-failure error (labelled by `what`) when this rank succeeded.
pub fn agree_abort<T>(
    local: anyhow::Result<T>,
    agree: impl FnOnce(usize) -> anyhow::Result<usize>,
    what: &str,
) -> anyhow::Result<T> {
    let global = agree(usize::from(local.is_ok()))?;
    if global == 0 {
        return Err(match local {
            Err(err) => err,
            Ok(_) => anyhow::anyhow!("peer rank failed {what}"),
        });
    }
    // min == 1 ⇒ every rank succeeded ⇒ this rank succeeded.
    match local {
        Ok(value) => Ok(value),
        Err(_) => unreachable!("min-reduce returned {global} despite a local failure"),
    }
}

/// One agree-or-rollback round. On disagreement a locally successful attempt
/// is undone via `rollback` (which runs only then) and the round reports
/// `false`; a failed rollback errors.
pub fn agree_rollback(
    local: bool,
    agree: impl FnOnce(usize) -> anyhow::Result<usize>,
    rollback: impl FnOnce() -> anyhow::Result<()>,
) -> anyhow::Result<bool> {
    let global = agree(usize::from(local))?;
    if global == 0 {
        if local {
            rollback()?;
        }
        return Ok(false);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake min-reduce: hands back a canned verdict and records the local
    /// input, so a test can assert the collective ran and saw the right value.
    fn fake_agree(
        verdict: usize,
        seen: &mut Option<usize>,
    ) -> impl FnOnce(usize) -> anyhow::Result<usize> + '_ {
        move |ok| {
            *seen = Some(ok);
            Ok(verdict)
        }
    }

    #[test]
    fn abort_commits_when_all_ranks_agree() {
        let mut seen = None;
        let out = agree_abort(Ok(42usize), fake_agree(1, &mut seen), "round").unwrap();
        assert_eq!(out, 42);
        assert_eq!(seen, Some(1));
    }

    #[test]
    fn abort_reports_peer_failure_when_this_rank_succeeded() {
        let mut seen = None;
        let err = agree_abort(Ok(1usize), fake_agree(0, &mut seen), "the round")
            .unwrap_err()
            .to_string();
        assert!(err.contains("peer rank failed the round"), "{err}");
        assert_eq!(seen, Some(1));
    }

    #[test]
    fn abort_runs_the_collective_even_when_the_local_attempt_failed() {
        // The lockstep property: a local failure still feeds 0 into the
        // reduce and surfaces the local error.
        let mut seen = None;
        let err = agree_abort::<usize>(
            Err(anyhow::anyhow!("local boom")),
            fake_agree(0, &mut seen),
            "round",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("local boom"), "{err}");
        assert_eq!(seen, Some(0));
    }

    #[test]
    #[should_panic(expected = "despite a local failure")]
    fn abort_panics_on_a_lying_collective() {
        let mut seen = None;
        let _ = agree_abort::<usize>(
            Err(anyhow::anyhow!("local boom")),
            fake_agree(1, &mut seen),
            "round",
        );
    }

    #[test]
    fn rollback_commits_without_touching_rollback() {
        let mut seen = None;
        let mut undone = false;
        let committed = agree_rollback(true, fake_agree(1, &mut seen), || {
            undone = true;
            Ok(())
        })
        .unwrap();
        assert!(committed);
        assert!(!undone);
        assert_eq!(seen, Some(1));
    }

    #[test]
    fn rollback_undoes_a_local_success_on_disagreement() {
        let mut seen = None;
        let mut undone = false;
        let committed = agree_rollback(true, fake_agree(0, &mut seen), || {
            undone = true;
            Ok(())
        })
        .unwrap();
        assert!(!committed);
        assert!(undone);
        assert_eq!(seen, Some(1));
    }

    #[test]
    fn rollback_skips_undo_when_this_rank_already_failed() {
        let mut seen = None;
        let mut undone = false;
        let committed = agree_rollback(false, fake_agree(0, &mut seen), || {
            undone = true;
            Ok(())
        })
        .unwrap();
        assert!(!committed);
        assert!(!undone);
        assert_eq!(seen, Some(0));
    }

    #[test]
    fn rollback_propagates_a_failed_undo() {
        let mut seen = None;
        let err = agree_rollback(true, fake_agree(0, &mut seen), || {
            Err(anyhow::anyhow!("undo boom"))
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("undo boom"), "{err}");
    }
}
