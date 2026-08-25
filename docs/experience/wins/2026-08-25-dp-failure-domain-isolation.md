# DP failure domain isolation — infer-server, 2026-08-25

> Status: pending-remote

## Goal

A single rank's crash kills only its own group's in-flight requests; other DP
groups continue serving (#210). Pod gate: kill one rank in a 2-group
deployment, assert the other group's requests complete and its step wall is
unaffected.

## What landed

`CoordinatorHandle.dead: Arc<AtomicBool>` set by the lockstep teardown path.
`DpCoordinator::select` filters dead groups; `streaming_submit` fails fast on
a dead group ("torn down") instead of hanging on a closed channel.

`dp_coordinator_router` passes `None` serve-shutdown to each group when
relays.len() > 1, so a group teardown no longer co-kills the deployment.
Single-group routers keep the shutdown (worker-guard reaping is the only
teardown mechanism there).

Unit tests (cpu lane, `cargo test -p infer-server dp_failure_domain`):
select skips a dead group; streaming_submit rejects a dead group.

## Parameters

```bash
# pending-remote: 2-group CUDA serve
# - kill -9 one rank of group 0 mid-flight
# - assert group 0's in-flight requests fail, group 1's requests complete
# - assert group 1 step wall within 5% of solo baseline
```

- Baseline: `663235636` (teardown co-kills all groups via serve shutdown)
- Treatment: this commit (dead flag isolates the group)
- Trials: pending-remote

## Environment

- Host / GPU: H20 pod, 2x TP group (pending-remote)

## Rule

A per-group dead flag is the isolation seam: routing skips it, submits fail
fast on it, and the serve-wide shutdown stays out of the group teardown path
in multi-group deployments.
