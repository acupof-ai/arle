# W3 503 capacity error — source identified, root cause non-obvious

> Continues `a672b08` errors entry(W3 short-multiturn 135/136 turns 503 at c=16)。
> Code grep traces 503 source。Direct admission cap not the bug — there's
> another capacity mechanism between handler and scheduler。

## 503 source path

`infer/src/http_server/handlers.rs:291-296`:

```rust
if let Err(e) = state.handle.submit(incoming) {
    warn!("Scheduler at capacity: {e}");
    return Err(ApiError::service_unavailable(
        "Server is at capacity, please retry later",
    ));
}
```

`handle.submit()` returns `Err(SchedulerFull)` when:
- `infer/src/scheduler/types.rs:735` SchedulerHandle::submit
- Calls `admission_allows(current_waiting)`
- → calls `QueueBoundAdmission { max_queued_requests: max_waiting }.allow(...)`

## Default max_waiting

`infer/src/scheduler/types.rs:213`:
```rust
max_waiting_requests: 256,
```

c=16 burst ≪ 256 → admission_allows should return true。**But 135/136 turns failed**。

## Observations from `/v1/stats` (per `a672b08`)

```
active=16, prefill_queue=15  ← all 16 admitted, 15 queued in prefill
kv_util=1.1%                 ← KV barely used
```

→ **16 sessions admitted but only 1 actively prefilling**(prefill_queue 持有 15 阻塞)

## Hypothesis: admission counts `active`, not just `waiting`

If `admission_allows` 实际看 `total_inflight = active + waiting`(not just waiting),and there's a hidden cap = num_slots OR `prefill_max_requests`:

- `--num-slots 16` set explicitly in W3 setup
- Active fills to 16 from first burst
- Turn 2 of any session tries to submit → admission_allows sees `total >= num_slots` → 503

Or:

- `prefill_max_requests` defaults to 1(scheduler config that limits concurrent **prefills**,not slots)
- All 16 admit OK,但 prefill_queue 排队 → 顺序 prefill 各 ~1-2s @4k context
- New turn submissions during 16×prefill backlog → some other check causes 503

## Codex investigation needed

1. Check `infer/src/main.rs` for `--max-waiting` CLI flag(if absent,use default 256 — should not be the cap)
2. Check `prefill_max_requests` default(line 215+ in types.rs)— possibly = 1
3. Check `admission_allows` 是否 also factors `prefill_max_requests` or `active_count`(not just `waiting_count`)
4. Run W3 with `--max-waiting 1024` if CLI flag exists,check if 503 disappears
5. If not capacity-cap related,grep for other 503 sources in submit chain(channel send error,wakeup_tx full,etc)

## Why this matters(strategic)

W3 short-multiturn is **master strategy §2.1 真 agent workload**(c=1-8 typical,c=16 burst tier)。10 KILL paths are all on canonical 4-shape benchmark `不反映 agent 痛点`。

W3 503 is the **first time we tried real agent workload bench** — and ARLE fails with capacity error before producing any number。This is a **production blocker** for axis 1(agent workload主战场)。

Fix is likely small(adjust admission cap or expose --max-waiting CLI flag),
high ROI:unblock entire agent workload bench cycle(currently 0/136 turns produce data)。

## Cross-references

- W3 errors entry: [`a672b08`](../experience/errors/2026-05-08-w3-bench-capacity-503-admission-backlog.md)
- handlers.rs 503 source: line 291-296
- SchedulerHandle::submit: `infer/src/scheduler/types.rs:735`
- admission_allows: `types.rs:628`
- max_waiting default: `types.rs:213` (256)
- bench_agent_trace.py: `scripts/bench_agent_trace.py`(W3 driver)
