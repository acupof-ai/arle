//! Agent-workflow benchmark harness — the AI-PC north-star metric.
//!
//! Unlike a tok/s sweep, this drives the engine with a **multi-turn agent
//! workflow** (a shared system prompt + a sequence of user turns, each growing
//! the shared context so cross-turn KV reuse via the radix cache applies — the
//! realistic agent shape) and measures end-to-end task behavior: per-turn
//! latency / ticks and aggregate throughput.
//!
//! It is generic over the backend ([`infer_seam::BackendExecutor`] +
//! [`infer_seam::KvPool`]). With the bundled [`EchoExecutor`] it benchmarks the
//! device-neutral **scheduler layer** today (CPU only); pointed at
//! `infer_metal::MetalExecutor` (once R3b lands the real generation loop) it
//! produces the real on-device agent-workflow + OS-impact report (G3 of the
//! rewrite verification targets).

use std::time::{Duration, Instant};

use anyhow::Result;
use infer_core::{Engine, SchedulerConfig};
use infer_plan::{ForwardPlan, SlotToken, StepOutput};
use infer_seam::{BackendExecutor, KvPool, PollResult};

pub use infer_metal::MetalKvPool;

/// Deterministic, synchronous executor for scheduler-layer benchmarking.
///
/// Produces one placeholder token per scheduled row (decode: last+1; prefill:
/// last-prompt-token+1). It isolates the engine's CPU scheduling cost from any
/// model compute, so workflow timings here are the scheduler's contribution.
#[derive(Debug, Default)]
pub struct EchoExecutor;

impl BackendExecutor for EchoExecutor {
    type Inflight = StepOutput;

    fn submit(&mut self, plan: &ForwardPlan, _kv: &mut dyn KvPool) -> Result<Self::Inflight> {
        let mut tokens = Vec::with_capacity(plan.prefill_rows.len() + plan.decode_rows.len());
        for row in &plan.prefill_rows {
            tokens.push(SlotToken {
                slot: row.slot,
                token: row.tokens.last().copied().unwrap_or(0).wrapping_add(1),
                logprob: None,
                finish: None,
            });
        }
        for row in &plan.decode_rows {
            tokens.push(SlotToken {
                slot: row.slot,
                token: row.last_token.wrapping_add(1),
                logprob: None,
                finish: None,
            });
        }
        Ok(StepOutput { tokens })
    }

    fn poll(&mut self, inflight: Self::Inflight) -> Result<PollResult<Self::Inflight>> {
        Ok(PollResult::Ready(inflight))
    }
}

/// One turn of an agent workflow: the user message and how many tokens the
/// assistant generates in reply.
#[derive(Debug, Clone)]
pub struct AgentTurn {
    pub user_tokens: Vec<u32>,
    pub gen_tokens: usize,
}

/// A multi-turn agent workflow with a shared system prompt.
#[derive(Debug, Clone)]
pub struct AgentWorkflow {
    pub system_tokens: Vec<u32>,
    pub turns: Vec<AgentTurn>,
}

impl AgentWorkflow {
    /// A representative coding-agent shape: a `system_len` system prompt and
    /// `num_turns` turns of (`user_len` user tokens -> `gen_len` reply tokens).
    /// Each turn appends to the shared context, so later turns reuse the radix
    /// prefix of earlier ones — the agent KV-reuse case.
    #[must_use]
    pub fn synthetic(system_len: usize, num_turns: usize, user_len: usize, gen_len: usize) -> Self {
        let system_tokens: Vec<u32> = (0..system_len).map(|i| (i as u32 % 30_000) + 2).collect();
        let turns = (0..num_turns)
            .map(|t| AgentTurn {
                // distinct per-turn user tokens (offset by turn so they differ)
                user_tokens: (0..user_len)
                    .map(|i| ((t * user_len + i) as u32 % 30_000) + 40_000)
                    .collect(),
                gen_tokens: gen_len,
            })
            .collect();
        Self {
            system_tokens,
            turns,
        }
    }

    fn total_user_tokens(&self) -> usize {
        self.turns.iter().map(|t| t.user_tokens.len()).sum()
    }
}

/// Per-turn measurement.
#[derive(Debug, Clone)]
pub struct TurnMetric {
    pub turn: usize,
    pub prompt_len: usize,
    pub generated: usize,
    pub ticks: u64,
    pub wall: Duration,
}

/// Aggregate workflow measurement.
#[derive(Debug, Clone)]
pub struct WorkflowMetrics {
    pub turns: Vec<TurnMetric>,
    pub total_wall: Duration,
    pub total_generated: usize,
    pub total_ticks: u64,
}

impl WorkflowMetrics {
    /// Scheduler ticks per generated token (lower is leaner).
    #[must_use]
    pub fn ticks_per_token(&self) -> f64 {
        self.total_ticks as f64 / self.total_generated.max(1) as f64
    }
}

/// Drive a single-agent (c=1) workflow turn by turn through `engine`, returning
/// per-turn and aggregate metrics. The context grows each turn so the engine's
/// radix cache reuses the shared prefix across turns.
pub fn run_agent_workflow<E, K>(
    engine: &mut Engine<E, K>,
    workflow: &AgentWorkflow,
) -> Result<WorkflowMetrics>
where
    E: BackendExecutor,
    K: KvPool,
{
    let mut context: Vec<u32> = workflow.system_tokens.clone();
    let mut turns = Vec::with_capacity(workflow.turns.len());
    let mut total_generated = 0usize;
    let mut total_ticks = 0u64;
    let total_start = Instant::now();

    for (i, turn) in workflow.turns.iter().enumerate() {
        let mut prompt = context.clone();
        prompt.extend_from_slice(&turn.user_tokens);
        let prompt_len = prompt.len();

        let handle = engine.submit_request(prompt, turn.gen_tokens);
        let start = Instant::now();
        let mut ticks = 0u64;
        while !engine.is_idle() {
            engine.step()?;
            ticks += 1;
        }
        let wall = start.elapsed();

        let generated = engine
            .completed(handle)
            .map(|c| c.generated_tokens.clone())
            .unwrap_or_default();

        // Append the user turn + assistant reply to the shared context so the
        // next turn reuses this prefix (radix KV reuse).
        context.extend_from_slice(&turn.user_tokens);
        context.extend_from_slice(&generated);

        total_generated += generated.len();
        total_ticks += ticks;
        turns.push(TurnMetric {
            turn: i,
            prompt_len,
            generated: generated.len(),
            ticks,
            wall,
        });
    }

    Ok(WorkflowMetrics {
        turns,
        total_wall: total_start.elapsed(),
        total_generated,
        total_ticks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench_engine() -> Engine<EchoExecutor, MetalKvPool> {
        let mut config = SchedulerConfig::for_slots(4);
        config.max_prompt_tokens = 32_768;
        config.max_total_tokens = 65_536;
        config.chunked_prefill_size = 512;
        // page_size 16, 8192 pages -> 131072 token capacity
        Engine::with_config(EchoExecutor, MetalKvPool::new(4, 8192, 16), config)
    }

    #[test]
    fn agent_workflow_runs_and_grows_context() -> Result<()> {
        let mut engine = bench_engine();
        let wf = AgentWorkflow::synthetic(64, 3, 8, 4);
        let m = run_agent_workflow(&mut engine, &wf)?;
        assert_eq!(m.turns.len(), 3);
        // Each turn's prompt is longer than the previous (context grows).
        assert!(m.turns[1].prompt_len > m.turns[0].prompt_len);
        assert!(m.turns[2].prompt_len > m.turns[1].prompt_len);
        // Each turn generated its requested reply length.
        assert!(m.turns.iter().all(|t| t.generated == 4));
        assert_eq!(m.total_generated, 12);
        Ok(())
    }

    #[test]
    #[ignore = "benchmark; run with --release -- --ignored --nocapture"]
    fn bench_agent_workflow_scheduler() -> Result<()> {
        // Representative coding-agent shape: 512-token system prompt, 6 turns,
        // 64-token user msgs, 96-token replies. Context grows each turn.
        let wf = AgentWorkflow::synthetic(512, 6, 64, 96);
        let mut engine = bench_engine();
        let m = run_agent_workflow(&mut engine, &wf)?;
        eprintln!(
            "[agent-workflow scheduler] turns={} system={} total_user={} total_gen={} \
             total_wall={:?} total_ticks={} ticks_per_token={:.3}",
            wf.turns.len(),
            wf.system_tokens.len(),
            wf.total_user_tokens(),
            m.total_generated,
            m.total_wall,
            m.total_ticks,
            m.ticks_per_token(),
        );
        for t in &m.turns {
            eprintln!(
                "  turn {} prompt_len={} gen={} ticks={} wall={:?} (per-turn task latency)",
                t.turn, t.prompt_len, t.generated, t.ticks, t.wall
            );
        }
        Ok(())
    }
}
