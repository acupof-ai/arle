//! Pipeline-parallel metadata and stage-boundary transfer scaffolding.
//!
//! F3 keeps this CPU-verifiable and disconnected from model forward paths. The
//! production F4 wiring will replace the placeholder send/recv ops with NCCL or
//! P2P transport-backed tensor movement.

use anyhow::{Result, bail};

use crate::scheduler::IntermediateTensors;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineGroup {
    pub rank: usize,
    pub world_size: usize,
    pub stage_count: usize,
    pub current_stage: usize,
}

impl PipelineGroup {
    pub fn new(
        rank: usize,
        world_size: usize,
        stage_count: usize,
        current_stage: usize,
    ) -> Result<Self> {
        if world_size == 0 {
            bail!("pipeline world_size must be >= 1");
        }
        if stage_count == 0 {
            bail!("pipeline stage_count must be >= 1");
        }
        if rank >= world_size {
            bail!("pipeline rank {rank} must be < world_size {world_size}");
        }
        if current_stage >= stage_count {
            bail!("pipeline stage {current_stage} must be < stage_count {stage_count}");
        }
        if stage_count > world_size {
            bail!("pipeline stage_count {stage_count} must be <= world_size {world_size}");
        }
        if !world_size.is_multiple_of(stage_count) {
            bail!(
                "pipeline world_size {world_size} must be divisible by stage_count {stage_count}"
            );
        }
        Ok(Self {
            rank,
            world_size,
            stage_count,
            current_stage,
        })
    }

    pub fn is_first_stage(&self) -> bool {
        self.current_stage == 0
    }

    pub fn is_last_stage(&self) -> bool {
        self.current_stage + 1 == self.stage_count
    }

    pub fn previous_stage(&self) -> Option<usize> {
        self.current_stage.checked_sub(1)
    }

    pub fn next_stage(&self) -> Option<usize> {
        (!self.is_last_stage()).then_some(self.current_stage + 1)
    }

    pub fn inter_stage_send(&self, tensors: IntermediateTensors) -> Result<IntermediateTensors> {
        let Some(next_stage) = self.next_stage() else {
            bail!("pipeline stage {} has no next stage", self.current_stage);
        };
        if tensors.source_pp_rank != self.current_stage || tensors.target_pp_rank != next_stage {
            bail!(
                "pipeline send expected {} -> {}, got {} -> {}",
                self.current_stage,
                next_stage,
                tensors.source_pp_rank,
                tensors.target_pp_rank
            );
        }
        Ok(tensors)
    }

    pub fn inter_stage_recv(&self, tensors: IntermediateTensors) -> Result<IntermediateTensors> {
        let Some(previous_stage) = self.previous_stage() else {
            bail!(
                "pipeline stage {} has no previous stage",
                self.current_stage
            );
        };
        if tensors.source_pp_rank != previous_stage || tensors.target_pp_rank != self.current_stage
        {
            bail!(
                "pipeline recv expected {} -> {}, got {} -> {}",
                previous_stage,
                self.current_stage,
                tensors.source_pp_rank,
                tensors.target_pp_rank
            );
        }
        Ok(tensors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_layer(layer_idx: usize, hidden: &mut [f32], residual: &mut [f32], positions: &[u32]) {
        let layer_scale = (layer_idx + 1) as f32;
        for ((hidden_value, residual_value), position) in hidden
            .iter_mut()
            .zip(residual.iter_mut())
            .zip(positions.iter().copied())
        {
            *hidden_value = (*hidden_value + layer_scale) * 1.01 + position as f32 * 0.001;
            *residual_value += *hidden_value * 0.1 + layer_scale * 0.01;
        }
    }

    fn run_layers(
        start: usize,
        end: usize,
        mut hidden: Vec<f32>,
        mut residual: Vec<f32>,
        positions: &[u32],
    ) -> (Vec<f32>, Vec<f32>) {
        for layer_idx in start..end {
            mock_layer(layer_idx, &mut hidden, &mut residual, positions);
        }
        (hidden, residual)
    }

    #[test]
    fn pipeline_group_validates_stage_boundaries() {
        assert!(PipelineGroup::new(0, 2, 2, 0).is_ok());
        assert!(PipelineGroup::new(2, 2, 2, 0).is_err());
        assert!(PipelineGroup::new(0, 2, 2, 2).is_err());
        assert!(PipelineGroup::new(0, 2, 3, 0).is_err());
        assert!(PipelineGroup::new(0, 3, 2, 0).is_err());
    }

    #[test]
    fn mock_two_stage_pipeline_matches_single_stage() {
        let layer_count = 6;
        let split = layer_count / 2;
        let hidden = vec![0.25, 0.5, 0.75, 1.0];
        let residual = vec![0.0; hidden.len()];
        let positions = vec![0, 1, 2, 3];

        let single_stage = run_layers(0, layer_count, hidden.clone(), residual.clone(), &positions);

        let stage0 = PipelineGroup::new(0, 2, 2, 0).unwrap();
        let stage1 = PipelineGroup::new(1, 2, 2, 1).unwrap();
        let (stage0_hidden, stage0_residual) = run_layers(0, split, hidden, residual, &positions);

        let boundary = IntermediateTensors::new(0, 1, 7)
            .with_hidden_states(vec![1, stage0_hidden.len()], stage0_hidden)
            .with_residual(vec![1, stage0_residual.len()], stage0_residual)
            .with_position_ids(vec![1, positions.len()], positions.clone());
        let sent = stage0.inter_stage_send(boundary).unwrap();
        let received = stage1.inter_stage_recv(sent).unwrap();

        let (pipeline_hidden, pipeline_residual) = run_layers(
            split,
            layer_count,
            received.hidden_states.unwrap().values,
            received.residual.unwrap().values,
            &received.position_ids.unwrap().values,
        );

        assert_eq!(pipeline_hidden, single_stage.0);
        assert_eq!(pipeline_residual, single_stage.1);
    }
}
