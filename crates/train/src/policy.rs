use autograd::{Result, Tape, TensorId, TensorStore};

pub trait GrpoPolicyConfig {
    fn max_seq_len(&self) -> usize;
    fn vocab_size(&self) -> usize;
}

pub trait GrpoPolicy: Sized {
    type Config: GrpoPolicyConfig;

    fn config(&self) -> &Self::Config;

    fn forward_single(
        &self,
        input_ids: &[usize],
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId>;

    fn forward_batch_tokens(
        &self,
        input_ids: &[usize],
        batch: usize,
        seq_len: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId>;

    fn forward_batch_tokens_with_positions(
        &self,
        input_ids: &[usize],
        position_ids: &[usize],
        batch: usize,
        store: &mut TensorStore,
        tape: &mut Tape,
    ) -> Result<TensorId> {
        self.forward_batch_tokens(input_ids, batch, position_ids.len(), store, tape)
    }

    fn all_parameter_ids(&self) -> Vec<TensorId>;

    fn clone_frozen(&self, store: &mut TensorStore) -> Self;
}
