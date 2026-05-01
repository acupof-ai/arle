//! Type-only forward batch surface for future TP/PP execution.
//!
//! F0.7 adds this as inert metadata so later pipeline-parallel work has a
//! stable slot for stage-boundary tensors. Existing CUDA/model forward
//! signatures do not consume this module yet.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForwardBatchKind {
    Decode,
    Prefill,
    Mixed,
    SpecVerify,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntermediateTensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl IntermediateTensorMeta {
    pub fn new(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        dtype: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype: dtype.into(),
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntermediateTensors {
    pub source_pp_rank: usize,
    pub target_pp_rank: usize,
    pub microbatch_id: u64,
    pub tensors: Vec<IntermediateTensorMeta>,
}

impl IntermediateTensors {
    pub fn new(source_pp_rank: usize, target_pp_rank: usize, microbatch_id: u64) -> Self {
        Self {
            source_pp_rank,
            target_pp_rank,
            microbatch_id,
            tensors: Vec::new(),
        }
    }

    pub fn with_tensor(mut self, tensor: IntermediateTensorMeta) -> Self {
        self.tensors.push(tensor);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ForwardBatch {
    pub kind: ForwardBatchKind,
    pub slot_indices: Vec<usize>,
    pub decode_tokens: Vec<u32>,
    pub prefill_token_counts: Vec<usize>,
    pub pp_proxy: Option<IntermediateTensors>,
}

impl ForwardBatch {
    pub fn new(kind: ForwardBatchKind) -> Self {
        Self {
            kind,
            slot_indices: Vec::new(),
            decode_tokens: Vec::new(),
            prefill_token_counts: Vec::new(),
            pp_proxy: None,
        }
    }

    pub fn decode(slot_indices: impl Into<Vec<usize>>, tokens: impl Into<Vec<u32>>) -> Self {
        Self {
            kind: ForwardBatchKind::Decode,
            slot_indices: slot_indices.into(),
            decode_tokens: tokens.into(),
            prefill_token_counts: Vec::new(),
            pp_proxy: None,
        }
    }

    pub fn prefill(
        slot_indices: impl Into<Vec<usize>>,
        token_counts: impl Into<Vec<usize>>,
    ) -> Self {
        Self {
            kind: ForwardBatchKind::Prefill,
            slot_indices: slot_indices.into(),
            decode_tokens: Vec::new(),
            prefill_token_counts: token_counts.into(),
            pp_proxy: None,
        }
    }

    pub fn with_pp_proxy(mut self, proxy: IntermediateTensors) -> Self {
        self.pp_proxy = Some(proxy);
        self
    }

    pub fn clear_pp_proxy(&mut self) {
        self.pp_proxy = None;
    }

    pub fn batch_size(&self) -> usize {
        self.slot_indices.len()
    }

    pub fn has_pp_proxy(&self) -> bool {
        self.pp_proxy.is_some()
    }
}

impl Default for ForwardBatch {
    fn default() -> Self {
        Self::new(ForwardBatchKind::Decode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_forward_batch_defaults_to_no_pp_proxy() {
        let batch = ForwardBatch::new(ForwardBatchKind::Mixed);

        assert_eq!(batch.kind, ForwardBatchKind::Mixed);
        assert_eq!(batch.batch_size(), 0);
        assert!(batch.pp_proxy.is_none());
        assert!(!batch.has_pp_proxy());
    }

    #[test]
    fn decode_constructor_preserves_slots_and_tokens() {
        let batch = ForwardBatch::decode(vec![2, 4], vec![11, 12]);

        assert_eq!(batch.kind, ForwardBatchKind::Decode);
        assert_eq!(batch.slot_indices, vec![2, 4]);
        assert_eq!(batch.decode_tokens, vec![11, 12]);
        assert!(batch.prefill_token_counts.is_empty());
        assert!(batch.pp_proxy.is_none());
    }

    #[test]
    fn pp_proxy_can_be_attached_and_cleared() {
        let proxy = IntermediateTensors::new(0, 1, 42).with_tensor(IntermediateTensorMeta::new(
            "hidden_states",
            vec![4, 4096],
            "bf16",
        ));
        let mut batch = ForwardBatch::prefill(vec![0], vec![128]).with_pp_proxy(proxy);

        assert!(batch.has_pp_proxy());
        assert_eq!(batch.pp_proxy.as_ref().unwrap().tensor_count(), 1);
        assert_eq!(batch.pp_proxy.as_ref().unwrap().tensors[0].numel(), 16_384);

        batch.clear_pp_proxy();

        assert!(batch.pp_proxy.is_none());
    }
}
