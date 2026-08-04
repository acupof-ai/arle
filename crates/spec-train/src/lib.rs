//! Offline speculative-decoding draft training.
//!
//! Separate from `train` (OPD-only) because the two share only the autograd
//! substrate: the data recipe, loss, block-mask construction and eval are all
//! different. What lives here trains draft models; what lives in `train` distils
//! a policy.
//!
//! Today this is the artifact layer only — reading and writing the DSpark Markov
//! head, and the fixed-spectrum (ISO) frames. The trainer itself is not built.
//!
//! The online train sidecar this replaced was deleted 2026-08-04: it saw 120
//! training rows per optimizer step against the reference's 1.8M, because
//! DSpark's 512x amplification comes from a training-time attention mask over
//! sampled anchors, which a path that only observes what the serve actually
//! drafted structurally cannot have. See
//! `docs/experience/errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md`.

#[path = "iso_spectrum.rs"]
pub mod iso_spectrum;
#[path = "markov_head.rs"]
pub mod markov_head;
