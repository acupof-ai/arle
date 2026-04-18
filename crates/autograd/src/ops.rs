#[path = "ops/activation.rs"]
pub mod activation;
#[path = "ops/broadcast.rs"]
pub mod broadcast;
#[path = "ops/elementwise.rs"]
pub mod elementwise;
#[path = "ops/embed.rs"]
pub mod embed;
#[path = "ops/gather.rs"]
pub mod gather;
#[path = "ops/layout.rs"]
pub mod layout;
#[path = "ops/matmul.rs"]
pub mod matmul;
#[path = "ops/norm.rs"]
pub mod norm;
#[path = "ops/reduce.rs"]
pub mod reduce;
#[path = "ops/softmax.rs"]
pub mod softmax;

pub use activation::gelu;
pub use broadcast::add_broadcast;
pub use elementwise::{add, mul, mul_scalar};
pub use embed::embedding;
pub use gather::gather_last_dim;
pub use layout::{reshape, transpose};
pub use matmul::matmul;
pub use norm::rmsnorm;
pub use reduce::{mean, sum};
pub use softmax::{log_softmax, softmax};

pub(crate) use activation::gelu_backward;
pub(crate) use broadcast::add_broadcast_backward;
pub(crate) use elementwise::{add_backward, mul_backward, mul_scalar_backward};
pub(crate) use embed::embedding_backward;
pub(crate) use gather::gather_last_dim_backward;
pub(crate) use layout::{reshape_backward, transpose_backward};
pub(crate) use matmul::matmul_backward;
pub(crate) use norm::rmsnorm_backward;
pub(crate) use reduce::{mean_backward, sum_backward};
pub(crate) use softmax::{log_softmax_backward, softmax_backward};
