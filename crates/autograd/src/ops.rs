#[path = "ops/elementwise.rs"]
pub mod elementwise;
#[path = "ops/reduce.rs"]
pub mod reduce;

pub use elementwise::{add, mul, mul_scalar};
pub use reduce::sum;

pub(crate) use elementwise::{add_backward, mul_backward, mul_scalar_backward};
pub(crate) use reduce::sum_backward;
