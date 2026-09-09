//! ARLE binary library surface: the runtime backend registry so examples and
//! tests can register backends without going through `main`.

pub mod backends;

pub use backends::register_all;
