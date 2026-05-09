//! Thin Rust wrapper over the upstream `mlc-ai/xgrammar` C++ matcher.
//!
//! The default build is a no-op scaffold so the workspace remains independent
//! of network-fetched native sources. Enable `--features real` and set
//! `XGRAMMAR_SOURCE_DIR=/path/to/xgrammar` to compile the pinned upstream C++
//! implementation through `cc`.

use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "real")]
use std::ffi::{CStr, CString};
#[cfg(feature = "real")]
use std::os::raw::{c_char, c_int};
#[cfg(feature = "real")]
use std::ptr::{self, NonNull};

#[derive(Debug, thiserror::Error)]
pub enum XGrammarError {
    #[error("xgrammar real backend is not compiled; rebuild xgrammar-sys with --features real")]
    Unavailable,
    #[error("xgrammar input contains an interior nul byte")]
    InteriorNul(#[from] std::ffi::NulError),
    #[error("xgrammar FFI error: {0}")]
    Ffi(String),
    #[error("xgrammar returned a null handle for {0}")]
    NullHandle(&'static str),
    #[error("bitmask buffer too small: got {got}, need {need}")]
    BitmaskTooSmall { got: usize, need: usize },
    #[error("vocab size {0} exceeds i32::MAX")]
    VocabTooLarge(usize),
}

pub type Result<T> = std::result::Result<T, XGrammarError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum VocabType {
    Raw = 0,
    ByteFallback = 1,
    ByteLevel = 2,
}

#[derive(Clone, Debug)]
pub struct CompilerConfig {
    pub vocab_type: VocabType,
    pub vocab_size: Option<usize>,
    pub stop_token_ids: Vec<i32>,
    pub add_prefix_space: bool,
    pub max_threads: i32,
    pub cache_enabled: bool,
    pub max_memory_bytes: i64,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            vocab_type: VocabType::Raw,
            vocab_size: None,
            stop_token_ids: Vec::new(),
            add_prefix_space: false,
            max_threads: 8,
            cache_enabled: true,
            max_memory_bytes: -1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatcherConfig {
    pub override_stop_tokens: Vec<i32>,
    pub terminate_without_stop_token: bool,
    pub max_rollback_tokens: i32,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            override_stop_tokens: Vec::new(),
            terminate_without_stop_token: false,
            max_rollback_tokens: -1,
        }
    }
}

#[cfg(feature = "real")]
mod ffi {
    use super::{c_char, c_int};

    pub enum ArleXGrammarCompiler {}
    pub enum ArleXGrammarCompiledGrammar {}
    pub enum ArleXGrammarMatcher {}

    unsafe extern "C" {
        pub fn arle_xgrammar_version() -> *const c_char;
        pub fn arle_xgrammar_free_error(message: *mut c_char);
        pub fn arle_xgrammar_bitmask_size(vocab_size: i32) -> i32;
        pub fn arle_xgrammar_compiler_new(
            encoded_vocab: *const *const c_char,
            encoded_vocab_len: usize,
            vocab_type: i32,
            vocab_size: i32,
            stop_token_ids: *const i32,
            stop_token_ids_len: usize,
            add_prefix_space: u8,
            max_threads: i32,
            cache_enabled: u8,
            max_memory_bytes: i64,
            out: *mut *mut ArleXGrammarCompiler,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn arle_xgrammar_compiler_free(compiler: *mut ArleXGrammarCompiler);
        pub fn arle_xgrammar_compile_json_schema(
            compiler: *mut ArleXGrammarCompiler,
            schema: *const c_char,
            strict_mode: u8,
            out: *mut *mut ArleXGrammarCompiledGrammar,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn arle_xgrammar_compile_ebnf(
            compiler: *mut ArleXGrammarCompiler,
            grammar: *const c_char,
            root_rule_name: *const c_char,
            out: *mut *mut ArleXGrammarCompiledGrammar,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn arle_xgrammar_compiled_grammar_free(grammar: *mut ArleXGrammarCompiledGrammar);
        pub fn arle_xgrammar_matcher_new(
            grammar: *const ArleXGrammarCompiledGrammar,
            override_stop_tokens: *const i32,
            override_stop_tokens_len: usize,
            terminate_without_stop_token: u8,
            max_rollback_tokens: i32,
            out: *mut *mut ArleXGrammarMatcher,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn arle_xgrammar_matcher_free(matcher: *mut ArleXGrammarMatcher);
        pub fn arle_xgrammar_matcher_fill_next_token_bitmask(
            matcher: *mut ArleXGrammarMatcher,
            bitmask: *mut u32,
            bitmask_len: usize,
            need_apply: *mut u8,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn arle_xgrammar_matcher_accept_token(
            matcher: *mut ArleXGrammarMatcher,
            token_id: i32,
            accepted: *mut u8,
            error: *mut *mut c_char,
        ) -> c_int;
        pub fn arle_xgrammar_matcher_is_terminated(matcher: *const ArleXGrammarMatcher) -> u8;
        pub fn arle_xgrammar_matcher_is_completed(matcher: *const ArleXGrammarMatcher) -> u8;
    }
}

pub fn compiled_backend_version() -> &'static str {
    #[cfg(feature = "real")]
    {
        // SAFETY: The shim returns a process-lifetime string literal.
        let ptr = unsafe { ffi::arle_xgrammar_version() };
        if ptr.is_null() {
            return "mlc-ai/xgrammar real backend";
        }
        // SAFETY: Non-null pointer is owned by the C++ library for process lifetime.
        unsafe { CStr::from_ptr(ptr) }
            .to_str()
            .unwrap_or("mlc-ai/xgrammar")
    }
    #[cfg(not(feature = "real"))]
    {
        "stub"
    }
}

pub fn bitmask_size(vocab_size: usize) -> Result<usize> {
    let vocab_i32 =
        i32::try_from(vocab_size).map_err(|_| XGrammarError::VocabTooLarge(vocab_size))?;
    #[cfg(feature = "real")]
    {
        // SAFETY: Pure upstream helper with no pointer arguments.
        let out = unsafe { ffi::arle_xgrammar_bitmask_size(vocab_i32) };
        Ok(out.max(0) as usize)
    }
    #[cfg(not(feature = "real"))]
    {
        let _ = vocab_i32;
        Ok(vocab_size.div_ceil(32))
    }
}

#[derive(Debug)]
pub struct GrammarCompiler {
    #[cfg(feature = "real")]
    inner: NonNull<ffi::ArleXGrammarCompiler>,
    vocab_size: usize,
    _not_sync: PhantomData<*mut ()>,
}

// The upstream compiler owns heap state and is safe to move between scheduler
// setup threads, but it is not exposed as Sync because cache mutation is
// internal to compile calls.
unsafe impl Send for GrammarCompiler {}

impl GrammarCompiler {
    pub fn new<S: AsRef<str>>(encoded_vocab: &[S], config: CompilerConfig) -> Result<Self> {
        #[cfg(not(feature = "real"))]
        {
            let _ = encoded_vocab;
            let _ = config;
            Err(XGrammarError::Unavailable)
        }
        #[cfg(feature = "real")]
        {
            let c_vocab: Vec<CString> = encoded_vocab
                .iter()
                .map(|token| CString::new(token.as_ref()))
                .collect::<std::result::Result<_, _>>()?;
            let ptrs: Vec<*const c_char> = c_vocab.iter().map(|token| token.as_ptr()).collect();
            let vocab_size = config.vocab_size.unwrap_or(encoded_vocab.len());
            let vocab_size_i32 =
                i32::try_from(vocab_size).map_err(|_| XGrammarError::VocabTooLarge(vocab_size))?;
            let mut out = ptr::null_mut();
            let mut error = ptr::null_mut();
            // SAFETY: All pointers are valid for the duration of the call; the shim copies
            // vocabulary and stop-token inputs into C++ owned storage.
            let status = unsafe {
                ffi::arle_xgrammar_compiler_new(
                    ptrs.as_ptr(),
                    ptrs.len(),
                    config.vocab_type as i32,
                    vocab_size_i32,
                    nullable_i32_ptr(&config.stop_token_ids),
                    config.stop_token_ids.len(),
                    u8::from(config.add_prefix_space),
                    config.max_threads,
                    u8::from(config.cache_enabled),
                    config.max_memory_bytes,
                    &mut out,
                    &mut error,
                )
            };
            check_status(status, error)?;
            let inner =
                NonNull::new(out).ok_or(XGrammarError::NullHandle("GrammarCompiler::new"))?;
            Ok(Self {
                inner,
                vocab_size,
                _not_sync: PhantomData,
            })
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn compile_json_schema(
        &mut self,
        schema: &str,
        strict_mode: bool,
    ) -> Result<CompiledGrammar> {
        #[cfg(not(feature = "real"))]
        {
            let _ = schema;
            let _ = strict_mode;
            Err(XGrammarError::Unavailable)
        }
        #[cfg(feature = "real")]
        {
            let schema = CString::new(schema)?;
            let mut out = ptr::null_mut();
            let mut error = ptr::null_mut();
            // SAFETY: `self.inner` is a live compiler handle and `schema` is NUL-terminated.
            let status = unsafe {
                ffi::arle_xgrammar_compile_json_schema(
                    self.inner.as_ptr(),
                    schema.as_ptr(),
                    u8::from(strict_mode),
                    &mut out,
                    &mut error,
                )
            };
            check_status(status, error)?;
            CompiledGrammar::from_raw(out, self.vocab_size)
        }
    }

    pub fn compile_ebnf(&mut self, grammar: &str, root_rule_name: &str) -> Result<CompiledGrammar> {
        #[cfg(not(feature = "real"))]
        {
            let _ = grammar;
            let _ = root_rule_name;
            Err(XGrammarError::Unavailable)
        }
        #[cfg(feature = "real")]
        {
            let grammar = CString::new(grammar)?;
            let root = CString::new(root_rule_name)?;
            let mut out = ptr::null_mut();
            let mut error = ptr::null_mut();
            // SAFETY: `self.inner` is live; grammar/root C strings are valid for the call.
            let status = unsafe {
                ffi::arle_xgrammar_compile_ebnf(
                    self.inner.as_ptr(),
                    grammar.as_ptr(),
                    root.as_ptr(),
                    &mut out,
                    &mut error,
                )
            };
            check_status(status, error)?;
            CompiledGrammar::from_raw(out, self.vocab_size)
        }
    }
}

#[cfg(feature = "real")]
impl Drop for GrammarCompiler {
    fn drop(&mut self) {
        // SAFETY: `inner` is owned by this wrapper and freed exactly once.
        unsafe { ffi::arle_xgrammar_compiler_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct CompiledGrammar {
    #[cfg(feature = "real")]
    inner: NonNull<ffi::ArleXGrammarCompiledGrammar>,
    vocab_size: usize,
}

unsafe impl Send for CompiledGrammar {}
unsafe impl Sync for CompiledGrammar {}

impl CompiledGrammar {
    #[cfg(feature = "real")]
    fn from_raw(raw: *mut ffi::ArleXGrammarCompiledGrammar, vocab_size: usize) -> Result<Self> {
        Ok(Self {
            inner: NonNull::new(raw).ok_or(XGrammarError::NullHandle("CompiledGrammar"))?,
            vocab_size,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(feature = "real")]
impl Drop for CompiledGrammar {
    fn drop(&mut self) {
        // SAFETY: `inner` is owned by this wrapper and freed exactly once.
        unsafe { ffi::arle_xgrammar_compiled_grammar_free(self.inner.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct GrammarMatcher {
    #[cfg(feature = "real")]
    inner: NonNull<ffi::ArleXGrammarMatcher>,
    grammar: Arc<CompiledGrammar>,
    vocab_size: usize,
    _not_sync: PhantomData<*mut ()>,
}

unsafe impl Send for GrammarMatcher {}

impl GrammarMatcher {
    pub fn new(grammar: Arc<CompiledGrammar>, config: MatcherConfig) -> Result<Self> {
        #[cfg(not(feature = "real"))]
        {
            let _ = grammar;
            let _ = config;
            Err(XGrammarError::Unavailable)
        }
        #[cfg(feature = "real")]
        {
            let mut out = ptr::null_mut();
            let mut error = ptr::null_mut();
            // SAFETY: The compiled grammar handle is live and retained by this wrapper.
            let status = unsafe {
                ffi::arle_xgrammar_matcher_new(
                    grammar.inner.as_ptr(),
                    nullable_i32_ptr(&config.override_stop_tokens),
                    config.override_stop_tokens.len(),
                    u8::from(config.terminate_without_stop_token),
                    config.max_rollback_tokens,
                    &mut out,
                    &mut error,
                )
            };
            check_status(status, error)?;
            let vocab_size = grammar.vocab_size();
            Ok(Self {
                inner: NonNull::new(out).ok_or(XGrammarError::NullHandle("GrammarMatcher"))?,
                grammar,
                vocab_size,
                _not_sync: PhantomData,
            })
        }
    }

    pub fn fill_next_token_bitmask(&mut self, bitmask: &mut [u32]) -> Result<bool> {
        let need = bitmask_size(self.vocab_size)?;
        if bitmask.len() < need {
            return Err(XGrammarError::BitmaskTooSmall {
                got: bitmask.len(),
                need,
            });
        }
        #[cfg(not(feature = "real"))]
        {
            let _ = bitmask;
            Err(XGrammarError::Unavailable)
        }
        #[cfg(feature = "real")]
        {
            let mut need_apply = 0_u8;
            let mut error = ptr::null_mut();
            // SAFETY: `inner` is live; bitmask points to at least `need` u32 values.
            let status = unsafe {
                ffi::arle_xgrammar_matcher_fill_next_token_bitmask(
                    self.inner.as_ptr(),
                    bitmask.as_mut_ptr(),
                    need,
                    &mut need_apply,
                    &mut error,
                )
            };
            check_status(status, error)?;
            Ok(need_apply != 0)
        }
    }

    pub fn accept_token(&mut self, token_id: u32) -> Result<bool> {
        #[cfg(not(feature = "real"))]
        {
            let _ = token_id;
            Err(XGrammarError::Unavailable)
        }
        #[cfg(feature = "real")]
        {
            let mut accepted = 0_u8;
            let mut error = ptr::null_mut();
            // SAFETY: `inner` is live and `accepted` is a valid out pointer.
            let status = unsafe {
                ffi::arle_xgrammar_matcher_accept_token(
                    self.inner.as_ptr(),
                    token_id as i32,
                    &mut accepted,
                    &mut error,
                )
            };
            check_status(status, error)?;
            Ok(accepted != 0)
        }
    }

    pub fn is_terminated(&self) -> bool {
        #[cfg(not(feature = "real"))]
        {
            false
        }
        #[cfg(feature = "real")]
        {
            // SAFETY: `inner` is live for `self`.
            unsafe { ffi::arle_xgrammar_matcher_is_terminated(self.inner.as_ptr()) != 0 }
        }
    }

    pub fn is_completed(&self) -> bool {
        #[cfg(not(feature = "real"))]
        {
            false
        }
        #[cfg(feature = "real")]
        {
            // SAFETY: `inner` is live for `self`.
            unsafe { ffi::arle_xgrammar_matcher_is_completed(self.inner.as_ptr()) != 0 }
        }
    }

    pub fn grammar(&self) -> &Arc<CompiledGrammar> {
        &self.grammar
    }
}

#[cfg(feature = "real")]
impl Drop for GrammarMatcher {
    fn drop(&mut self) {
        // SAFETY: `inner` is owned by this wrapper and freed exactly once.
        unsafe { ffi::arle_xgrammar_matcher_free(self.inner.as_ptr()) };
    }
}

#[cfg(feature = "real")]
fn nullable_i32_ptr(values: &[i32]) -> *const i32 {
    if values.is_empty() {
        ptr::null()
    } else {
        values.as_ptr()
    }
}

#[cfg(feature = "real")]
fn check_status(status: c_int, error: *mut c_char) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = if error.is_null() {
        "unknown xgrammar FFI failure".to_string()
    } else {
        // SAFETY: The shim returns a NUL-terminated malloc-allocated string
        // for errors, and `arle_xgrammar_free_error` releases it.
        let out = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: `error` was allocated by the shim for this exact purpose.
        unsafe { ffi::arle_xgrammar_free_error(error) };
        out
    };
    Err(XGrammarError::Ffi(message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitmask_size_rounds_up_to_u32_words() {
        assert_eq!(bitmask_size(0).unwrap(), 0);
        assert_eq!(bitmask_size(1).unwrap(), 1);
        assert_eq!(bitmask_size(32).unwrap(), 1);
        assert_eq!(bitmask_size(33).unwrap(), 2);
    }

    #[cfg(not(feature = "real"))]
    #[test]
    fn compiler_reports_unavailable_without_real_feature() {
        let err = GrammarCompiler::new(&["a", "b"], CompilerConfig::default()).unwrap_err();
        assert!(matches!(err, XGrammarError::Unavailable));
    }

    #[cfg(feature = "real")]
    #[test]
    fn real_backend_compiles_ebnf_and_fills_bitmask() {
        let mut config = CompilerConfig {
            vocab_size: Some(3),
            stop_token_ids: vec![2],
            ..CompilerConfig::default()
        };
        config.max_threads = 1;
        let mut compiler = GrammarCompiler::new(&["a", "b", ""], config).unwrap();
        let grammar = Arc::new(compiler.compile_ebnf("root ::= \"a\"", "root").unwrap());
        let mut matcher = GrammarMatcher::new(grammar, MatcherConfig::default()).unwrap();
        let mut bitmask = vec![0_u32; bitmask_size(3).unwrap()];
        let _ = matcher.fill_next_token_bitmask(&mut bitmask).unwrap();
        assert!(matcher.accept_token(0).unwrap());
        assert!(matcher.is_completed());
    }
}
