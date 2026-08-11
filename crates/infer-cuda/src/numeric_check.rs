//! NaN/Inf detection for CUDA forward outputs.
//!
//! Controlled by `ARLE_NUMERIC_CHECK` (default off). When set, samples the
//! first and last 1024 bf16 elements of a device buffer, copies them to the
//! host, and logs an error if any NaN or Inf is found. The sample size keeps
//! the overhead bounded even for large (vocab-sized) logits buffers.

use std::sync::OnceLock;

use cuda_kernels::prelude::DeviceContext;
use cudarc::driver::CudaSlice;
use half::bf16;

fn numeric_check_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ARLE_NUMERIC_CHECK").is_ok_and(|v| {
            !v.is_empty() && v != "0" && v.to_lowercase() != "false"
        })
    })
}

/// Check a device bf16 buffer for NaN/Inf. No-op unless `ARLE_NUMERIC_CHECK`
/// is set. Samples the first and last [`SAMPLE`] elements (or the whole buffer
/// when smaller) to bound the D2H + scan cost.
pub(crate) fn check_numeric(ctx: &DeviceContext, data: &CudaSlice<bf16>, name: &str) {
    if !numeric_check_enabled() {
        return;
    }
    let len = data.len();
    if len == 0 {
        return;
    }
    const SAMPLE: usize = 1024;
    let n = len.min(SAMPLE);

    let scan = |view: cudarc::driver::CudaView<'_, bf16>| -> (usize, usize) {
        let host = match ctx.stream.clone_dtoh(&view) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("[numeric] D2H failed for {name}: {e}");
                return (0, 0);
            }
        };
        if let Err(e) = ctx.sync() {
            log::warn!("[numeric] sync failed for {name}: {e}");
            return (0, 0);
        }
        let mut nan = 0usize;
        let mut inf = 0usize;
        for x in host {
            let f = x.to_f32();
            if f.is_nan() {
                nan += 1;
            } else if f.is_infinite() {
                inf += 1;
            }
        }
        (nan, inf)
    };

    let (mut nan, mut inf) = scan(data.slice(..n));
    if len > SAMPLE {
        let (tail_nan, tail_inf) = scan(data.slice(len - n..len));
        nan += tail_nan;
        inf += tail_inf;
    }

    if nan > 0 || inf > 0 {
        let sampled = if len > SAMPLE { 2 * n } else { n };
        log::error!(
            "[numeric] {name} (len={len}) contains NaN={nan} Inf={inf} \
             (sampled {sampled} of {len} elements)"
        );
    }
}
