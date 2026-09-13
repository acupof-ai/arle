//! Shared test helpers for the on-device Vulkan gates.

/// Obtain a [`VulkanContext`], or skip when no device/ICD exists. These are
/// manual-only device gates: no CI lane provisions a Vulkan ICD, so a missing
/// device is a plain skip, never a pass and never a failure. The note goes to
/// stderr (libtest captures it unless `--nocapture` is used); on a box with
/// an ICD — set VK_ICD_FILENAMES to a valid MoltenVK json if needed — run with
/// `--nocapture` to see the gate execute.
pub fn require_device() -> Option<vulkan_sys::VulkanContext> {
    match vulkan_sys::VulkanContext::create() {
        Ok(ctx) => {
            eprintln!("ARLE Vulkan device gate on: {}", ctx.device_name());
            Some(ctx)
        }
        Err(e) => {
            eprintln!(
                "skipping: no Vulkan device/ICD available ({e}); \
                 set VK_ICD_FILENAMES to a valid ICD, e.g. a MoltenVK json"
            );
            None
        }
    }
}
