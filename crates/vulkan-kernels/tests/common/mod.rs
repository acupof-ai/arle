//! Shared test helpers for the on-device Vulkan gates.

/// Obtain a [`VulkanContext`], or SKIP the test cleanly when no device exists.
///
/// CI sets `ARLE_REQUIRE_VULKAN_DEVICE=1`: with that set, a missing device
/// PANICS instead of silently passing (an uninstalled ICD otherwise makes every
/// device gate a no-op, so a green suite would prove nothing).
pub fn require_device() -> Option<vulkan_sys::VulkanContext> {
    match vulkan_sys::VulkanContext::create() {
        Ok(ctx) => {
            eprintln!("ARLE Vulkan device gate on: {}", ctx.device_name());
            Some(ctx)
        }
        Err(e) => {
            if std::env::var_os("ARLE_REQUIRE_VULKAN_DEVICE").is_some() {
                panic!(
                    "ARLE_REQUIRE_VULKAN_DEVICE set but no Vulkan device is available: {e}\n\
                     (set VK_ICD_FILENAMES to a valid ICD, e.g. a MoltenVK json)"
                );
            }
            eprintln!("no Vulkan device available ({e}); skipping device test");
            None
        }
    }
}
