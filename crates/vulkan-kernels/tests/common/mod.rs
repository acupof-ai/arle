//! Shared test helpers for the on-device Vulkan gates.

/// Obtain a [`VulkanContext`], or SKIP the test cleanly when no device exists.
///
/// No CI workflow sets `ARLE_REQUIRE_VULKAN_DEVICE`, and no CI lane provisions
/// a Vulkan ICD (the Apple-Silicon lane is Metal-only — no MoltenVK/VK_ICD
/// setup), so these device gates skip in every current lane. The knob is the
/// mechanism by which a runner that DOES provide a device makes a missing one
/// panic instead of silently pass; it is unbacked until such a lane exists.
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
