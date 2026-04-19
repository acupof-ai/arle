// Env-gated MTLCaptureManager hook for Qwen3.5 decode-step GPU capture.
//
// Activated via environment variables — default OFF so production paths pay no
// cost beyond a single thread-safe getenv cache check + atomic increment.
//
// Env vars:
//   INFER_CAPTURE_STEP=N    Capture the Nth invocation (0-indexed; N=0 = first
//                           call after the hook is first observed). Unset =
//                           capture disabled entirely (zero work on the hot
//                           path beyond one relaxed atomic load).
//   INFER_CAPTURE_PATH=...  Destination `.gputrace` path. Default:
//                           /tmp/qwen35_step_<unix_timestamp>.gputrace
//   MTL_CAPTURE_ENABLED=1   Required by Apple's toolchain before programmatic
//                           capture is allowed; we don't check this ourselves
//                           but startCaptureWithDescriptor: will fail without
//                           it.
//
// C ABI consumed by mlx_qwen35_model.cpp::qwen35_compiled_step_session.
//
// See docs/plans/metal-gdr-kernel-xcode-capture.md §Step 2b.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>

namespace {

// Cached capture target parsed once on first entry. -1 means "env var missing"
// (disable capture cheaply on every subsequent call). >= 0 means "fire when
// counter reaches this value".
struct CaptureConfig {
    int32_t target_step;   // -1 = disabled
    std::string out_path;
};

const CaptureConfig& capture_config() {
    static const CaptureConfig cfg = [] {
        CaptureConfig c;
        const char* env = std::getenv("INFER_CAPTURE_STEP");
        if (!env || *env == '\0') {
            c.target_step = -1;
            return c;
        }
        char* end = nullptr;
        long v = std::strtol(env, &end, 10);
        if (end == env || v < 0) {
            c.target_step = -1;
            return c;
        }
        c.target_step = static_cast<int32_t>(v);

        const char* path = std::getenv("INFER_CAPTURE_PATH");
        if (path && *path != '\0') {
            c.out_path = path;
        } else {
            char buf[128];
            std::snprintf(buf, sizeof(buf), "/tmp/qwen35_step_%ld.gputrace",
                          static_cast<long>(std::time(nullptr)));
            c.out_path = buf;
        }
        return c;
    }();
    return cfg;
}

std::atomic<int32_t> g_step_counter{0};
std::atomic<bool> g_capture_active{false};

}  // namespace

extern "C" {

// Returns non-zero if a capture was started for this step. The caller must
// invoke maybe_capture_qwen35_step_end() with the same return value.
int32_t maybe_capture_qwen35_step_begin(void) {
    const auto& cfg = capture_config();
    if (cfg.target_step < 0) {
        return 0;  // env var not set; no-op
    }
    int32_t n = g_step_counter.fetch_add(1, std::memory_order_relaxed);
    if (n != cfg.target_step) {
        return 0;
    }

    @autoreleasepool {
        MTLCaptureManager* cm = [MTLCaptureManager sharedCaptureManager];
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            std::fprintf(stderr,
                         "[capture] MTLCreateSystemDefaultDevice returned nil; "
                         "skipping capture\n");
            return 0;
        }

        if (![cm supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
            std::fprintf(stderr,
                         "[capture] MTLCaptureDestinationGPUTraceDocument "
                         "unsupported. Did you export MTL_CAPTURE_ENABLED=1?\n");
            return 0;
        }

        MTLCaptureDescriptor* desc = [[MTLCaptureDescriptor alloc] init];
        desc.captureObject = device;
        desc.destination = MTLCaptureDestinationGPUTraceDocument;
        NSString* path = [NSString stringWithUTF8String:cfg.out_path.c_str()];
        desc.outputURL = [NSURL fileURLWithPath:path];

        NSError* err = nil;
        if (![cm startCaptureWithDescriptor:desc error:&err]) {
            std::fprintf(stderr,
                         "[capture] startCaptureWithDescriptor failed: %s\n",
                         err ? [[err localizedDescription] UTF8String]
                             : "(null error)");
            return 0;
        }
        std::fprintf(stderr, "[capture] started GPU trace at step %d → %s\n",
                     cfg.target_step, cfg.out_path.c_str());
        g_capture_active.store(true, std::memory_order_release);
        return 1;
    }
}

void maybe_capture_qwen35_step_end(int32_t started) {
    if (!started) {
        return;
    }
    if (!g_capture_active.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    @autoreleasepool {
        MTLCaptureManager* cm = [MTLCaptureManager sharedCaptureManager];
        [cm stopCapture];
        std::fprintf(stderr,
                     "[capture] stopped GPU trace → %s\n",
                     capture_config().out_path.c_str());
    }
}

}  // extern "C"
