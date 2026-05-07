#include <nvtx3/nvToolsExt.h>

void arle_nvtx_range_push(const char *message) {
    nvtxRangePushA(message);
}

void arle_nvtx_range_pop(void) {
    nvtxRangePop();
}
