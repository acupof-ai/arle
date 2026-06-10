// hipcc-only stand-in for <cuda.h> so unmodified cuda-kernels sources
// preprocess; the force-included arle_hip_shim.h supplies the surface.
// Never on an nvcc include path.
#pragma once
