#include <cuda.h>

#ifdef ARLE_ENABLE_DEEPGEMM_NATIVE

#include <cuda_runtime.h>

#include <sys/file.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <csrc/jit_kernels/heuristics/sm90_mega_moe_core.hpp>
#include <deep_gemm/layout/sym_buffer.cuh>

#ifndef ARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT
#define ARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT "vendor/deepgemm/deep_gemm"
#endif

#ifndef ARLE_DEEPGEMM_DEFAULT_CUDA_HOME
#define ARLE_DEEPGEMM_DEFAULT_CUDA_HOME "/usr/local/cuda"
#endif

#ifndef ARLE_DEEPGEMM_DEFAULT_CUTLASS_INCLUDE
#define ARLE_DEEPGEMM_DEFAULT_CUTLASS_INCLUDE ""
#endif

namespace {

constexpr int kScaleGranK = 128;
constexpr int kSm90SmemCapacity = 232448;
constexpr int kMaxStages = 16;

enum class Major { K, MN };

struct Layout {
  int block_m;
  int block_n;
  int block_k;
  int cluster_m;
  int cluster_n;

  int cluster_size() const { return cluster_m * cluster_n; }
};

struct StorageConfig {
  int load_block_m;
  int load_block_n;
  int store_block_m;
  int store_block_n;
  int swizzle_a_mode;
  int swizzle_b_mode;
  int swizzle_cd_mode;
};

struct PipelineConfig {
  int smem_size;
  int num_stages;
};

struct LaunchConfig {
  int num_sms;
  int num_threads;
  int num_tma_threads;
  int num_math_threads;
};

struct GemmConfig {
  Layout layout;
  StorageConfig storage;
  PipelineConfig pipeline;
  LaunchConfig launch;
};

struct GemmDesc {
  int m;
  int n;
  int k;
  int num_groups;
  int num_sms;
  int expected_m;
  int expected_n;
  int expected_k;
  int expected_num_groups;
  // Upper bound on block_m for MGroupedContiguous: the scheduler resolves the
  // B group once per BLOCK_M tile from m_indices[tile_start], so block_m must
  // not exceed the caller's per-group row alignment. 128 (the legacy contiguous
  // alignment) keeps every pre-existing call site unchanged; the DSv4 decode
  // band passes 64 so groups can be packed 64-aligned (half the pad rows).
  int max_block_m = 128;
};

struct Sm90MegaMoeWorkspaceLayout {
  uint64_t num_bytes;
  uint64_t x;
  uint64_t x_sf;
  uint64_t topk_idx;
  uint64_t topk_weights;
  uint64_t l1_acts;
  uint64_t l1_acts_sf;
  uint64_t l1_topk_weights;
  uint64_t l2_acts;
  uint64_t l2_acts_sf;
  uint64_t combine;
  int num_max_pool_tokens;
  int num_padded_sf_pool_tokens;
  int num_max_tokens_per_rank;
};

struct LayoutInfo {
  int64_t num_cycles;
  Layout layout;
};

void* cuda_driver_handle() {
  static void* handle = nullptr;
  if (handle == nullptr) {
    handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      throw std::runtime_error("failed to load CUDA driver libcuda.so.1");
    }
  }
  return handle;
}

#define ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(name)       \
  template <typename... Args>                           \
  auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&(name));                 \
    static FuncType func = nullptr;                     \
    if (func == nullptr) {                              \
      func = reinterpret_cast<FuncType>(dlsym(cuda_driver_handle(), #name)); \
      if (func == nullptr) {                            \
        throw std::runtime_error("failed to load CUDA driver API " #name); \
      }                                                 \
    }                                                   \
    return func(std::forward<Args>(args)...);           \
  }

ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuCtxGetCurrent);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuTensorMapEncodeTiled);

ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
#if CUDART_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;
#else
using LibraryHandle = CUmodule;
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;
#endif

void unload_library(const LibraryHandle& library);

struct NativeRuntime {
  LibraryHandle library{};
  KernelHandle kernel{};
  bool loaded = false;

  ~NativeRuntime() {
    if (!loaded) return;
    try {
      unload_library(library);
    } catch (...) {
    }
  }
};

std::mutex g_runtime_mu;
std::unordered_map<std::string, std::shared_ptr<NativeRuntime>> g_runtimes;
std::unordered_map<std::string, std::string> g_runtime_failures;

template <typename T>
T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
T align_to(T a, T b) {
  return ceil_div(a, b) * b;
}

const char* non_empty_env(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' ? value : fallback;
}

std::filesystem::path cuda_home_path() {
  const char* arle_jit_cuda_home = std::getenv("ARLE_DEEPGEMM_JIT_CUDA_HOME");
  if (arle_jit_cuda_home != nullptr && arle_jit_cuda_home[0] != '\0') {
    return arle_jit_cuda_home;
  }
  const char* dg_jit_cuda_home = std::getenv("DG_JIT_CUDA_HOME");
  if (dg_jit_cuda_home != nullptr && dg_jit_cuda_home[0] != '\0') {
    return dg_jit_cuda_home;
  }
  const char* cuda_home = std::getenv("CUDA_HOME");
  if (cuda_home != nullptr && cuda_home[0] != '\0') return cuda_home;
  const char* cuda_path = std::getenv("CUDA_PATH");
  if (cuda_path != nullptr && cuda_path[0] != '\0') return cuda_path;
  return ARLE_DEEPGEMM_DEFAULT_CUDA_HOME;
}

std::filesystem::path nvcc_ccbin_path() {
  const char* dg_ccbin = std::getenv("DG_JIT_NVCC_CCBIN");
  if (dg_ccbin != nullptr && dg_ccbin[0] != '\0') return dg_ccbin;
  const char* nvcc_ccbin = std::getenv("NVCC_CCBIN");
  if (nvcc_ccbin != nullptr && nvcc_ccbin[0] != '\0') return nvcc_ccbin;
  return {};
}

std::filesystem::path deepgemm_library_root_path() {
  return std::filesystem::path(non_empty_env(
      "ARLE_DEEPGEMM_LIBRARY_ROOT", ARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT));
}

std::filesystem::path deepgemm_source_root_path(
    const std::filesystem::path& library_root) {
  return library_root.parent_path();
}

std::filesystem::path deepgemm_cutlass_include_path(
    const std::filesystem::path& source_root) {
  const char* cutlass_env = std::getenv("ARLE_DEEPGEMM_CUTLASS_INCLUDE");
  if (cutlass_env != nullptr && cutlass_env[0] != '\0') {
    return std::filesystem::path(cutlass_env);
  }
  constexpr const char* default_cutlass = ARLE_DEEPGEMM_DEFAULT_CUTLASS_INCLUDE;
  if (default_cutlass[0] != '\0') {
    return std::filesystem::path(default_cutlass);
  }
  return source_root / "third-party/cutlass/include";
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  int parsed = fallback;
  return std::sscanf(value, "%d", &parsed) == 1 ? parsed : fallback;
}

CUresult current_device_prop(cudaDeviceProp* prop_out) {
  int device = 0;
  cudaError_t cuda_err = cudaGetDevice(&device);
  if (cuda_err != cudaSuccess) return static_cast<CUresult>(cuda_err);

  struct CachedProp {
    int device = -1;
    cudaDeviceProp prop{};
  };
  static thread_local CachedProp cache;
  if (cache.device != device) {
    cuda_err = cudaGetDeviceProperties(&cache.prop, device);
    if (cuda_err != cudaSuccess) return static_cast<CUresult>(cuda_err);
    cache.device = device;
  }
  *prop_out = cache.prop;
  return CUDA_SUCCESS;
}

std::string shell_quote(const std::filesystem::path& path) {
  std::string input = path.string();
  std::string out = "'";
  for (char c : input) {
    if (c == '\'') {
      out += "'\\''";
    } else {
      out += c;
    }
  }
  out += "'";
  return out;
}

std::tuple<int, std::string> run_capture(const std::string& command) {
  const std::string full = command + " 2>&1";
  FILE* pipe = popen(full.c_str(), "r");
  if (pipe == nullptr) throw std::runtime_error("popen failed: " + command);
  std::array<char, 512> buffer{};
  std::string output;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = pclose(pipe);
  const int exit_code =
      WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  return {exit_code, output};
}

std::filesystem::path cache_root();

class ScopedFileLock {
 public:
  explicit ScopedFileLock(const std::filesystem::path& path) {
    std::filesystem::create_directories(path.parent_path());
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) {
      throw std::runtime_error(
          "failed to open DeepGEMM JIT lock " + path.string() + ": " +
          std::strerror(errno));
    }
    while (::flock(fd_, LOCK_EX) != 0) {
      if (errno == EINTR) continue;
      const std::string err = std::strerror(errno);
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(
          "failed to lock DeepGEMM JIT lock " + path.string() + ": " + err);
    }
  }

  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;

  ~ScopedFileLock() {
    if (fd_ >= 0) {
      ::flock(fd_, LOCK_UN);
      ::close(fd_);
    }
  }

 private:
  int fd_ = -1;
};

bool path_is_directory(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

bool path_is_regular_file(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

void write_c_string(char* out, size_t out_len, const std::string& value) {
  if (out == nullptr || out_len == 0) return;
  const size_t n = std::min(out_len - 1, value.size());
  std::memcpy(out, value.data(), n);
  out[n] = '\0';
}

struct PreflightReport {
  bool ok = true;
  std::string message;
};

PreflightReport deepgemm_preflight_report() {
  const auto library_root = deepgemm_library_root_path();
  const auto source_root = deepgemm_source_root_path(library_root);
  const auto cuda_home = cuda_home_path();
  const auto nvcc = cuda_home / "bin" / "nvcc";
  const auto cuobjdump = cuda_home / "bin" / "cuobjdump";
  const auto nvcc_ccbin = nvcc_ccbin_path();
  const auto cutlass_include = deepgemm_cutlass_include_path(source_root);
  const auto deepgemm_include = library_root / "include";
  const auto deepgemm_header =
      deepgemm_include / "deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh";
  const auto cutlass_barrier = cutlass_include / "cutlass/arch/barrier.h";

  std::vector<std::string> missing;
  if (!path_is_directory(deepgemm_include)) {
    missing.push_back("deepgemm_include=" + deepgemm_include.string());
  }
  if (!path_is_regular_file(deepgemm_header)) {
    missing.push_back("deepgemm_header=" + deepgemm_header.string());
  }
  if (!path_is_directory(cutlass_include)) {
    missing.push_back("cutlass_include=" + cutlass_include.string());
  }
  if (!path_is_regular_file(cutlass_barrier)) {
    missing.push_back("cutlass_barrier=" + cutlass_barrier.string());
  }
  if (!path_is_regular_file(nvcc)) {
    missing.push_back("nvcc=" + nvcc.string());
  }
  if (!path_is_regular_file(cuobjdump)) {
    missing.push_back("cuobjdump=" + cuobjdump.string());
  }
  if (!nvcc_ccbin.empty() && !path_is_regular_file(nvcc_ccbin)) {
    missing.push_back("nvcc_ccbin=" + nvcc_ccbin.string());
  }

  std::ostringstream out;
  out << "library_root=" << library_root
      << " source_root=" << source_root
      << " cutlass_include=" << cutlass_include
      << " cuda_home=" << cuda_home
      << " nvcc=" << nvcc
      << " nvcc_ccbin=" << (nvcc_ccbin.empty() ? "<default>" : nvcc_ccbin.string())
      << " cuobjdump=" << cuobjdump
      << " jit_cache=" << cache_root();
  if (missing.empty()) {
    return {true, "status=ok " + out.str()};
  }

  out << " missing=";
  for (size_t i = 0; i < missing.size(); ++i) {
    if (i != 0) out << ",";
    out << missing[i];
  }
  return {false, "status=failed " + out.str()};
}

uint64_t fnv1a(const std::string& data, uint64_t seed) {
  uint64_t h = seed;
  constexpr uint64_t prime = 0x100000001b3ull;
  for (unsigned char c : data) {
    h ^= c;
    h *= prime;
  }
  return h;
}

uint64_t split_mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

std::string hex_digest(const std::string& data) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16)
      << split_mix(fnv1a(data, 0xc6a4a7935bd1e995ull)) << std::setw(16)
      << split_mix(fnv1a(data, 0x9e3779b97f4a7c15ull));
  return out.str();
}

void write_binary_file(const std::filesystem::path& path, const std::string& data) {
  std::ofstream out(path, std::ios::binary);
  if (!out.write(data.data(), static_cast<std::streamsize>(data.size()))) {
    throw std::runtime_error("failed to write " + path.string());
  }
}

void check_driver(CUresult result, const char* what) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* info = nullptr;
  lazy_cuGetErrorName(result, &name);
  lazy_cuGetErrorString(result, &info);
  std::ostringstream out;
  out << what << " failed: " << static_cast<int>(result);
  if (name != nullptr) out << " (" << name << ")";
  if (info != nullptr) out << ": " << info;
  throw std::runtime_error(out.str());
}

std::string current_context_key() {
  CUcontext ctx = nullptr;
  check_driver(lazy_cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
  if (ctx == nullptr) {
    throw std::runtime_error("DeepGEMM native bridge requires a current CUDA context");
  }
  std::ostringstream out;
  out << "ctx:" << reinterpret_cast<std::uintptr_t>(ctx);
  return out.str();
}

KernelHandle load_kernel(
    const std::filesystem::path& cubin_path,
    const std::string& func_name,
    LibraryHandle* library_out) {
  LibraryHandle library{};
  KernelHandle kernel{};
#if CUDART_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
  const auto load_error = cudaLibraryLoadFromFile(
      &library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (load_error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(load_error));
  const auto kernel_error = cudaLibraryGetKernel(&kernel, library, func_name.c_str());
  if (kernel_error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(kernel_error));
#else
  check_driver(lazy_cuModuleLoad(&library, cubin_path.c_str()), "cuModuleLoad");
  check_driver(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()),
               "cuModuleGetFunction");
#endif
  if (library_out != nullptr) *library_out = library;
  return kernel;
}

void unload_library(const LibraryHandle& library) {
#if CUDART_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
  const auto err = cudaLibraryUnload(library);
  if (err != cudaSuccess && err != cudaErrorCudartUnloading)
    throw std::runtime_error(cudaGetErrorString(err));
#else
  const auto err = lazy_cuModuleUnload(library);
  if (err != CUDA_SUCCESS && err != CUDA_ERROR_DEINITIALIZED) {
    check_driver(err, "CUDA library unload");
  }
#endif
}

LaunchConfigHandle construct_launch_config(
    KernelHandle kernel,
    cudaStream_t stream,
    int smem_size,
    dim3 grid_dim,
    dim3 block_dim,
    int cluster_dim,
    bool enable_pdl) {
#if CUDART_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
  if (smem_size > 0) {
    const auto err = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  }
  LaunchConfigHandle config{};
  config.gridDim = grid_dim;
  config.blockDim = block_dim;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  static thread_local LaunchAttrHandle attrs[2];
  config.numAttrs = 0;
  config.attrs = attrs;
  if (cluster_dim > 1) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
  }
  if (enable_pdl) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
  }
#else
  if (smem_size > 0) {
    check_driver(
        lazy_cuFuncSetAttribute(
            kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size),
        "cuFuncSetAttribute");
  }

  LaunchConfigHandle config{};
  config.gridDimX = grid_dim.x;
  config.gridDimY = grid_dim.y;
  config.gridDimZ = grid_dim.z;
  config.blockDimX = block_dim.x;
  config.blockDimY = block_dim.y;
  config.blockDimZ = block_dim.z;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;

  static thread_local LaunchAttrHandle attrs[2];
  config.numAttrs = 0;
  config.attrs = attrs;

  if (cluster_dim > 1) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    attr.value.clusterDim.x = static_cast<unsigned>(cluster_dim);
    attr.value.clusterDim.y = 1;
    attr.value.clusterDim.z = 1;
  }

  if (enable_pdl) {
    auto& attr = attrs[config.numAttrs++];
    attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr.value.programmaticStreamSerializationAllowed = 1;
  }
#endif

  return config;
}

template <typename... Args>
CUresult launch_kernel(KernelHandle kernel, const LaunchConfigHandle& config, Args&&... args) {
  void* ptr_args[] = {
      const_cast<void*>(reinterpret_cast<const void*>(&args))...,
  };
#if CUDART_VERSION >= 12080 && defined(DG_JIT_USE_RUNTIME_API)
  return static_cast<CUresult>(cudaLaunchKernelExC(&config, kernel, ptr_args));
#else
  return lazy_cuLaunchKernelEx(
      const_cast<LaunchConfigHandle*>(&config), kernel, ptr_args, nullptr);
#endif
}

int get_tma_aligned_size(int x, int elem_size) {
  constexpr int kTmaAlignmentBytes = 16;
  if ((kTmaAlignmentBytes % elem_size) != 0) {
    throw std::runtime_error("invalid TMA element size");
  }
  return align_to(x, kTmaAlignmentBytes / elem_size);
}

int get_swizzle_mode(int block_size, int elem_size) {
  for (int mode : {128, 64, 32, 16}) {
    if ((block_size * elem_size) % mode == 0) return mode;
  }
  throw std::runtime_error("unsupported DeepGEMM swizzle shape");
}

StorageConfig get_storage_config(const Layout& layout) {
  const int load_block_m = layout.block_m;
  const int load_block_n = layout.block_n;
  const int store_block_m = layout.block_m;
  const int store_block_n = layout.block_n;
  return {
      load_block_m,
      load_block_n,
      store_block_m,
      store_block_n,
      get_swizzle_mode(layout.block_k, 1),
      get_swizzle_mode(layout.block_k, 1),
      get_swizzle_mode(store_block_n, 2),
  };
}

PipelineConfig get_pipeline_config(
    const GemmDesc& desc,
    const Layout& layout,
    const StorageConfig& storage) {
  const int smem_cd = align_to(layout.block_m * layout.block_n * 2, 1024);
  const int smem_barriers = kMaxStages * 8 * 2;
  const int smem_a_per_stage = storage.load_block_m * layout.block_k;
  const int smem_b_per_stage = storage.load_block_n * layout.block_k;
  const int smem_sfa_per_stage = align_to(layout.block_m * 4, 128);
  const int use_uniform_sfb = layout.block_k % layout.block_n == 0 ? 1 : 2;
  const int smem_extra_sfb =
      align_to(ceil_div(desc.k, layout.block_k) * 4 * use_uniform_sfb, 8);
  const int smem_extra = smem_cd + smem_barriers + smem_extra_sfb;
  const int smem_per_stage =
      smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage;
  const int num_stages =
      std::min((kSm90SmemCapacity - smem_extra) / smem_per_stage, kMaxStages);
  return {smem_extra + num_stages * smem_per_stage, num_stages};
}

LayoutInfo get_layout_info(const GemmDesc& desc, const Layout& layout) {
  const int64_t num_blocks =
      static_cast<int64_t>(ceil_div(desc.expected_m, layout.block_m)) *
      ceil_div(desc.expected_n, layout.block_n) * desc.expected_num_groups;
  const int64_t num_waves = ceil_div(num_blocks, static_cast<int64_t>(desc.num_sms));
  const double l2_bandwidth_per_cycle =
      std::min(64.0 * static_cast<double>(desc.num_sms), 8e6 / 1.3e3);
  const double l1_bandwidth_per_cycle = 128.0 * static_cast<double>(desc.num_sms);
  const int64_t num_bytes_l2_ab =
      static_cast<int64_t>(desc.expected_k) *
      (layout.block_m / layout.cluster_n + layout.block_n / layout.cluster_m);
  const int64_t num_bytes_l1_ab =
      static_cast<int64_t>(desc.expected_k) * (layout.block_m + layout.block_n);
  const int64_t num_bytes_l1_tc =
      static_cast<int64_t>(desc.expected_k) *
          (std::max(64, layout.block_m) + layout.block_n) +
      static_cast<int64_t>(layout.block_m) * layout.block_n * 2;
  const int64_t num_bytes_l1_l2_cd =
      static_cast<int64_t>(layout.block_m) * layout.block_n * 2;
  const double num_l2_cycles =
      static_cast<double>(num_bytes_l2_ab + num_bytes_l1_l2_cd) *
      static_cast<double>(num_blocks) / l2_bandwidth_per_cycle;
  const double num_l1_cycles =
      static_cast<double>(num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) *
      static_cast<double>(num_blocks) / l1_bandwidth_per_cycle;
  const double wave_efficiency =
      static_cast<double>(num_blocks) /
      static_cast<double>(num_waves * desc.num_sms);
  int64_t cycles =
      static_cast<int64_t>(std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency);
  if (layout.cluster_size() > 1 && num_waves <= 1) {
    cycles = std::numeric_limits<int64_t>::max();
  }
  return {cycles, layout};
}

std::vector<Layout> get_layout_candidates(const GemmDesc& desc) {
  std::vector<Layout> candidates;
  const int block_k = kScaleGranK;
  const int step = std::lcm(16, 1);
  // ARLE_DEEPGEMM_CONSERVATIVE_LAYOUT (b335 H20 NOT_SUPPORTED probe): force the
  // smem-light, cluster-free config (block_m=64, cluster=1) to isolate whether
  // the larger block_m=128 / cluster=2 config is what H20 rejects at the unpadded
  // prefill max_m (max_m=56..96 → CUDA_ERROR_NOT_SUPPORTED, while the decode
  // max_m=8 config works). If the GEMM succeeds with this set, the config (smem
  // or cluster) is the cause, not the SFA TMA / scale layout.
  const bool conservative = env_int("ARLE_DEEPGEMM_CONSERVATIVE_LAYOUT", 0) != 0;
  const int max_cluster = conservative ? 1 : 2;
  for (int cluster_m = 1; cluster_m <= max_cluster; ++cluster_m) {
    for (int cluster_n = 1; cluster_n <= max_cluster; ++cluster_n) {
      if (cluster_m * cluster_n > 2) continue;
      if (desc.num_sms % (cluster_m * cluster_n) != 0) continue;
      for (int block_m : {64, 128}) {
        if (conservative && block_m != 64) continue;
        if (block_m > desc.max_block_m) continue;
        for (int block_n = step; block_n <= 192; block_n += step) {
          if (block_n > block_k) {
            const int diff = block_n - block_k;
            if (block_n % diff != 0 && block_k % diff != 0) continue;
          }
          if (ceil_div(desc.n, block_n) % (cluster_m * cluster_n) != 0) {
            continue;
          }
          const Layout layout{block_m, block_n, block_k, cluster_m, cluster_n};
          const auto storage = get_storage_config(layout);
          if (storage.swizzle_a_mode % 64 != 0 || storage.swizzle_b_mode % 64 != 0) {
            continue;
          }
          const int stages = get_pipeline_config(desc, layout, storage).num_stages;
          if (stages < 3 || (block_m * block_n < 128 * 192 && stages < 4)) {
            continue;
          }
          candidates.push_back(layout);
        }
      }
    }
  }
  return candidates;
}

GemmConfig get_best_config(const GemmDesc& desc) {
  const auto candidates = get_layout_candidates(desc);
  if (candidates.empty()) throw std::runtime_error("no DeepGEMM layout candidates");
  LayoutInfo best = get_layout_info(desc, candidates[0]);
  for (size_t i = 1; i < candidates.size(); ++i) {
    const auto candidate = get_layout_info(desc, candidates[i]);
    if (candidate.num_cycles < best.num_cycles) best = candidate;
  }
  const auto storage = get_storage_config(best.layout);
  const auto pipeline = get_pipeline_config(desc, best.layout, storage);
  const int num_math_threads = best.layout.block_m <= 64 ? 128 : 256;
  return {
      best.layout,
      storage,
      pipeline,
      {desc.num_sms, 128 + num_math_threads, 128, num_math_threads},
  };
}

std::pair<int, int> get_inner_outer_dims(Major major, int k, int mn) {
  return major == Major::K ? std::make_pair(k, mn) : std::make_pair(mn, k);
}

CUtensorMapSwizzle mode_into_tensor_map_swizzle(int mode) {
  switch (mode) {
    case 0:
    case 16:
      return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
      return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
      return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
      return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
      throw std::runtime_error("unsupported TMA swizzle mode");
  }
}

CUtensorMap make_tma_2d_desc_raw(
    const void* ptr,
    CUtensorMapDataType dtype,
    int elem_size,
    int gmem_inner_dim,
    int gmem_outer_dim,
    int smem_inner_dim,
    int smem_outer_dim,
    int gmem_outer_stride,
    int swizzle_mode) {
  if (swizzle_mode != 0) smem_inner_dim = swizzle_mode / elem_size;
  CUtensorMap tensor_map{};
  const cuuint64_t gmem_dims[2] = {
      static_cast<cuuint64_t>(gmem_inner_dim),
      static_cast<cuuint64_t>(gmem_outer_dim),
  };
  const cuuint32_t smem_dims[2] = {
      static_cast<cuuint32_t>(smem_inner_dim),
      static_cast<cuuint32_t>(smem_outer_dim),
  };
  const cuuint64_t gmem_strides[1] = {
      static_cast<cuuint64_t>(gmem_outer_stride * elem_size),
  };
  const cuuint32_t elem_strides[2] = {1, 1};
  check_driver(
      lazy_cuTensorMapEncodeTiled(
          &tensor_map, dtype, 2, const_cast<void*>(ptr), gmem_dims, gmem_strides,
          smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          mode_into_tensor_map_swizzle(swizzle_mode),
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

CUtensorMap make_tma_sf_desc_raw(
    const void* ptr,
    int shape_mn,
    int shape_k,
    int block_mn,
    int gran_k) {
  shape_mn = get_tma_aligned_size(shape_mn, sizeof(float));
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sizeof(float), shape_mn,
      ceil_div(shape_k, gran_k), block_mn, 1, shape_mn, 0);
}

CUtensorMap make_tma_3d_desc_raw(
    const void* ptr,
    CUtensorMapDataType dtype,
    int elem_size,
    int gmem_dim_0,
    int gmem_dim_1,
    int gmem_dim_2,
    int smem_dim_0,
    int smem_dim_1,
    int smem_dim_2,
    int gmem_stride_0,
    int gmem_stride_1,
    int swizzle_mode) {
  if (swizzle_mode != 0) smem_dim_0 = swizzle_mode / elem_size;
  CUtensorMap tensor_map{};
  const cuuint64_t gmem_dims[3] = {
      static_cast<cuuint64_t>(gmem_dim_0),
      static_cast<cuuint64_t>(gmem_dim_1),
      static_cast<cuuint64_t>(gmem_dim_2),
  };
  const cuuint32_t smem_dims[3] = {
      static_cast<cuuint32_t>(smem_dim_0),
      static_cast<cuuint32_t>(smem_dim_1),
      static_cast<cuuint32_t>(smem_dim_2),
  };
  const cuuint64_t gmem_strides[2] = {
      static_cast<cuuint64_t>(gmem_stride_0 * elem_size),
      static_cast<cuuint64_t>(gmem_stride_1 * elem_size),
  };
  const cuuint32_t elem_strides[3] = {1, 1, 1};
  check_driver(
      lazy_cuTensorMapEncodeTiled(
          &tensor_map, dtype, 3, const_cast<void*>(ptr), gmem_dims, gmem_strides,
          smem_dims, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
          mode_into_tensor_map_swizzle(swizzle_mode),
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");
  return tensor_map;
}

CUtensorMap make_tma_a_desc(
    const void* ptr,
    int shape_m,
    int shape_k,
    int block_m,
    int block_k,
    int outer_stride,
    int num_groups,
    int swizzle_mode) {
  const auto [gmem_inner_dim, gmem_outer_dim] =
      get_inner_outer_dims(Major::K, shape_k, shape_m * num_groups);
  const auto [smem_inner_dim, smem_outer_dim] =
      get_inner_outer_dims(Major::K, block_k, block_m);
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, gmem_inner_dim, gmem_outer_dim,
      smem_inner_dim, smem_outer_dim, outer_stride, swizzle_mode);
}

CUtensorMap make_tma_b_desc(
    const void* ptr,
    int shape_n,
    int shape_k,
    int block_n,
    int block_k,
    int outer_stride,
    int num_groups,
    int swizzle_mode) {
  const auto [gmem_inner_dim, gmem_outer_dim] =
      get_inner_outer_dims(Major::K, shape_k, shape_n);
  const auto [smem_inner_dim, smem_outer_dim] =
      get_inner_outer_dims(Major::K, block_k, block_n);
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, gmem_inner_dim,
      gmem_outer_dim * num_groups, smem_inner_dim, smem_outer_dim, outer_stride,
      swizzle_mode);
}

CUtensorMap make_tma_d_desc(
    const void* ptr,
    int shape_m,
    int shape_n,
    int block_m,
    int block_n,
    int outer_stride,
    int num_groups,
    int swizzle_mode) {
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, shape_n, shape_m * num_groups,
      block_n, block_m, outer_stride, swizzle_mode);
}

CUtensorMap make_tma_sfa_desc(
    const void* ptr,
    int shape_m,
    int shape_k,
    int block_m,
    int gran_k,
    int num_groups,
    int outer_stride) {
  const int aligned_m = get_tma_aligned_size(shape_m, 4);
  if (outer_stride < aligned_m) {
    throw std::runtime_error("SFA stride is smaller than DeepGEMM TMA alignment");
  }
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, aligned_m,
      ceil_div(shape_k, gran_k) * num_groups, block_m, 1, outer_stride, 0);
}

std::string bool_lit(bool value) { return value ? "true" : "false"; }

// ── SM90 BF16 m-grouped GEMM (KernelNoSF) ──────────────────────────────────
// Heuristics ported from vendor/deepgemm csrc/jit_kernels/heuristics/sm90.hpp
// for the BF16 (KernelNoSF) kernel family. Differences vs the FP8 1D2D port
// above, per the upstream decision table:
//   * block_k = 128 / elem_size(bf16) = 64 (FP8 uses 128);
//   * no SFA/SFB shared memory and no extra-SFB smem term in the pipeline;
//   * block_n enumerates 16..256 step 16 (KernelNoSF; 1D2D caps at 192) with
//     no 1D2D `block_n > block_k` unroll constraint, but block_m and block_n
//     may not BOTH exceed 128 (register budget);
//   * MGroupedContiguous pins block_m to the contiguous-layout MK alignment
//     (128 on SM90, heuristics/runtime.hpp); MGroupedMasked enumerates
//     {64, 128};
//   * masked multicast legality: ceil_div(n, block_n) divisible by the
//     cluster size (contiguous validates peers at runtime in the scheduler);
//   * cycle model bytes scale by elem_size(bf16) = 2 on the A/B side.

constexpr int kBf16BlockK = 64;
constexpr int kBf16ElemSize = 2;
// SM90 contiguous-layout MK alignment (heuristics/runtime.hpp
// kLegacyMKAlignmentForContiguousLayout). The contiguous scheduler resolves
// the per-tile B group from `m_indices[m_block_idx * BLOCK_M]`
// (scheduler/gemm.cuh `get_global_idx`), so every per-group row segment MUST
// start at a multiple of this; BLOCK_M for MGroupedContiguous is pinned to it.
constexpr int kBf16ContiguousAlignment = 128;

PipelineConfig get_bf16_pipeline_config(const Layout& layout) {
  const int smem_cd =
      align_to(layout.block_m * layout.block_n * kBf16ElemSize, 1024);
  const int smem_barriers = kMaxStages * 8 * 2;
  const int smem_a_per_stage = layout.block_m * layout.block_k * kBf16ElemSize;
  const int smem_b_per_stage = layout.block_n * layout.block_k * kBf16ElemSize;
  const int smem_extra = smem_cd + smem_barriers;
  const int smem_per_stage = smem_a_per_stage + smem_b_per_stage;
  const int num_stages =
      std::min((kSm90SmemCapacity - smem_extra) / smem_per_stage, kMaxStages);
  return {smem_extra + num_stages * smem_per_stage, num_stages};
}

LayoutInfo get_bf16_layout_info(const GemmDesc& desc, const Layout& layout) {
  const int64_t num_blocks =
      static_cast<int64_t>(ceil_div(desc.expected_m, layout.block_m)) *
      ceil_div(desc.expected_n, layout.block_n) * desc.expected_num_groups;
  const int64_t num_waves = ceil_div(num_blocks, static_cast<int64_t>(desc.num_sms));
  const double l2_bandwidth_per_cycle =
      std::min(64.0 * static_cast<double>(desc.num_sms), 8e6 / 1.3e3);
  const double l1_bandwidth_per_cycle = 128.0 * static_cast<double>(desc.num_sms);
  const int64_t num_bytes_l2_ab =
      static_cast<int64_t>(desc.expected_k) * kBf16ElemSize *
      (layout.block_m / layout.cluster_n + layout.block_n / layout.cluster_m);
  const int64_t num_bytes_l1_ab = static_cast<int64_t>(desc.expected_k) *
                                  kBf16ElemSize *
                                  (layout.block_m + layout.block_n);
  const int64_t num_bytes_l1_tc =
      static_cast<int64_t>(desc.expected_k) * kBf16ElemSize *
          (std::max(64, layout.block_m) + layout.block_n) +
      static_cast<int64_t>(layout.block_m) * layout.block_n * kBf16ElemSize;
  const int64_t num_bytes_l1_l2_cd =
      static_cast<int64_t>(layout.block_m) * layout.block_n * kBf16ElemSize;
  const double num_l2_cycles =
      static_cast<double>(num_bytes_l2_ab + num_bytes_l1_l2_cd) *
      static_cast<double>(num_blocks) / l2_bandwidth_per_cycle;
  const double num_l1_cycles =
      static_cast<double>(num_bytes_l1_ab + num_bytes_l1_tc + num_bytes_l1_l2_cd) *
      static_cast<double>(num_blocks) / l1_bandwidth_per_cycle;
  const double wave_efficiency =
      static_cast<double>(num_blocks) /
      static_cast<double>(num_waves * desc.num_sms);
  int64_t cycles =
      static_cast<int64_t>(std::max(num_l1_cycles, num_l2_cycles) / wave_efficiency);
  if (layout.cluster_size() > 1 && num_waves <= 1) {
    cycles = std::numeric_limits<int64_t>::max();
  }
  return {cycles, layout};
}

std::vector<Layout> get_bf16_layout_candidates(const GemmDesc& desc, bool masked) {
  std::vector<Layout> candidates;
  const int block_k = kBf16BlockK;
  std::vector<int> block_m_candidates =
      masked ? std::vector<int>{64, 128}
             : std::vector<int>{kBf16ContiguousAlignment};
  for (int cluster_m = 1; cluster_m <= 2; ++cluster_m) {
    for (int cluster_n = 1; cluster_n <= 2; ++cluster_n) {
      if (cluster_m * cluster_n > 2) continue;
      if (desc.num_sms % (cluster_m * cluster_n) != 0) continue;
      for (int block_m : block_m_candidates) {
        for (int block_n = 16; block_n <= 256; block_n += 16) {
          // Masked multicast legality (upstream sm90.hpp): the N tiling must
          // divide evenly into clusters.
          if (masked &&
              ceil_div(desc.n, block_n) % (cluster_m * cluster_n) != 0) {
            continue;
          }
          // Register budget: at least one block dim must be <= 128.
          if (block_m > 128 && block_n > 128) continue;
          const Layout layout{block_m, block_n, block_k, cluster_m, cluster_n};
          const int swizzle_ab = get_swizzle_mode(block_k, kBf16ElemSize);
          if (swizzle_ab % 64 != 0) continue;
          // STSM epilogue requires a swizzled D store.
          if (get_swizzle_mode(block_n, kBf16ElemSize) == 16) continue;
          const int stages = get_bf16_pipeline_config(layout).num_stages;
          if (stages < 3 || (block_m * block_n < 128 * 192 && stages < 4)) {
            continue;
          }
          candidates.push_back(layout);
        }
      }
    }
  }
  return candidates;
}

// Defined below with the FP8 host path; the BF16 launcher precedes it in
// file order, so forward-declare here.
std::shared_ptr<NativeRuntime> get_or_build_runtime(
    const std::string& name,
    const std::string& code,
    int major,
    int minor);

GemmConfig get_best_bf16_config(const GemmDesc& desc, bool masked) {
  const auto candidates = get_bf16_layout_candidates(desc, masked);
  if (candidates.empty()) {
    throw std::runtime_error("no DeepGEMM BF16 layout candidates");
  }
  LayoutInfo best = get_bf16_layout_info(desc, candidates[0]);
  for (size_t i = 1; i < candidates.size(); ++i) {
    const auto candidate = get_bf16_layout_info(desc, candidates[i]);
    if (candidate.num_cycles < best.num_cycles) best = candidate;
  }
  const int swizzle_ab = get_swizzle_mode(kBf16BlockK, kBf16ElemSize);
  const StorageConfig storage{
      best.layout.block_m,
      best.layout.block_n,
      best.layout.block_m,
      best.layout.block_n,
      swizzle_ab,
      swizzle_ab,
      get_swizzle_mode(best.layout.block_n, kBf16ElemSize),
  };
  const auto pipeline = get_bf16_pipeline_config(best.layout);
  const int num_math_threads = best.layout.block_m <= 64 ? 128 : 256;
  return {
      best.layout,
      storage,
      pipeline,
      {desc.num_sms, 128 + num_math_threads, 128, num_math_threads},
  };
}

std::string generate_bf16_kernel_code(
    const GemmDesc& desc,
    const GemmConfig& config,
    const char* gemm_type) {
  std::ostringstream code;
  code << R"(
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_bf16_gemm_impl<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        0, )" << desc.n << R"(, )" << desc.k << R"(,
        )" << desc.num_groups << R"(,
        )" << config.layout.block_m << R"(, )" << config.layout.block_n << R"(, )"
       << config.layout.block_k << R"(,
        )" << config.storage.swizzle_a_mode << R"(, )"
       << config.storage.swizzle_b_mode << R"(, )"
       << config.storage.swizzle_cd_mode << R"(,
        )" << config.pipeline.num_stages << R"(,
        )" << config.launch.num_tma_threads << R"(, )"
       << config.launch.num_math_threads << R"(,
        )" << config.layout.cluster_size() << R"(, )"
       << bool_lit(config.layout.cluster_n > 1) << R"(,
        )" << config.launch.num_sms << R"(, GemmType::)" << gemm_type << R"(,
        false,
        cutlass::bfloat16_t
    >);
}
)";
  return code.str();
}

// K-major BF16 A/B operand descriptor: gmem `[shape_mn * num_groups, shape_k]`
// row-major, smem box `[block_mn, block_k]` (swizzle overrides the inner dim).
CUtensorMap make_tma_ab_desc_bf16(
    const void* ptr,
    int shape_mn,
    int shape_k,
    int block_mn,
    int block_k,
    int outer_stride,
    int num_groups,
    int swizzle_mode) {
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kBf16ElemSize, shape_k,
      shape_mn * num_groups, block_k, block_mn, outer_stride, swizzle_mode);
}

CUresult launch_sm90_bf16_grouped(
    const unsigned short* a,
    const unsigned short* b,
    unsigned short* d,
    const int* grouped_layout,
    int num_groups,
    int m,
    int n,
    int k,
    int expected_m,
    bool masked,
    CUstream stream) {
  cudaDeviceProp prop{};
  CUresult prop_err = current_device_prop(&prop);
  if (prop_err != CUDA_SUCCESS) return prop_err;
  if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;

  int num_sms = env_int("DG_NUM_SMS", prop.multiProcessorCount);
  if (num_sms <= 0 || num_sms > prop.multiProcessorCount || (num_sms % 2) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  // expected_m is a heuristics-only hint (upstream masked API): the per-group
  // expected valid-row count used by the cycle model to pick block sizes. The
  // contiguous variant expects one flat M; the masked variant weights the
  // model by `num_groups`.
  const GemmDesc desc{
      m,
      n,
      k,
      num_groups,
      num_sms,
      std::min(std::max(expected_m, 1), m),
      n,
      k,
      masked ? num_groups : 1,
  };
  const auto config = get_best_bf16_config(desc, masked);
  const auto code = generate_bf16_kernel_code(
      desc, config, masked ? "MGroupedMasked" : "MGroupedContiguous");
  const auto runtime = get_or_build_runtime(
      masked ? "sm90_bf16_m_grouped_gemm_masked_native"
             : "sm90_bf16_m_grouped_gemm_contiguous_native",
      code, prop.major, prop.minor);

  // Masked: A and D are per-group bands `[num_groups, m, *]`; contiguous: one
  // flat `[m, *]` (the scheduler offsets B/D by the per-tile m_indices group).
  const int ad_groups = masked ? num_groups : 1;
  const auto tensor_map_a = make_tma_ab_desc_bf16(
      a, m, k, config.storage.load_block_m, config.layout.block_k, k, ad_groups,
      config.storage.swizzle_a_mode);
  const auto tensor_map_b = make_tma_ab_desc_bf16(
      b, n, k, config.storage.load_block_n, config.layout.block_k, k, num_groups,
      config.storage.swizzle_b_mode);
  const auto tensor_map_d = make_tma_d_desc(
      d, m, n, config.storage.store_block_m, config.storage.store_block_n, n,
      ad_groups, config.storage.swizzle_cd_mode);

  dim3 grid(static_cast<unsigned>(config.launch.num_sms), 1, 1);
  dim3 block(static_cast<unsigned>(config.launch.num_threads), 1, 1);
  auto launch_config = construct_launch_config(
      runtime->kernel, reinterpret_cast<cudaStream_t>(stream), config.pipeline.smem_size,
      grid, block, config.layout.cluster_size(), false);

  void* grouped_arg = const_cast<int*>(grouped_layout);
  check_driver(
      launch_kernel(
          runtime->kernel, launch_config, grouped_arg, m, n, k, tensor_map_a,
          tensor_map_b, tensor_map_d),
      "DeepGEMM BF16 grouped launch");
  return CUDA_SUCCESS;
}

std::string generate_paged_mqa_metadata_code(
    int aligned_batch_size,
    int split_kv,
    int num_sms) {
  std::ostringstream code;
  code << R"(
#include <deep_gemm/scheduler/paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sched::smxx_paged_mqa_logits_metadata<
        )" << aligned_batch_size << R"(, )" << split_kv << R"(, )" << num_sms << R"(, false
    >);
}
)";
  return code.str();
}

std::string generate_fp8_paged_mqa_logits_code(
    int next_n,
    int num_heads,
    int head_dim,
    int block_kv,
    int num_q_stages,
    int num_kv_stages,
    int split_kv,
    int num_tma_threads,
    int num_math_threads) {
  std::ostringstream code;
  code << R"(
#include <deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_paged_mqa_logits<
        )" << next_n << R"(, )" << num_heads << R"(,
        )" << head_dim << R"(, )" << block_kv << R"(,
        true, false,
        )" << num_q_stages << R"(, )" << num_kv_stages << R"(,
        )" << split_kv << R"(,
        )" << num_tma_threads << R"(, )" << num_math_threads << R"(,
        float
    >);
}
)";
  return code.str();
}

std::string generate_kernel_code(
    const GemmDesc& desc,
    const GemmConfig& config,
    const char* gemm_type) {
  std::ostringstream code;
  code << R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K,
        0,
        )" << desc.n << R"(,
        )" << desc.k << R"(,
        )" << desc.num_groups << R"(,
        )" << config.layout.block_m << R"(, )" << config.layout.block_n << R"(, )"
       << config.layout.block_k << R"(,
        )" << config.storage.swizzle_a_mode << R"(, )"
       << config.storage.swizzle_b_mode << R"(, )"
       << config.storage.swizzle_cd_mode << R"(,
        )" << config.pipeline.num_stages << R"(,
        )" << config.launch.num_tma_threads << R"(, )"
       << config.launch.num_math_threads << R"(,
        )" << config.layout.cluster_size() << R"(, )"
       << bool_lit(config.layout.cluster_n > 1) << R"(,
        )" << config.launch.num_sms << R"(, GemmType::)" << gemm_type << R"(,
        epilogue::transform::EpilogueIdentity
    >);
}
)";
  return code.str();
}

std::filesystem::path cache_root() {
  const char* env_cache = std::getenv("DG_JIT_CACHE_DIR");
  if (env_cache != nullptr && env_cache[0] != '\0') {
    return std::filesystem::path(env_cache);
  }
  const char* home = non_empty_env("HOME", "/tmp");
  return std::filesystem::path(home) / ".deep_gemm";
}

std::filesystem::path lock_path_for_kernel(
    const std::string& name,
    const std::string& digest) {
  return cache_root() / "locks" / ("kernel." + name + "." + digest + ".lock");
}

std::filesystem::path unique_tmp_path_for_kernel(const std::string& digest) {
  static std::atomic<uint64_t> counter{0};
  const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
  return cache_root() / "tmp" /
         ("arle-" + digest + "-" + std::to_string(static_cast<long long>(::getpid())) +
          "-" + std::to_string(static_cast<long long>(now_ns)) + "-" +
          std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
}

std::string arch_flag(int major, int minor, int cuda_major, int cuda_minor) {
  if (major == 10 && minor != 1) {
    return (cuda_major > 12 || (cuda_major == 12 && cuda_minor >= 9)) ? "100a"
                                                                      : "100";
  }
  return std::to_string(major * 10 + minor) + "a";
}

void compile_with_nvcc(
    const std::filesystem::path& code_path,
    const std::filesystem::path& cubin_path,
    int major,
    int minor) {
  const auto library_root = deepgemm_library_root_path();
  const auto source_root = deepgemm_source_root_path(library_root);
  const auto cuda_home = cuda_home_path();
  const auto nvcc = cuda_home / "bin" / "nvcc";
  const auto nvcc_ccbin = nvcc_ccbin_path();
  const auto cutlass_include = deepgemm_cutlass_include_path(source_root);

  std::ostringstream command;
  command << shell_quote(nvcc)
          << " " << shell_quote(code_path)
          << " -cubin"
          << " -o " << shell_quote(cubin_path)
          << " -O3"
          // -std=c++17 (NOT c++20). c++20 makes nvcc's cicc frontend choke on
          // the flashmla CUTLASS `is_any_of` fold expression
          // (`... || std::is_same_v<T, Us>` → "pack expansion does not make
          // use of any argument packs"). c++17 parses the same headers fine
          // (the build itself uses -std=c++17). The earlier c++17+-fconcepts
          // combo failed because -fconcepts pulled gcc's C++20 <type_traits>
          // `requires` clauses into device code; without -fconcepts, c++17
          // libstdc++ stays requires-free. SM90-only path, CUDA 12.9.
          << " -std=c++17"
          << " --expt-relaxed-constexpr"
          << " --expt-extended-lambda"
          << " --diag-suppress=39,161,174,177,186,940"
          << " --ptxas-options=--register-usage-level=10"
          << " --compiler-options=-fPIC,-O3,-Wno-deprecated-declarations,-Wno-abi"
          << " --gpu-architecture=sm_" << arch_flag(major, minor, 12, 9);
  if (!nvcc_ccbin.empty()) {
    command << " -ccbin=" << shell_quote(nvcc_ccbin);
  }
  command
          << " -I" << shell_quote(library_root / "include")
          << " -I" << shell_quote(cutlass_include)
          << " -I" << shell_quote(cuda_home / "include");

  const auto [exit_code, output] = run_capture(command.str());
  if (exit_code != 0) {
    throw std::runtime_error("NVCC DeepGEMM compile failed:\n" + output);
  }
}

std::string parse_kernel_symbol(const std::filesystem::path& cubin_path) {
  const auto cuda_home = cuda_home_path();
  const auto cuobjdump = cuda_home / "bin" / "cuobjdump";
  auto [exit_code, output] =
      run_capture(shell_quote(cuobjdump) + " -symbols " + shell_quote(cubin_path));
  if (exit_code != 0) {
    throw std::runtime_error("cuobjdump failed:\n" + output);
  }
  std::vector<std::string> symbols;
  std::istringstream in(output);
  for (std::string line; std::getline(in, line);) {
    if (line.find("STT_FUNC") == 0 &&
        line.find("STO_ENTRY") != std::string::npos &&
        line.find("vprintf") == std::string::npos &&
        line.find("__instantiate_kernel") == std::string::npos &&
        line.find("__internal") == std::string::npos &&
        line.find("__assertfail") == std::string::npos) {
      const auto pos = line.rfind(' ');
      if (pos != std::string::npos) symbols.push_back(line.substr(pos + 1));
    }
  }
  if (symbols.size() != 1) {
    std::ostringstream err;
    err << "expected exactly one DeepGEMM kernel symbol, got " << symbols.size()
        << "\n" << output;
    throw std::runtime_error(err.str());
  }
  return symbols[0];
}

std::shared_ptr<NativeRuntime> get_or_build_runtime(
    const std::string& name,
    const std::string& code,
    int major,
    int minor) {
  const std::string key = name + "$$" + arch_flag(major, minor, 12, 9) + "$$" + code;
  const std::string digest = hex_digest(key);

  const auto dir = cache_root() / "cache" / ("kernel." + name + "." + digest);
  const auto cubin_path = dir / "kernel.cubin";
  const auto code_path = dir / "kernel.cu";
  std::lock_guard<std::mutex> lock(g_runtime_mu);
  if (auto it = g_runtime_failures.find(digest); it != g_runtime_failures.end()) {
    throw std::runtime_error(
        "DeepGEMM JIT disabled after prior compile/load failure for this kernel:\n" +
        it->second);
  }
  if (!std::filesystem::exists(cubin_path) || !std::filesystem::exists(code_path)) {
    ScopedFileLock jit_lock(lock_path_for_kernel(name, digest));
    if (!std::filesystem::exists(cubin_path)) {
      if (std::filesystem::exists(dir)) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
      }
      const auto tmp = unique_tmp_path_for_kernel(digest);
      std::filesystem::create_directories(tmp);
      const auto tmp_cubin = tmp / "kernel.cubin";
      const auto tmp_code = tmp / "kernel.cu";
      write_binary_file(tmp_code, code);
      try {
        compile_with_nvcc(tmp_code, tmp_cubin, major, minor);
      } catch (const std::exception& err) {
        std::error_code ec;
        std::filesystem::remove_all(tmp, ec);
        g_runtime_failures.emplace(digest, err.what());
        throw;
      }
      std::filesystem::create_directories(dir.parent_path());
      std::error_code ec;
      std::filesystem::rename(tmp, dir, ec);
      if (ec) {
        std::filesystem::remove_all(tmp, ec);
        if (!std::filesystem::exists(cubin_path)) {
          g_runtime_failures.emplace(
              digest,
              "DeepGEMM JIT cache publish failed for " + dir.string() + ": " +
                  ec.message());
          throw std::runtime_error(g_runtime_failures.at(digest));
        }
      }
    }
    if (!std::filesystem::exists(code_path)) {
      write_binary_file(code_path, code);
    }
  }

  const std::string runtime_key = digest + "$$" + current_context_key();
  if (auto it = g_runtimes.find(runtime_key); it != g_runtimes.end()) {
    return it->second;
  }

  auto runtime = std::make_shared<NativeRuntime>();
  try {
    const std::string symbol = parse_kernel_symbol(cubin_path);
    runtime->kernel = load_kernel(cubin_path, symbol, &runtime->library);
  } catch (const std::exception& err) {
    g_runtime_failures.emplace(digest, err.what());
    throw;
  }
  runtime->loaded = true;
  g_runtimes.emplace(runtime_key, runtime);
  return runtime;
}

CUresult launch_sm90_grouped_masked(
    const unsigned char* a,
    const float* sfa,
    const unsigned char* b,
    const float* sfb,
    unsigned short* d,
    const int* masked_m,
    int num_groups,
    int m,
    int n,
    int k,
    int sfa_aligned_m,
    CUstream stream) {
  cudaDeviceProp prop{};
  CUresult prop_err = current_device_prop(&prop);
  if (prop_err != CUDA_SUCCESS) return prop_err;
  if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;

  int num_sms = env_int("DG_NUM_SMS", prop.multiProcessorCount);
  if (num_sms <= 0 || num_sms > prop.multiProcessorCount || (num_sms % 2) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const GemmDesc desc{
      m,
      n,
      k,
      num_groups,
      num_sms,
      m,
      n,
      k,
      num_groups,
  };
  const auto config = get_best_config(desc);
  const auto code = generate_kernel_code(desc, config, "MGroupedMasked");
  const auto runtime = get_or_build_runtime(
      "sm90_fp8_m_grouped_gemm_masked_1d2d_native", code, prop.major, prop.minor);

  const auto tensor_map_a = make_tma_a_desc(
      a, m, k, config.storage.load_block_m, config.layout.block_k, k, num_groups,
      config.storage.swizzle_a_mode);
  const auto tensor_map_b = make_tma_b_desc(
      b, n, k, config.storage.load_block_n, config.layout.block_k, k, num_groups,
      config.storage.swizzle_b_mode);
  const auto tensor_map_d = make_tma_d_desc(
      d, m, n, config.storage.store_block_m, config.storage.store_block_n, n,
      num_groups, config.storage.swizzle_cd_mode);
  const auto tensor_map_sfa = make_tma_sfa_desc(
      sfa, m, k, config.layout.block_m, config.layout.block_k, num_groups,
      sfa_aligned_m);

  dim3 grid(static_cast<unsigned>(config.launch.num_sms), 1, 1);
  dim3 block(static_cast<unsigned>(config.launch.num_threads), 1, 1);
  auto launch_config = construct_launch_config(
      runtime->kernel, reinterpret_cast<cudaStream_t>(stream), config.pipeline.smem_size,
      grid, block, config.layout.cluster_size(), false);

  void* sfb_arg = const_cast<float*>(sfb);
  void* masked_arg = const_cast<int*>(masked_m);
  check_driver(
      launch_kernel(
          runtime->kernel, launch_config, sfb_arg, masked_arg, m, n, k,
          tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa),
      "DeepGEMM launch");
  return CUDA_SUCCESS;
}

CUresult launch_sm90_grouped_contiguous(
    const unsigned char* a,
    const float* sfa,
    const unsigned char* b,
    const float* sfb,
    unsigned short* d,
    const int* m_indices,
    int num_groups,
    int m,
    int n,
    int k,
    int sfa_aligned_m,
    int mk_align,
    CUstream stream) {
  cudaDeviceProp prop{};
  CUresult prop_err = current_device_prop(&prop);
  if (prop_err != CUDA_SUCCESS) return prop_err;
  if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;

  int num_sms = env_int("DG_NUM_SMS", prop.multiProcessorCount);
  if (num_sms <= 0 || num_sms > prop.multiProcessorCount || (num_sms % 2) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const GemmDesc desc{
      m,
      n,
      k,
      num_groups,
      num_sms,
      m,
      n,
      k,
      1,
      mk_align,
  };
  const auto config = get_best_config(desc);
  const auto code = generate_kernel_code(desc, config, "MGroupedContiguous");
  const auto runtime = get_or_build_runtime(
      "sm90_fp8_m_grouped_gemm_contiguous_1d2d_native", code, prop.major, prop.minor);

  const auto tensor_map_a = make_tma_a_desc(
      a, m, k, config.storage.load_block_m, config.layout.block_k, k, 1,
      config.storage.swizzle_a_mode);
  const auto tensor_map_b = make_tma_b_desc(
      b, n, k, config.storage.load_block_n, config.layout.block_k, k, num_groups,
      config.storage.swizzle_b_mode);
  const auto tensor_map_d = make_tma_d_desc(
      d, m, n, config.storage.store_block_m, config.storage.store_block_n, n, 1,
      config.storage.swizzle_cd_mode);
  const auto tensor_map_sfa = make_tma_sfa_desc(
      sfa, m, k, config.layout.block_m, config.layout.block_k, 1,
      sfa_aligned_m);

  dim3 grid(static_cast<unsigned>(config.launch.num_sms), 1, 1);
  dim3 block(static_cast<unsigned>(config.launch.num_threads), 1, 1);
  auto launch_config = construct_launch_config(
      runtime->kernel, reinterpret_cast<cudaStream_t>(stream), config.pipeline.smem_size,
      grid, block, config.layout.cluster_size(), false);

  void* sfb_arg = const_cast<float*>(sfb);
  void* m_indices_arg = const_cast<int*>(m_indices);
  check_driver(
      launch_kernel(
          runtime->kernel, launch_config, sfb_arg, m_indices_arg, m, n, k,
          tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa),
      "DeepGEMM contiguous launch");
  return CUDA_SUCCESS;
}

CUresult launch_sm90_dense_nt(
    const unsigned char* a,
    const float* sfa,
    const unsigned char* b,
    const float* sfb,
    unsigned short* d,
    int m,
    int n,
    int k,
    int sfa_aligned_m,
    CUstream stream) {
  cudaDeviceProp prop{};
  CUresult prop_err = current_device_prop(&prop);
  if (prop_err != CUDA_SUCCESS) return prop_err;
  if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;

  int num_sms = env_int("DG_NUM_SMS", prop.multiProcessorCount);
  if (num_sms <= 0 || num_sms > prop.multiProcessorCount || (num_sms % 2) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  const GemmDesc desc{
      m,
      n,
      k,
      1,
      num_sms,
      m,
      n,
      k,
      1,
  };
  const auto config = get_best_config(desc);
  const auto code = generate_kernel_code(desc, config, "Normal");
  const auto runtime = get_or_build_runtime(
      "sm90_fp8_gemm_nt_1d2d_native", code, prop.major, prop.minor);

  const auto tensor_map_a = make_tma_a_desc(
      a, m, k, config.storage.load_block_m, config.layout.block_k, k, 1,
      config.storage.swizzle_a_mode);
  const auto tensor_map_b = make_tma_b_desc(
      b, n, k, config.storage.load_block_n, config.layout.block_k, k, 1,
      config.storage.swizzle_b_mode);
  const auto tensor_map_d = make_tma_d_desc(
      d, m, n, config.storage.store_block_m, config.storage.store_block_n, n, 1,
      config.storage.swizzle_cd_mode);
  const auto tensor_map_sfa = make_tma_sfa_desc(
      sfa, m, k, config.layout.block_m, config.layout.block_k, 1, sfa_aligned_m);

  dim3 grid(static_cast<unsigned>(config.launch.num_sms), 1, 1);
  dim3 block(static_cast<unsigned>(config.launch.num_threads), 1, 1);
  auto launch_config = construct_launch_config(
      runtime->kernel, reinterpret_cast<cudaStream_t>(stream), config.pipeline.smem_size,
      grid, block, config.layout.cluster_size(), false);

  void* sfb_arg = const_cast<float*>(sfb);
  int* grouped_arg = nullptr;
  check_driver(
      launch_kernel(
          runtime->kernel, launch_config, sfb_arg, grouped_arg, m, n, k,
          tensor_map_a, tensor_map_b, tensor_map_d, tensor_map_sfa),
      "DeepGEMM dense launch");
  return CUDA_SUCCESS;
}

CUresult launch_sm90_paged_mqa_logits_metadata(
    const int* context_lens,
    int* schedule_metadata,
    int batch_size,
    int next_n,
    int block_kv,
    int num_sms,
    CUstream stream) {
  cudaDeviceProp prop{};
  CUresult prop_err = current_device_prop(&prop);
  if (prop_err != CUDA_SUCCESS) return prop_err;
  if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;
  if (context_lens == nullptr || schedule_metadata == nullptr || stream == nullptr ||
      batch_size <= 0 || next_n <= 0 || block_kv != 64 || num_sms <= 0 ||
      num_sms > prop.multiProcessorCount) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int split_kv = 256;
  const int aligned_batch_size = align_to(batch_size, 32);
  const int smem_size = (3 * aligned_batch_size + 1) * static_cast<int>(sizeof(int));
  if (smem_size > kSm90SmemCapacity) return CUDA_ERROR_INVALID_VALUE;

  const auto code =
      generate_paged_mqa_metadata_code(aligned_batch_size, split_kv, num_sms);
  const auto runtime = get_or_build_runtime(
      "sm90_fp8_paged_mqa_logits_metadata_native", code, prop.major, prop.minor);

  dim3 grid(1, 1, 1);
  dim3 block(32, 1, 1);
  auto launch_config = construct_launch_config(
      runtime->kernel, reinterpret_cast<cudaStream_t>(stream), smem_size, grid, block, 1,
      false);
  const int* indices = nullptr;
  check_driver(
      launch_kernel(
          runtime->kernel, launch_config, batch_size, next_n, true, context_lens,
          indices, schedule_metadata),
      "DeepGEMM paged MQA metadata launch");
  return CUDA_SUCCESS;
}

CUresult launch_sm90_fp8_paged_mqa_logits(
    const unsigned char* q,
    const unsigned char* kv_cache,
    const float* kv_cache_scales,
    const float* weights,
    const int* context_lens,
    const int* block_table,
    const int* schedule_meta,
    float* logits,
    int batch_size,
    int next_n,
    int num_heads,
    int head_dim,
    int num_kv_blocks,
    int block_kv,
    int max_context_len,
    int logits_stride,
    int block_table_stride,
    int kv_cache_stride_bytes,
    int num_sms,
    CUstream stream) {
  cudaDeviceProp prop{};
  CUresult prop_err = current_device_prop(&prop);
  if (prop_err != CUDA_SUCCESS) return prop_err;
  if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;
  if (q == nullptr || kv_cache == nullptr || kv_cache_scales == nullptr ||
      weights == nullptr || context_lens == nullptr || block_table == nullptr ||
      schedule_meta == nullptr || logits == nullptr || stream == nullptr ||
      batch_size <= 0 || next_n <= 0 || (num_heads != 32 && num_heads != 64) ||
      (head_dim != 32 && head_dim != 64 && head_dim != 128) || num_kv_blocks <= 0 ||
      block_kv != 64 || max_context_len <= 0 || logits_stride < max_context_len ||
      block_table_stride <= 0 || kv_cache_stride_bytes < block_kv * (head_dim + 4) ||
      (kv_cache_stride_bytes % static_cast<int>(sizeof(float))) != 0 ||
      num_sms <= 0 || num_sms > prop.multiProcessorCount) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  constexpr int split_kv = 256;
  if ((logits_stride % split_kv) != 0) return CUDA_ERROR_INVALID_VALUE;
  const int num_specialized_threads = 128;
  const int mma_m = 64;
  const int num_math_warp_groups = split_kv / mma_m;
  const int num_math_threads = num_math_warp_groups * 128;
  const int num_q_stages = 3;
  const int num_kv_stages = 3;

  const int swizzle_alignment = head_dim * 8;
  const int smem_q_size_per_stage = next_n * num_heads * head_dim;
  const int aligned_smem_weight_size_per_stage =
      align_to(next_n * num_heads * static_cast<int>(sizeof(float)), swizzle_alignment);
  const int smem_q_pipe_size =
      num_q_stages *
          (smem_q_size_per_stage + aligned_smem_weight_size_per_stage) +
      align_to(num_q_stages * 8 * 2, swizzle_alignment);

  const int smem_kv_size_per_stage = block_kv * head_dim;
  const int aligned_smem_kv_scale_size_per_stage =
      align_to(block_kv * static_cast<int>(sizeof(float)), swizzle_alignment);
  const int smem_kv_pipe_size =
      num_kv_stages *
          (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) +
      align_to(num_kv_stages * 8 * 2, swizzle_alignment);

  const int smem_umma_barriers = num_math_warp_groups * 2 * 8;
  const int smem_tmem_ptr = 4;
  const int smem_size = smem_q_pipe_size +
                        num_math_warp_groups * smem_kv_pipe_size +
                        smem_umma_barriers + smem_tmem_ptr;
  if (smem_size > kSm90SmemCapacity) return CUDA_ERROR_INVALID_VALUE;

  const auto code = generate_fp8_paged_mqa_logits_code(
      next_n, num_heads, head_dim, block_kv, num_q_stages, num_kv_stages, split_kv,
      num_specialized_threads, num_math_threads);
  const auto runtime = get_or_build_runtime(
      "sm90_fp8_paged_mqa_logits_native", code, prop.major, prop.minor);

  const auto tensor_map_q = make_tma_2d_desc_raw(
      q, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, head_dim, batch_size * next_n * num_heads,
      head_dim, next_n * num_heads, head_dim, head_dim);
  const auto tensor_map_kv = make_tma_3d_desc_raw(
      kv_cache, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, head_dim, block_kv, num_kv_blocks,
      head_dim, block_kv, 1, head_dim, kv_cache_stride_bytes, head_dim);
  const auto tensor_map_kv_scales = make_tma_2d_desc_raw(
      kv_cache_scales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, block_kv, num_kv_blocks,
      block_kv, 1, kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 0);
  const auto tensor_map_weights = make_tma_2d_desc_raw(
      weights, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, next_n * num_heads, batch_size,
      next_n * num_heads, 1, next_n * num_heads, 0);

  dim3 grid(static_cast<unsigned>(num_sms), 1, 1);
  dim3 block(static_cast<unsigned>(num_specialized_threads + num_math_threads), 1, 1);
  auto launch_config = construct_launch_config(
      runtime->kernel, reinterpret_cast<cudaStream_t>(stream), smem_size, grid, block, 1,
      false);
  const int* indices = nullptr;
  check_driver(
      launch_kernel(
          runtime->kernel, launch_config, batch_size, static_cast<uint64_t>(logits_stride),
          static_cast<uint64_t>(block_table_stride), context_lens, logits, block_table,
          indices, schedule_meta, tensor_map_q, tensor_map_kv, tensor_map_kv_scales,
          tensor_map_weights),
      "DeepGEMM FP8 paged MQA logits launch");
  return CUDA_SUCCESS;
}

}  // namespace

extern "C" CUresult dsv4_deepgemm_native_preflight_cuda(
    char* out,
    size_t out_len) {
  try {
    const auto report = deepgemm_preflight_report();
    write_c_string(out, out_len, report.message);
    return report.ok ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
  } catch (const std::exception& err) {
    write_c_string(out, out_len, std::string("status=failed error=") + err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

extern "C" CUresult dsv4_sm90_mega_moe_workspace_layout_cuda(
    int num_ranks,
    int num_experts,
    int requested_max_tokens_per_rank,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    Sm90MegaMoeWorkspaceLayout* out) {
  if (out == nullptr || num_ranks <= 0 || num_experts <= 0 ||
      requested_max_tokens_per_rank <= 0 || num_topk <= 0 || hidden <= 0 ||
      intermediate_hidden <= 0 || intermediate_hidden > 4096 ||
      hidden % 128 != 0 || intermediate_hidden % 128 != 0 ||
      requested_max_tokens_per_rank >
          std::numeric_limits<int>::max() - deep_gemm::layout::kLCMCandidateBlockM + 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  try {
    constexpr int kTokenAlignment = deep_gemm::layout::kLCMCandidateBlockM;
    const int max_tokens = align_to(requested_max_tokens_per_rank, kTokenAlignment);
    const auto layout = deep_gemm::get_sm90_mega_moe_workspace_layout(
        num_ranks, num_experts, max_tokens, num_topk, hidden, intermediate_hidden);
    *out = {
        layout.num_bytes,
        layout.x,
        layout.x_sf,
        layout.topk_idx,
        layout.topk_weights,
        layout.l1_acts,
        layout.l1_acts_sf,
        layout.l1_topk_weights,
        layout.l2_acts,
        layout.l2_acts_sf,
        layout.combine,
        layout.num_max_pool_tokens,
        layout.num_padded_sf_pool_tokens,
        max_tokens,
    };
    return CUDA_SUCCESS;
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM SM90 MegaMoE workspace failed: %s\n", err.what());
    return CUDA_ERROR_INVALID_VALUE;
  }
}

extern "C" CUresult dsv4_sm90_mega_moe_pre_dispatch_cuda(
    const unsigned short* hidden_states,
    const int* route_indices,
    const float* route_weights,
    unsigned char* workspace_x,
    float* workspace_x_sf,
    int64_t* workspace_topk_idx,
    float* workspace_topk_weights,
    int num_tokens,
    int padded_max_tokens,
    int hidden,
    int num_topk,
    int enable_pdl,
    CUstream stream) {
  if (hidden_states == nullptr || route_indices == nullptr || route_weights == nullptr ||
      workspace_x == nullptr || workspace_x_sf == nullptr ||
      workspace_topk_idx == nullptr || workspace_topk_weights == nullptr ||
      stream == nullptr || num_tokens < 0 || padded_max_tokens <= 0 ||
      num_tokens > padded_max_tokens || hidden <= 0 || hidden % 128 != 0 ||
      hidden / 8 > 1024 || num_topk <= 0 || num_topk > hidden / 8 ||
      enable_pdl < 0 || enable_pdl > 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    cudaDeviceProp prop{};
    const CUresult prop_err = current_device_prop(&prop);
    if (prop_err != CUDA_SUCCESS) return prop_err;
    if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;

    std::ostringstream source;
    source << "#include <deep_gemm/impls/sm90_mega_moe_pre_dispatch.cuh>\n"
              "using namespace deep_gemm;\n"
              "static void __instantiate_kernel() { auto ptr = reinterpret_cast<void*>("
              "&sm90_mega_moe_pre_dispatch_kernel<128,"
           << (enable_pdl != 0 ? "true" : "false") << ">); }\n";
    const auto runtime = get_or_build_runtime(
        "sm90_mega_moe_pre_dispatch_native", source.str(), prop.major, prop.minor);

    const uint64_t num_threads = static_cast<uint64_t>(hidden / 8);
    const uint64_t pad_slots =
        static_cast<uint64_t>(padded_max_tokens - num_tokens) * num_topk;
    const uint64_t pad_blocks = pad_slots == 0 ? 0 : ceil_div(pad_slots, num_threads);
    const uint64_t total_blocks = static_cast<uint64_t>(num_tokens) + pad_blocks;
    if (total_blocks == 0 || total_blocks > std::numeric_limits<unsigned>::max()) {
      return total_blocks == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    }

    const auto launch_config = construct_launch_config(
        runtime->kernel, reinterpret_cast<cudaStream_t>(stream), 0,
        dim3(static_cast<unsigned>(total_blocks), 1, 1),
        dim3(static_cast<unsigned>(num_threads), 1, 1), 1, enable_pdl != 0);
    constexpr float kAlreadyScaledRouteWeights = 1.0f;
    return launch_kernel(
        runtime->kernel, launch_config, hidden_states, route_indices, route_weights,
        workspace_x, workspace_x_sf, workspace_topk_idx, workspace_topk_weights,
        static_cast<uint32_t>(num_tokens), static_cast<uint32_t>(padded_max_tokens),
        static_cast<uint32_t>(hidden), static_cast<uint32_t>(hidden / 128),
        static_cast<uint32_t>(num_topk), kAlreadyScaledRouteWeights);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM SM90 MegaMoE pre-dispatch failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

extern "C" CUresult dsv4_sm90_mega_moe_launch_cuda(
    void* y,
    int* cumulative_local_expert_recv_stats,
    const uint64_t* peer_buffer_ptrs,
    void* local_workspace,
    int num_ranks,
    int rank_idx,
    int num_max_tokens_per_rank,
    int num_tokens,
    int num_experts,
    int num_topk,
    int hidden,
    int intermediate_hidden,
    float activation_clamp,
    int fast_math,
    int enable_pdl,
    const unsigned char* l1_weights,
    int l1_weight_stride,
    const float* l1_weights_sf,
    const unsigned char* l2_weights,
    int l2_weight_stride,
    const float* l2_weights_sf,
    CUstream stream) {
  if (y == nullptr || peer_buffer_ptrs == nullptr || local_workspace == nullptr ||
      l1_weights == nullptr || l1_weights_sf == nullptr || l2_weights == nullptr ||
      l2_weights_sf == nullptr || stream == nullptr ||
      num_ranks <= 0 || num_ranks > static_cast<int>(deep_gemm::layout::kNumMaxRanks) ||
      rank_idx < 0 || rank_idx >= num_ranks || num_max_tokens_per_rank <= 0 ||
      num_max_tokens_per_rank % deep_gemm::layout::kLCMCandidateBlockM != 0 ||
      num_tokens <= 0 || num_tokens > num_max_tokens_per_rank || num_experts <= 0 ||
      num_experts % num_ranks != 0 || num_topk <= 0 || hidden <= 0 ||
      intermediate_hidden <= 0 || intermediate_hidden > 4096 ||
      hidden % 128 != 0 || intermediate_hidden % 128 != 0 ||
      l1_weight_stride < hidden ||
      l2_weight_stride < intermediate_hidden || fast_math < 0 || fast_math > 1 ||
      enable_pdl < 0 || enable_pdl > 1 || std::isnan(activation_clamp) ||
      activation_clamp < 0.0f) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  for (int rank = 0; rank < num_ranks; ++rank) {
    if (peer_buffer_ptrs[rank] == 0) return CUDA_ERROR_INVALID_VALUE;
  }
  if (peer_buffer_ptrs[rank_idx] != reinterpret_cast<uint64_t>(local_workspace)) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    cudaDeviceProp prop{};
    const CUresult prop_err = current_device_prop(&prop);
    if (prop_err != CUDA_SUCCESS) return prop_err;
    if (prop.major != 9) return CUDA_ERROR_NOT_SUPPORTED;
    const int num_sms = env_int("DG_NUM_SMS", prop.multiProcessorCount);
    if (num_sms <= 0 || num_sms > prop.multiProcessorCount) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    const auto workspace = deep_gemm::get_sm90_mega_moe_workspace_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden,
        intermediate_hidden);
    const auto spec = deep_gemm::get_sm90_fp8_mega_moe_kernel_spec(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, hidden,
        intermediate_hidden, workspace.num_padded_sf_pool_tokens, activation_clamp,
        fast_math != 0, num_sms, kSm90SmemCapacity,
        env_int("DG_SM90_FP8_SWAP_AB", 1) != 0);
    const auto runtime = get_or_build_runtime(
        "sm90_fp8_mega_moe_native",
        deep_gemm::generate_sm90_fp8_mega_moe_source(spec), prop.major, prop.minor);
    std::vector<int64_t> peers(peer_buffer_ptrs, peer_buffer_ptrs + num_ranks);
    const deep_gemm::layout::SymBuffer<> sym_buffer(peers, rank_idx);
    const auto& config = spec.config;
    const auto at = [local_workspace](uint64_t offset) {
      return static_cast<unsigned char*>(local_workspace) + offset;
    };
    const auto tensor_map_l1_acts = make_tma_2d_desc_raw(
        at(workspace.l1_acts), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, hidden,
        config.num_max_pool_tokens, config.block_k, config.block_m, hidden,
        config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc_raw(
        at(workspace.l1_acts_sf), config.num_padded_sf_pool_tokens, hidden,
        config.block_m, 128);
    const auto tma = deep_gemm::get_sm90_mega_moe_tma_config(config);
    const auto tensor_map_l1_weights = make_tma_2d_desc_raw(
        l1_weights, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, hidden,
        (num_experts / num_ranks) * intermediate_hidden * 2, config.block_k,
        tma.weight_block_n, l1_weight_stride, config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc_raw(
        at(workspace.l2_acts), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        intermediate_hidden, config.num_max_pool_tokens, tma.l1_output_box_n,
        tma.l1_output_box_m, intermediate_hidden, 0);
    const auto tensor_map_l2_acts = make_tma_2d_desc_raw(
        at(workspace.l2_acts), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        intermediate_hidden, config.num_max_pool_tokens, config.block_k,
        config.block_m, intermediate_hidden, config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc_raw(
        at(workspace.l2_acts_sf), config.num_padded_sf_pool_tokens,
        intermediate_hidden, config.block_m, 64);
    const auto tensor_map_l2_weights = make_tma_2d_desc_raw(
        l2_weights, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, intermediate_hidden,
        (num_experts / num_ranks) * hidden, config.block_k, tma.weight_block_n,
        l2_weight_stride, config.swizzle_weights_mode);
    const auto launch_config = construct_launch_config(
        runtime->kernel, reinterpret_cast<cudaStream_t>(stream), config.smem_size,
        dim3(static_cast<unsigned>(num_sms), 1, 1),
        dim3(static_cast<unsigned>(config.num_dispatch_threads +
                                   config.num_non_epilogue_threads +
                                   config.num_epilogue_threads),
             1, 1),
        config.cluster_size, enable_pdl != 0);
    return launch_kernel(
        runtime->kernel, launch_config, y, cumulative_local_expert_recv_stats, num_tokens,
        sym_buffer, tensor_map_l1_acts, tensor_map_l1_acts_sf,
        tensor_map_l1_weights, l1_weights_sf, tensor_map_l1_output,
        tensor_map_l2_acts, tensor_map_l2_acts_sf, tensor_map_l2_weights,
        l2_weights_sf);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM SM90 MegaMoE launch failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

extern "C" CUresult dsv4_deepgemm_m_grouped_fp8_gemm_nt_masked_cuda(
    const unsigned char* a,
    const float* sfa,
    const unsigned char* b,
    const float* sfb,
    unsigned short* d,
    const int* masked_m,
    int num_groups,
    int m,
    int n,
    int k,
    int sfa_aligned_m,
    CUstream stream) {
  if (a == nullptr || sfa == nullptr || b == nullptr || sfb == nullptr ||
      d == nullptr || masked_m == nullptr || stream == nullptr || num_groups <= 0 ||
      m <= 0 || n <= 0 || k <= 0 || sfa_aligned_m < m) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((k % kScaleGranK) != 0 || (n % 8) != 0 ||
      sfa_aligned_m < get_tma_aligned_size(m, sizeof(float))) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    return launch_sm90_grouped_masked(
        a, sfa, b, sfb, d, masked_m, num_groups, m, n, k, sfa_aligned_m, stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM native bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

extern "C" CUresult dsv4_deepgemm_m_grouped_fp8_gemm_nt_contiguous_cuda(
    const unsigned char* a,
    const float* sfa,
    const unsigned char* b,
    const float* sfb,
    unsigned short* d,
    const int* m_indices,
    int num_groups,
    int m,
    int n,
    int k,
    int sfa_aligned_m,
    int mk_align,
    CUstream stream) {
  if (a == nullptr || sfa == nullptr || b == nullptr || sfb == nullptr ||
      d == nullptr || m_indices == nullptr || stream == nullptr || num_groups <= 0 ||
      m <= 0 || n <= 0 || k <= 0 || sfa_aligned_m < m) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((k % kScaleGranK) != 0 || (n % 8) != 0 ||
      sfa_aligned_m < get_tma_aligned_size(m, sizeof(float))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  // mk_align caps the block_m candidates (see GemmDesc.max_block_m) so an
  // output tile can never span two mk_align-aligned groups. The per-group row
  // alignment itself stays the caller's packing contract, as before.
  if (mk_align != 64 && mk_align != 128) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    return launch_sm90_grouped_contiguous(
        a, sfa, b, sfb, d, m_indices, num_groups, m, n, k, sfa_aligned_m, mk_align,
        stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM native contiguous bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// SM90 BF16 m-grouped masked GEMM (KernelNoSF, NT): per group g,
// `d[g, 0..masked_m[g], :] = a[g, 0..masked_m[g], :] @ b[g]^T`.
// `a` is `[num_groups, m, k]`, `b` `[num_groups, n, k]`, `d` `[num_groups, m, n]`,
// all row-major BF16. `m` is the per-group band capacity: it MUST be a
// multiple of 128 so a BLOCK_M (64 or 128) output tile can never cross into
// the next group's band (the TMA store writes the full tile). `expected_m` is
// a heuristics-only hint (per-group expected valid rows).
extern "C" CUresult deepgemm_m_grouped_bf16_gemm_nt_masked_cuda(
    const unsigned short* a,
    const unsigned short* b,
    unsigned short* d,
    const int* masked_m,
    int num_groups,
    int m,
    int n,
    int k,
    int expected_m,
    CUstream stream) {
  if (a == nullptr || b == nullptr || d == nullptr || masked_m == nullptr ||
      stream == nullptr || num_groups <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((k % kBf16BlockK) != 0 || (n % 8) != 0 || (m % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    return launch_sm90_bf16_grouped(
        a, b, d, masked_m, num_groups, m, n, k, expected_m, true, stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM native BF16 masked bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

// SM90 BF16 m-grouped contiguous GEMM (KernelNoSF, NT): `d = a @ b[g]^T` where
// each activation row's group is `m_indices[row]` (-1 = padding row). `a` is
// `[m, k]`, `b` `[num_groups, n, k]`, `d` `[m, n]`, all row-major BF16.
// CONTRACT (scheduler/gemm.cuh): the kernel resolves the B group ONCE per
// BLOCK_M=128 output tile from `m_indices[tile_start]`, so every group's row
// segment must start 128-aligned and pad rows must carry -1; pad-row outputs
// are computed against `max(0, -1)` = group 0 and are garbage — callers must
// exclude them (route-slot sentinel). `m` must be a multiple of 128.
extern "C" CUresult deepgemm_m_grouped_bf16_gemm_nt_contiguous_cuda(
    const unsigned short* a,
    const unsigned short* b,
    unsigned short* d,
    const int* m_indices,
    int num_groups,
    int m,
    int n,
    int k,
    CUstream stream) {
  if (a == nullptr || b == nullptr || d == nullptr || m_indices == nullptr ||
      stream == nullptr || num_groups <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((k % kBf16BlockK) != 0 || (n % 8) != 0 ||
      (m % kBf16ContiguousAlignment) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    return launch_sm90_bf16_grouped(
        a, b, d, m_indices, num_groups, m, n, k, m, false, stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM native BF16 contiguous bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

extern "C" CUresult dsv4_deepgemm_fp8_gemm_nt_cuda(
    const unsigned char* a,
    const float* sfa,
    const unsigned char* b,
    const float* sfb,
    unsigned short* d,
    int m,
    int n,
    int k,
    int sfa_aligned_m,
    CUstream stream) {
  // stream==nullptr is CUDA's legal default stream (handle 0); the training
  // caller runs on it (autograd holds ctx.default_stream()). pack_quantize
  // accepts it too — only the launch APIs are consulted, not this guard.
  if (a == nullptr || sfa == nullptr || b == nullptr || sfb == nullptr ||
      d == nullptr || m <= 0 || n <= 0 || k <= 0 || sfa_aligned_m < m) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((k % kScaleGranK) != 0 || (n % 8) != 0 ||
      sfa_aligned_m < get_tma_aligned_size(m, sizeof(float))) {
    return CUDA_ERROR_INVALID_VALUE;
  }

  try {
    return launch_sm90_dense_nt(a, sfa, b, sfb, d, m, n, k, sfa_aligned_m, stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM native dense bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

extern "C" CUresult dsv4_deepgemm_paged_mqa_logits_metadata_cuda(
    const int* context_lens,
    int* schedule_metadata,
    int batch_size,
    int next_n,
    int block_kv,
    int num_sms,
    CUstream stream) {
  try {
    return launch_sm90_paged_mqa_logits_metadata(
        context_lens, schedule_metadata, batch_size, next_n, block_kv, num_sms, stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM paged MQA metadata bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}
extern "C" CUresult dsv4_deepgemm_fp8_paged_mqa_logits_fused_cache_cuda(
    const unsigned char* q,
    const unsigned char* kv_cache_with_scale,
    const float* weights,
    const int* context_lens,
    const int* block_table,
    const int* schedule_meta,
    float* logits,
    int batch_size,
    int next_n,
    int num_heads,
    int head_dim,
    int num_kv_blocks,
    int block_kv,
    int max_context_len,
    int logits_stride,
    int block_table_stride,
    int kv_cache_stride_bytes,
    int num_sms,
    CUstream stream) {
  if (kv_cache_with_scale == nullptr || block_kv != 64 || head_dim <= 0 ||
      kv_cache_stride_bytes < block_kv * (head_dim + static_cast<int>(sizeof(float)))) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const auto* kv_cache_scales =
      reinterpret_cast<const float*>(kv_cache_with_scale + block_kv * head_dim);
  try {
    return launch_sm90_fp8_paged_mqa_logits(
        q, kv_cache_with_scale, kv_cache_scales, weights, context_lens, block_table,
        schedule_meta, logits, batch_size, next_n, num_heads, head_dim, num_kv_blocks,
        block_kv, max_context_len, logits_stride, block_table_stride,
        kv_cache_stride_bytes, num_sms, stream);
  } catch (const std::exception& err) {
    std::fprintf(stderr, "DeepGEMM FP8 paged MQA fused-cache bridge failed: %s\n", err.what());
    return CUDA_ERROR_UNKNOWN;
  }
}

#endif  // ARLE_ENABLE_DEEPGEMM_NATIVE
