#include <cuda.h>

#ifdef ARLE_ENABLE_DEEPGEMM_NATIVE

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <cstdio>
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

#ifndef ARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT
#define ARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT "vendor/deepgemm/deep_gemm"
#endif

#ifndef ARLE_DEEPGEMM_DEFAULT_CUDA_HOME
#define ARLE_DEEPGEMM_DEFAULT_CUDA_HOME "/usr/local/cuda"
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
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuTensorMapEncodeTiled);

#if CUDA_VERSION >= 12040
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryLoadFromFile);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryUnload);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryGetKernelCount);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryEnumerateKernels);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuKernelGetFunction);
using LibraryHandle = CUlibrary;
#else
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
ARLE_DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
using LibraryHandle = CUmodule;
#endif

using KernelHandle = CUfunction;

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
  const char* cuda_home = std::getenv("CUDA_HOME");
  if (cuda_home != nullptr && cuda_home[0] != '\0') return cuda_home;
  const char* cuda_path = std::getenv("CUDA_PATH");
  if (cuda_path != nullptr && cuda_path[0] != '\0') return cuda_path;
  return ARLE_DEEPGEMM_DEFAULT_CUDA_HOME;
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  int parsed = fallback;
  return std::sscanf(value, "%d", &parsed) == 1 ? parsed : fallback;
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

KernelHandle load_kernel(
    const std::filesystem::path& cubin_path,
    const std::string& func_name,
    LibraryHandle* library_out) {
  LibraryHandle library{};
  KernelHandle kernel{};
#if CUDA_VERSION >= 12040
  check_driver(
      lazy_cuLibraryLoadFromFile(
          &library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0),
      "cuLibraryLoadFromFile");
  unsigned int num_kernels = 0;
  check_driver(lazy_cuLibraryGetKernelCount(&num_kernels, library),
               "cuLibraryGetKernelCount");
  if (num_kernels != 1) {
    throw std::runtime_error("DeepGEMM CUBIN should contain exactly one kernel");
  }
  CUkernel cu_kernel{};
  check_driver(lazy_cuLibraryEnumerateKernels(&cu_kernel, 1, library),
               "cuLibraryEnumerateKernels");
  check_driver(lazy_cuKernelGetFunction(&kernel, cu_kernel), "cuKernelGetFunction");
#else
  check_driver(lazy_cuModuleLoad(&library, cubin_path.c_str()), "cuModuleLoad");
  check_driver(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()),
               "cuModuleGetFunction");
#endif
  if (library_out != nullptr) *library_out = library;
  return kernel;
}

void unload_library(const LibraryHandle& library) {
#if CUDA_VERSION >= 12040
  const auto err = lazy_cuLibraryUnload(library);
#else
  const auto err = lazy_cuModuleUnload(library);
#endif
  if (err != CUDA_SUCCESS && err != CUDA_ERROR_DEINITIALIZED) {
    check_driver(err, "CUDA library unload");
  }
}

CUlaunchConfig construct_launch_config(
    KernelHandle kernel,
    cudaStream_t stream,
    int smem_size,
    dim3 grid_dim,
    dim3 block_dim,
    int cluster_dim,
    bool enable_pdl) {
  if (smem_size > 0) {
    check_driver(
        lazy_cuFuncSetAttribute(
            kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size),
        "cuFuncSetAttribute");
  }

  CUlaunchConfig config{};
  config.gridDimX = grid_dim.x;
  config.gridDimY = grid_dim.y;
  config.gridDimZ = grid_dim.z;
  config.blockDimX = block_dim.x;
  config.blockDimY = block_dim.y;
  config.blockDimZ = block_dim.z;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;

  static thread_local CUlaunchAttribute attrs[2];
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

  return config;
}

template <typename... Args>
CUresult launch_kernel(KernelHandle kernel, const CUlaunchConfig& config, Args&&... args) {
  void* ptr_args[] = {
      const_cast<void*>(reinterpret_cast<const void*>(&args))...,
  };
  return lazy_cuLaunchKernelEx(
      const_cast<CUlaunchConfig*>(&config), kernel, ptr_args, nullptr);
}

void check_nvrtc(nvrtcResult result, const char* what) {
  if (result == NVRTC_SUCCESS) return;
  std::ostringstream out;
  out << what << " failed: " << nvrtcGetErrorString(result);
  throw std::runtime_error(out.str());
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
  for (int cluster_m = 1; cluster_m <= 2; ++cluster_m) {
    for (int cluster_n = 1; cluster_n <= 2; ++cluster_n) {
      if (cluster_m * cluster_n > 2) continue;
      if (desc.num_sms % (cluster_m * cluster_n) != 0) continue;
      for (int block_m : {64, 128}) {
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
  if (outer_stride != aligned_m) {
    throw std::runtime_error("SFA stride does not match DeepGEMM TMA alignment");
  }
  return make_tma_2d_desc_raw(
      ptr, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, aligned_m,
      ceil_div(shape_k, gran_k) * num_groups, block_m, 1, outer_stride, 0);
}

std::string bool_lit(bool value) { return value ? "true" : "false"; }

std::string generate_kernel_code(
    const GemmDesc& desc,
    const GemmConfig& config) {
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
        )" << config.launch.num_sms << R"(, GemmType::MGroupedMasked,
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

std::string arch_flag(int major, int minor, int nvrtc_major, int nvrtc_minor) {
  if (major == 10 && minor != 1) {
    return (nvrtc_major > 12 || (nvrtc_major == 12 && nvrtc_minor >= 9)) ? "100a"
                                                                         : "100";
  }
  return std::to_string(major * 10 + minor) + "a";
}

std::string compile_with_nvrtc(
    const std::string& code,
    const std::filesystem::path& cubin_path,
    int major,
    int minor) {
  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  check_nvrtc(nvrtcVersion(&nvrtc_major, &nvrtc_minor), "nvrtcVersion");

  const auto library_root =
      std::filesystem::path(non_empty_env(
          "ARLE_DEEPGEMM_LIBRARY_ROOT", ARLE_DEEPGEMM_DEFAULT_LIBRARY_ROOT));
  const auto source_root = library_root.parent_path();
  const auto cuda_home = cuda_home_path();
  const char* cutlass_env = std::getenv("ARLE_DEEPGEMM_CUTLASS_INCLUDE");
  const auto cutlass_include =
      cutlass_env != nullptr && cutlass_env[0] != '\0'
          ? std::filesystem::path(cutlass_env)
          : source_root / "third-party/cutlass/include";

  std::vector<std::string> options = {
      "-std=c++20",
      "--diag-suppress=39,161,174,177,186,940",
      "--ptxas-options=--register-usage-level=10",
      "--expt-relaxed-constexpr",
      "--device-int128",
      "-default-device",
      "--gpu-architecture=sm_" + arch_flag(major, minor, nvrtc_major, nvrtc_minor),
      "-I" + (library_root / "include").string(),
      "-I" + cutlass_include.string(),
      "-I" + (cuda_home / "include").string(),
  };
  if (nvrtc_major > 12 || (nvrtc_major == 12 && nvrtc_minor >= 8)) {
    options.push_back("--pch");
  }

  std::vector<const char*> option_ptrs;
  option_ptrs.reserve(options.size());
  for (const auto& option : options) option_ptrs.push_back(option.c_str());

  nvrtcProgram program = nullptr;
  check_nvrtc(
      nvrtcCreateProgram(&program, code.c_str(), "deepgemm_native.cu", 0, nullptr, nullptr),
      "nvrtcCreateProgram");
  nvrtcResult compile_result =
      nvrtcCompileProgram(program, static_cast<int>(option_ptrs.size()), option_ptrs.data());

  size_t log_size = 0;
  check_nvrtc(nvrtcGetProgramLogSize(program, &log_size), "nvrtcGetProgramLogSize");
  std::string log;
  if (log_size > 1) {
    log.resize(log_size);
    check_nvrtc(nvrtcGetProgramLog(program, log.data()), "nvrtcGetProgramLog");
  }
  if (compile_result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    throw std::runtime_error("NVRTC DeepGEMM compile failed:\n" + log);
  }

  size_t cubin_size = 0;
  check_nvrtc(nvrtcGetCUBINSize(program, &cubin_size), "nvrtcGetCUBINSize");
  std::string cubin(cubin_size, '\0');
  check_nvrtc(nvrtcGetCUBIN(program, cubin.data()), "nvrtcGetCUBIN");
  check_nvrtc(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
  write_binary_file(cubin_path, cubin);
  return log;
}

#ifndef DG_JIT_USE_LIBRARY_ENUM_KERNELS
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
#endif

std::shared_ptr<NativeRuntime> get_or_build_runtime(
    const std::string& name,
    const std::string& code,
    int major,
    int minor) {
  const std::string key = name + "$$" + arch_flag(major, minor, 12, 9) + "$$" + code;
  const std::string digest = hex_digest(key);

  std::lock_guard<std::mutex> lock(g_runtime_mu);
  if (auto it = g_runtimes.find(digest); it != g_runtimes.end()) {
    return it->second;
  }

  const auto dir = cache_root() / "cache" / ("kernel." + name + "." + digest);
  const auto cubin_path = dir / "kernel.cubin";
  const auto code_path = dir / "kernel.cu";
  if (!std::filesystem::exists(cubin_path)) {
    const auto tmp = cache_root() / "tmp" / ("arle-" + digest);
    std::filesystem::create_directories(tmp);
    const auto tmp_cubin = tmp / "kernel.cubin";
    const auto tmp_code = tmp / "kernel.cu";
    write_binary_file(tmp_code, code);
    compile_with_nvrtc(code, tmp_cubin, major, minor);
    std::filesystem::create_directories(dir.parent_path());
    std::error_code ec;
    std::filesystem::rename(tmp, dir, ec);
    if (ec) {
      std::filesystem::remove_all(tmp, ec);
    }
  }
  if (!std::filesystem::exists(code_path)) {
    write_binary_file(code_path, code);
  }

  auto runtime = std::make_shared<NativeRuntime>();
#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
  const std::string symbol;
#else
  const std::string symbol = parse_kernel_symbol(cubin_path);
#endif
  runtime->kernel = load_kernel(cubin_path, symbol, &runtime->library);
  runtime->loaded = true;
  g_runtimes.emplace(digest, runtime);
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
  int device = 0;
  cudaError_t cuda_err = cudaGetDevice(&device);
  if (cuda_err != cudaSuccess) return static_cast<CUresult>(cuda_err);

  cudaDeviceProp prop{};
  cuda_err = cudaGetDeviceProperties(&prop, device);
  if (cuda_err != cudaSuccess) return static_cast<CUresult>(cuda_err);
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
  const auto code = generate_kernel_code(desc, config);
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

}  // namespace

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
      sfa_aligned_m != get_tma_aligned_size(m, sizeof(float))) {
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

#endif  // ARLE_ENABLE_DEEPGEMM_NATIVE
