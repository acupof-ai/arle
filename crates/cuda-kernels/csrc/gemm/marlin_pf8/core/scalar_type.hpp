#pragma once

// Adapted from vLLM core/scalar_type.hpp constants (Apache-2.0).
//
// Minimal vLLM scalar type shim for the PF8.3 Marlin specialization.
//
// The upstream Marlin template references vllm::ScalarType and the public
// scalar constants from core/scalar_type.hpp. ARLE's cuda-kernels crate does
// not depend on vLLM or Torch, so this local shim keeps the copied template
// self-contained while preserving the upstream type ids used by generated
// specializations.
namespace vllm {

using ScalarTypeId = long;

struct ScalarType {
  ScalarTypeId id_;
  int bits_;

  constexpr ScalarTypeId id() const { return id_; }
  constexpr int size_bits() const { return bits_; }

  constexpr bool operator==(const ScalarType& other) const {
    return id_ == other.id_;
  }
  constexpr bool operator!=(const ScalarType& other) const {
    return id_ != other.id_;
  }

  static constexpr ScalarType from_id(ScalarTypeId id) {
    return id == 0 ? ScalarType{0, 16}
         : id == 1 ? ScalarType{1, 16}
         : id == 2 ? ScalarType{2, 8}
         : id == 3 ? ScalarType{3, 8}
         : id == 4 ? ScalarType{4, 4}
         : id == 5 ? ScalarType{5, 4}
         : id == 6 ? ScalarType{6, 8}
         : id == 7 ? ScalarType{7, 8}
         : id == 8 ? ScalarType{8, 4}
         : id == 9 ? ScalarType{9, 4}
         : id == 10 ? ScalarType{10, 8}
                    : ScalarType{-1, 0};
  }
};

inline constexpr ScalarType kFloat16{0, 16};
inline constexpr ScalarType kBFloat16{1, 16};
inline constexpr ScalarType kFE4M3fn{2, 8};
inline constexpr ScalarType kS8{3, 8};
inline constexpr ScalarType kU4B8{4, 4};
inline constexpr ScalarType kU4{5, 4};
inline constexpr ScalarType kU8B128{6, 8};
inline constexpr ScalarType kU8{7, 8};
inline constexpr ScalarType kS4{8, 4};
inline constexpr ScalarType kFE2M1f{9, 4};
inline constexpr ScalarType kFE8M0fnu{10, 8};

}  // namespace vllm
