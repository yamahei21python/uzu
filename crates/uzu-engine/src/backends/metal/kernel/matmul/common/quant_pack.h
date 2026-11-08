#pragma once

#include <metal_stdlib>

namespace uzu {
namespace gemm {

template <int bits, int word_size_bits = 8>
METAL_FUNC constexpr short get_pack_factor() {
  // 6-bit codes do not land on byte boundaries: a pack is 4 values in 3 bytes.
  if constexpr (bits == 6) {
    return 4;
  } else {
    return word_size_bits / bits;
  }
}

template <int bits, int word_size_bits = 8>
METAL_FUNC constexpr short get_bytes_per_pack() {
  if constexpr (bits == 6) {
    return 3;
  } else {
    return word_size_bits / 8;
  }
}

} // namespace gemm
} // namespace uzu
