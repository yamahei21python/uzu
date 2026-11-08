#pragma once

#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../common/packed.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <ushort BITS>
METAL_FUNC constexpr uint symmetric_zero_point() {
  return 1u << (BITS - 1);
}

template <ushort BITS>
METAL_FUNC uint decode_zero_point(uint8_t packed, uint group_index) {
  static_assert(BITS == 4 || BITS == 6 || BITS == 8, "zero points are 4, 6 or 8 bit");
  if constexpr (BITS == 4) {
    return read_packed<4>(packed, group_index);
  } else {
    // 6- and 8-bit zero points take a whole byte each, so the index is unused.
    (void)group_index;
    return uint(packed);
  }
}

template <ushort BITS>
METAL_FUNC uint decode_zero_point(const device uint8_t* zero_points_row, uint group_index) {
  static_assert(BITS == 4 || BITS == 6 || BITS == 8, "zero points are 4, 6 or 8 bit");
  if constexpr (BITS == 4) {
    return read_packed<4>(zero_points_row, group_index);
  } else {
    return uint(zero_points_row[group_index]);
  }
}

UZU_CONST ushort W4_BITS = 4;
UZU_CONST uint W4_SIGN_MASK = symmetric_zero_point<W4_BITS>() * 0x11111111u;
UZU_CONST uint W8_SIGN_MASK = symmetric_zero_point<8>() * 0x01010101u;
UZU_CONST uint W4_NIBBLE_MASK = 0x0F0F0F0Fu;
UZU_CONST uint ZERO_POINTS_PER_BYTE = 2;
UZU_CONST uint W8_MAGIC_FLOAT_OFFSET = 0x4B400000u;

UZU_CONST uint W4_PAIR_HALF_BITS = 16;
UZU_CONST uint W4_PAIR_MANTISSA_BITS = 10;
UZU_CONST uint W4_PAIR_EXPONENT_BIAS = 15;

static METAL_FUNC constexpr uint w4_pair_offset(uint component) {
  constexpr uint COMPONENTS = W4_PAIR_HALF_BITS / W4_BITS;
  return W4_BITS * (component < COMPONENTS ? component : COMPONENTS - 1);
}

static METAL_FUNC constexpr uint w4_pair_position(uint component) {
  const uint offset = w4_pair_offset(component);
  return offset + W4_BITS <= W4_PAIR_MANTISSA_BITS ? offset : offset - (W4_PAIR_HALF_BITS - W4_PAIR_MANTISSA_BITS);
}

template <uint COMPONENT>
static METAL_FUNC half2 w4_pair_at(uint word) {
  constexpr uint offset = w4_pair_offset(COMPONENT);
  constexpr uint position = w4_pair_position(COMPONENT);
  constexpr uint mask = ((1u << W4_BITS) - 1) << position;
  constexpr uint exponent = (W4_PAIR_EXPONENT_BIAS + W4_PAIR_MANTISSA_BITS - position) << W4_PAIR_MANTISSA_BITS;
  return as_type<half2>(
      ((word >> (offset - position)) & (mask | (mask << W4_PAIR_HALF_BITS))) |
      (exponent | (exponent << W4_PAIR_HALF_BITS))
  );
}

static METAL_FUNC constexpr float w4_pair_bias_value(uint component) {
  return float(1u << (W4_PAIR_MANTISSA_BITS - w4_pair_position(component)));
}

static METAL_FUNC half2 w4_pair(uint word, ushort component) {
  switch (component) {
  case 0:
    return w4_pair_at<0>(word);
  case 1:
    return w4_pair_at<1>(word);
  case 2:
    return w4_pair_at<2>(word);
  default:
    return w4_pair_at<3>(word);
  }
}

template <uint ORIGIN>
METAL_FUNC float magic_code_float(uint code) {
  return as_type<float>(code + W8_MAGIC_FLOAT_OFFSET - ORIGIN);
}

METAL_FUNC float magic_dequant_bias(float scale) { return -12582912.0f * scale; }

template <typename StageT, typename ComputeT, int N, int bits>
inline void dequantize(
    const device uint8_t* w,
    float scale,
    float bias,
    threadgroup StageT* w_local,
    const bool signed_codes
) {
  static_assert(bits == 4 || bits == 6 || bits == 8, "Only int4, int6 and int8 supported");
  static_assert(bits == 8 || N % 2 == 0, "int4 values are unpacked in pairs");
  constexpr bool IS_16_BIT_COMPUTE = sizeof(ComputeT) == 2;
  constexpr bool READ_CODE_WORDS = bits == 4 && N % 8 == 0 && IS_16_BIT_COMPUTE;
  threadgroup vec<StageT, 2>* pairs = reinterpret_cast<threadgroup vec<StageT, 2>*>(w_local);
  const ComputeT compute_scale = ComputeT(scale);
  const ComputeT compute_bias = ComputeT(bias);
  auto dequantize_code = [&](uint code) { return fma(ComputeT(code), compute_scale, compute_bias); };
  if constexpr (READ_CODE_WORDS) {
    const device uint* words = reinterpret_cast<const device uint*>(w);
    for (int i = 0; i < N / 8; i++) {
      const uint word = signed_codes ? words[i] ^ W4_SIGN_MASK : words[i];
      for (int j = 0; j < 4; j++) {
        pairs[i * 4 + j] = vec<StageT, 2>(
            vec<ComputeT, 2>(dequantize_code((word >> (8 * j)) & 0x0fu), dequantize_code((word >> (8 * j + 4)) & 0x0fu))
        );
      }
    }
  } else if constexpr (bits == 4) {
    const uint sign_flip = signed_codes ? 0x88u : 0u;
    for (int i = 0; i < (N / 2); i++) {
      const uint code = uint(w[i]) ^ sign_flip;
      pairs[i] = vec<StageT, 2>(vec<ComputeT, 2>(dequantize_code(code & 0x0fu), dequantize_code(code >> 4)));
    }
  } else if constexpr (bits == 6) {
    // Six-bit codes form an LSB-first bit stream: value i lives at bit 6 * i, so one
    // pack of four values spans three bytes. Six-bit codes are always unsigned
    // (asymmetric zero-point only), so signed_codes must be false here; a byte-wide
    // sign flip cannot align with 6-bit code boundaries.
    (void)signed_codes;
    for (int i = 0; i < (N / 4); i++) {
      const uint b0 = uint(w[3 * i]);
      const uint b1 = uint(w[3 * i + 1]);
      const uint b2 = uint(w[3 * i + 2]);
      pairs[2 * i] = vec<StageT, 2>(
          vec<ComputeT, 2>(
              dequantize_code(b0 & 0x3Fu),
              dequantize_code(((b0 >> 6) | (b1 << 2)) & 0x3Fu)
          )
      );
      pairs[2 * i + 1] = vec<StageT, 2>(
          vec<ComputeT, 2>(
              dequantize_code(((b1 >> 4) | (b2 << 4)) & 0x3Fu),
              dequantize_code((b2 >> 2) & 0x3Fu)
          )
      );
    }
    static_assert(N % 4 == 0, "6-bit dequantize works on complete four-value packs");
  } else {
    const uint sign_flip = signed_codes ? 0x80u : 0u;
    for (int i = 0; i < (N / 2); i++) {
      pairs[i] = vec<StageT, 2>(
          vec<ComputeT, 2>(dequantize_code(uint(w[2 * i]) ^ sign_flip), dequantize_code(uint(w[2 * i + 1]) ^ sign_flip))
      );
    }
    if constexpr (N % 2 == 1) {
      w_local[N - 1] = StageT(fma(float(uint(w[N - 1]) ^ sign_flip), scale, bias));
    }
  }
}

// A byte ored with 0x6400 is the half 1024 plus the byte, so the subtraction is exact.
METAL_FUNC float4 bytes_to_float(const uint word, const float offset) {
  const half2 bias = half2(half(1024.0f + offset));
  const half2 even = as_type<half2>((word & 0x00FF00FFu) | 0x64006400u) - bias;
  const half2 odd = as_type<half2>(((word >> 8) & 0x00FF00FFu) | 0x64006400u) - bias;
  return float4(float(even.x), float(odd.x), float(even.y), float(odd.y));
}

METAL_FUNC float4 int8_to_float(const uint word) { return bytes_to_float(word ^ 0x80808080u, 128.0f); }

} // namespace gemm
} // namespace uzu
