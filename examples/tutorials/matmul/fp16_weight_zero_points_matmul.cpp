/// @example fp16_weight_zero_points_matmul.cpp
/// @brief C++ API example demonstrating MatMul with u4 weights,
///        fp16 per-group scales, and fp16 per-group zero points.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

namespace {

constexpr int64_t kM = 3;
constexpr int64_t kK = 64;
constexpr int64_t kN = 4;
constexpr int64_t kGroupK = 32;
constexpr float kTolerance = 0.35f;

uint16_t float_to_half_bits(float value) {
    union {
        float f;
        uint32_t u;
    } bits {value};

    const uint32_t sign = (bits.u >> 16) & 0x8000u;
    uint32_t exp = (bits.u >> 23) & 0xFFu;
    uint32_t man = bits.u & 0x7FFFFFu;

    if (exp == 0xFFu)
        return static_cast<uint16_t>(sign | 0x7C00u | (man ? 0x0200u : 0));
    if (exp == 0) return static_cast<uint16_t>(sign);

    int new_exp = static_cast<int>(exp) - 127 + 15;
    if (new_exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    if (new_exp <= 0) return static_cast<uint16_t>(sign);

    return static_cast<uint16_t>(
            sign | (static_cast<uint32_t>(new_exp) << 10)
            | ((man + 0x1000u) >> 13));
}

float half_bits_to_float(uint16_t bits) {
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1Fu;
    uint32_t man = static_cast<uint32_t>(bits) & 0x03FFu;

    if (exp == 0x1Fu) {
        union {
            uint32_t u;
            float f;
        } result {sign | 0x7F800000u | (man << 13)};
        return result.f;
    }

    if (exp == 0) {
        if (man == 0) {
            union {
                uint32_t u;
                float f;
            } result {sign};
            return result.f;
        }

        while ((man & 0x0400u) == 0) {
            man <<= 1;
            exp--;
        }
        exp++;
        man &= 0x03FFu;
    }

    union {
        uint32_t u;
        float f;
    } result {sign | ((exp - 15 + 127) << 23) | (man << 13)};
    return result.f;
}

std::vector<uint8_t> pack_u4_values(const std::vector<uint8_t> &values) {
    std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
    for (size_t index = 0; index < values.size(); ++index) {
        const uint8_t nibble = static_cast<uint8_t>(values[index] & 0x0F);
        if ((index & 1u) == 0)
            packed[index / 2] = nibble;
        else
            packed[index / 2] |= static_cast<uint8_t>(nibble << 4);
    }
    return packed;
}

void write_fp16_values(const std::vector<float> &values, memory &mem) {
    std::vector<uint16_t> fp16_bits(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        fp16_bits[index] = float_to_half_bits(values[index]);
    }
    write_to_dnnl_memory(fp16_bits.data(), mem);
}

std::vector<float> read_fp16_values(memory &mem) {
    const size_t count = mem.get_desc().get_size() / sizeof(uint16_t);
    std::vector<uint16_t> fp16_bits(count);
    read_from_dnnl_memory(fp16_bits.data(), mem);

    std::vector<float> values(count);
    for (size_t index = 0; index < count; ++index) {
        values[index] = half_bits_to_float(fp16_bits[index]);
    }
    return values;
}

matmul::primitive_desc create_matmul_pd(const engine &eng) {
    memory::desc src_md({kM, kK}, memory::data_type::f16, {kK, 1});
    memory::desc weights_md(
            {kK, kN}, memory::data_type::u4, memory::format_tag::any);
    memory::desc dst_md({kM, kN}, memory::data_type::f16, {kN, 1});

    primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {kGroupK, 1},
            memory::data_type::f16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {kGroupK, 1},
            memory::data_type::f16);
    attr.set_fpmath_mode(fpmath_mode::f16, true);

    return matmul::primitive_desc(eng, src_md, weights_md, dst_md, attr);
}

std::vector<float> build_src_values() {
    std::vector<float> src(kM * kK);
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t k = 0; k < kK; ++k) {
            const float signed_pattern = static_cast<float>((k % 7) - 3);
            src[m * kK + k] = 0.0625f * static_cast<float>(m + 1)
                    * (signed_pattern + 0.5f * static_cast<float>(k & 1));
        }
    }
    return src;
}

std::vector<uint8_t> build_u4_weights() {
    std::vector<uint8_t> weights(kK * kN);
    for (int64_t k = 0; k < kK; ++k) {
        for (int64_t n = 0; n < kN; ++n) {
            weights[k * kN + n]
                    = static_cast<uint8_t>((3 * k + 5 * n + (k / 3)) & 0x0F);
        }
    }
    return weights;
}

std::vector<float> build_group_scales() {
    const int64_t groups = kK / kGroupK;
    std::vector<float> scales(groups * kN);
    for (int64_t group = 0; group < groups; ++group) {
        for (int64_t n = 0; n < kN; ++n) {
            scales[group * kN + n] = 0.125f * static_cast<float>(group + 1)
                    + 0.03125f * static_cast<float>(n + 1);
        }
    }
    return scales;
}

std::vector<float> build_group_zero_points() {
    const int64_t groups = kK / kGroupK;
    std::vector<float> zero_points(groups * kN);
    for (int64_t group = 0; group < groups; ++group) {
        for (int64_t n = 0; n < kN; ++n) {
            const float base = (group == 0) ? 0.375f : -0.4375f;
            zero_points[group * kN + n]
                    = base + 0.0625f * static_cast<float>(n);
        }
    }
    return zero_points;
}

std::vector<float> compute_reference(const std::vector<float> &src,
        const std::vector<uint8_t> &weights_u4,
        const std::vector<float> &group_scales,
        const std::vector<float> &group_zero_points) {
    std::vector<float> dst(kM * kN, 0.0f);
    for (int64_t m = 0; m < kM; ++m) {
        for (int64_t n = 0; n < kN; ++n) {
            float acc = 0.0f;
            for (int64_t k = 0; k < kK; ++k) {
                const int64_t group = k / kGroupK;
                const float scale = group_scales[group * kN + n];
                const float zero_point = group_zero_points[group * kN + n];
                const float weight = scale
                                * static_cast<float>(weights_u4[k * kN + n])
                        - zero_point;
                acc += src[m * kK + k] * weight;
            }
            dst[m * kN + n] = acc;
        }
    }
    return dst;
}

void run_example(engine::kind engine_kind) {
    if (engine_kind != engine::kind::gpu) {
        throw example_allows_unimplemented(
                "This example requires the GPU engine.");
    }

    engine eng(engine_kind, 0);
    stream strm(eng);

    try {
        auto matmul_pd = create_matmul_pd(eng);

        auto src_values = build_src_values();
        auto weights_u4 = build_u4_weights();
        auto group_scales = build_group_scales();
        auto group_zero_points = build_group_zero_points();
        auto reference = compute_reference(
                src_values, weights_u4, group_scales, group_zero_points);

        memory src_mem({{kM, kK}, memory::data_type::f16, {kK, 1}}, eng);
        write_fp16_values(src_values, src_mem);

        memory scale_mem({{kN, kK / kGroupK}, memory::data_type::f16, {1, kN}},
                eng);
        write_fp16_values(group_scales, scale_mem);

        memory zero_point_mem(
                {{kN, kK / kGroupK}, memory::data_type::f16, {1, kN}}, eng);
        write_fp16_values(group_zero_points, zero_point_mem);

        auto packed_weights = pack_u4_values(weights_u4);
        memory weights_plain_mem(
                {{kK, kN}, memory::data_type::u4, {kN, 1}}, eng);
        write_to_dnnl_memory(packed_weights.data(), weights_plain_mem);

        memory weights_mem(matmul_pd.weights_desc(), eng);
        reorder(weights_plain_mem, weights_mem)
                .execute(strm, weights_plain_mem, weights_mem);
        strm.wait();

        memory dst_mem({{kM, kN}, memory::data_type::f16, {kN, 1}}, eng);
        matmul matmul_prim(matmul_pd);
        matmul_prim.execute(strm,
                {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
                        {DNNL_ARG_DST, dst_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_mem},
                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                                zero_point_mem}});
        strm.wait();

        auto actual = read_fp16_values(dst_mem);

        float max_abs_diff = 0.0f;
        size_t max_index = 0;
        for (size_t index = 0; index < actual.size(); ++index) {
            const float diff = std::fabs(actual[index] - reference[index]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_index = index;
            }
        }

        if (max_abs_diff > kTolerance) {
            std::ostringstream message;
            message << "reference mismatch at index " << max_index
                    << ": actual=" << actual[max_index]
                    << ", reference=" << reference[max_index]
                    << ", max_abs_diff=" << max_abs_diff;
            throw std::runtime_error(message.str());
        }

        std::cout
                << "Verified u4 MatMul with fp16 group zero points. Max abs diff = "
                << max_abs_diff << std::endl;
    } catch (const error &e) {
        if (e.status == dnnl_unimplemented) {
            throw example_allows_unimplemented(
                    "The current GPU build does not expose fp16 weight zero-point decomposition support.");
        }
        throw;
    }
}

} // namespace

int main(int argc, char **argv) {
    return handle_example_errors(run_example, parse_engine_kind(argc, argv));
}