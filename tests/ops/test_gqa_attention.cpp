// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"
#include "ops/gqa_q4_prompt_control.h"
#include "ops/gqa_q4_decode_control.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim       = 256;
constexpr std::int32_t kQuantGroup    = 64;
constexpr std::int32_t kQuantGroups   = kHeadDim / kQuantGroup;
constexpr float kAttentionScale       = 0.0625f;
constexpr std::uint16_t kOutputCanary = 0x7fc1u;
bool benchmark_prompt_staging = false;
bool benchmark_prompt_pipeline = false;

// BF16 and quantized compute profiles. Q4 uses the unchanged INT8-profile criterion.
// A1 and A3 use the same criterion for a given
// profile; token count, geometry, execution envelope, and private launch route do not select it.
constexpr ReductionCriterion kAttentionBf16Criterion{
    /*relative_l2*/ 2.8e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 2.7e-3,
};

constexpr ReductionCriterion kAttentionInt8Criterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ 2.2e-3,
};

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

constexpr Geometry kGeometries[] = {
    {"qwen3_6_27b", 24, 4},
    {"qwen3_6_35b_a3b", 16, 2},
};

struct AttentionCase {
    std::int32_t tokens;
    std::int32_t base;
    std::uint32_t envelope_max;
    std::uint32_t seed;
};

enum class MappingPattern { Identity, Offset, Fragmented };

const char* mapping_name(MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return "identity";
    case MappingPattern::Offset:
        return "offset";
    case MappingPattern::Fragmented:
        return "fragmented";
    }
    return "unknown";
}

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::int32_t physical_page_count(std::int32_t logical_pages, MappingPattern pattern) {
    switch (pattern) {
    case MappingPattern::Identity:
        return logical_pages;
    case MappingPattern::Offset:
        return logical_pages + 2;
    case MappingPattern::Fragmented:
        return 2 * logical_pages + 1;
    }
    return 0;
}

std::vector<std::int32_t> make_block_table(std::int32_t logical_pages, MappingPattern pattern) {
    std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages));
    switch (pattern) {
    case MappingPattern::Identity:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page; }
        break;
    case MappingPattern::Offset:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = page + 1; }
        break;
    case MappingPattern::Fragmented:
        for (std::int32_t page = 0; page < logical_pages; ++page) { table[page] = 2 * page + 1; }
        break;
    }
    return table;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::size_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::size_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::size_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::size_t>(physical_page)));
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity,
                             std::span<const std::int32_t> block_table,
                             std::int32_t physical_pages) {
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(physical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

template <typename T>
void scatter_paged_into(const std::vector<T>& logical, std::int32_t leading_extent,
                        const Geometry& geometry, std::int32_t logical_capacity,
                        std::span<const std::int32_t> block_table, std::vector<T>& physical) {
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
}

template <typename T>
std::vector<T> gather_paged(std::span<const T> physical, std::int32_t leading_extent,
                            const Geometry& geometry, std::int32_t logical_capacity,
                            std::span<const std::int32_t> block_table) {
    std::vector<T> logical(static_cast<std::size_t>(leading_extent) * logical_capacity *
                           static_cast<std::size_t>(geometry.kv_heads));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page =
                block_table[static_cast<std::size_t>(position) / kPagedKVPageSize];
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t target = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) * head);
                logical[target] =
                    physical[paged_index(leading_extent, geometry, page, head, position, leading)];
            }
        }
    }
    return logical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> bf16_bits_to_double(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        values[i] = static_cast<double>(bf16_to_f32(bits[i]));
    }
    return values;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }

    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

std::int32_t code_extent(DType dtype) { return dtype == DType::U8 ? kHeadDim / 2 : kHeadDim; }

// Independent signed-nibble codec. HostCache keeps unpacked logical signed codes so the
// mathematical oracle never depends on a production staging layout or device helper.
std::vector<std::int8_t> pack_q4(const std::vector<std::int8_t>& codes) {
    std::vector<std::int8_t> packed(codes.size() / 2);
    for (std::size_t i = 0; i < packed.size(); ++i) {
        const unsigned lo = static_cast<unsigned>(codes[2 * i]) & 15u;
        const unsigned hi = static_cast<unsigned>(codes[2 * i + 1]) & 15u;
        packed[i] = static_cast<std::int8_t>(lo | (hi << 4));
    }
    return packed;
}

std::vector<std::int8_t> unpack_q4(const std::vector<std::int8_t>& packed) {
    std::vector<std::int8_t> codes(packed.size() * 2);
    for (std::size_t i = 0; i < packed.size(); ++i) {
        const unsigned byte = static_cast<std::uint8_t>(packed[i]);
        const int lo = static_cast<int>(byte & 15u);
        const int hi = static_cast<int>(byte >> 4);
        codes[2 * i] = static_cast<std::int8_t>(lo >= 8 ? lo - 16 : lo);
        codes[2 * i + 1] = static_cast<std::int8_t>(hi >= 8 ? hi - 16 : hi);
    }
    return codes;
}

std::vector<std::int8_t> storage_codes(const std::vector<std::int8_t>& codes, DType dtype) {
    return dtype == DType::U8 ? pack_q4(codes) : codes;
}

void encode_group(const std::vector<float>& source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset,
                  std::int32_t maximum_code = 127) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }

    const float unrounded_scale    = absmax / static_cast<float>(maximum_code);
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -maximum_code, maximum_code);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

HostCache make_cache(const Geometry& geometry, DType dtype, std::int32_t max_context,
                     std::uint32_t seed) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, dtype, max_context, logical_capacity};
    if (dtype == DType::BF16) {
        cache.k_bf16 = to_bf16_bits(logical_k);
        cache.v_bf16 = to_bf16_bits(logical_v);
        return cache;
    }

    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    const std::size_t scales = scale_elements(geometry, logical_capacity);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_k, code, cache.k_i8, code, cache.k_scale, scale,
                             dtype == DType::U8 ? 7 : 127);
                encode_group(logical_v, code, cache.v_i8, code, cache.v_scale, scale,
                             dtype == DType::U8 ? 7 : 127);
            }
        }
    }
    return cache;
}

void append_cache(HostCache& cache, const std::vector<float>& k, const std::vector<float>& v,
                  const std::vector<std::int32_t>& positions) {
    const Geometry& geometry = cache.geometry;
    for (std::int32_t token = 0; token < static_cast<std::int32_t>(positions.size()); ++token) {
        const std::int32_t position = positions[static_cast<std::size_t>(token)];
        for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
            if (cache.dtype == DType::BF16) {
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    const std::size_t source = kv_input_index(geometry, head, d, token);
                    const std::size_t target =
                        cache_index(geometry, cache.logical_capacity, head, position, d);
                    cache.k_bf16[target] = f32_to_bf16(k[source]);
                    cache.v_bf16[target] = f32_to_bf16(v[source]);
                }
                continue;
            }

            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d     = group * kQuantGroup;
                const std::size_t source = kv_input_index(geometry, head, d, token);
                const std::size_t target =
                    cache_index(geometry, cache.logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, cache.logical_capacity, head, position, group);
                encode_group(k, source, cache.k_i8, target, cache.k_scale, scale,
                             cache.dtype == DType::U8 ? 7 : 127);
                encode_group(v, source, cache.v_i8, target, cache.v_scale, scale,
                             cache.dtype == DType::U8 ? 7 : 127);
            }
        }
    }
}

double cache_value(const HostCache& cache, bool key, std::int32_t head, std::int32_t position,
                   std::int32_t d) {
    const std::size_t code = cache_index(cache.geometry, cache.logical_capacity, head, position, d);
    if (cache.dtype == DType::BF16) {
        return static_cast<double>(bf16_to_f32(key ? cache.k_bf16[code] : cache.v_bf16[code]));
    }

    const std::size_t scale =
        scale_index(cache.geometry, cache.logical_capacity, head, position, d / kQuantGroup);
    const auto& codes   = key ? cache.k_i8 : cache.v_i8;
    const auto& scales  = key ? cache.k_scale : cache.v_scale;
    const float decoded = static_cast<float>(codes[code]) * f16_bits_to_f32(scales[scale]);
    return static_cast<double>(decoded);
}

std::vector<double> ideal_attention(const std::vector<float>& q, const HostCache& cache,
                                    const std::vector<std::int32_t>& positions) {
    const Geometry& geometry  = cache.geometry;
    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(static_cast<std::size_t>(kHeadDim) *
                               static_cast<std::size_t>(geometry.q_heads) *
                               static_cast<std::size_t>(tokens));

    std::vector<double> scores(static_cast<std::size_t>(positions.back()) + 1);
    std::vector<double> probabilities(scores.size());
    // Decode the represented cache once per KV head, independently of the production Q8,
    // BF16-P/V and partial-reduction stages. The formula and per-element reduction order stay
    // naive FP64. Contiguous value-row traversal makes full-128K oracle samples practical.
    std::vector<double> keys(scores.size() * kHeadDim);
    std::vector<double> values(keys.size());
    for (std::int32_t kv_head = 0; kv_head < geometry.kv_heads; ++kv_head) {
        for (std::int32_t position = 0; position <= positions.back(); ++position) {
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                const std::size_t index = static_cast<std::size_t>(position) * kHeadDim + d;
                keys[index] = cache_value(cache, true, kv_head, position, d);
                values[index] = cache_value(cache, false, kv_head, position, d);
            }
        }
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
            for (std::int32_t local_q = 0; local_q < geometry.query_group(); ++local_q) {
            const std::int32_t q_head = kv_head * geometry.query_group() + local_q;
            double max_score           = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    dot += static_cast<double>(q[q_index(geometry, q_head, d, token)]) *
                           keys[static_cast<std::size_t>(position) * kHeadDim + d];
                }
                const double score = dot * static_cast<double>(kAttentionScale);
                scores[static_cast<std::size_t>(position)] = score;
                max_score                                  = std::max(max_score, score);
            }

            double sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                const double probability =
                    std::exp(scores[static_cast<std::size_t>(position)] - max_score);
                probabilities[static_cast<std::size_t>(position)] = probability;
                sum += probability;
            }
            for (std::int32_t position = 0; position < visible; ++position) {
                probabilities[static_cast<std::size_t>(position)] /= sum;
            }

            const std::size_t output_base = q_index(geometry, q_head, 0, token);
            for (std::int32_t position = 0; position < visible; ++position) {
                const double probability = probabilities[static_cast<std::size_t>(position)];
                const std::size_t value_base = static_cast<std::size_t>(position) * kHeadDim;
                for (std::int32_t d = 0; d < kHeadDim; ++d) {
                    output[output_base + d] += probability * values[value_base + d];
                }
            }
            }
        }
    }
    return output;
}

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

std::uint64_t byte_digest(const void* data, std::size_t bytes) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto* input = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { hash = (hash ^ input[i]) * 1099511628211ull; }
    return hash;
}

void emit_output_digest(const std::string& label, std::span<const std::uint16_t> output) {
    std::ostringstream name;
    name << std::hex << std::setfill('0') << std::setw(16)
         << byte_digest(label.data(), label.size());
    std::cout << "PARITY_DIGEST case=" << name.str() << " elements=" << output.size()
              << " fnv1a64=" << std::hex << std::setfill('0') << std::setw(16)
              << byte_digest(output.data(), output.size_bytes()) << std::dec
              << " label=" << label << '\n';
    if (const char* directory = std::getenv("NINFER_GQA_DUMP_DIR")) {
        const std::filesystem::path root(directory);
        std::filesystem::create_directories(root);
        std::ofstream file(root / (name.str() + ".bf16"), std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()),
                   static_cast<std::streamsize>(output.size_bytes()));
        if (!file) { throw std::runtime_error("cannot write GQA parity output"); }
        std::ofstream metadata(root / (name.str() + ".txt"));
        metadata << label << '\n';
        if (!metadata) { throw std::runtime_error("cannot write GQA parity metadata"); }
    }
}

template <class Launch>
int verify_graph_replay(const std::string& label, Launch&& launch,
                        const GuardedDeviceBuffer& output, std::size_t elements) {
    const auto eager = copy_from_guarded<std::uint16_t>(output, elements);
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t execution = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "GQA graph stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "GQA begin graph");
    launch(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "GQA end graph");
    cuda_check(cudaGraphInstantiate(&execution, graph, 0), "GQA instantiate graph");
    int failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        cuda_check(cudaGraphLaunch(execution, stream), "GQA graph replay");
        cuda_check(cudaStreamSynchronize(stream), "GQA graph synchronize");
        failures += verify_exact((label + " graph/eager BF16 bit parity").c_str(),
                                  copy_from_guarded<std::uint16_t>(output, elements), eager);
    }
    cuda_check(cudaGraphExecDestroy(execution), "GQA destroy graph execution");
    cuda_check(cudaGraphDestroy(graph), "GQA destroy graph");
    cuda_check(cudaStreamDestroy(stream), "GQA destroy stream");
    emit_output_digest(label, eager);
    return failures;
}

class DeviceCache {
public:
    DeviceCache(const HostCache& cache, MappingPattern mapping)
        : geometry_(cache.geometry), dtype_(cache.dtype), max_context_(cache.max_context),
          logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(physical_page_count(logical_pages_, mapping)),
          block_table_host_(make_block_table(logical_pages_, mapping)),
          code_elements_(static_cast<std::size_t>(code_extent(dtype_)) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ != DType::BF16 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ != DType::BF16 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_table_(block_table_host_.size() * sizeof(std::int32_t)) {
        block_table_.copy_from_host(block_table_host_.data(),
                                    block_table_host_.size() * sizeof(std::int32_t));
        if (dtype_ == DType::BF16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto v_physical =
                scatter_paged(cache.v_bf16, kHeadDim, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else {
            const auto k_physical =
                scatter_paged(storage_codes(cache.k_i8, dtype_), code_extent(dtype_), geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto v_physical =
                scatter_paged(storage_codes(cache.v_i8, dtype_), code_extent(dtype_), geometry_, logical_capacity_, block_table_host_,
                              physical_pages_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_,
                              block_table_host_, physical_pages_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        }
    }

    PagedKVLayerView view() {
        PagedKVLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {code_extent(dtype_), kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {code_extent(dtype_), kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_table  = Tensor(block_table_.data(), DType::I32, {logical_pages_});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ != DType::BF16) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    PagedKVBatchLayerView batch_view() {
        const PagedKVLayerView direct = view();
        return {
            .k_pages       = direct.k_pages,
            .v_pages       = direct.v_pages,
            .k_scale_pages = direct.k_scale_pages,
            .v_scale_pages = direct.v_scale_pages,
            .block_tables  = direct.block_table.view({logical_pages_, 1}),
            .head_dim      = direct.head_dim,
            .num_kv_heads  = direct.num_kv_heads,
            .dtype         = direct.dtype,
            .quant_group   = direct.quant_group,
        };
    }

    HostCache snapshot() const {
        HostCache cache{geometry_, dtype_, max_context_, logical_capacity_};
        if (dtype_ == DType::BF16) {
            const auto k_physical = copy_from_guarded<std::uint16_t>(k_, code_elements_);
            const auto v_physical = copy_from_guarded<std::uint16_t>(v_, code_elements_);
            cache.k_bf16          = gather_paged<std::uint16_t>(k_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
            cache.v_bf16          = gather_paged<std::uint16_t>(v_physical, kHeadDim, geometry_,
                                                                logical_capacity_, block_table_host_);
        } else {
            const auto k_physical  = copy_from_guarded<std::int8_t>(k_, code_elements_);
            const auto v_physical  = copy_from_guarded<std::int8_t>(v_, code_elements_);
            const auto ks_physical = copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_);
            const auto vs_physical = copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_);
            cache.k_i8             = gather_paged<std::int8_t>(k_physical, code_extent(dtype_), geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.v_i8             = gather_paged<std::int8_t>(v_physical, code_extent(dtype_), geometry_,
                                                               logical_capacity_, block_table_host_);
            cache.k_scale = gather_paged<std::uint16_t>(ks_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            cache.v_scale = gather_paged<std::uint16_t>(vs_physical, kQuantGroups, geometry_,
                                                        logical_capacity_, block_table_host_);
            if (dtype_ == DType::U8) {
                cache.k_i8 = unpack_q4(cache.k_i8);
                cache.v_i8 = unpack_q4(cache.v_i8);
            }
        }
        return cache;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (dtype_ != DType::BF16) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        failures += block_table_.verify_guards((label + " block-table").c_str());
        failures +=
            verify_exact((label + " block-table unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_table_, block_table_host_.size()),
                         block_table_host_);
        return failures;
    }

private:
    Geometry geometry_;
    DType dtype_;
    std::int32_t max_context_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_table_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

class BatchDeviceCache {
public:
    BatchDeviceCache(std::span<const HostCache> rows, MappingPattern mapping)
        : geometry_(rows.front().geometry), dtype_(rows.front().dtype), rows_(rows.size()),
          logical_capacity_(rows.front().logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          physical_pages_(mapping == MappingPattern::Fragmented
                              ? 2 * static_cast<std::int32_t>(rows_) * logical_pages_ + 1
                              : static_cast<std::int32_t>(rows_) * logical_pages_),
          block_tables_host_(rows_ * static_cast<std::size_t>(logical_pages_)),
          code_elements_(static_cast<std::size_t>(code_extent(dtype_)) * kPagedKVPageSize *
                         geometry_.kv_heads * physical_pages_),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * physical_pages_),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ != DType::BF16 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ != DType::BF16 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_tables_(block_tables_host_.size() * sizeof(std::int32_t)) {
        for (std::size_t row = 0; row < rows_; ++row) {
            const HostCache& cache = rows[row];
            if (cache.geometry.q_heads != geometry_.q_heads ||
                cache.geometry.kv_heads != geometry_.kv_heads || cache.dtype != dtype_ ||
                cache.logical_capacity != logical_capacity_) {
                throw std::invalid_argument("batch cache rows must share one physical geometry");
            }
            for (std::int32_t logical = 0; logical < logical_pages_; ++logical) {
                const std::int32_t linear =
                    static_cast<std::int32_t>(row) * logical_pages_ + logical;
                block_tables_host_[row * static_cast<std::size_t>(logical_pages_) + logical] =
                    mapping == MappingPattern::Fragmented ? 2 * linear + 1 : linear;
            }
        }
        block_tables_.copy_from_host(block_tables_host_.data(),
                                     block_tables_host_.size() * sizeof(std::int32_t));
        upload_rows(rows);
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {code_extent(dtype_), kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {code_extent(dtype_), kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
        result.block_tables = Tensor(block_tables_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(rows_)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ != DType::BF16) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, physical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    int verify(const std::string& label, std::span<const HostCache> expected) const {
        if (expected.size() != rows_) {
            std::cerr << label << ": expected cache row count mismatch\n";
            return 1;
        }
        int failures = 0;
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> expected_k(code_elements_, 0);
            std::vector<std::uint16_t> expected_v(code_elements_, 0);
            scatter_bf16_rows(expected, expected_k, expected_v);
            failures +=
                verify_exact((label + " cache-k").c_str(),
                             copy_from_guarded<std::uint16_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v").c_str(),
                             copy_from_guarded<std::uint16_t>(v_, code_elements_), expected_v);
        } else {
            std::vector<std::int8_t> expected_k(code_elements_, 0);
            std::vector<std::int8_t> expected_v(code_elements_, 0);
            std::vector<std::uint16_t> expected_ks(scale_elements_, 0);
            std::vector<std::uint16_t> expected_vs(scale_elements_, 0);
            for (std::size_t row = 0; row < rows_; ++row) {
                const std::span<const std::int32_t> table = row_table(row);
                scatter_paged_into(storage_codes(expected[row].k_i8, dtype_), code_extent(dtype_), geometry_, logical_capacity_,
                                   table, expected_k);
                scatter_paged_into(storage_codes(expected[row].v_i8, dtype_), code_extent(dtype_), geometry_, logical_capacity_,
                                   table, expected_v);
                scatter_paged_into(expected[row].k_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_ks);
                scatter_paged_into(expected[row].v_scale, kQuantGroups, geometry_,
                                   logical_capacity_, table, expected_vs);
            }
            failures +=
                verify_exact((label + " cache-k-code").c_str(),
                             copy_from_guarded<std::int8_t>(k_, code_elements_), expected_k);
            failures +=
                verify_exact((label + " cache-v-code").c_str(),
                             copy_from_guarded<std::int8_t>(v_, code_elements_), expected_v);
            failures += verify_exact((label + " cache-k-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(k_scale_, scale_elements_),
                                     expected_ks);
            failures += verify_exact((label + " cache-v-scale").c_str(),
                                     copy_from_guarded<std::uint16_t>(v_scale_, scale_elements_),
                                     expected_vs);
        }
        failures +=
            verify_exact((label + " block tables unchanged").c_str(),
                         copy_from_guarded<std::int32_t>(block_tables_, block_tables_host_.size()),
                         block_tables_host_);
        failures += k_.verify_guards((label + " cache-k guard").c_str());
        failures += v_.verify_guards((label + " cache-v guard").c_str());
        if (dtype_ != DType::BF16) {
            failures += k_scale_.verify_guards((label + " cache-k-scale guard").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale guard").c_str());
        }
        failures += block_tables_.verify_guards((label + " block tables guard").c_str());
        return failures;
    }

private:
    [[nodiscard]] std::span<const std::int32_t> row_table(std::size_t row) const {
        return std::span<const std::int32_t>(block_tables_host_.data() +
                                                 row * static_cast<std::size_t>(logical_pages_),
                                             static_cast<std::size_t>(logical_pages_));
    }

    void scatter_bf16_rows(std::span<const HostCache> rows, std::vector<std::uint16_t>& k,
                           std::vector<std::uint16_t>& v) const {
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(rows[row].k_bf16, kHeadDim, geometry_, logical_capacity_, table, k);
            scatter_paged_into(rows[row].v_bf16, kHeadDim, geometry_, logical_capacity_, table, v);
        }
    }

    void upload_rows(std::span<const HostCache> rows) {
        if (dtype_ == DType::BF16) {
            std::vector<std::uint16_t> physical_k(code_elements_, 0);
            std::vector<std::uint16_t> physical_v(code_elements_, 0);
            scatter_bf16_rows(rows, physical_k, physical_v);
            k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::uint16_t));
            v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::uint16_t));
            return;
        }
        std::vector<std::int8_t> physical_k(code_elements_, 0);
        std::vector<std::int8_t> physical_v(code_elements_, 0);
        std::vector<std::uint16_t> physical_ks(scale_elements_, 0);
        std::vector<std::uint16_t> physical_vs(scale_elements_, 0);
        for (std::size_t row = 0; row < rows_; ++row) {
            const std::span<const std::int32_t> table = row_table(row);
            scatter_paged_into(storage_codes(rows[row].k_i8, dtype_), code_extent(dtype_), geometry_, logical_capacity_, table,
                               physical_k);
            scatter_paged_into(storage_codes(rows[row].v_i8, dtype_), code_extent(dtype_), geometry_, logical_capacity_, table,
                               physical_v);
            scatter_paged_into(rows[row].k_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_ks);
            scatter_paged_into(rows[row].v_scale, kQuantGroups, geometry_, logical_capacity_, table,
                               physical_vs);
        }
        k_.copy_from_host(physical_k.data(), physical_k.size() * sizeof(std::int8_t));
        v_.copy_from_host(physical_v.data(), physical_v.size() * sizeof(std::int8_t));
        k_scale_.copy_from_host(physical_ks.data(), physical_ks.size() * sizeof(std::uint16_t));
        v_scale_.copy_from_host(physical_vs.data(), physical_vs.size() * sizeof(std::uint16_t));
    }

    Geometry geometry_;
    DType dtype_;
    std::size_t rows_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::int32_t physical_pages_;
    std::vector<std::int32_t> block_tables_host_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_tables_;
};

int verify_cache(const std::string& label, const HostCache& got, const HostCache& expected) {
    int failures = 0;
    if (expected.dtype == DType::BF16) {
        failures += verify_exact((label + " cache-k").c_str(), got.k_bf16, expected.k_bf16);
        failures += verify_exact((label + " cache-v").c_str(), got.v_bf16, expected.v_bf16);
    } else {
        failures += verify_exact((label + " cache-k-code").c_str(), got.k_i8, expected.k_i8);
        failures += verify_exact((label + " cache-v-code").c_str(), got.v_i8, expected.v_i8);
        failures += verify_exact((label + " cache-k-scale").c_str(), got.k_scale, expected.k_scale);
        failures += verify_exact((label + " cache-v-scale").c_str(), got.v_scale, expected.v_scale);
    }
    return failures;
}

int verify_input(const std::string& label, const GuardedDeviceBuffer& device,
                 const std::vector<std::uint16_t>& expected) {
    int failures = verify_exact(
        label.c_str(), copy_from_guarded<std::uint16_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

int verify_positions(const std::string& label, const GuardedDeviceBuffer& device,
                     const std::vector<std::int32_t>& expected) {
    int failures = verify_exact(label.c_str(),
                                copy_from_guarded<std::int32_t>(device, expected.size()), expected);
    failures += device.verify_guards((label + " guard").c_str());
    return failures;
}

const char* cache_name(DType dtype) {
    if (dtype == DType::BF16) return "bf16";
    return dtype == DType::U8 ? "int4-g64" : "int8-g64";
}

ReductionCriterion attention_criterion(DType dtype) {
    return dtype == DType::BF16 ? kAttentionBf16Criterion : kAttentionInt8Criterion;
}

int verify_attention(const std::string& label, const std::vector<double>& actual,
                     const std::vector<double>& reference, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, reference, criterion);
}

std::string case_label(const char* entry, const Geometry& geometry, DType dtype,
                       const AttentionCase& test_case, MappingPattern mapping) {
    return std::string(entry) + " " + geometry.name + " " + cache_name(dtype) +
           " mapping=" + mapping_name(mapping) + " T=" + std::to_string(test_case.tokens) +
           " keys=" + std::to_string(test_case.base + test_case.tokens) +
           " envelope_max=" + std::to_string(test_case.envelope_max) +
           " seed=" + std::to_string(test_case.seed);
}

void inject_codec_edges(const Geometry& geometry, std::int32_t tokens, std::vector<float>& k,
                        std::vector<float>& v) {
    if (tokens == 0) return;
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        k[kv_input_index(geometry, 0, d, 0)]               = 0.0f;
        v[kv_input_index(geometry, 0, kQuantGroup + d, 0)] = 0.0f;
    }
    k[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = -1.0f;
    v[kv_input_index(geometry, geometry.kv_heads - 1, 0, tokens - 1)] = 1.0f;
}

void inject_q4_codec_edges(const Geometry& geometry, std::vector<float>& k,
                           std::vector<float>& v) {
    // Stored scale 1 makes these exact BF16 values exercise ties-to-even and both signs.
    constexpr float values[] = {7.0f, -7.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f};
    for (std::int32_t d = 0; d < kQuantGroup; ++d) {
        const std::size_t index = kv_input_index(geometry, geometry.kv_heads - 1,
                                                kQuantGroup + d, 0);
        k[index] = d < 8 ? values[d] : 0.0f;
        v[index] = d < 8 ? -values[d] : 0.0f;
        // FP16 scale rounds to zero: encoded cache must be exactly zero.
        k[kv_input_index(geometry, geometry.kv_heads - 1, 2 * kQuantGroup + d, 0)] =
            std::ldexp(1.0f, -30);
    }
}

int verify_q4_cpu_codec() {
    int failures = 0;
    std::vector<std::int8_t> all_bytes(256);
    std::vector<std::int8_t> expected_codes(512);
    for (int byte = 0; byte < 256; ++byte) {
        all_bytes[byte] = static_cast<std::int8_t>(byte);
        const int low = byte % 16;
        const int high = byte / 16;
        expected_codes[2 * byte] = static_cast<std::int8_t>(low < 8 ? low : low - 16);
        expected_codes[2 * byte + 1] = static_cast<std::int8_t>(high < 8 ? high : high - 16);
    }
    failures += verify_exact("Q4 all signed-nibble byte decodes", unpack_q4(all_bytes), expected_codes);
    failures += verify_exact("Q4 all signed-nibble byte encodes", pack_q4(expected_codes), all_bytes);
    std::vector<float> source(64, 0.0f);
    const float ties[] = {7.0f, -7.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f};
    std::copy(std::begin(ties), std::end(ties), source.begin());
    std::vector<std::int8_t> codes(64, 0), expected(64, 0);
    const std::int8_t rounded[] = {7, -7, 0, 0, 2, -2, 2, -2};
    std::copy(std::begin(rounded), std::end(rounded), expected.begin());
    std::vector<std::uint16_t> scales(1);
    encode_group(source, 0, codes, 0, scales, 0, 7);
    failures += verify_exact("Q4 CPU ties-to-even and signed extrema", codes, expected);
    failures += verify_exact("Q4 CPU stored FP16 scale one", scales, std::vector<std::uint16_t>{0x3c00u});
    std::fill(source.begin(), source.end(), std::ldexp(1.0f, -30));
    encode_group(source, 0, codes, 0, scales, 0, 7);
    failures += verify_exact("Q4 CPU scale-underflow codes", codes, std::vector<std::int8_t>(64, 0));
    failures += verify_exact("Q4 CPU scale-underflow bits", scales, std::vector<std::uint16_t>{0});
    std::fill(source.begin(), source.end(), 0.0f);
    encode_group(source, 0, codes, 0, scales, 0, 7);
    failures += verify_exact("Q4 CPU zero group codes", codes, std::vector<std::int8_t>(64, 0));
    failures += verify_exact("Q4 CPU zero group scale", scales, std::vector<std::uint16_t>{0});
    return failures;
}

int run_append_case(const Geometry& geometry, DType dtype, MappingPattern mapping,
                    std::uint32_t seed, std::int32_t tokens = 3, std::int32_t base = 63) {
    const std::int32_t max_context = base + tokens + 4;
    const std::size_t elements =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(geometry.kv_heads) * tokens;
    std::vector<float> k = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);
    inject_codec_edges(geometry, tokens, k, v);
    if (dtype == DType::U8) { inject_q4_codec_edges(geometry, k, v); }
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    DeviceCache cache(initial, mapping);

    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dpositions(positions.size() * sizeof(std::int32_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dpositions.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, tokens});
    Tensor tp(dpositions.data(), DType::I32, {tokens});

    ops::gqa_kv_append(tk, tv, tp, cache.view(), nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_kv_append ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(mapping);
    int failures = verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dpositions, positions);
    failures += cache.verify_guards(label);
    return failures;
}

template <typename Value>
std::vector<Value> select_query_rows(const std::vector<Value>& source, const Geometry& geometry,
                                     const std::vector<std::int32_t>& rows) {
    if (rows.empty()) return source;
    const std::size_t column = static_cast<std::size_t>(geometry.q_heads) * kHeadDim;
    std::vector<Value> selected(rows.size() * column);
    for (std::size_t index = 0; index < rows.size(); ++index) {
        std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(rows[index] * column), column,
                    selected.begin() + static_cast<std::ptrdiff_t>(index * column));
    }
    return selected;
}

int verify_decode_expanded_control(const std::string& label, const Tensor& q,
    const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    PagedKVBatchLayerView cache, ops::GqaExecutionEnvelope envelope,
    const std::vector<std::uint16_t>& native_output) {
    const int width = q.ne[2];
    if (cache.dtype != DType::U8 || q.ne[1] != 24 || width < 4 || width > 6 ||
        (width == 6 && envelope.max_visible_keys > 2054 && envelope.max_visible_keys <= 8198)) {
        return 0;
    }
    GuardedDeviceBuffer control(native_output.size() * sizeof(std::uint16_t));
    Tensor control_out(control.data(), DType::BF16, {q.ne[0], q.ne[1], width, q.ne[3]});
    const std::size_t bytes = ops::gqa_attention_workspace_capacity_bytes(
        q.ne[1], cache.dtype, envelope, q.ne[3], width, width);
    GuardedDeviceBuffer storage(std::max<std::size_t>(bytes, 256));
    WorkspaceArena workspace(DeviceSpan{storage.data(), storage.bytes()});
    auto launch = [&](cudaStream_t stream) {
        gqa_q4_decode_expanded_control(q, positions, valid_columns, table_rows,
            kAttentionScale, cache, envelope, workspace, control_out, stream);
    };
    launch(nullptr);
    cuda_synchronize();
    int failures = verify_exact((label + " expanded/native K BF16 all rows").c_str(),
        copy_from_guarded<std::uint16_t>(control, native_output.size()), native_output);
    if (q.ne[3] == 1 && (width == 4 || width == 5) && envelope.max_visible_keys > 1029) {
        gqa_q4_decode_copy8_control(q, positions, valid_columns, table_rows,
            kAttentionScale, cache, envelope, workspace, control_out, nullptr);
        cuda_synchronize();
        failures += verify_exact((label + " packed8/packed16 BF16 all rows").c_str(),
            copy_from_guarded<std::uint16_t>(control, native_output.size()), native_output);
    }
    failures += verify_graph_replay(label + " expanded K control", launch, control,
                                    native_output.size());
    failures += control.verify_guards((label + " expanded K control output").c_str());
    failures += storage.verify_guards((label + " expanded K control workspace").c_str());
    return failures;
}

PagedKVBatchLayerView decode_control_batch_view(const PagedKVLayerView& cache) {
    return {.k_pages = cache.k_pages, .v_pages = cache.v_pages,
            .k_scale_pages = cache.k_scale_pages, .v_scale_pages = cache.v_scale_pages,
            .block_tables = cache.block_table.view({cache.block_table.ne[0], 1}),
            .head_dim = cache.head_dim, .num_kv_heads = cache.num_kv_heads,
            .dtype = cache.dtype, .quant_group = cache.quant_group};
}

int verify_prompt_expanded_control(const std::string& label, const Tensor& q,
                                    const Tensor& positions, const PagedKVLayerView& cache,
                                    const std::vector<std::uint16_t>& native_output) {
    if (q.ne[1] != 24 || q.ne[2] <= 6 || cache.dtype != DType::U8) return 0;
    GuardedDeviceBuffer control(native_output.size() * sizeof(std::uint16_t));
    Tensor control_out(control.data(), DType::BF16, {q.ne[0], q.ne[1], q.ne[2], q.ne[3]});
    gqa_q4_prompt_expanded_control(q, positions, kAttentionScale, cache, control_out);
    cuda_synchronize();
    int failures = verify_exact((label + " v1.5 expanded/native BF16 all rows").c_str(),
        native_output, copy_from_guarded<std::uint16_t>(control, native_output.size()));
    gqa_q4_prompt_native_control(q, positions, kAttentionScale, cache, control_out);
    cuda_synchronize();
    failures += verify_exact((label + " accepted-native64/pipeline BF16 all rows").c_str(),
        native_output, copy_from_guarded<std::uint16_t>(control, native_output.size()));
    gqa_q4_prompt_pipeline512_control(q, positions, kAttentionScale, cache, control_out);
    cuda_synchronize();
    failures += verify_exact((label + " pipeline512/pipeline384 BF16 all rows").c_str(),
        native_output, copy_from_guarded<std::uint16_t>(control, native_output.size()));
    failures += control.verify_guards((label + " expanded control output").c_str());
    return failures;
}

int run_a1_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    const std::size_t kv_elements = static_cast<std::size_t>(kHeadDim) *
                                    static_cast<std::size_t>(geometry.kv_heads) *
                                    static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k = make_bf16_values(kv_elements, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v = make_bf16_values(kv_elements, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, test_case.tokens, k, v);
    if (dtype == DType::U8) { inject_q4_codec_edges(geometry, k, v); }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache initial = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    HostCache expected      = initial;
    append_cache(expected, k, v, positions);
    const std::vector<double> reference = ideal_attention(q, expected, positions);
    DeviceCache cache(initial, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale, cache.batch_view(),
                       envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = case_label("gqa_attention", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    if (dtype == DType::U8) {
        failures += verify_decode_expanded_control(label, tq, tp, Tensor{}, ttable_row,
            cache.batch_view(), envelope, output_bits);
        failures += verify_prompt_expanded_control(label, tq, tp, cache.view(), output_bits);
        failures += verify_graph_replay(label, [&](cudaStream_t stream) {
            ops::gqa_attention(tq, tk, tv, tp, Tensor{}, ttable_row, kAttentionScale,
                               cache.batch_view(), envelope, workspace, tout, stream);
        }, dout, q_bits.size());
    }
    failures += verify_cache(label, cache.snapshot(), expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += verify_positions(label + " table row unchanged", dtable_row, {table_row});
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    return failures;
}

int run_a3_case(const Geometry& geometry, DType dtype, const AttentionCase& test_case,
                MappingPattern mapping, const std::vector<std::int32_t>& oracle_rows = {}) {
    const std::int32_t total       = test_case.base + test_case.tokens;
    const std::int32_t max_context = static_cast<std::int32_t>(
        std::max<std::uint32_t>(static_cast<std::uint32_t>(total + 3), test_case.envelope_max));
    const std::size_t q_elements = static_cast<std::size_t>(kHeadDim) *
                                   static_cast<std::size_t>(geometry.q_heads) *
                                   static_cast<std::size_t>(test_case.tokens);
    std::vector<float> q = make_bf16_values(q_elements, test_case.seed, -0.25f, 0.25f);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(test_case.tokens));
    for (std::int32_t token = 0; token < test_case.tokens; ++token) {
        positions[static_cast<std::size_t>(token)] = test_case.base + token;
    }

    const HostCache cache_host = make_cache(geometry, dtype, max_context, test_case.seed + 10u);
    // The actual Op still receives every query row. For large-context probes,
    // evaluate the unchanged complete FP64 formula on explicit selected rows,
    // retaining all heads, all dimensions, and every visible causal key.
    std::vector<std::int32_t> oracle_positions;
    if (oracle_rows.empty()) oracle_positions = positions;
    else {
        for (const auto row : oracle_rows) oracle_positions.push_back(positions.at(row));
    }
    const std::vector<double> reference = ideal_attention(
        select_query_rows(q, geometry, oracle_rows), cache_host, oracle_positions);
    DeviceCache cache(cache_host, mapping);

    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    Tensor tp(dp.data(), DType::I32, {test_case.tokens});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(total),
                                             test_case.envelope_max};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, 1, test_case.tokens, test_case.tokens);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope, workspace, tout,
                              nullptr);
    cuda_synchronize();

    const std::string label =
        case_label("gqa_attention_cached", geometry, dtype, test_case, mapping);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label,
                                    bf16_bits_to_double(select_query_rows(output_bits, geometry,
                                                                          oracle_rows)), reference,
                                    attention_criterion(dtype));
    if (dtype == DType::U8) {
        failures += verify_decode_expanded_control(label, tq, tp, Tensor{}, Tensor{},
            decode_control_batch_view(cache.view()), envelope, output_bits);
        failures += verify_prompt_expanded_control(label, tq, tp, cache.view(), output_bits);
        failures += verify_graph_replay(label, [&](cudaStream_t stream) {
            ops::gqa_attention_cached(tq, tp, kAttentionScale, cache.view(), envelope,
                                      workspace, tout, stream);
        }, dout, q_bits.size());
    }
    failures += verify_cache(label + " cache unchanged", cache.snapshot(), cache_host);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    failures += cache.verify_guards(label);
    if (!oracle_rows.empty()) {
        std::cout << "FP64_ORACLE_ROWS width=" << test_case.tokens << " keys=" << total
                  << " heads=" << geometry.q_heads << " dimensions=" << kHeadDim << " rows=";
        for (std::size_t index = 0; index < oracle_rows.size(); ++index) {
            std::cout << (index == 0 ? "" : ",") << oracle_rows[index];
        }
        std::cout << " (all visible causal keys per selected row; "
                     "native/expanded equality checks every actual row)\n";
        if (benchmark_prompt_staging) {
            benchmark_q4_prompt_staging(tq, tp, kAttentionScale, cache.view(), tout);
        }
        if (benchmark_prompt_pipeline) {
            benchmark_q4_prompt_pipeline(tq, tp, kAttentionScale, cache.view(), tout);
        }
    }
    return failures;
}

struct BatchAttentionCase {
    std::int32_t width;
    std::vector<std::int32_t> contexts;
    std::vector<std::int32_t> valid_columns;
    std::vector<std::int32_t> table_rows;
    MappingPattern mapping;
    std::uint32_t seed;
};

std::vector<float> extract_request_columns(const std::vector<float>& source,
                                           std::size_t column_elements, std::int32_t width,
                                           std::int32_t request, std::int32_t valid) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::vector<float> result(static_cast<std::size_t>(valid) * column_elements);
    std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(begin), result.size(), result.begin());
    return result;
}

void insert_request_columns(const std::vector<double>& source, std::size_t column_elements,
                            std::int32_t width, std::int32_t request,
                            std::vector<double>& destination) {
    const std::size_t begin = static_cast<std::size_t>(request) * width * column_elements;
    std::copy(source.begin(), source.end(),
              destination.begin() + static_cast<std::ptrdiff_t>(begin));
}

int verify_invalid_columns_zero(const std::string& label, std::span<const std::uint16_t> output,
                                const Geometry& geometry, std::int32_t width,
                                std::span<const std::int32_t> valid_columns) {
    int failures                      = 0;
    const std::size_t column_elements = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    for (std::size_t batch = 0; batch < valid_columns.size(); ++batch) {
        for (std::int32_t token = valid_columns[batch]; token < width; ++token) {
            const std::size_t begin =
                (batch * static_cast<std::size_t>(width) + token) * column_elements;
            for (std::size_t element = 0; element < column_elements; ++element) {
                if (output[begin + element] != 0) {
                    if (failures == 0) {
                        std::cerr << label << ": invalid output column is not BF16 zero at row "
                                  << batch << " column " << token << '\n';
                    }
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_batch_case(const Geometry& geometry, DType dtype, const BatchAttentionCase& test_case) {
    const std::int32_t batch = static_cast<std::int32_t>(test_case.contexts.size());
    if (batch <= 0 || test_case.valid_columns.size() != static_cast<std::size_t>(batch) ||
        test_case.table_rows.size() != static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("invalid GQA batch test profile");
    }

    std::int32_t maximum_visible = 1;
    for (std::int32_t row = 0; row < batch; ++row) {
        maximum_visible =
            std::max(maximum_visible, test_case.contexts[static_cast<std::size_t>(row)] +
                                          test_case.valid_columns[static_cast<std::size_t>(row)]);
    }
    const std::int32_t max_context       = maximum_visible + 3;
    const std::size_t q_column_elements  = static_cast<std::size_t>(kHeadDim) * geometry.q_heads;
    const std::size_t kv_column_elements = static_cast<std::size_t>(kHeadDim) * geometry.kv_heads;
    const std::size_t columns            = static_cast<std::size_t>(test_case.width) * batch;
    std::vector<float> q =
        make_bf16_values(q_column_elements * columns, test_case.seed, -0.25f, 0.25f);
    std::vector<float> k =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 1u, -0.25f, 0.25f);
    std::vector<float> v =
        make_bf16_values(kv_column_elements * columns, test_case.seed + 2u, -1.0f, 1.0f);
    inject_codec_edges(geometry, static_cast<std::int32_t>(columns), k, v);
    if (dtype == DType::U8) { inject_q4_codec_edges(geometry, k, v); }

    std::vector<std::int32_t> positions(columns, 0);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < valid; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] =
                test_case.contexts[static_cast<std::size_t>(row)] + token;
        }
        const std::int32_t padding_position =
            valid == 0 ? 0 : test_case.contexts[static_cast<std::size_t>(row)] + valid - 1;
        for (std::int32_t token = valid; token < test_case.width; ++token) {
            positions[static_cast<std::size_t>(row) * test_case.width + token] = padding_position;
        }
    }

    std::vector<HostCache> initial;
    initial.reserve(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial.push_back(
            make_cache(geometry, dtype, max_context, test_case.seed + 20u + 3u * row));
    }
    std::vector<HostCache> expected = initial;
    std::vector<double> reference(q_column_elements * columns, 0.0);
    for (std::int32_t request = 0; request < batch; ++request) {
        const std::int32_t valid = test_case.valid_columns[static_cast<std::size_t>(request)];
        if (valid == 0) { continue; }
        const std::int32_t table_row = test_case.table_rows[static_cast<std::size_t>(request)];
        std::vector<std::int32_t> row_positions(static_cast<std::size_t>(valid));
        std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(request * test_case.width),
                    valid, row_positions.begin());
        const std::vector<float> row_q =
            extract_request_columns(q, q_column_elements, test_case.width, request, valid);
        const std::vector<float> row_k =
            extract_request_columns(k, kv_column_elements, test_case.width, request, valid);
        const std::vector<float> row_v =
            extract_request_columns(v, kv_column_elements, test_case.width, request, valid);
        append_cache(expected[static_cast<std::size_t>(table_row)], row_k, row_v, row_positions);
        insert_request_columns(
            ideal_attention(row_q, expected[static_cast<std::size_t>(table_row)], row_positions),
            q_column_elements, test_case.width, request, reference);
    }

    BatchDeviceCache cache(initial, test_case.mapping);
    const std::vector<std::uint16_t> q_bits = to_bf16_bits(q);
    const std::vector<std::uint16_t> k_bits = to_bf16_bits(k);
    const std::vector<std::uint16_t> v_bits = to_bf16_bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(k_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(v_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dvalid(test_case.valid_columns.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dtable_rows(test_case.table_rows.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer dout(q_bits.size() * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data(), q_bits.size() * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data(), k_bits.size() * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data(), v_bits.size() * sizeof(std::uint16_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));
    dvalid.copy_from_host(test_case.valid_columns.data(),
                          test_case.valid_columns.size() * sizeof(std::int32_t));
    dtable_rows.copy_from_host(test_case.table_rows.data(),
                               test_case.table_rows.size() * sizeof(std::int32_t));
    std::vector<std::uint16_t> output_canary(q_bits.size(), kOutputCanary);
    dout.copy_from_host(output_canary.data(), output_canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, geometry.kv_heads, test_case.width, batch});
    Tensor tp(dp.data(), DType::I32, {test_case.width, batch});
    Tensor tvalid(dvalid.data(), DType::I32, {batch});
    Tensor ttable_rows(dtable_rows.data(), DType::I32, {batch});
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, geometry.q_heads, test_case.width, batch});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(maximum_visible),
                                             static_cast<std::uint32_t>(maximum_visible)};
    const std::size_t workspace_bytes = ops::gqa_attention_workspace_capacity_bytes(
        geometry.q_heads, dtype, envelope, batch, test_case.width, test_case.width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    const bool masked = std::any_of(test_case.valid_columns.begin(), test_case.valid_columns.end(),
                                    [&](std::int32_t valid) { return valid != test_case.width; });
    ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows, kAttentionScale,
                       cache.view(), envelope, workspace, tout, nullptr);
    cuda_synchronize();

    const std::string label = std::string("gqa_attention batch ") + geometry.name + " " +
                              cache_name(dtype) + " mapping=" + mapping_name(test_case.mapping) +
                              " B=" + std::to_string(batch) +
                              " W=" + std::to_string(test_case.width) +
                              " seed=" + std::to_string(test_case.seed);
    const std::vector<std::uint16_t> output_bits =
        copy_from_guarded<std::uint16_t>(dout, q_bits.size());
    int failures = verify_attention(label, bf16_bits_to_double(output_bits), reference,
                                    attention_criterion(dtype));
    if (dtype == DType::U8) {
        if (batch == 1 && geometry.q_heads == 24 && test_case.width > 6) {
            GuardedDeviceBuffer control(output_bits.size() * sizeof(std::uint16_t));
            Tensor control_out(control.data(), DType::BF16,
                {kHeadDim, geometry.q_heads, test_case.width, batch});
            gqa_q4_prompt_native_batch_control(tq, tp, tvalid, ttable_rows, masked,
                kAttentionScale, cache.view(), control_out);
            cuda_synchronize();
            failures += verify_exact((label + " accepted-native64/pipeline full BF16").c_str(),
                output_bits, copy_from_guarded<std::uint16_t>(control, output_bits.size()));
            gqa_q4_prompt_pipeline512_batch_control(tq, tp, tvalid, ttable_rows, masked,
                kAttentionScale, cache.view(), control_out);
            cuda_synchronize();
            failures += verify_exact((label + " pipeline512/pipeline384 full BF16").c_str(),
                output_bits, copy_from_guarded<std::uint16_t>(control, output_bits.size()));
            failures += control.verify_guards((label + " native batch control").c_str());
        }
        failures += verify_decode_expanded_control(label, tq, tp, masked ? tvalid : Tensor{},
            ttable_rows, cache.view(), envelope, output_bits);
        failures += verify_graph_replay(label, [&](cudaStream_t stream) {
            ops::gqa_attention(tq, tk, tv, tp, masked ? tvalid : Tensor{}, ttable_rows,
                               kAttentionScale, cache.view(), envelope, workspace, tout, stream);
        }, dout, q_bits.size());
    }
    failures += verify_invalid_columns_zero(label, output_bits, geometry, test_case.width,
                                            test_case.valid_columns);
    failures += cache.verify(label, expected);
    failures += verify_input(label + " q unchanged", dq, q_bits);
    failures += verify_input(label + " k unchanged", dk, k_bits);
    failures += verify_input(label + " v unchanged", dv, v_bits);
    failures += verify_positions(label + " positions unchanged", dp, positions);
    if (masked) {
        failures +=
            verify_positions(label + " valid columns unchanged", dvalid, test_case.valid_columns);
    }
    failures +=
        verify_positions(label + " table rows unchanged", dtable_rows, test_case.table_rows);
    failures += dout.verify_guards((label + " output").c_str());
    failures += workspace_buffer.verify_guards((label + " workspace").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_batch_cases() {
    int failures = 0;
    failures += run_batch_case(kGeometries[0], DType::I8,
                               {6, {127}, {3}, {0}, MappingPattern::Identity, 499u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {16, {49}, {7}, {0}, MappingPattern::Identity, 500u});
    failures += run_batch_case(kGeometries[0], DType::BF16,
                               {1, {63, 2048}, {1, 1}, {1, 0}, MappingPattern::Fragmented, 501u});
    failures += run_batch_case(kGeometries[1], DType::I8,
                               {1,
                                {0, 31, 63, 127, 511, 1023, 2047, 4095},
                                {1, 1, 1, 1, 1, 1, 1, 1},
                                {7, 0, 5, 2, 6, 1, 4, 3},
                                MappingPattern::Identity,
                                502u});
    failures +=
        run_batch_case(kGeometries[0], DType::I8,
                       {6, {61, 127, 511}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, 503u});
    failures += run_batch_case(kGeometries[1], DType::BF16,
                               {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Identity, 504u});
    return failures;
}

int run_geometry(const Geometry& geometry) {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        for (const MappingPattern mapping :
             {MappingPattern::Identity, MappingPattern::Offset, MappingPattern::Fragmented}) {
            failures += run_append_case(geometry, dtype, mapping, 100u + geometry.q_heads);
            failures += run_a1_case(geometry, dtype, {6, 61, 67, 190u}, mapping);
            failures += run_a3_case(geometry, dtype, {1, 128, 129, 191u}, mapping);
        }
        if (dtype == DType::I8) {
            failures += run_append_case(geometry, dtype, MappingPattern::Fragmented,
                                        150u + geometry.q_heads, 129, 61);
        }

        const AttentionCase a1_cases[] = {
            {1, 0, 1, 201u},    {6, 17, 23, 202u},   {7, 17, 512, 203u},
            {17, 31, 48, 204u}, {66, 63, 129, 205u},
        };
        for (const AttentionCase& test_case : a1_cases) {
            failures += run_a1_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        const AttentionCase a3_cases[] = {
            {1, 31, 32, 301u},
            {7, 17, 512, 302u},
            {17, 31, 48, 303u},
        };
        for (const AttentionCase& test_case : a3_cases) {
            failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Identity);
        }

        if (geometry.q_heads == 16) {
            // Loose execution envelopes straddle the two registered host-resource frontiers.
            // Device positions, not these bounds, continue to define the oracle result.
            failures += run_a1_case(geometry, dtype, {7, 17, 513, 401u}, MappingPattern::Identity);
            failures += run_a3_case(geometry, dtype, {7, 17, 513, 402u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1024, 403u}, MappingPattern::Identity);
            failures +=
                run_a3_case(geometry, dtype, {16, 17, 1025, 404u}, MappingPattern::Identity);
        }
    }
    return failures;
}

int run_q4_cases(bool full_context) {
    const Geometry& geometry = kGeometries[0];
    constexpr DType dtype = DType::U8;
    int failures = 0;
    for (MappingPattern mapping : {MappingPattern::Identity, MappingPattern::Offset,
                                    MappingPattern::Fragmented}) {
        failures += run_append_case(geometry, dtype, mapping, 610u, 3, 63);
    }
    failures += run_append_case(geometry, dtype, MappingPattern::Fragmented, 611u, 129, 61);
    for (std::int32_t tokens : {4, 5, 6}) {
        failures += run_a1_case(geometry, dtype, {tokens, 67 - tokens, 67, 620u + tokens},
                                MappingPattern::Fragmented);
        // Loose envelope crosses T4's 1029 launch-resource frontier without altering math.
        failures += run_a3_case(geometry, dtype, {tokens, 1029 - tokens, 1030, 630u + tokens},
                                MappingPattern::Identity);
    }
    failures += run_a1_case(geometry, dtype, {5, 1025, 1030, 641u}, MappingPattern::Fragmented);
    // Bc64 control remains unchanged; the second sample selects affected Bc32.
    failures += run_a3_case(geometry, dtype, {6, 2049, 2055, 642u}, MappingPattern::Identity);
    failures += run_a3_case(geometry, dtype, {6, 8193, 8199, 643u}, MappingPattern::Fragmented);
    failures += run_batch_case(geometry, dtype,
                               {5, {61, 1027, 511}, {5, 2, 0}, {2, 0, 1},
                                MappingPattern::Fragmented, 650u});
    failures += run_batch_case(geometry, dtype,
                               {6, {61, 8196, 1023}, {6, 3, 0}, {2, 0, 1},
                                MappingPattern::Fragmented, 651u});
    if (full_context) {
        // Full logical cache contains independently encoded nonzero K/V. The complete FP64
        // oracle checks all query heads, all output dimensions and every causal key.
        failures += run_a1_case(geometry, dtype, {5, 131067, 131072, 660u},
                                MappingPattern::Fragmented);
        failures += run_a3_case(geometry, dtype, {6, 131066, 131072, 661u},
                                MappingPattern::Identity);
    }
    return failures;
}

int run_q4_pipeline_edge_cases() {
    const Geometry& geometry = kGeometries[0];
    constexpr DType dtype = DType::U8;
    int failures = 0;
    // Single-tile startup/drain, an exactly full physical page, one valid key
    // in the final page, and CTAs with different causal key-block counts.
    const AttentionCase cases[] = {
        {7, 0, 7, 810u}, {7, 57, 64, 811u}, {7, 58, 65, 812u},
        {65, 0, 65, 813u}, {65, 63, 128, 814u}};
    for (const AttentionCase& test_case : cases) {
        failures += run_a3_case(geometry, dtype, test_case, MappingPattern::Fragmented);
    }
    // Entire zero-valid CTAs must return uniformly before either barrier;
    // adjacent valid/zero tiles must never participate in each other's phases.
    failures += run_batch_case(geometry, dtype,
        {65, {127}, {0}, {0}, MappingPattern::Fragmented, 815u});
    failures += run_batch_case(geometry, dtype,
        {65, {64}, {1}, {0}, MappingPattern::Offset, 816u});
    failures += run_batch_case(geometry, dtype,
        {188, {127}, {65}, {0}, MappingPattern::Fragmented, 817u});
    return failures;
}

int run_q4_prompt_cases(bool full_context) {
    const Geometry& geometry = kGeometries[0];
    constexpr DType dtype = DType::U8;
    int failures = 0;
    failures += run_q4_pipeline_edge_cases();
    // Full-query FP64 coverage includes both sides of Br64 and logical-page
    // boundaries, the actual append widths, and the partial final score tile.
    for (const std::int32_t width : {7, 31, 32, 33, 63, 64, 65, 188}) {
        failures += run_a3_case(geometry, dtype, {width, 63, 63u + width, 700u + width},
                                MappingPattern::Fragmented);
    }
    failures += run_a3_case(geometry, dtype, {581, 0, 581, 790u}, MappingPattern::Offset);
    failures += run_a3_case(geometry, dtype, {936, 0, 936, 791u}, MappingPattern::Fragmented);
    // Public append + Prompt uses the batch-metadata<false> production variant.
    failures += run_a1_case(geometry, dtype, {65, 63, 128, 792u}, MappingPattern::Fragmented);
    failures += run_a1_case(geometry, dtype, {188, 61, 249, 793u}, MappingPattern::Offset);
    // B=1 masked metadata selects Prompt, validating complete/partial invalid
    // row tiles, nonzero cache append, and exact zero output padding.
    failures += run_batch_case(geometry, dtype,
        {188, {63}, {137}, {0}, MappingPattern::Fragmented, 794u});
    failures += run_batch_case(geometry, dtype,
        {65, {61}, {33}, {0}, MappingPattern::Offset, 795u});
    if (full_context) {
        auto rows = [](std::int32_t width) {
            return std::vector<std::int32_t>{0, 15, 16, 31, 32, 63, 64, width - 2, width - 1};
        };
        // These widths and cached prefixes reproduce the profiled real Agent
        // suffix geometries. Long-cache tests sample query rows only: K is full.
        failures += run_a3_case(geometry, dtype, {188, 102437, 102625, 796u},
                                MappingPattern::Fragmented, rows(188));
        failures += run_a3_case(geometry, dtype, {581, 102654, 103235, 797u},
                                MappingPattern::Offset, rows(581));
        failures += run_a3_case(geometry, dtype, {936, 103261, 104197, 798u},
                                MappingPattern::Fragmented, rows(936));
        failures += run_a3_case(geometry, dtype, {936, 130136, 131072, 799u},
                                MappingPattern::Fragmented, rows(936));
    }
    return failures;
}

int verify_workspace_capacity_contract() {
    int failures = 0;
    for (const DType dtype : {DType::BF16, DType::I8}) {
        constexpr ops::GqaExecutionEnvelope envelope{1, 1025};
        const std::size_t interval =
            ops::gqa_attention_workspace_capacity_bytes(16, dtype, envelope, 1, 1, 17);
        std::size_t witness = 0;
        for (std::int32_t tokens = 1; tokens <= 17; ++tokens) {
            witness = std::max(witness, ops::gqa_attention_workspace_capacity_bytes(
                                            16, dtype, envelope, 1, tokens, tokens));
        }
        if (interval != witness) {
            std::cerr << "gqa_attention interval capacity has no exact route witness\n";
            ++failures;
        }
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys}, 1, 1, 1);
    } catch (const std::invalid_argument&) {
        std::cerr << "gqa_attention rejected its maximum visible-key envelope\n";
        ++failures;
    }
    try {
        (void)ops::gqa_attention_workspace_capacity_bytes(
            16, DType::BF16, {1, ops::kGqaAttentionMaximumVisibleKeys + 1}, 1, 1, 1);
        std::cerr << "gqa_attention accepted an envelope outside the launcher domain\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    bool q4_only = false;
    bool full_context = true;
    bool codec_only = false;
    bool prompt_only = false;
    bool fragment_only = false;
    bool pipeline_edges_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--q4-only") q4_only = true;
        else if (option == "--q4-short") { q4_only = true; full_context = false; }
        else if (option == "--cpu-codec-only") codec_only = true;
        else if (option == "--q4-prompt-only") { q4_only = true; prompt_only = true; }
        else if (option == "--q4-prompt-short") {
            q4_only = true; prompt_only = true; full_context = false;
        }
        else if (option == "--q4-prompt-bench") {
            q4_only = true; prompt_only = true; benchmark_prompt_staging = true;
        }
        else if (option == "--q4-fragment-only") fragment_only = true;
        else if (option == "--q4-pipeline-edges") pipeline_edges_only = true;
        else if (option == "--q4-pipeline-bench") {
            q4_only = true; prompt_only = true; benchmark_prompt_pipeline = true;
        }
        else {
            std::cerr << "usage: ninfer_gqa_attention_test [--q4-only|--q4-short|"
                         "--q4-prompt-only|--q4-prompt-short|--q4-prompt-bench|"
                         "--q4-fragment-only|--q4-pipeline-edges|--q4-pipeline-bench|"
                         "--cpu-codec-only]\n";
            return 2;
        }
    }
    int failures = verify_q4_cpu_codec();
    if (codec_only) { return failures == 0 ? 0 : 1; }
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    failures += verify_q4_prompt_native_matrix_fragment();
    if (fragment_only) return failures == 0 ? 0 : 1;
    if (pipeline_edges_only) {
        failures += run_q4_pipeline_edge_cases();
        std::cout << (failures == 0 ? "PASS" : "FAIL") << " Q4 Prompt pipeline edges\n";
        return failures == 0 ? 0 : 1;
    }
    if (!q4_only) {
        failures += verify_workspace_capacity_contract();
        for (const Geometry& geometry : kGeometries) { failures += run_geometry(geometry); }
        failures += run_batch_cases();
    }
    if (!prompt_only) failures += run_q4_cases(full_context);
    failures += run_q4_prompt_cases(full_context);
    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " gqa_attention public-contract correctness\n";
    return failures == 0 ? 0 : 1;
}
