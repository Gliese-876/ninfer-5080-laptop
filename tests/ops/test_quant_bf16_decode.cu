// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear_swiglu/q3/q3_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct alignas(16) Q3Case {
    std::uint8_t bytes[24];
    int pair_index;
};

struct DecodeResult {
    unsigned actual;
    unsigned reference;
};

union PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

// Retain the previous arithmetic route as a second, supplementary comparison.
__device__ unsigned reference_q3(const std::uint8_t* codes, int pair_index) {
    const int bit = pair_index * 6;
    const int word_index = bit >> 5;
    const auto* words = reinterpret_cast<const std::uint32_t*>(codes);
    const unsigned packed = __funnelshift_r(words[word_index],
        word_index < 5 ? words[word_index + 1] : 0u,
        static_cast<unsigned>(bit & 31)) & 0x3fu;
    const int q0 = (static_cast<int>(packed & 7u) ^ 4) - 4;
    const int q1 = (static_cast<int>((packed >> 3) & 7u) ^ 4) - 4;
    PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

__device__ unsigned reference_q4(std::uint8_t packed) {
    const int q0 = (static_cast<int>(packed & 15u) ^ 8) - 8;
    const int q1 = (static_cast<int>(packed >> 4) ^ 8) - 8;
    PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

__global__ void decode_q3(const Q3Case* cases, DecodeResult* results, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        const Q3Case& c = cases[index];
        results[index] = {ninfer::ops::detail::q3_small_t_bf16_pair(c.bytes, c.pair_index),
                          reference_q3(c.bytes, c.pair_index)};
    }
}

__global__ void decode_q4(DecodeResult* results) {
    const int index = static_cast<int>(threadIdx.x);
    const auto packed = static_cast<std::uint8_t>(index);
    results[index] = {ninfer::ops::detail::q4_small_t_bf16_pair(packed), reference_q4(packed)};
}

// Independent exact oracle: interpret a two's-complement code, then construct
// the IEEE BF16 sign/exponent/significand of the represented integer.
std::uint16_t integer_bf16_bits(int value) {
    if (value == 0) return 0;
    const unsigned sign = value < 0 ? 0x8000u : 0u;
    const unsigned magnitude = static_cast<unsigned>(value < 0 ? -value : value);
    unsigned exponent = 0;
    while ((1u << (exponent + 1)) <= magnitude) ++exponent;
    return static_cast<std::uint16_t>(sign | ((127u + exponent) << 7) |
        ((magnitude - (1u << exponent)) << (7u - exponent)));
}

unsigned pair_oracle(unsigned packed, unsigned width) {
    const unsigned mask = (1u << width) - 1u;
    const int sign = 1 << (width - 1);
    const auto decode = [=](unsigned code) {
        return static_cast<int>(code) >= sign
            ? static_cast<int>(code) - (1 << width) : static_cast<int>(code);
    };
    return static_cast<unsigned>(integer_bf16_bits(decode(packed & mask))) |
        (static_cast<unsigned>(integer_bf16_bits(decode((packed >> width) & mask))) << 16);
}

unsigned extract_q3_oracle(const Q3Case& c) {
    unsigned packed = 0;
    for (unsigned offset = 0; offset < 6; ++offset) {
        const unsigned bit = static_cast<unsigned>(c.pair_index) * 6 + offset;
        packed |= ((static_cast<unsigned>(c.bytes[bit / 8]) >> (bit % 8)) & 1u) << offset;
    }
    return packed;
}

void check_host_formula() {
    for (unsigned width : {3u, 4u}) {
        const unsigned sign = 1u << (width - 1);
        for (unsigned packed = 0; packed < (1u << (2 * width)); ++packed) {
            unsigned bits = 0;
            for (unsigned lane = 0; lane < 2; ++lane) {
                const unsigned raw = (packed >> (lane * width)) & ((1u << width) - 1u);
                const auto biased_bits = static_cast<std::uint16_t>(0x4300u | (raw ^ sign));
                const float biased = std::bit_cast<float>(static_cast<unsigned>(biased_bits) << 16);
                const float decoded = biased - static_cast<float>(128u + sign);
                const unsigned fp32 = std::bit_cast<unsigned>(decoded);
                if ((fp32 & 0xffffu) != 0) throw std::runtime_error("nonexact host decode");
                bits |= (fp32 >> 16) << (16 * lane);
            }
            if (bits != pair_oracle(packed, width))
                throw std::runtime_error("host formula disagrees with signed-code oracle");
        }
    }
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

struct DeviceBuffer {
    void* data = nullptr;
    explicit DeviceBuffer(std::size_t size) { cuda_check(cudaMalloc(&data, size), "cudaMalloc"); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void check_result(const DecodeResult& result, unsigned expected, const std::string& label) {
    if (result.actual != expected || result.reference != expected) {
        throw std::runtime_error(label + " BF16 bit mismatch: actual=" +
            std::to_string(result.actual) + " reference=" + std::to_string(result.reference) +
            " oracle=" + std::to_string(expected));
    }
}

void check_gpu() {
    std::vector<Q3Case> cases;
    // Exhaust every six-bit pair at every G64 position. Different backgrounds
    // expose accidental dependence on adjacent codes across 32-bit boundaries.
    for (unsigned background = 0; background < 3; ++background) {
        for (unsigned position = 0; position < 32; ++position) {
            for (unsigned packed = 0; packed < 64; ++packed) {
                Q3Case c{};
                c.pair_index = static_cast<int>(position);
                for (unsigned byte = 0; byte < 24; ++byte) {
                    c.bytes[byte] = background == 0 ? 0 : background == 1 ? 255 :
                        static_cast<std::uint8_t>(byte * 73u + packed * 29u + position * 17u);
                }
                for (unsigned offset = 0; offset < 6; ++offset) {
                    const unsigned bit = position * 6 + offset;
                    const unsigned mask = 1u << (bit % 8);
                    c.bytes[bit / 8] = static_cast<std::uint8_t>(
                        (c.bytes[bit / 8] & ~mask) | (((packed >> offset) & 1u) << (bit % 8)));
                }
                cases.push_back(c);
            }
        }
    }
    std::vector<DecodeResult> q3_results(cases.size());
    std::vector<DecodeResult> q4_results(256);
    DeviceBuffer device_cases(cases.size() * sizeof(Q3Case));
    DeviceBuffer device_q3(q3_results.size() * sizeof(DecodeResult));
    DeviceBuffer device_q4(q4_results.size() * sizeof(DecodeResult));
    cuda_check(cudaMemcpy(device_cases.data, cases.data(), cases.size() * sizeof(Q3Case),
                          cudaMemcpyHostToDevice), "copy Q3 cases");
    decode_q3<<<static_cast<unsigned>((cases.size() + 255) / 256), 256>>>(
        static_cast<const Q3Case*>(device_cases.data),
        static_cast<DecodeResult*>(device_q3.data), static_cast<int>(cases.size()));
    cuda_check(cudaGetLastError(), "decode_q3 launch");
    decode_q4<<<1, 256>>>(static_cast<DecodeResult*>(device_q4.data));
    cuda_check(cudaGetLastError(), "decode_q4 launch");
    cuda_check(cudaMemcpy(q3_results.data(), device_q3.data, q3_results.size() * sizeof(DecodeResult),
                          cudaMemcpyDeviceToHost), "read Q3 decode");
    cuda_check(cudaMemcpy(q4_results.data(), device_q4.data, q4_results.size() * sizeof(DecodeResult),
                          cudaMemcpyDeviceToHost), "read Q4 decode");
    for (std::size_t index = 0; index < cases.size(); ++index)
        check_result(q3_results[index], pair_oracle(extract_q3_oracle(cases[index]), 3),
                     "Q3 case " + std::to_string(index));
    for (unsigned packed = 0; packed < 256; ++packed)
        check_result(q4_results[packed], pair_oracle(packed, 4), "Q4 byte " + std::to_string(packed));
    std::cout << "PASS exact BF16: " << cases.size()
              << " Q3 cases (64 pairs x 32 positions x 3 backgrounds), 256 Q4 bytes; "
              << "positive zero, cross-word and final-word extraction included\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        check_host_formula();
        if (argc == 2 && std::string(argv[1]) == "--host-only") {
            std::cout << "PASS host exact signed-code oracle: 64 Q3 pairs, 256 Q4 pairs\n";
            return 0;
        }
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && count == 0)) {
            std::cout << "SKIP: CUDA device unavailable\n";
            return 77;
        }
        cuda_check(status, "cudaGetDeviceCount");
        check_gpu();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
