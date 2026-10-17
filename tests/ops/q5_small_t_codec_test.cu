// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/linear/q5/q5_small_t_mma.cuh"
#include "ops/op_tester.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr int kCases = 4 * 256 * 256;
__global__ void decode_all_pairs(std::uint32_t* output) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < kCases) {
        output[index] = ninfer::ops::detail::q5_small_t_bf16_pair(
            static_cast<std::uint8_t>(index), static_cast<std::uint8_t>(index >> 8),
            2 * (index >> 16));
    }
}
}

int verify_exact_q5_small_t_codec() {
    using namespace ninfer::test;
    GuardedDeviceBuffer device(static_cast<std::size_t>(kCases) * sizeof(std::uint32_t));
    decode_all_pairs<<<kCases / 256, 256>>>(static_cast<std::uint32_t*>(device.data()));
    cuda_check(cudaGetLastError(), "Q5 exhaustive codec launch");
    cuda_check(cudaDeviceSynchronize(), "Q5 exhaustive codec synchronize");
    std::vector<std::uint32_t> actual(kCases), expected(kCases);
    device.copy_to_host(actual.data(), actual.size() * sizeof(std::uint32_t));
    for (int index = 0; index < kCases; ++index) {
        const int low = index & 255;
        const int high = (index >> 8) & 255;
        const int position = 2 * (index >> 16);
        const int unsigned0 = (low & 15) + 16 * ((high / (1 << position)) % 2);
        const int unsigned1 = (low >> 4) + 16 * ((high / (1 << (position + 1))) % 2);
        const int signed0 = unsigned0 < 16 ? unsigned0 : unsigned0 - 32;
        const int signed1 = unsigned1 < 16 ? unsigned1 : unsigned1 - 32;
        expected[index] = f32_to_bf16(static_cast<float>(signed0)) |
            (static_cast<std::uint32_t>(f32_to_bf16(static_cast<float>(signed1))) << 16);
    }
    const int failures = device.verify_guards("Q5 exhaustive code guard") +
        verify_exact("Q5 exhaustive low/high/bit-position BF16 pairs", actual, expected);
    std::cout << (failures ? "FAIL" : "OK") << " Q5 exhaustive 262144 packed pairs\n";
    return failures;
}
