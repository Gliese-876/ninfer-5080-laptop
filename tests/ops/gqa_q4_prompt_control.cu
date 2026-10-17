// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/gqa_q4_prompt_control.h"

#include "core/device.h"
#include "ops/kernel/gqa_attention_prefill_q4.cuh"
#include "ops/kernel/gqa_attention_prefill_q4_pipeline.cuh"
#include "ops/launcher/gqa_attention.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace ninfer::test {
namespace {

using Geometry = ops::Gqa27Geometry;
using Metadata = ops::GqaPrefillDirectMetadata;

template <bool Packed>
void launch_staging(const Tensor& q, const Tensor& positions, float scale,
                    const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    constexpr int smem = Packed ? ops::kGqaPrefillQ4SmemBytes : ops::kGqaPrefillI8SmemBytes;
    static const cudaError_t attribute = cudaFuncSetAttribute(
        ops::gqa_attention_prefill_q4_kernel<Geometry, Metadata, Packed>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    CUDA_CHECK(attribute);
    const int width = q.ne[2];
    const dim3 grid((width + ops::kGqaPrefillI8Br - 1) / ops::kGqaPrefillI8Br,
                    Geometry::QHeads);
    const Metadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    ops::gqa_attention_prefill_q4_kernel<Geometry, Metadata, Packed>
        <<<grid, ops::kGqaPrefillI8Threads, smem, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), width);
    CUDA_CHECK(cudaGetLastError());
}

void launch_pipeline512(const Tensor& q, const Tensor& positions, float scale,
                      const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    static const cudaError_t attribute = cudaFuncSetAttribute(
        ops::gqa_attention_prefill_q4_pipeline_kernel<Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, ops::kGqaPrefillQ4SmemBytes);
    CUDA_CHECK(attribute);
    const int width = q.ne[2];
    const dim3 grid((width + ops::kGqaPrefillI8Br - 1) / ops::kGqaPrefillI8Br,
                    Geometry::QHeads);
    const Metadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    ops::gqa_attention_prefill_q4_pipeline_kernel<Metadata>
        <<<grid, ops::kGqaPrefillI8Threads, ops::kGqaPrefillQ4SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache.k_pages.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), width);
    CUDA_CHECK(cudaGetLastError());
}

void launch_pipeline384(const Tensor& q, const Tensor& positions, float scale,
                       const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    // Launch the exact production leaf rather than emitting a duplicate CUDA
    // template stub from this non-RDC test translation unit.
    ops::detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

// Each CTA covers 512 consecutive 16-bit patterns. Across 128 CTAs every
// possible four-code packed word is expanded, including the signed -8 code.
// All eight dimension slices, eight key rows and both matrix registers use
// the exact packed shared layout and helper consumed by production QK.
__global__ void native_fragment_patterns(unsigned* output) {
    __shared__ __align__(16) std::uint8_t packed[8 * 128];
    const int lane = threadIdx.x;
    for (int pair = lane; pair < 512; pair += 32) {
        const int row = pair / 64;
        const int byte = (pair % 64) * 2;
        const auto pattern = static_cast<std::uint16_t>(blockIdx.x * 512 + pair);
        *reinterpret_cast<std::uint16_t*>(
            packed + row * 128 + ops::gqa_prefill_q4_k_swz(row, byte)) = pattern;
    }
    __syncthreads();
    const int gid = lane >> 2;
    const int lid = lane & 3;
    for (int slice = 0; slice < 8; ++slice) {
        const int address_row = lane & 7;
        const int address_byte = slice * 16 + ((lane >> 3) & 1) * 8;
        unsigned b0, b1;
        ops::gqa_prefill_q4_ldmatrix_s8(
            b0, b1, ops::smem_addr(packed + address_row * 128 +
                ops::gqa_prefill_q4_k_swz(address_row, address_byte)));
        const int pair = gid * 64 + slice * 8 + lid;
        output[blockIdx.x * 512 + pair] = b0;
        output[blockIdx.x * 512 + pair + 4] = b1;
    }
}

unsigned naive_unpack(std::uint16_t pattern) {
    unsigned word = 0;
    for (int code = 0; code < 4; ++code) {
        int value = (pattern >> (4 * code)) & 15;
        if (value >= 8) value -= 16;
        word |= static_cast<unsigned>(static_cast<std::uint8_t>(value)) << (8 * code);
    }
    return word;
}

} // namespace

void gqa_q4_prompt_expanded_control(const Tensor& q, const Tensor& positions, float scale,
                                    const PagedKVLayerView& cache, Tensor& out,
                                    cudaStream_t stream) {
    launch_staging<false>(q, positions, scale, cache, out, stream);
}

void gqa_q4_prompt_native_control(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache, Tensor& out,
                                  cudaStream_t stream) {
    launch_staging<true>(q, positions, scale, cache, out, stream);
}

void gqa_q4_prompt_native_batch_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, bool masked, float scale,
    const PagedKVBatchLayerView& cache, Tensor& out, cudaStream_t stream) {
    auto launch = [&]<bool Masked>() {
        using BatchMetadata = ops::GqaPrefillBatchMetadata<Masked>;
        static const cudaError_t attribute = cudaFuncSetAttribute(
            ops::gqa_attention_prefill_q4_kernel<Geometry, BatchMetadata>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ops::kGqaPrefillQ4SmemBytes);
        CUDA_CHECK(attribute);
        const int width = q.ne[2];
        const dim3 grid((width + ops::kGqaPrefillI8Br - 1) / ops::kGqaPrefillI8Br,
                        Geometry::QHeads);
        const BatchMetadata metadata{
            static_cast<const std::int32_t*>(cache.block_tables.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0]};
        ops::gqa_attention_prefill_q4_kernel<Geometry, BatchMetadata>
            <<<grid, ops::kGqaPrefillI8Threads, ops::kGqaPrefillQ4SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache.k_pages.data),
                static_cast<const std::uint8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), width);
        CUDA_CHECK(cudaGetLastError());
    };
    if (masked) launch.template operator()<true>();
    else launch.template operator()<false>();
}

void gqa_q4_prompt_pipeline512_control(const Tensor& q, const Tensor& positions, float scale,
    const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    launch_pipeline512(q, positions, scale, cache, out, stream);
}

void gqa_q4_prompt_pipeline512_batch_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, bool masked, float scale,
    const PagedKVBatchLayerView& cache, Tensor& out, cudaStream_t stream) {
    auto launch = [&]<bool Masked>() {
        using BatchMetadata = ops::GqaPrefillBatchMetadata<Masked>;
        static const cudaError_t attribute = cudaFuncSetAttribute(
            ops::gqa_attention_prefill_q4_pipeline_kernel<BatchMetadata>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, ops::kGqaPrefillQ4SmemBytes);
        CUDA_CHECK(attribute);
        const int width = q.ne[2];
        const dim3 grid((width + ops::kGqaPrefillI8Br - 1) / ops::kGqaPrefillI8Br,
                        Geometry::QHeads);
        const BatchMetadata metadata{
            static_cast<const std::int32_t*>(cache.block_tables.data),
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(table_rows.data), cache.block_tables.ne[0]};
        ops::gqa_attention_prefill_q4_pipeline_kernel<BatchMetadata>
            <<<grid, ops::kGqaPrefillI8Threads, ops::kGqaPrefillQ4SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache.k_pages.data),
                static_cast<const std::uint8_t*>(cache.v_pages.data),
                static_cast<const __half*>(cache.k_scale_pages.data),
                static_cast<const __half*>(cache.v_scale_pages.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), width);
        CUDA_CHECK(cudaGetLastError());
    };
    if (masked) launch.template operator()<true>();
    else launch.template operator()<false>();
}

int verify_q4_prompt_native_matrix_fragment() {
    unsigned* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, 65536 * sizeof(unsigned)));
    native_fragment_patterns<<<128, 32>>>(device);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned> actual(65536);
    CUDA_CHECK(cudaMemcpy(actual.data(), device, actual.size() * sizeof(unsigned),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device));
    for (unsigned pattern = 0; pattern < 65536; ++pattern) {
        if (actual[pattern] != naive_unpack(static_cast<std::uint16_t>(pattern))) {
            std::cerr << "FAIL native Q4 matrix fragment pattern=" << pattern << '\n';
            return 1;
        }
    }
    std::cout << "PASS native Q4 matrix fragment: all 65536 signed packed words, "
                 "all key rows/dimension slices/register positions\n";
    return 0;
}

void benchmark_q4_prompt_staging(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache, Tensor& out) {
    constexpr int warmup = 2;
    constexpr int repeats = 20;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (bool packed : std::array<bool, 4>{false, true, true, false}) {
        auto launch = [&] {
            if (packed) launch_staging<true>(q, positions, scale, cache, out, nullptr);
            else launch_staging<false>(q, positions, scale, cache, out, nullptr);
        };
        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        std::cout << "Q4_PROMPT_STAGING_BENCH width=" << q.ne[2]
                  << " policy=" << (packed ? "native-packed-async" : "expanded-v1.5")
                  << " repeats=" << repeats << " mean_ms=" << elapsed / repeats << '\n';
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_q4_prompt_pipeline(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache, Tensor& out) {
    constexpr int warmup = 2;
    constexpr int repeats = 20;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (bool pipeline : std::array<bool, 4>{false, true, true, false}) {
        auto launch = [&] {
            if (pipeline) launch_pipeline384(q, positions, scale, cache, out, nullptr);
            else launch_pipeline512(q, positions, scale, cache, out, nullptr);
        };
        for (int i = 0; i < warmup; ++i) launch();
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        std::cout << "Q4_PROMPT_PIPELINE_BENCH width=" << q.ne[2]
                  << " policy=" << (pipeline ? "warp-pipeline384-cap168" : "warp-pipeline512")
                  << " repeats=" << repeats << " mean_ms=" << elapsed / repeats << '\n';
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

} // namespace ninfer::test
