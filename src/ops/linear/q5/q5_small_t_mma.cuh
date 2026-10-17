// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Decode the stored little-endian low nibble/high-bit pair without applying
// scale. Every signed Q5 integer is exactly representable in BF16.
__device__ __forceinline__ unsigned q5_small_t_bf16_pair(std::uint8_t low,
                                                        std::uint8_t high, int shift) {
    const int raw0 = (low & 15u) | (((high >> shift) & 1u) << 4);
    const int raw1 = (low >> 4) | (((high >> (shift + 1)) & 1u) << 4);
    const int q0 = (raw0 ^ 16) - 16;
    const int q1 = (raw1 ^ 16) - 16;
    union { __nv_bfloat162 pair; unsigned bits; } decoded;
    decoded.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return decoded.bits;
}

// Four lanes own the eight token columns of one logical row. All 32 lanes of
// the final warp participate; only the quartet's first lane publishes a row.
template <int Tokens, bool Bottom>
__device__ __forceinline__ void q5_small_t_collect_row(float4 value, float (&row)[Tokens]) {
    const int quartet = (static_cast<int>(threadIdx.x) & 31) & ~3;
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float source = Bottom ? ((token & 1) ? value.w : value.z)
                                    : ((token & 1) ? value.y : value.x);
        row[token] = __shfl_sync(0xffffffffu, source, quartet + token / 2);
    }
}

template <int SplitRow>
struct Q5SmallTStridedStore {
    __nv_bfloat16* head;
    int head_ld;
    __nv_bfloat16* tail;
    int tail_ld;

    __device__ __forceinline__ void put(int row, int column, float value) const {
        if (row < SplitRow) {
            head[static_cast<std::int64_t>(column) * head_ld + row] = __float2bfloat16_rn(value);
        } else {
            tail[static_cast<std::int64_t>(column) * tail_ld + row - SplitRow] =
                __float2bfloat16_rn(value);
        }
    }

    template <int Tokens>
    __device__ __forceinline__ void store(int row, int column, float4 value) const {
        if (column < Tokens) {
            put(row, column, value.x);
            put(row + 8, column, value.z);
        }
        if (column + 1 < Tokens) {
            put(row, column + 1, value.y);
            put(row + 8, column + 1, value.w);
        }
    }
};

// Closed canonical T4/T5/T6 route: raw signed-code BF16 MMA, FP32 G64 partials,
// original FP16 scales applied in FP32. No scaled-weight BF16 materialization.
template <int OutputRows, int Tokens, class Epilogue, bool TriggerPdl = false,
          bool JoinPdl = false>
__global__ __launch_bounds__(256, 4)
void q5_small_t_mma_kernel(const __nv_bfloat16* __restrict__ x,
                           const std::uint8_t* __restrict__ codes,
                           const std::uint8_t* __restrict__ high,
                           const std::uint8_t* __restrict__ scales, Epilogue epilogue) {
    static_assert(OutputRows == 7168 || OutputRows == 12288);
    static_assert(Tokens >= 4 && Tokens <= 6);
    constexpr int kHidden = 5120;
    constexpr int kWarps = 8;
    constexpr int kRows = 16;
    constexpr int kOuterK = 512;
    constexpr int kTileK = 64;
    constexpr int kGroupsPerRow = 80;
    constexpr int kCodeStride = 256 + 16;
    constexpr int kHighStride = 64 + 16;
    union SharedStorage {
        struct {
            // Both paddings retain 16-byte copy alignment and separate rows'
            // shared-memory banks. They do not change artifact plane strides.
            std::uint8_t codes[kRows][kCodeStride];
            std::uint8_t high[kRows][kHighStride];
            __nv_bfloat16 activations[kWarps][8 * kTileK];
            std::uint16_t scales[kRows][kWarps];
        } staging;
        float partial[kWarps][32][4];
    };
    static_assert(sizeof(SharedStorage) == 14080);
    __shared__ __align__(16) SharedStorage shared;
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kRows;
    if constexpr (TriggerPdl) {
        if (tid == 0) { pdl::trigger_dependents(); }
    }
    const auto swizzle = [](int row, int column) {
        return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
    };
    const auto stage = [&](int outer_k0) {
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int row = warp * 2 + item;
            if (lane < 16) {
                cp_async<16, Cache::cg>(shared.staging.codes[row] + lane * 16,
                    codes + static_cast<std::int64_t>(row0 + row) * (kHidden / 2) +
                        outer_k0 / 2 + lane * 16);
            }
            if (lane < 4) {
                cp_async<16, Cache::cg>(shared.staging.high[row] + lane * 16,
                    high + static_cast<std::int64_t>(row0 + row) * (kHidden / 8) +
                        outer_k0 / 8 + lane * 16);
            }
        }
        if (tid < kRows) {
            cp_async<16>(shared.staging.scales[tid],
                scales + (static_cast<std::int64_t>(row0 + tid) * kGroupsPerRow +
                          outer_k0 / 64) * 2);
        }
        for (int item = lane; item < Tokens * 8; item += 32) {
            const int column = item / 8;
            const int k8 = item & 7;
            cp_async<16>(&shared.staging.activations[warp][column * kTileK +
                                                          swizzle(column, k8 * 8)],
                x + static_cast<std::int64_t>(column) * kHidden + outer_k0 +
                    warp * kTileK + k8 * 8);
        }
    };
    float acc[4]{};
    stage(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int outer = 0; outer < kHidden / kOuterK; ++outer) {
        float partial[4]{};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int byte = warp * 32 + ks * 8 + lid;
            const int hi_byte = byte >> 2;
            const int shift = 2 * lid;
            const unsigned a0 = q5_small_t_bf16_pair(shared.staging.codes[gid][byte],
                                                     shared.staging.high[gid][hi_byte], shift);
            const unsigned a1 = q5_small_t_bf16_pair(shared.staging.codes[gid + 8][byte],
                                                     shared.staging.high[gid + 8][hi_byte], shift);
            const unsigned a2 = q5_small_t_bf16_pair(shared.staging.codes[gid][byte + 4],
                                                     shared.staging.high[gid][hi_byte + 1], shift);
            const unsigned a3 = q5_small_t_bf16_pair(shared.staging.codes[gid + 8][byte + 4],
                                                     shared.staging.high[gid + 8][hi_byte + 1], shift);
            unsigned b0, b1;
            const int br = lane & 7;
            const int bk = ks * 16 + (((lane >> 3) & 1) << 3);
            ldmatrix_x2(b0, b1, smem_addr(&shared.staging.activations[warp][
                                            br * kTileK + swizzle(br, bk)]));
            mma_bf16(partial[0], partial[1], partial[2], partial[3],
                     a0, a1, a2, a3, b0, b1);
        }
        const float top = __half2float(__ushort_as_half(shared.staging.scales[gid][warp]));
        const float bottom = __half2float(__ushort_as_half(shared.staging.scales[gid + 8][warp]));
        acc[0] = fmaf(partial[0], top, acc[0]);
        acc[1] = fmaf(partial[1], top, acc[1]);
        acc[2] = fmaf(partial[2], bottom, acc[2]);
        acc[3] = fmaf(partial[3], bottom, acc[3]);
        if (outer + 1 < kHidden / kOuterK) {
            __syncthreads();
            stage((outer + 1) * kOuterK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }
    __syncthreads();
    if (warp & 1) {
        store_vec(shared.partial[warp][lane], make_float4(acc[0], acc[1], acc[2], acc[3]));
    }
    __syncthreads();
    if (!(warp & 1)) {
        const float4 partner = load_vec<float4>(shared.partial[warp + 1][lane]);
        acc[0] += partner.x; acc[1] += partner.y;
        acc[2] += partner.z; acc[3] += partner.w;
        if (warp != 0) {
            store_vec(shared.partial[warp][lane], make_float4(acc[0], acc[1], acc[2], acc[3]));
        }
    }
    __syncthreads();
    if (warp == 0) {
        float4 sum = make_float4(acc[0], acc[1], acc[2], acc[3]);
#pragma unroll
        for (int split = 2; split < kWarps; split += 2) {
            const float4 value = load_vec<float4>(shared.partial[split][lane]);
            sum.x += value.x; sum.y += value.y;
            sum.z += value.z; sum.w += value.w;
        }
        epilogue.template store<Tokens>(row0 + gid, 2 * lid, sum);
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

} // namespace ninfer::ops::detail
