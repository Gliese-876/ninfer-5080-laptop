// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows   = 131072;
constexpr int kHidden = 5120;
using Geometry        = Q4LinearGeometry<kRows, kHidden>;

// Output rows retain shortlist order. Exact integer BF16 MMA uses the original
// FP16 group scales in FP32, with no scaled-weight BF16 rounding.
struct IndexedRows {
    static constexpr int kOutputRowsPerCta = 16;
    const std::int32_t* ids;

    __device__ __forceinline__ int weight_row(int output_row) const { return ids[output_row]; }

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return ids[output_row0 + local_row];
    }
};

using GatherLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, cudaStream_t);

void launch_one(const Tensor& x, const Weight& source, const Tensor& row_ids, Tensor& out,
                cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule;
    constexpr int kBlocks = kRows / Schedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, 8, 1, Q4SmallTMmaStoreEpilogue, IndexedRows>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(source.qdata),
        static_cast<const std::uint8_t*>(source.scales), static_cast<__nv_bfloat16*>(out.data),
        Q4SmallTMmaStoreEpilogue{},
        {static_cast<const std::int32_t*>(row_ids.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <int Columns>
void launch_small(const Tensor& x, const Weight& source, const Tensor& row_ids, Tensor& out,
                  cudaStream_t stream) {
    static_assert(Columns >= 2 && Columns <= 8);
    using Schedule        = Q4DraftSmallTSchedule;
    constexpr int kBlocks = kRows / Schedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, 8, Columns, Q4SmallTMmaStoreEpilogue, IndexedRows>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(source.qdata),
            static_cast<const std::uint8_t*>(source.scales), static_cast<__nv_bfloat16*>(out.data),
            Q4SmallTMmaStoreEpilogue{}, {static_cast<const std::int32_t*>(row_ids.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<GatherLaunch, sizeof...(Offsets)>{
        &launch_small<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kSmallLaunchers = make_launchers(std::make_index_sequence<7>{});

template <int Columns>
void prewarm_small() {
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes,
        q4_small_t_mma_kernel<Geometry, 8, Columns, Q4SmallTMmaStoreEpilogue, IndexedRows>));
}

template <std::size_t... Offsets>
void prewarm_small_all(std::index_sequence<Offsets...>) {
    (prewarm_small<2 + static_cast<int>(Offsets)>(), ...);
}

} // namespace

void q4_gather_rows_prewarm() {
    prewarm_small<1>();
    prewarm_small_all(std::make_index_sequence<7>{});
}

void launch_q4_gather_rows(const Tensor& x, const Weight& source, const Tensor& row_ids,
                           Tensor& out, cudaStream_t stream) {
    // MTP reaches T=1..8. Larger legal column extents use the same arithmetic
    // in independent bounded column groups, without a device allocation.
    for (std::int32_t first = 0; first < x.ne[1];) {
        const std::int32_t columns = std::min<std::int32_t>(8, x.ne[1] - first);
        Tensor input               = x.slice(1, first, columns);
        Tensor output              = out.slice(1, first, columns);
        if (columns == 1) {
            launch_one(input, source, row_ids, output, stream);
        } else {
            kSmallLaunchers[static_cast<std::size_t>(columns - 2)](input, source, row_ids, output,
                                                                   stream);
        }
        first += columns;
    }
}

} // namespace ninfer::ops::detail
