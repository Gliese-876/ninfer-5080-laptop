// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstSmallT    = 2;
constexpr int kLastFullT      = 8;
constexpr int kLastOptimizedT = 20;
constexpr int kFirstQwen38T   = 4;
constexpr int kLastQwen38T    = 8;
using FullGeometry            = Q4DraftHeadGeometry<5120>;
using OptimizedGeometry       = Q4DraftHeadGeometry<2048>;
using Qwen38DownGeometry      = Q4LinearGeometry<5120, 17408>;
using Qwen38GdnOutGeometry    = Q4LinearGeometry<5120, 6144>;
using Qwen38HeadGeometry      = Q4LinearGeometry<248320, 5120>;

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, TileTokens, ActiveTokens>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, ((First + static_cast<int>(Offsets) + 7) / 8) * 8,
                      First + static_cast<int>(Offsets)>...};
}

constexpr auto kFullLaunchers = make_launchers<FullGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});
constexpr auto kOptimizedLaunchers = make_launchers<OptimizedGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastOptimizedT - kFirstSmallT + 1>{});

template <class Geometry>
bool matches(const Tensor& x, const Weight& weight) {
    return weight.n == Geometry::kOutputRows && weight.k == Geometry::kInputRows &&
           weight.padded_shape[1] == Geometry::kInputRows && x.ne[1] >= kFirstSmallT;
}

template <class Geometry>
void launch_qwen38_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    if (!matches<Geometry>(x, weight) || x.ne[1] < kFirstQwen38T || x.ne[1] > kLastQwen38T) {
        throw std::invalid_argument("Q4 Qwen3.8 small-T: unsupported exact problem");
    }
    static constexpr auto launchers = make_launchers<Geometry, kFirstQwen38T>(
        std::make_index_sequence<kLastQwen38T - kFirstQwen38T + 1>{});
    launchers[static_cast<std::size_t>(x.ne[1] - kFirstQwen38T)](x, weight, out, stream);
}

template <class Geometry, std::size_t... Offsets>
void prewarm_qwen38_small_t(std::index_sequence<Offsets...>) {
    cudaFuncAttributes attr{};
    (CUDA_CHECK(cudaFuncGetAttributes(
         &attr, q4_small_t_mma_kernel<Geometry, 8, kFirstQwen38T + static_cast<int>(Offsets)>)),
     ...);
}

} // namespace

void q4_small_t_mma_prewarm() {
    cudaFuncAttributes attr{};

    // Force CUDA residency for the Qwen3.8 Q4 small-T module before
    // explicit KV/runtime capacity is resolved. Include every T=4..8
    // specialization reached by target verification / speculative decode.
    CUDA_CHECK(cudaFuncGetAttributes(&attr, q4_small_t_mma_kernel<FullGeometry, 8, 4>));

    CUDA_CHECK(cudaFuncGetAttributes(&attr, q4_small_t_mma_kernel<OptimizedGeometry, 8, 4>));

    using Qwen38Widths = std::make_index_sequence<kLastQwen38T - kFirstQwen38T + 1>;
    prewarm_qwen38_small_t<Qwen38DownGeometry>(Qwen38Widths{});
    prewarm_qwen38_small_t<Qwen38GdnOutGeometry>(Qwen38Widths{});
    prewarm_qwen38_small_t<Qwen38HeadGeometry>(Qwen38Widths{});
}

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    if (matches<FullGeometry>(x, weight) && x.ne[1] <= kLastFullT) {
        kFullLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    if (matches<OptimizedGeometry>(x, weight) && x.ne[1] <= kLastOptimizedT) {
        kOptimizedLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out,
                                                                              stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}


void launch_q4_qwen38_down_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    launch_qwen38_small_t<Qwen38DownGeometry>(x, weight, out, stream);
}

void launch_q4_qwen38_gdn_out_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    launch_qwen38_small_t<Qwen38GdnOutGeometry>(x, weight, out, stream);
}

void launch_q4_qwen38_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream) {
    launch_qwen38_small_t<Qwen38HeadGeometry>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
