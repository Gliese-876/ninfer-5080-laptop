// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/gdn_input_proj/gdn_projected_conv.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <int Hidden, int QueryRows, int KeyRows, int ValueRows, int ZRows, int FullSlabs>
struct GdnSnapshotGeometry {
    static constexpr int kHidden      = Hidden;
    static constexpr int kQueryRows   = QueryRows;
    static constexpr int kKeyRows     = KeyRows;
    static constexpr int kValueRows   = ValueRows;
    static constexpr int kZRows       = ZRows;
    static constexpr int kValueZRows  = ValueRows + ZRows;
    static constexpr int kQkRows      = QueryRows + KeyRows;
    static constexpr int kChannels    = QueryRows + KeyRows + ValueRows;
    static constexpr int kValueOffset = QueryRows + KeyRows;
    static constexpr int kFullSlabs   = FullSlabs;
};

using GdnSnapshotGeometry27 = GdnSnapshotGeometry<5120, 2048, 2048, 6144, 6144, 5>;
using GdnSnapshotGeometry9  = GdnSnapshotGeometry<4096, 2048, 2048, 4096, 4096, 4>;

using Q4ScheduleC4 = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4ScheduleC8 = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

enum class PdlOrder {
    Q4ThenQ5,
    Q5ThenQ4,
};

template <class Geometry, class Publish>
GdnConvEpilogue<Publish> make_epilogue(const Tensor& conv_weight, const Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& query, Tensor& key, Tensor& value,
                                       int global_row_offset, Publish publish) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        Geometry::kChannels,
        Geometry::kQueryRows,
        Geometry::kKeyRows,
        Geometry::kValueRows,
        global_row_offset,
        static_cast<std::int32_t>(query.ne[1]),
        0,
        publish,
    };
}

template <class Geometry, class Publish>
struct Q4GdnDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        const float projected[1]{value};
        conv.store(row, projected);
    }
};

template <class Geometry, int Tokens, class Publish>
struct Q4GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
        if (active_cols == Tokens) { conv.store(row, projected); }
    }
};

template <class Geometry, class Publish>
struct Q5GdnDecodeEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        if (row < Geometry::kValueRows) {
            const float projected[1]{value};
            conv.store(row, projected);
        } else {
            z[row - Geometry::kValueRows] = __float2bfloat16_rn(value);
        }
    }
};

template <class Geometry, int Tokens, class Publish>
struct Q5GdnSmallTEpilogue {
    GdnConvEpilogue<Publish> conv;
    __nv_bfloat16* z;

    template <bool, int, int ProducedTokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                               std::int32_t, std::int32_t row,
                                               const float (&values)[ProducedTokens]) const {
        static_assert(ProducedTokens == Tokens);
        if (row < Geometry::kValueRows) {
            conv.store(row, values);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * Geometry::kZRows + row - Geometry::kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <class Geometry, int Tokens, class Publish>
struct Q5GdnMmaEpilogue {
    Q5GdnSmallTEpilogue<Geometry, Tokens, Publish> rows;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, int column, float4 value) const {
        static_assert(ActiveCols == Tokens && (Tokens == 5 || Tokens == 6));
        static_assert(Geometry::kValueRows % 16 == 0);
        if (row < Geometry::kValueRows) {
            // Keep the full FP32 projected row through convolution/history.
            // Every lane joins each shuffle; only one owner publishes each row.
            float projected[Tokens];
            q5_small_t_collect_row<Tokens, false>(value, projected);
            if ((threadIdx.x & 3) == 0) {
                rows.template operator()<true, Geometry::kValueRows>(
                    nullptr, nullptr, Geometry::kValueZRows, Geometry::kValueRows, row, projected);
            }
            q5_small_t_collect_row<Tokens, true>(value, projected);
            if ((threadIdx.x & 3) == 0) {
                rows.template operator()<true, Geometry::kValueRows>(
                    nullptr, nullptr, Geometry::kValueZRows, Geometry::kValueRows, row + 8,
                    projected);
            }
        } else {
            const Q5SmallTStridedStore<Geometry::kValueRows> z_store{
                nullptr, 0, rows.z, Geometry::kZRows};
            z_store.template store<Tokens>(row, column, value);
        }
    }
};

template <class Geometry, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_t1(const Tensor& x, const Weight& qk_weight,
                  const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query, cudaStream_t stream) {
    using Schedule = std::conditional_t<Geometry::kHidden == 4096,
                                        Q4GemvR1W8DirectK64Schedule, Q4GemvR1W8DirectSchedule>;
    constexpr int q4_threads = Schedule::kThreads;
    constexpr int q4_blocks  = Geometry::kQkRows / Schedule::kRowsPerCta;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q4_blocks), dim3(q4_threads), 0, stream},
            q4_rowsplit_gemv_kernel<Schedule, false, 0,
                                    Q4GdnDecodeEpilogue<Geometry, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQkRows, Geometry::kHidden,
            Q4GdnDecodeEpilogue<Geometry, Publish>{qk_epilogue}, Q4GemvIdentityRows{}));
    } else {
        q4_rowsplit_gemv_kernel<Schedule, false, 0, Q4GdnDecodeEpilogue<Geometry, Publish>,
                                TriggerPdl, JoinPdl><<<q4_blocks, q4_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQkRows, Geometry::kHidden,
            Q4GdnDecodeEpilogue<Geometry, Publish>{qk_epilogue}, Q4GemvIdentityRows{});
    }
}

template <class Geometry, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_t1(const Tensor& x, const Weight& value_z_weight,
                  const GdnConvEpilogue<Publish>& value_epilogue, Tensor& value, Tensor& z,
                  cudaStream_t stream) {
    constexpr int q5_rows_per_block = 16;
    constexpr int q5_threads        = q5_rows_per_block * 32;
    constexpr int q5_blocks         = Geometry::kValueZRows / q5_rows_per_block;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q5_blocks), dim3(q5_threads), 0, stream},
            q5_rowsplit_gemv_kernel<Geometry::kValueZRows, Geometry::kHidden, q5_rows_per_block, 2, true, false, true,
                                    Geometry::kValueRows, Q5GdnDecodeEpilogue<Geometry, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnDecodeEpilogue<Geometry, Publish>{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q5_rowsplit_gemv_kernel<Geometry::kValueZRows, Geometry::kHidden, q5_rows_per_block, 2, true, false, true,
                                Geometry::kValueRows, Q5GdnDecodeEpilogue<Geometry, Publish>, TriggerPdl, JoinPdl>
            <<<q5_blocks, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Q5GdnDecodeEpilogue<Geometry, Publish>{value_epilogue, static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <class Geometry, int Tokens, class Q4Schedule, class Publish, bool TriggerPdl, bool JoinPdl,
          bool Dependent>
void launch_q4_small_t(const Tensor& x, const Weight& qk_weight,
                       const GdnConvEpilogue<Publish>& qk_epilogue, Tensor& query,
                       cudaStream_t stream) {
    const dim3 q4_grid(Geometry::kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                         Q4GdnSmallTEpilogue<Geometry, Tokens, Publish>, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQueryRows, 0, Geometry::kQkRows, Geometry::kHidden,
            Tokens, Geometry::kHidden, Q4GdnSmallTEpilogue<Geometry, Tokens, Publish>{qk_epilogue}));
    } else {
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0,
                                     Q4GdnSmallTEpilogue<Geometry, Tokens, Publish>, TriggerPdl, JoinPdl>
            <<<q4_grid, Q4Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, Geometry::kQueryRows, 0, Geometry::kQkRows, Geometry::kHidden,
                Tokens, Geometry::kHidden, Q4GdnSmallTEpilogue<Geometry, Tokens, Publish>{qk_epilogue});
    }
}

template <class Geometry, int Tokens, class Publish, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_small_t(const Tensor& x, const Weight& value_z_weight,
                       const GdnConvEpilogue<Publish>& value_epilogue, Tensor& value, Tensor& z,
                       cudaStream_t stream) {
    if constexpr (Geometry::kHidden == 5120 && (Tokens == 5 || Tokens == 6)) {
        using Epilogue = Q5GdnMmaEpilogue<Geometry, Tokens, Publish>;
        const Epilogue epilogue{{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}};
        const dim3 grid(Geometry::kValueZRows / 16);
        if constexpr (Dependent) {
            CUDA_CHECK(pdl::launch_dependent(
                {grid, dim3(256), 0, stream},
                q5_small_t_mma_kernel<Geometry::kValueZRows, Tokens, Epilogue,
                                      TriggerPdl, JoinPdl>,
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales), epilogue));
        } else {
            q5_small_t_mma_kernel<Geometry::kValueZRows, Tokens, Epilogue,
                                  TriggerPdl, JoinPdl><<<grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales), epilogue);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    constexpr int q5_threads = 4 * 32;
    const dim3 q5_grid(Geometry::kValueZRows, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q5_grid, dim3(q5_threads), 0, stream},
            q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, Geometry::kFullSlabs, Geometry::kHidden, true,
                                                Geometry::kValueRows, Q5GdnSmallTEpilogue<Geometry, Tokens, Publish>,
                                                TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Geometry::kValueZRows, Geometry::kValueRows, Geometry::kHidden, Tokens, Geometry::kHidden, Geometry::kFullSlabs,
            Q5GdnSmallTEpilogue<Geometry, Tokens, Publish>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            }));
    } else {
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, Geometry::kFullSlabs, Geometry::kHidden, true,
                                            Geometry::kValueRows, Q5GdnSmallTEpilogue<Geometry, Tokens, Publish>,
                                            TriggerPdl, JoinPdl>
            <<<q5_grid, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Geometry::kValueZRows, Geometry::kValueRows, Geometry::kHidden, Tokens, Geometry::kHidden, Geometry::kFullSlabs,
                Q5GdnSmallTEpilogue<Geometry, Tokens, Publish>{
                    value_epilogue,
                    static_cast<__nv_bfloat16*>(z.data),
                });
    }
}

template <class Geometry, PdlOrder Order, class Publish>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvEpilogue<Publish>& qk_epilogue,
               const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_t1<Geometry, Publish, true, false, false>(x, value_z_weight, value_epilogue, value, z,
                                                            stream);
        launch_q4_t1<Geometry, Publish, false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<Geometry, Publish, true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_q5_t1<Geometry, Publish, false, true, true>(x, value_z_weight, value_epilogue, value, z,
                                                           stream);
    }
}

template <class Geometry, int Tokens, class Q4Schedule, PdlOrder Order, class Publish>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvEpilogue<Publish>& qk_epilogue,
                             const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_small_t<Geometry, Tokens, Publish, true, false, false>(x, value_z_weight, value_epilogue,
                                                                         value, z, stream);
        launch_q4_small_t<Geometry, Tokens, Q4Schedule, Publish, false, true, true>(x, qk_weight, qk_epilogue,
                                                                                    query, stream);
    } else {
        launch_q4_small_t<Geometry, Tokens, Q4Schedule, Publish, true, false, false>(
            x, qk_weight, qk_epilogue, query, stream);
        launch_q5_small_t<Geometry, Tokens, Publish, false, true, true>(x, value_z_weight, value_epilogue,
                                                                        value, z, stream);
    }
}

template <class Geometry, int Tokens, PdlOrder Order, class Publish>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvEpilogue<Publish>& qk_epilogue,
                    const GdnConvEpilogue<Publish>& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Geometry, Tokens, Q4ScheduleC4, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Geometry, Tokens, Q4ScheduleC8, Order, Publish>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <class Geometry, PdlOrder Order, class Publish>
void launch_conv(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                 const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
                 const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                 Publish publish, cudaStream_t stream) {
    const GdnConvEpilogue<Publish> qk_epilogue = make_epilogue<Geometry>(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, 0, publish);
    const GdnConvEpilogue<Publish> value_epilogue =
        make_epilogue<Geometry>(conv_weight, conv_states, valid_columns, initial_slot, query, key, value,
                                Geometry::kValueOffset, publish);

    switch (x.ne[1]) {
    case 1:
        launch_t1<Geometry, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                            value, z, stream);
        break;
    case 2:
        launch_small_t<Geometry, 2, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                    query, value, z, stream);
        break;
    case 3:
        launch_small_t<Geometry, 3, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                    query, value, z, stream);
        break;
    case 5:
        launch_small_t<Geometry, 5, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                    query, value, z, stream);
        break;
    case 6:
        launch_small_t<Geometry, 6, Order, Publish>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue,
                                                    query, value, z, stream);
        break;
    default:
        throw std::invalid_argument("Q4/Q5 projection-epilogue GDN conv requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

template <PdlOrder Order, class Publish>
void launch_conv_by_geometry(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             Publish publish, cudaStream_t stream) {
    if (x.ne[0] == 4096) {
        launch_conv<GdnSnapshotGeometry9, Order>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                                valid_columns, initial_slot, query, key, value, z, publish, stream);
    } else if (x.ne[0] == 5120) {
        launch_conv<GdnSnapshotGeometry27, Order>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                                 valid_columns, initial_slot, query, key, value, z, publish, stream);
    } else {
        throw std::invalid_argument("GDN Q4/Q5 projection-epilogue conv: unsupported input width");
    }
}

} // namespace

void q4_q5_gdn_input_conv_t5_mma_prewarm() {
    cudaFuncAttributes attributes{};
    using Snapshot = Q5GdnMmaEpilogue<GdnSnapshotGeometry27, 5, SnapshotHistoryPublish>;
    using Record = Q5GdnMmaEpilogue<GdnSnapshotGeometry27, 5, RecordColumnPublish>;
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, q5_small_t_mma_kernel<12288, 5, Snapshot, true, false>));
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, q5_small_t_mma_kernel<12288, 5, Record, true, false>));
    using Snapshot6 = Q5GdnMmaEpilogue<GdnSnapshotGeometry27, 6, SnapshotHistoryPublish>;
    using Record6 = Q5GdnMmaEpilogue<GdnSnapshotGeometry27, 6, RecordColumnPublish>;
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, q5_small_t_mma_kernel<12288, 6, Snapshot6, true, false>));
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, q5_small_t_mma_kernel<12288, 6, Record6, true, false>));
}

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    const std::int32_t channels = (x.ne[0] == 4096) ? 8192 : 10240;
    const SnapshotHistoryPublish publish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         channels};
    if (x.ne[1] == 2) {
        launch_conv_by_geometry<PdlOrder::Q4ThenQ5>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z, publish, stream);
    } else {
        launch_conv_by_geometry<PdlOrder::Q5ThenQ4>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z, publish, stream);
    }
}

void q4_q5_gdn_input_conv_record_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, const Tensor& conv_weight,
                                        const Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_slot, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        cudaStream_t stream) {
    const std::int32_t channels = (x.ne[0] == 4096) ? 8192 : 10240;
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data), channels,
                                      static_cast<std::int32_t>(x.ne[1])};
    if (x.ne[1] == 2) {
        launch_conv_by_geometry<PdlOrder::Q4ThenQ5>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z, publish, stream);
    } else {
        launch_conv_by_geometry<PdlOrder::Q5ThenQ4>(
            x, qk_weight, value_z_weight, conv_weight, conv_states, valid_columns, initial_slot,
            query, key, value, z, publish, stream);
    }
}

} // namespace ninfer::ops::detail
