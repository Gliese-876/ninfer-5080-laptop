// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/kernel/gqa_attention_decode_q4.cuh"
#include "ops/launcher/gqa_attention.h"
#include "ninfer/ops/gqa_attention.h"
#include "core/device.h"
#include <stdexcept>

namespace ninfer::test {
using namespace ninfer::ops;
namespace {

bool native_route(const Tensor& q, DType dtype, int kv_heads, GqaExecutionEnvelope envelope) {
    const int width = q.ne[2];
    if (dtype != DType::U8 || q.ne[1] != 24 || kv_heads != 4 || width < 4 || width > 6) return false;
    // Preserve the production T6 Bc64 route; only Bc32 is this experiment's candidate.
    return !(width == 6 && envelope.max_visible_keys > 2054 && envelope.max_visible_keys <= 8198);
}

template <int Tokens, bool MultiBatch, bool Masked, class Input, bool NativeK = false>
void launch_expanded(const Tensor& q, Input input, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto scope = workspace.scope();
    const int batch = q.ne[3];
    const int splits = ops::detail::gqa_attention_split_capacity(24, Tokens, DType::U8, envelope);
    Tensor acc = workspace.alloc(DType::BF16, {256, 24, Tokens, splits * batch});
    Tensor m = workspace.alloc(DType::FP32, {24, Tokens, splits * batch});
    Tensor l = workspace.alloc(DType::FP32, {24, Tokens, splits * batch});
    const dim3 grid(4, splits, batch);
    auto partial = [&]<int Warps, int MinBlocks>() {
        gqa_attention_decode_q4_tiled_kernel<Gqa27Geometry, Tokens, Warps, MinBlocks,
            32, false, MultiBatch, Masked, Input, NativeK, false><<<grid, Warps * 32, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(positions.data),
                static_cast<std::uint8_t*>(cache.k_pages.data),
                static_cast<std::uint8_t*>(cache.v_pages.data),
                static_cast<__half*>(cache.k_scale_pages.data),
                static_cast<__half*>(cache.v_scale_pages.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
                table_rows.data ? static_cast<const std::int32_t*>(table_rows.data) : nullptr,
                cache.block_tables.ne[0], Tokens, 0, envelope.max_visible_keys, scale,
                static_cast<__nv_bfloat16*>(acc.data), static_cast<float*>(m.data),
                static_cast<float*>(l.data));
    };
    const int window = envelope.max_visible_keys;
    if constexpr (Tokens == 4) {
        if (window <= 1029) partial.template operator()<16, 1>();
        else partial.template operator()<8, 2>();
    } else if constexpr (Tokens == 5) {
        if (window > 128 && window <= 512) partial.template operator()<32, 1>();
        else if (window <= 1029) partial.template operator()<16, 1>();
        else partial.template operator()<8, 2>();
    } else {
        if (window > 128 && window <= 160) partial.template operator()<24, 1>();
        else if (window <= 2054) partial.template operator()<12, 1>();
        else partial.template operator()<6, 2>();
    }
    CUDA_CHECK(cudaGetLastError());
    const dim3 reduce_grid(24, 4, Tokens * batch);
    gqa_attention_small_t_reduce_output_kernel<Gqa27Geometry, 64, true, MultiBatch, Masked,
        false><<<reduce_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(acc.data), static_cast<const float*>(m.data),
            static_cast<const float*>(l.data), static_cast<const std::int32_t*>(positions.data),
            Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            Tokens, Tokens, 0, batch, splits, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Input>
void dispatch_expanded(const Tensor& q, Input input, const Tensor& positions,
                     const Tensor& valid_columns, const Tensor& table_rows, float scale,
                     PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                     WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto launch = [&]<int Tokens>() {
        auto profile = [&]<bool MultiBatch, bool Masked>() {
            launch_expanded<Tokens, MultiBatch, Masked>(q, input, positions, valid_columns,
                table_rows, scale, cache, envelope, workspace, out, stream);
        };
        if (q.ne[3] == 1) {
            if (valid_columns.data) profile.template operator()<false, true>();
            else profile.template operator()<false, false>();
        } else {
            if (valid_columns.data) profile.template operator()<true, true>();
            else profile.template operator()<true, false>();
        }
    };
    switch (q.ne[2]) {
        case 4: launch.template operator()<4>(); break;
        case 5: launch.template operator()<5>(); break;
        case 6: launch.template operator()<6>(); break;
        default: throw std::invalid_argument("private native decode width");
    }
}

} // namespace

void gqa_q4_decode_expanded_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, ops::GqaExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    if (!native_route(q, cache.dtype, cache.num_kv_heads, envelope)) {
        throw std::invalid_argument("expanded decode control requires canonical T4..6/Bc32");
    }
    dispatch_expanded(q, GqaCachedInput{}, positions, valid_columns, table_rows, scale,
                      cache, envelope, workspace, out, stream);
}
void gqa_q4_decode_copy8_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, ops::GqaExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    if (!native_route(q, cache.dtype, cache.num_kv_heads, envelope) || q.ne[3] != 1 ||
        (q.ne[2] != 4 && q.ne[2] != 5) || envelope.max_visible_keys <= 1029) {
        throw std::invalid_argument("copy8 control requires canonical long C1 T4/T5");
    }
    auto launch = [&]<int Tokens>() {
        if (valid_columns.data) {
            launch_expanded<Tokens, false, true, GqaCachedInput, true>(q, GqaCachedInput{},
                positions, valid_columns, table_rows, scale, cache, envelope, workspace, out, stream);
        } else {
            launch_expanded<Tokens, false, false, GqaCachedInput, true>(q, GqaCachedInput{},
                positions, valid_columns, table_rows, scale, cache, envelope, workspace, out, stream);
        }
    };
    if (q.ne[2] == 4) launch.template operator()<4>();
    else launch.template operator()<5>();
}

} // namespace ninfer::test