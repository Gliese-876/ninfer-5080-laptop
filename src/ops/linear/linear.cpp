// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ninfer/ops/linear.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_dispatch.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q6/q6_dispatch.h"
#include "ops/linear/w8/w8_dispatch.h"

#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

std::int64_t checked_numel(const Tensor& tensor, const char* label) {
    std::int64_t total = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0) {
            throw std::invalid_argument(std::string("linear: ") + label +
                                        " dimensions must be positive");
        }
        if (total > std::numeric_limits<std::int64_t>::max() / extent) {
            throw std::overflow_error("linear: tensor size overflows int64");
        }
        total *= extent;
    }
    return total;
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_linear_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear: invalid compute policy");
}

void validate_linear_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                               LinearPolicy policy) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear: x/out must be BF16");
    }
    (void)checked_numel(x, "x");
    (void)checked_numel(out, "out");
    if (x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("linear: x must have shape [K,T]");
    }
    if (out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear: out must have shape [N,T]");
    }
    if (w.n <= 0 || w.k <= 0) {
        throw std::invalid_argument("linear: weight n/k must be positive");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("linear: expected [K,T] x [N,K] -> [N,T]");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear: x/out must be non-null and 16-byte aligned");
    }
    validate_linear_policy(policy);
}

void dispatch_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    switch (w.qtype) {
    case QType::Q4G64_F16S:
        detail::q4_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q5G64_F16S:
        detail::q5_dispatch(x, w, out, policy, stream);
        return;
    case QType::Q6G64_F16S:
        detail::q6_dispatch(x, w, out, policy, stream);
        return;
    case QType::W8G32_F16S:
        detail::w8_dispatch(x, w, out, policy, stream);
        return;
    case QType::BF16_CTRL:
        detail::bf16_dispatch(x, w, out, policy, stream);
        return;
    case QType::NVFP4:
        detail::nvfp4_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP8_E4M3FN_ROW_BF16S:
        detail::fp8_dispatch(x, w, out, policy, workspace, stream);
        return;
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear: unsupported weight qtype");
}

} // namespace

std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                            std::int32_t input_rows, LinearPolicy policy,
                                            std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_linear_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("linear workspace: invalid token interval");
    }

    switch (qtype) {
    case QType::Q4G64_F16S:
        (void)detail::select_q4_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q4_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q5G64_F16S:
        (void)detail::select_q5_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q5_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::Q6G64_F16S:
        (void)detail::select_q6_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_q6_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::W8G32_F16S:
        (void)detail::select_w8_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_w8_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::BF16_CTRL:
        (void)detail::select_bf16_launch(output_rows, input_rows, min_tokens, policy);
        (void)detail::select_bf16_launch(output_rows, input_rows, max_tokens, policy);
        return 0;
    case QType::NVFP4:
        if (!detail::is_nvfp4_linear_problem(output_rows, input_rows) ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("linear workspace: unsupported NVFP4 profile");
        }
        return detail::nvfp4_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                             min_tokens, max_tokens);
    case QType::FP8_E4M3FN_ROW_BF16S:
        return detail::fp8_linear_workspace_capacity_bytes(output_rows, input_rows, policy,
                                                           min_tokens, max_tokens);
    case QType::FP32_CTRL:
    case QType::I32_CTRL:
        break;
    }
    throw std::invalid_argument("linear workspace: unsupported weight qtype");
}

void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, policy);
    dispatch_linear(x, w, out, policy, &workspace, stream);
}

void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    validate_linear_semantics(x, w, out, LinearPolicy::A16Only);
    dispatch_linear(x, w, out, LinearPolicy::A16Only, nullptr, stream);
}

void linear_gather_rows(const Tensor& x, const Weight& source, const Tensor& row_ids,
                         Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kSourceRows = 248320;
    constexpr std::int32_t kSelectedRows = 131072;
    constexpr std::int32_t kHidden = 5120;
    const bool valid_source =
        source.qtype == QType::Q4G64_F16S && source.layout == QuantLayout::RowSplit &&
        source.scale_dtype == DType::FP16 && source.group_size == 64 && source.group == 64 &&
        source.ndim == 2 && source.n == kSourceRows && source.k == kHidden &&
        source.shape[0] == kSourceRows && source.shape[1] == kHidden &&
        source.padded_shape[0] == kSourceRows && source.padded_shape[1] == kHidden &&
        source.qhigh == nullptr && source.high_plane_bytes == 0 &&
        aligned_to(source.qdata, 16) && aligned_to(source.scales, 16);
    if (!valid_source) {
        throw std::invalid_argument("linear_gather_rows: invalid Q4 source profile");
    }
    (void)checked_numel(x, "gather input");
    (void)checked_numel(out, "gather output");
    if (x.dtype != DType::BF16 || x.ne[0] != kHidden || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || !aligned_to(x.data, 16)) {
        throw std::invalid_argument("linear_gather_rows: invalid BF16 input [5120,T]");
    }
    if (out.dtype != DType::BF16 || out.ne[0] != kSelectedRows || out.ne[1] != x.ne[1] ||
        out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_gather_rows: invalid BF16 output [131072,T]");
    }
    if (row_ids.dtype != DType::I32 || row_ids.ne[0] != kSelectedRows ||
        row_ids.ne[1] != 1 || row_ids.ne[2] != 1 || row_ids.ne[3] != 1 ||
        !row_ids.is_contiguous() || !aligned_to(row_ids.data, 4)) {
        throw std::invalid_argument("linear_gather_rows: invalid I32 row ids [131072]");
    }
    struct Range {
        const void* data;
        std::size_t bytes;
    };
    const Range ranges[]{
        {x.data, x.bytes()},
        {row_ids.data, row_ids.bytes()},
        {out.data, out.bytes()},
        {source.qdata, static_cast<std::size_t>(kSourceRows) * kHidden / 2},
        {source.scales, static_cast<std::size_t>(kSourceRows) * (kHidden / 64) * 2}};
    for (std::size_t i = 0; i < std::size(ranges); ++i) {
        const auto begin = reinterpret_cast<std::uintptr_t>(ranges[i].data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(ranges[j].data);
            if (begin < other + ranges[j].bytes && other < begin + ranges[i].bytes) {
                throw std::invalid_argument("linear_gather_rows: live buffers overlap");
            }
        }
    }
    detail::launch_q4_gather_rows(x, source, row_ids, out, stream);
}

} // namespace ninfer::ops
