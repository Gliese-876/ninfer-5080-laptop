// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

namespace ninfer::test {

// Supplementary exact v1.5 control. Numerical qualification remains the
// independent FP64 attention formula in test_gqa_attention.cpp.
void gqa_q4_prompt_expanded_control(const Tensor& q, const Tensor& positions, float scale,
                                    const PagedKVLayerView& cache, Tensor& out,
                                    cudaStream_t stream = nullptr);
void gqa_q4_prompt_native_control(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache, Tensor& out,
                                  cudaStream_t stream = nullptr);
void gqa_q4_prompt_native_batch_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, bool masked, float scale,
    const PagedKVBatchLayerView& cache, Tensor& out, cudaStream_t stream = nullptr);
void gqa_q4_prompt_pipeline512_control(const Tensor& q, const Tensor& positions, float scale,
    const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream = nullptr);
void gqa_q4_prompt_pipeline512_batch_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, bool masked, float scale,
    const PagedKVBatchLayerView& cache, Tensor& out, cudaStream_t stream = nullptr);
int verify_q4_prompt_native_matrix_fragment();
void benchmark_q4_prompt_staging(const Tensor& q, const Tensor& positions, float scale,
                                  const PagedKVLayerView& cache, Tensor& out);
void benchmark_q4_prompt_pipeline(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache, Tensor& out);

} // namespace ninfer::test
