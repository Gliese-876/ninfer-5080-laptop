// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

#include "ninfer/ops/gqa_attention.h"

namespace ninfer::test {

// Supplementary accepted expanded-K control. Independent FP64 qualification remains
// in test_gqa_attention.cpp; this entry only accepts its canonical validated fixtures.
void gqa_q4_decode_expanded_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, ops::GqaExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream = nullptr);

// Exact original packed8 copy policy for qualified canonical long T4/T5 leaves.
void gqa_q4_decode_copy8_control(const Tensor& q, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale,
    PagedKVBatchLayerView cache, ops::GqaExecutionEnvelope envelope,
    WorkspaceArena& workspace, Tensor& out, cudaStream_t stream = nullptr);

} // namespace ninfer::test
