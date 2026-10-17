// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

// Canonical Q24/KV4/D256 Prompt: four QK/copy producers and eight PV consumers.
// Share the frozen mathematical helpers with the independent 512-thread control.
#include "ops/kernel/gqa_attention_prefill_q4_pipeline.cuh"

namespace ninfer::ops::q4_prompt_384 {
inline constexpr int kThreads = 384;
inline constexpr int kWarps = 12;

__device__ __forceinline__ void copied_kv() {
    asm volatile("barrier.cta.sync 1, 128;" : : : "memory");
}
__device__ __forceinline__ void joint() {
    asm volatile("barrier.cta.sync 0, 384;" : : : "memory");
}

template <typename Metadata, int RegisterCap>
__global__ __maxnreg__(RegisterCap) void kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_ks,
    const __half* __restrict__ cache_vs, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale,
    __nv_bfloat16* __restrict__ out, std::int32_t width) {
    static_assert(RegisterCap == 160 || RegisterCap == 168);
    using namespace q4_prompt_pipeline;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const Shared s{smem_raw};
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = blockIdx.x * kRows;
    const int q_head = blockIdx.y;
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) return;
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + kRows, width),
                                               tid, kThreads);
        return;
    }
    const int tile_rows = min(kRows, tokens - q0);
    const int base_pos = positions[0];
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks = max_query_abs / kKeys + 1;
    const int kv_head = q_head / Geometry::GroupSize;
    const auto* block_table = metadata.block_table();

    // Only independent (row,G64) ownership changes from 16 to 12 warps.
    // Every Q value, warp reduction, FP32 scale and INT8 code is unchanged.
    for (int unit = warp; unit < kRows * kGroups; unit += kWarps) {
        const int row = unit / kGroups;
        const int grp = unit % kGroups;
        const int d0 = grp * 64 + lane;
        const int d1 = d0 + 32;
        float x0 = 0.0f, x1 = 0.0f;
        if (row < tile_rows) {
            x0 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row)]);
            x1 = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d1, q0 + row)]);
        }
        float absmax = fmaxf(fabsf(x0), fabsf(x1));
        absmax = warp_max(absmax, kFullMask);
        const float qs = absmax > 0.0f ? absmax / 127.0f : 0.0f;
        const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
        gqa_prefill_i8_store_swz(s.q(), row, d0, gqa_kv_quant_code(x0, inv));
        gqa_prefill_i8_store_swz(s.q(), row, d1, gqa_kv_quant_code(x1, inv));
        if (lane == 0) s.qs()[row * kGroups + grp] = qs;
    }
    joint();
    if (warp < 4) {
        // These are the same 128 worker indices as the old dedicated loaders.
        issue_kv(s, tid, 0, max_query_abs, kv_head, block_table,
                 cache_k, cache_v, cache_ks, cache_vs);
    }
    joint();

    if (warp < 4) {
        ProbabilityState state;
        state.m0 = -CUDART_INF_F;
        state.m1 = -CUDART_INF_F;
        state.l0 = 0.0f;
        state.l1 = 0.0f;
        const int gid = lane >> 2;
        const int lid = lane & 3;
#pragma unroll
        for (int grp = 0; grp < 2; ++grp) {
            float qs0 = lid == 0 ? s.qs()[(warp * 16 + gid) * kGroups + grp] : 0.0f;
            float qs1 = lid == 0 ? s.qs()[(warp * 16 + gid + 8) * kGroups + grp] : 0.0f;
            state.qs0[grp] = __shfl_sync(kFullMask, qs0, gid * 4);
            state.qs1[grp] = __shfl_sync(kFullMask, qs1, gid * 4);
        }
        constexpr float Log2E = 1.4426950408889634074f;
        const float scale_l2 = scale * Log2E;
        prepare_probabilities(s, warp, lane, tile_rows, base_pos, q0, tokens, 0,
                               scale_l2, state);
        publish_probabilities(s, warp, lane, state);
        dequant_v(s, tid, lane, 0, max_query_abs);
        joint(); // Ready(0): all 12 warps jointly cover the original V work.
        for (int kb = 0; kb < key_blocks; ++kb) {
            const bool has_next = kb + 1 < key_blocks;
            if (has_next) {
                // Current PV reads only shared P/V_fp16/alpha. Current QK and
                // V dequant are complete, so packed KV/scales can be reused.
                issue_kv(s, tid, (kb + 1) * kKeys, max_query_abs, kv_head,
                         block_table, cache_k, cache_v, cache_ks, cache_vs);
                copied_kv();
                prepare_probabilities(s, warp, lane, tile_rows, base_pos, q0, tokens,
                                       (kb + 1) * kKeys, scale_l2, state);
            }
            joint(); // Consumed(kb): all eight PV warps have retired current P/V.
            if (has_next) {
                publish_probabilities(s, warp, lane, state);
                // Producer worker0..127 plus consumer worker128..383 retains
                // exactly the original stride384 dequant mapping, no gaps.
                dequant_v(s, tid, lane, (kb + 1) * kKeys, max_query_abs);
                joint();
            }
        }
        if (lid == 0) {
            s.final_l()[warp * 16 + gid] = state.l0;
            s.final_l()[warp * 16 + gid + 8] = state.l1;
        }
        joint();
    } else {
        const int consumer = warp - 4;
        float acc[kPVFragments][4];
#pragma unroll
        for (int n = 0; n < kPVFragments; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[n][i] = 0.0f;
        }
        dequant_v(s, tid, lane, 0, max_query_abs);
        joint();
        for (int kb = 0; kb < key_blocks; ++kb) {
            consume_pv(s, consumer, lane, acc);
            joint();
            if (kb + 1 < key_blocks) {
                dequant_v(s, tid, lane, (kb + 1) * kKeys, max_query_abs);
                joint();
            }
        }
        joint();
        const int gid = lane >> 2;
        const int lid = lane & 3;
        const int row_base = (consumer % 4) * 16;
        const int d_slice = consumer / 4;
        const int row0 = row_base + gid;
        const int row1 = row0 + 8;
        const float inv_l0 = s.final_l()[row0] > 0.0f ? __frcp_rn(s.final_l()[row0]) : 0.0f;
        const float inv_l1 = s.final_l()[row1] > 0.0f ? __frcp_rn(s.final_l()[row1]) : 0.0f;
#pragma unroll
        for (int n = 0; n < kPVFragments; ++n) {
            const int d0 = (d_slice * kPVFragments + n) * 8 + 2 * lid;
            if (row0 < tile_rows) {
                *reinterpret_cast<unsigned*>(out +
                    gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)) =
                    pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
            }
            if (row1 < tile_rows) {
                *reinterpret_cast<unsigned*>(out +
                    gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)) =
                    pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
            }
        }
    }
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + kRows, width),
                                           tid, kThreads);
}
} // namespace ninfer::ops::q4_prompt_384
