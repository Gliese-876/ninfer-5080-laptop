// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

// Canonical Q24/KV4/D256 Prompt schedule. The accepted native64 kernel remains
// the independent scheduling control; this leaf changes only warp ownership
// and when already-computed P/alpha are published to shared memory.
#include "ops/kernel/gqa_attention_prefill_q4.cuh"

namespace ninfer::ops {
namespace q4_prompt_pipeline {

inline constexpr int kRows = 64;
inline constexpr int kKeys = 64;
inline constexpr int kDim = 256;
inline constexpr int kGroups = 4;
inline constexpr int kQKTiles = 8;
inline constexpr int kPVFragments = 16;
inline constexpr unsigned kFullMask = 0xffffffffu;
using Geometry = Gqa27Geometry;
static_assert(kGqaPrefillI8Br == kRows && kGqaPrefillI8Bc == kKeys);
static_assert(kGqaPrefillHeadDim == kDim && kGqaPrefillI8Threads == 512);

// Byte ranges are identical to accepted native64. Current PV reads P/V_fp16/
// alpha while look-ahead copies touch only packed K/V and KV scales.
struct Shared {
    unsigned char* base;
    __device__ __forceinline__ std::int8_t* q() const {
        return reinterpret_cast<std::int8_t*>(base);
    }
    __device__ __forceinline__ float* qs() const {
        return reinterpret_cast<float*>(base + 16384);
    }
    __device__ __forceinline__ unsigned char* k() const { return base + 17408; }
    __device__ __forceinline__ unsigned char* v() const { return base + 25600; }
    __device__ __forceinline__ __half* vf() const {
        return reinterpret_cast<__half*>(base + 33792);
    }
    __device__ __forceinline__ __half* p() const {
        return reinterpret_cast<__half*>(base + 66560);
    }
    __device__ __forceinline__ __half* ks() const {
        return reinterpret_cast<__half*>(base + 74752);
    }
    __device__ __forceinline__ __half* vs() const {
        return reinterpret_cast<__half*>(base + 75264);
    }
    __device__ __forceinline__ float* alpha() const {
        return reinterpret_cast<float*>(base + 75776);
    }
    __device__ __forceinline__ float* final_l() const {
        return reinterpret_cast<float*>(base + 76032);
    }
};
static_assert(kGqaPrefillQ4SmemBytes == 76288);

struct ProbabilityState {
    float qs0[2];
    float qs1[2];
    float m0, m1, l0, l1;
    unsigned p0[kQKTiles], p1[kQKTiles];
    float alpha0, alpha1;
};

// Only the first eight warps participate: four QK producers and four loaders.
// NON-aligned PTX is intentional: the remaining eight warps execute current PV.
// Every loader drains its own cp.async group before arriving. This rendezvous
// then makes all completed writes visible to every participating QK producer.
__device__ __forceinline__ void copied_kv_barrier() {
    asm volatile("barrier.cta.sync 1, 256;" : : : "memory");
}

// Role loops reach the same hardware phase from different instruction sites.
// Use explicit NON-aligned PTX rather than __syncthreads inside divergent role
// scopes: all 512 threads participate, but their branch PCs intentionally differ.
__device__ __forceinline__ void joint_barrier() {
    asm volatile("barrier.cta.sync 0, 512;" : : : "memory");
}

__device__ __forceinline__ void issue_kv(
    Shared s, int worker, int k0, int max_query_abs, int kv_head,
    const std::int32_t* block_table, const std::uint8_t* cache_k,
    const std::uint8_t* cache_v, const __half* cache_ks, const __half* cache_vs) {
    // Exactly 128 loader threads own every copy and every logical-tail zero.
    // Every call starts at a valid logical page; no look-ahead call is made
    // after the last causal tile. Bc64 exactly equals the physical page size.
    const int page = block_table[k0 >> kPagedKVPageShift];
    for (int key_l = worker; key_l < kKeys; key_l += 128) {
        const int key = k0 + key_l;
        if (key <= max_query_abs) {
            const auto off = gqa_kv_quant_scale_index<Geometry>(
                page, kv_head, 0, key & kPagedKVPageMask);
            cp_async<8>(s.ks() + key_l * kGroups, cache_ks + off);
            cp_async<8>(s.vs() + key_l * kGroups, cache_vs + off);
        } else {
            store_vec(s.ks() + key_l * kGroups, make_int2(0, 0));
            store_vec(s.vs() + key_l * kGroups, make_int2(0, 0));
        }
    }
#pragma unroll 1
    for (int chunk = worker; chunk < kKeys * (kDim / 32); chunk += 128) {
        const int key_l = chunk / (kDim / 32);
        const int packed_byte = (chunk % (kDim / 32)) * 16;
        const int key = k0 + key_l;
        auto* kd = s.k() + key_l * (kDim / 2) +
                   gqa_prefill_q4_k_swz(key_l, packed_byte);
        auto* vd = s.v() + key_l * (kDim / 2) + packed_byte;
        if (key <= max_query_abs) {
            const auto off = gqa_kv_q4_code_index<Geometry>(
                page, kv_head, packed_byte * 2, key & kPagedKVPageMask);
            cp_async<16>(kd, cache_k + off);
            cp_async<16>(vd, cache_v + off);
        } else {
            store_vec(kd, make_int4(0, 0, 0, 0));
            store_vec(vd, make_int4(0, 0, 0, 0));
        }
    }
    cp_commit();
    cp_wait<0>();
}

__device__ __forceinline__ void dequant_v(Shared s, int worker, int lane,
                                          int k0, int max_query_abs) {
    // All twelve non-QK warps participate after the current-PV completion
    // barrier; their mapping and FP16 conversion match accepted native64.
#pragma unroll 1
    for (int chunk = worker; chunk < kKeys * (kDim / 8); chunk += 384) {
        const int key_l = chunk / (kDim / 8);
        const int d = (chunk % (kDim / 8)) * 8;
        auto* dst = s.vf() + key_l * kDim + gqa_prefill_swz(key_l, d);
        if (k0 + key_l <= max_query_abs) {
            const int grp = d >> 6;
            __half vs = __float2half_rn(0.0f);
            if ((lane & 7) == 0) vs = s.vs()[key_l * kGroups + grp];
            vs = __shfl_sync(kFullMask, vs, grp * 8);
            store_vec(dst, gqa_prefill_q4_dequant_f16x8(
                s.v() + key_l * (kDim / 2) + d / 2, vs));
        } else {
            store_vec(dst, make_int4(0, 0, 0, 0));
        }
    }
}

__device__ __forceinline__ void prepare_probabilities(
    Shared s, int producer, int lane, int tile_rows, int base_pos, int q0,
    int tokens, int k0, float scale_l2, ProbabilityState& state) {
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int row_base = producer * 16;
    const int a_rowoff = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_coloff = (lane >> 4) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const auto* q_b16 = reinterpret_cast<const __nv_bfloat16*>(s.q());
    float score[kQKTiles][4];
#pragma unroll
    for (int nt = 0; nt < kQKTiles; ++nt) {
        score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
    }
#pragma unroll
    for (int grp = 0; grp < kGroups; ++grp) {
        float qs0, qs1;
        if (grp < 2) {
            qs0 = state.qs0[grp];
            qs1 = state.qs1[grp];
        } else {
            qs0 = lid == 0 ? s.qs()[(row_base + gid) * kGroups + grp] : 0.0f;
            qs1 = lid == 0 ? s.qs()[(row_base + gid + 8) * kGroups + grp] : 0.0f;
            qs0 = __shfl_sync(kFullMask, qs0, gid * 4);
            qs1 = __shfl_sync(kFullMask, qs1, gid * 4);
        }
        unsigned af[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k = grp * 2 + kk;
            const int acol = k * 16 + a_coloff;
            ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                smem_addr(q_b16 + (row_base + a_rowoff) * (kDim / 2) +
                          gqa_prefill_swz(row_base + a_rowoff, acol)));
        }
#pragma unroll
        for (int nt = 0; nt < kQKTiles; ++nt) {
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int k = grp * 2 + kk;
                const int brow = nt * 8 + b_rin;
                const int bcol = k * 16 + b_koff;
                unsigned bf[2];
                gqa_prefill_q4_ldmatrix_s8(bf[0], bf[1],
                    smem_addr(s.k() + brow * (kDim / 2) +
                              gqa_prefill_q4_k_swz(brow, bcol)));
                mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                       bf[0], bf[1]);
            }
            const int keya = nt * 8 + 2 * lid;
            const int keyb = keya + 1;
            float ks0 = 0.0f, ks1 = 0.0f;
            if (gid == 0) {
                ks0 = __half2float(s.ks()[keya * kGroups + grp]);
                ks1 = __half2float(s.ks()[keyb * kGroups + grp]);
            }
            ks0 = __shfl_sync(kFullMask, ks0, lid);
            ks1 = __shfl_sync(kFullMask, ks1, lid);
            score[nt][0] = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
            score[nt][1] = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
            score[nt][2] = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
            score[nt][3] = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
        }
    }

    const int row0 = row_base + gid;
    const int row1 = row0 + 8;
    const int qabs0 = row0 < tile_rows ? base_pos + q0 + row0 : -1;
    const int qabs1 = row1 < tile_rows ? base_pos + q0 + row1 : -1;
    const bool full_score_tile = q0 + kRows <= tokens && k0 + kKeys - 1 <= base_pos + q0;
    float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
    for (int nt = 0; nt < kQKTiles; ++nt) {
        const int key0 = k0 + nt * 8 + 2 * lid;
        const int key1 = key0 + 1;
        if (!full_score_tile) {
            score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
            score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
            score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
            score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
        }
        bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
        bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
    }
    bm0 = warp_max<4>(bm0, kFullMask);
    bm1 = warp_max<4>(bm1, kFullMask);
    const float nm0 = fmaxf(state.m0, bm0);
    const float nm1 = fmaxf(state.m1, bm1);
    const float nm0_scaled = nm0 * scale_l2;
    const float nm1_scaled = nm1 * scale_l2;
    const float alpha0 = state.m0 == -CUDART_INF_F ? 0.0f :
        exp2_approx(__fmaf_rn(state.m0, scale_l2, -nm0_scaled));
    const float alpha1 = state.m1 == -CUDART_INF_F ? 0.0f :
        exp2_approx(__fmaf_rn(state.m1, scale_l2, -nm1_scaled));
    float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < kQKTiles; ++nt) {
        const float p00 = score[nt][0] > -CUDART_INF_F ?
            exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled)) : 0.0f;
        const float p01 = score[nt][1] > -CUDART_INF_F ?
            exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled)) : 0.0f;
        const float p10 = score[nt][2] > -CUDART_INF_F ?
            exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled)) : 0.0f;
        const float p11 = score[nt][3] > -CUDART_INF_F ?
            exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled)) : 0.0f;
        bl0 += p00 + p01;
        bl1 += p10 + p11;
        // Two independent FP16_RNE casts, retained in registers until current
        // PV has consumed the old shared P and alpha. Never round bl through P.
        state.p0[nt] = pack_f16x2(p00, p01);
        state.p1[nt] = pack_f16x2(p10, p11);
    }
    bl0 = warp_sum<4>(bl0, kFullMask);
    bl1 = warp_sum<4>(bl1, kFullMask);
    state.l0 = __fmaf_rn(state.l0, alpha0, bl0);
    state.l1 = __fmaf_rn(state.l1, alpha1, bl1);
    state.m0 = nm0;
    state.m1 = nm1;
    state.alpha0 = alpha0;
    state.alpha1 = alpha1;
}

__device__ __forceinline__ void publish_probabilities(Shared s, int producer, int lane,
                                                      const ProbabilityState& state) {
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int row0 = producer * 16 + gid;
    const int row1 = row0 + 8;
#pragma unroll
    for (int nt = 0; nt < kQKTiles; ++nt) {
        const int column = nt * 8 + 2 * lid;
        *reinterpret_cast<unsigned*>(s.p() + row0 * kKeys +
            gqa_prefill_i8_p_swz(row0, column)) = state.p0[nt];
        *reinterpret_cast<unsigned*>(s.p() + row1 * kKeys +
            gqa_prefill_i8_p_swz(row1, column)) = state.p1[nt];
    }
    if (lid == 0) {
        s.alpha()[row0] = state.alpha0;
        s.alpha()[row1] = state.alpha1;
    }
}

__device__ __forceinline__ void consume_pv(Shared s, int consumer, int lane,
                                          float (&acc)[kPVFragments][4]) {
    const int gid = lane >> 2;
    const int row_tile = consumer % 4;
    const int d_slice = consumer / 4;
    const int row_base = row_tile * 16;
    const float alpha0 = s.alpha()[row_base + gid];
    const float alpha1 = s.alpha()[row_base + gid + 8];
#pragma unroll
    for (int n = 0; n < kPVFragments; ++n) {
        acc[n][0] *= alpha0;
        acc[n][1] *= alpha0;
        acc[n][2] *= alpha1;
        acc[n][3] *= alpha1;
    }
    const int a_rowoff = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_coloff = (lane >> 4) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        unsigned pf[4];
        const int pcol = k * 16 + a_coloff;
        ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
            smem_addr(s.p() + (row_base + a_rowoff) * kKeys +
                      gqa_prefill_i8_p_swz(row_base + a_rowoff, pcol)));
#pragma unroll
        for (int n = 0; n < kPVFragments; ++n) {
            const int global_n = d_slice * kPVFragments + n;
            const int vrow = k * 16 + b_koff + b_rin;
            const int vcol = global_n * 8;
            unsigned vf[2];
            ldmatrix_x2_t(vf[0], vf[1],
                smem_addr(s.vf() + vrow * kDim + gqa_prefill_swz(vrow, vcol)));
            mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                    pf[0], pf[1], pf[2], pf[3], vf[0], vf[1]);
        }
    }
}

} // namespace q4_prompt_pipeline

template <typename Metadata>
__global__ __maxnreg__(128) void gqa_attention_prefill_q4_pipeline_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_ks,
    const __half* __restrict__ cache_vs, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale,
    __nv_bfloat16* __restrict__ out, std::int32_t width) {
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
                                               tid, 512);
        return;
    }
    const int tile_rows = min(kRows, tokens - q0);
    const int base_pos = positions[0];
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks = max_query_abs / kKeys + 1;
    const int kv_head = q_head / Geometry::GroupSize;
    const auto* block_table = metadata.block_table();

    // Keep Q quantization and its FP32 scales exactly as accepted native64.
    for (int unit = warp; unit < kRows * kGroups; unit += 16) {
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
    __syncthreads();
    if (warp >= 4 && warp < 8) {
        issue_kv(s, tid - 128, 0, max_query_abs, kv_head, block_table,
                 cache_k, cache_v, cache_ks, cache_vs);
    }
    __syncthreads(); // Initial packed KV/scales are visible to all twelve V workers.

    // These disjoint role scopes let ptxas overlay producer score/P registers
    // and consumer PV accumulators. Every role executes the same two full-CTA
    // phases per non-final tile and the same final-l phase.
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
        joint_barrier(); // Ready(0): initial P/V/alpha.
        for (int kb = 0; kb < key_blocks; ++kb) {
            const bool has_next = kb + 1 < key_blocks;
            if (has_next) {
                copied_kv_barrier();
                prepare_probabilities(s, warp, lane, tile_rows, base_pos, q0, tokens,
                                       (kb + 1) * kKeys, scale_l2, state);
            }
            joint_barrier(); // Consumed(kb): current PV finished, old P/V/alpha dead.
            if (has_next) {
                publish_probabilities(s, warp, lane, state);
                joint_barrier(); // Ready(kb+1): deferred P/alpha and next V published.
            }
        }
        if (lid == 0) {
            s.final_l()[warp * 16 + gid] = state.l0;
            s.final_l()[warp * 16 + gid + 8] = state.l1;
        }
        joint_barrier(); // Final-l: all PV is complete before normalization.
    } else if (warp < 8) {
        const int worker = tid - 128;
        dequant_v(s, worker, lane, 0, max_query_abs);
        joint_barrier(); // Ready(0).
        for (int kb = 0; kb < key_blocks; ++kb) {
            const bool has_next = kb + 1 < key_blocks;
            if (has_next) {
                issue_kv(s, worker, (kb + 1) * kKeys, max_query_abs, kv_head,
                         block_table, cache_k, cache_v, cache_ks, cache_vs);
                copied_kv_barrier();
            }
            joint_barrier(); // Consumed(kb).
            if (has_next) {
                dequant_v(s, worker, lane, (kb + 1) * kKeys, max_query_abs);
                joint_barrier(); // Ready(kb+1).
            }
        }
        joint_barrier(); // Final-l.
    } else {
        const int consumer = warp - 8;
        const int worker = tid - 128;
        float acc[kPVFragments][4];
#pragma unroll
        for (int n = 0; n < kPVFragments; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[n][i] = 0.0f;
        }
        dequant_v(s, worker, lane, 0, max_query_abs);
        joint_barrier(); // Ready(0).
        for (int kb = 0; kb < key_blocks; ++kb) {
            consume_pv(s, consumer, lane, acc);
            joint_barrier(); // Consumed(kb).
            if (kb + 1 < key_blocks) {
                dequant_v(s, worker, lane, (kb + 1) * kKeys, max_query_abs);
                joint_barrier(); // Ready(kb+1).
            }
        }
        joint_barrier(); // Final-l.
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
    gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + kRows, width), tid, 512);
}

} // namespace ninfer::ops
