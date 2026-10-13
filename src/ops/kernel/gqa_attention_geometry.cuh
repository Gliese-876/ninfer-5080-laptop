#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.
//
// BaseSplitsValue parameterizes the decode split ceiling: 85 base splits (one
// CTA per RTX 5080 SM plus one), scaled by DecodeSplitScaleValue. The 85
// default preserves every production geometry bit-for-bit. Gqa27S84Geometry
// (base 84 = the RTX 5080 SM count) exists only for the isolated 84-versus-85
// split experiment (issue #47) in bench/ops/causal_softmax_attention_bench.cu;
// no production route selects it.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue, int BaseSplitsValue = 85>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);
    static_assert(BaseSplitsValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int BaseSplits       = BaseSplitsValue;
    static constexpr int DecodeSplits     = BaseSplits * DecodeSplitScale;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;
using Gqa9Geometry  = GqaGeometry<16, 4, 1>;

// Benchmark-only experimental 27B geometry (issue #47): the 27B shape with the
// base split count pinned to the RTX 5080's 84 SMs. Consumed only by the
// isolated split-experiment launcher and the public-Op benchmark; production
// dispatch never selects it.
using Gqa27S84Geometry = GqaGeometry<24, 4, 1, 84>;

} // namespace ninfer::ops
