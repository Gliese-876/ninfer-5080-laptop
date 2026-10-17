// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/q4/q4_small_t_mma.cuh"
#include "ops/linear_swiglu/q3/q3_small_t_mma.cuh"
#include "ops/linear/linear_test_common.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace detail = ninfer::ops::detail;

struct Fixture {
    int k;
    quantized_weight::PackedWeight host;
    DeviceBuffer payload;
    DeviceBuffer activation;
    Weight weight;
    std::vector<std::uint16_t> input;

    Fixture(QType qtype, int n, int input_rows, unsigned seed)
        : k(input_rows), host(quantized_weight::make_patterned_weight(
                             qtype, n, k, seed,
                             {quantized_weight::RowSplitScalePattern::Small,
                              quantized_weight::RowSplitCodePattern::Hashed})),
          payload(host.payload.size()), activation(static_cast<std::size_t>(k) * 8 * 2),
          weight(host.device_weight(payload.p)), input(static_cast<std::size_t>(k) * 8) {
        payload.copy_from_host(host.payload.data(), payload.bytes);
        std::mt19937 random(seed + 1);
        for (auto& value : input)
            value = f32_to_bf16((static_cast<int>(random() % 2049) - 1024) / 2048.0F);
        upload(false);
    }

    void upload(bool negative) {
        auto values = input;
        if (negative)
            for (auto& value : values) value ^= 0x8000;
        activation.copy_from_host(values.data(), activation.bytes);
    }
};

template <class Launch>
void qualify_outputs(Fixture& fixture, int rows, int tokens, DeviceContext& context,
                     const std::string& label, Launch launch) {
    const auto words = static_cast<std::size_t>(rows) * tokens;
    GuardedDeviceBuffer old_output(words * 2), padded_output(words * 2), public_output(words * 2);
    Tensor x(fixture.activation.p, DType::BF16, {fixture.k, tokens});
    Tensor old(old_output.data(), DType::BF16, {rows, tokens});
    Tensor padded(padded_output.data(), DType::BF16, {rows, tokens});
    Tensor production(public_output.data(), DType::BF16, {rows, tokens});
    auto all    = [&] { launch(x, fixture.weight, old, padded, production, context.stream); };
    auto poison = [&] {
        old_output.fill(0xff);
        padded_output.fill(0xff);
        public_output.fill(0xff);
        cuda_check(cudaDeviceSynchronize(), "finish padding-output poisoning");
    };
    auto verify = [&] {
        cuda_check(cudaStreamSynchronize(context.stream), "finish padding comparison");
        if (old_output.verify_guards(label) + padded_output.verify_guards(label) +
                public_output.verify_guards(label) !=
            0)
            throw std::runtime_error(label + ": output guard changed");
        const auto a = from_device<std::uint16_t>(old.data, words);
        const auto b = from_device<std::uint16_t>(padded.data, words);
        const auto c = from_device<std::uint16_t>(production.data, words);
        for (std::size_t index = 0; index < words; ++index) {
            if ((a[index] & 0x7f80) == 0x7f80 || a[index] != b[index] || b[index] != c[index])
                throw std::runtime_error(label + ": BF16 mismatch/nonfinite at " +
                                         std::to_string(index));
        }
    };
    fixture.upload(false);
    poison();
    all();
    verify();
    if (tokens == 6) {
        DecodeGraphDefinition definition;
        definition.capture(context.stream, all);
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        for (bool negative : {false, true}) {
            fixture.upload(negative);
            poison();
            graph.launch(context.stream);
            verify();
        }
        fixture.upload(false);
    }
    std::cout << "PASS bit-exact " << label << " T=" << tokens << (tokens == 6 ? " Graph X/-X" : "")
              << '\n';
}

template <class Geometry, int T, int Padding, class Rows = detail::Q4SmallTMmaIdentityRows>
void launch_q4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream, Rows rows = {}) {
    detail::q4_small_t_mma_kernel<Geometry, 8, T, detail::Q4SmallTMmaStoreEpilogue, Rows, Padding>
        <<<Geometry::kOutputRows / Rows::kOutputRowsPerCta, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            detail::Q4SmallTMmaStoreEpilogue{}, rows);
    cuda_check_last_launch("Q4 padding probe");
}

template <class Function>
void widths(Function function) {
    function.template operator()<2>();
    function.template operator()<3>();
    function.template operator()<4>();
    function.template operator()<5>();
    function.template operator()<6>();
    function.template operator()<7>();
    function.template operator()<8>();
}

template <int N, int K>
void q4_shape(DeviceContext& context) {
    using Geometry = detail::Q4LinearGeometry<N, K>;
    Fixture fixture(QType::Q4G64_F16S, N, K, 731U + K);
    const auto label = "Q4[" + std::to_string(N) + "," + std::to_string(K) + "]";
    widths([&]<int T> {
        if constexpr (T >= 4) {
            qualify_outputs(fixture, N, T, context, label,
                            [](const Tensor& x, const Weight& w, Tensor& old, Tensor& padded,
                               Tensor& production, cudaStream_t stream) {
                                launch_q4<Geometry, T, 0>(x, w, old, stream);
                                launch_q4<Geometry, T, 16>(x, w, padded, stream);
                                ops::linear(x, w, production, stream);
                            });
        }
    });
}

struct GatherRows {
    static constexpr int kOutputRowsPerCta = 16;
    const std::int32_t* ids;

    __device__ __forceinline__ int weight_row(int first, int local) const {
        return ids[first + local];
    }
};

void q4_gather(DeviceContext& context) {
    constexpr int rows = 131072;
    using Geometry     = detail::Q4LinearGeometry<rows, 5120>;
    Fixture fixture(QType::Q4G64_F16S, 248320, 5120, 947U);
    std::vector<std::int32_t> ids(248077);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 random(953U);
    std::shuffle(ids.begin(), ids.end(), random);
    ids.resize(rows);
    DeviceBuffer map(ids.size() * sizeof(std::int32_t));
    map.copy_from_host(ids.data(), map.bytes);
    Tensor row_ids(map.p, DType::I32, {rows});
    widths([&]<int T> {
        qualify_outputs(fixture, rows, T, context, "Q4 indexed gather",
                        [&](const Tensor& x, const Weight& w, Tensor& old, Tensor& padded,
                            Tensor& production, cudaStream_t stream) {
                            GatherRows policy{static_cast<const std::int32_t*>(map.p)};
                            launch_q4<Geometry, T, 0>(x, w, old, stream, policy);
                            launch_q4<Geometry, T, 16>(x, w, padded, stream, policy);
                            ops::linear_gather_rows(x, w, row_ids, production, stream);
                        });
    });
}

struct Q3Geometry {
    static constexpr int kInputRows    = 5120;
    static constexpr int kGroupsPerRow = 80;
};

struct SwiGluRows {
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int first, int local) const {
        return first + (local & 7) + (local >= 8 ? 17408 : 0);
    }
};

struct SwiGluEpilogue {
    __nv_bfloat16* out;

    template <int T>
    __device__ __forceinline__ void store(int row, int col, float4 value) const {
        if (col < T)
            out[static_cast<std::int64_t>(col) * 17408 + row] =
                __float2bfloat16_rn(ops::silu(value.x) * value.z);
        if (col + 1 < T)
            out[static_cast<std::int64_t>(col + 1) * 17408 + row] =
                __float2bfloat16_rn(ops::silu(value.y) * value.w);
    }
};

template <int T, int Padding>
void launch_q3(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    detail::q3_small_t_mma_kernel<Q3Geometry, 8, T, SwiGluEpilogue, SwiGluRows, Padding>
        <<<2176, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            SwiGluEpilogue{static_cast<__nv_bfloat16*>(out.data)}, SwiGluRows{});
    cuda_check_last_launch("Q3 padding probe");
}

void q3_fused(DeviceContext& context) {
    Fixture fixture(QType::Q3G64_F16S, 34816, 5120, 967U);
    DeviceArena workspace(256);
    widths([&]<int T> {
        if constexpr (T != 3) {
            qualify_outputs(fixture, 17408, T, context, "Q3 fused SwiGLU",
                            [&](const Tensor& x, const Weight& w, Tensor& old, Tensor& padded,
                                Tensor& production, cudaStream_t stream) {
                                launch_q3<T, 0>(x, w, old, stream);
                                launch_q3<T, 16>(x, w, padded, stream);
                                ops::linear_swiglu(x, w, production, workspace, stream);
                            });
        }
    });
}

int independent_oracles() {
    namespace linear = ninfer::test::linear;
    namespace fused  = ninfer::test::linear_swiglu;
    constexpr std::array<linear::Invocation, 5> invocations{{{4}, {5}, {6}, {7}, {8}}};
    int failures = 0;
    for (auto shape : {std::pair{5120, 17408}, std::pair{5120, 6144}, std::pair{248320, 5120}})
        failures += linear::run_shape(
            "Q4 padding FP64", linear::ActivationCompute::A16, linear::make_q4g64_f16s_weight,
            {shape.first, shape.second, 977U, linear::Comparison::Full, true, invocations});
    constexpr std::array<std::int32_t, 6> tokens{2, 4, 5, 6, 7, 8};
    failures += fused::run_profile("Q3 padding FP64",
                                   {QType::Q3G64_F16S, 34816, 5120, 17408, 983U,
                                    fused::ActivationCompute::A16, fused::InputPattern::AllDense},
                                   tokens);
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        DeviceContext context;
        q4_shape<5120, 17408>(context);
        q4_shape<5120, 6144>(context);
        q4_shape<248320, 5120>(context);
        q4_gather(context);
        q3_fused(context);
        const int failures = independent_oracles();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " shared-code padding qualification\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "shared-code padding: " << error.what() << '\n';
        return 1;
    }
}
