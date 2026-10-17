// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int verify_exact_q5_small_t_codec();

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
constexpr int kK = 5120;
#ifndef NINFER_Q5_EXACT_TEST_TOKENS
#define NINFER_Q5_EXACT_TEST_TOKENS 5
#endif
constexpr int kT = NINFER_Q5_EXACT_TEST_TOKENS;
static_assert(kT >= 4 && kT <= 6);
constexpr int kChannels = 10240;
constexpr int kSlots = kT + 2;
constexpr int kInitial = 1;
constexpr int kBase = 1; // Supported overlap with this row's own initial history.
constexpr ReductionCriterion kAttention{2.9e-3, 4.0e-3, 4.5e-3};
constexpr ReductionCriterion kGdn{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kFused{3.15e-3, 4.0e-3, 3.2e-3};

quantized_weight::PackedWeight weight(QType type, int rows, std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.row_split_codes = quantized_weight::RowSplitCodePattern::Hashed;
    options.row_split_scale = quantized_weight::RowSplitScalePattern::Small;
    return quantized_weight::make_patterned_weight(type, rows, kK, seed, options);
}

std::vector<double> full_projection(const quantized_weight::PackedWeight& packed,
                                    const std::vector<float>& x) {
    const int rows = packed.weight.n;
    std::vector<double> result(static_cast<std::size_t>(rows) * kT);
    const int workers = std::min(rows, static_cast<int>(
        std::max(1u, std::thread::hardware_concurrency())));
    std::vector<std::thread> threads;
    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&, worker] {
            const int first = static_cast<std::int64_t>(rows) * worker / workers;
            const int last = static_cast<std::int64_t>(rows) * (worker + 1) / workers;
            for (int row = first; row < last; ++row) {
                for (int token = 0; token < kT; ++token) {
                    result[static_cast<std::size_t>(token) * rows + row] =
                        quantized_weight::dot_fp64(packed, row, x.data() + token * kK, kK);
                }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    return result;
}

std::vector<double> range(const std::vector<double>& parent, int rows, int first, int count) {
    std::vector<double> result(static_cast<std::size_t>(count) * kT);
    for (int token = 0; token < kT; ++token) {
        std::copy_n(parent.data() + static_cast<std::size_t>(token) * rows + first, count,
                    result.data() + static_cast<std::size_t>(token) * count);
    }
    return result;
}

std::vector<std::uint16_t> random_bits(std::size_t count, std::uint32_t seed, float bound) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -bound, bound);
    round_to_bf16(values);
    return bf16_bits(values);
}

template <class Launch, class Reset, class Verify, class Capture>
int eager_and_graph(const std::string& label, Launch&& launch, Reset&& reset,
                     Verify&& verify, Capture&& capture) {
    reset(nullptr);
    launch(nullptr);
    cuda_synchronize();
    int failures = verify(label + " eager");
    const auto eager = capture();
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t execution = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Q5 graph stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "Q5 graph begin");
    launch(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "Q5 graph end");
    cuda_check(cudaGraphInstantiate(&execution, graph, 0), "Q5 graph instantiate");
    for (int replay = 0; replay < 2; ++replay) {
        reset(stream);
        cuda_check(cudaGraphLaunch(execution, stream), "Q5 graph replay");
        cuda_check(cudaStreamSynchronize(stream), "Q5 graph synchronize");
        failures += verify(label + " graph");
        const auto actual = capture();
        for (std::size_t index = 0; index < actual.size(); ++index) {
            failures += verify_exact((label + " graph/eager BF16").c_str(), actual[index],
                                      eager[index]);
        }
    }
    cuda_check(cudaGraphExecDestroy(execution), "Q5 graph execution destroy");
    cuda_check(cudaGraphDestroy(graph), "Q5 graph destroy");
    cuda_check(cudaStreamDestroy(stream), "Q5 stream destroy");
    std::cout << (failures ? "FAIL " : "OK ") << label << " full FP64/Graph/guards\n";
    return failures;
}

struct Fixture {
    DevicePackedWeight aq{weight(QType::Q4G64_F16S, 7168, 2511U)};
    DevicePackedWeight av{weight(QType::Q5G64_F16S, 7168, 2513U)};
    DevicePackedWeight gq{weight(QType::Q4G64_F16S, 4096, 2517U)};
    DevicePackedWeight gv{weight(QType::Q5G64_F16S, 12288, 2521U)};
    std::vector<float> x = make_bf16_activation(kK, kT, 2523U);
    std::vector<std::uint16_t> x_bits = bf16_bits(x);
    DeviceBuffer device_x = to_device(x_bits);
    std::vector<double> attention_q = full_projection(aq.host, x);
    std::vector<double> attention_v = full_projection(av.host, x);
    std::vector<double> gdn_q = full_projection(gq.host, x);
    std::vector<double> gdn_v = full_projection(gv.host, x);
};

int independent(Fixture& fixture, bool attention) {
    const int q_rows = attention ? 6144 : 10240;
    const int tail_rows = attention ? 1024 : 6144;
    GuardedBf16Tensor q(q_rows, kT), gate(attention ? 6144 : 1, kT);
    GuardedBf16Tensor key(attention ? 1024 : 1, kT), value(tail_rows, kT);
    Tensor x(fixture.device_x.p, DType::BF16, {kK, kT});
    Tensor tq = q.tensor(), tg = gate.tensor(), tk = key.tensor(), tv = value.tensor();
    std::vector<GuardedBf16Tensor*> outputs = attention
        ? std::vector<GuardedBf16Tensor*>{&q, &gate, &key, &value}
        : std::vector<GuardedBf16Tensor*>{&q, &value};
    std::vector<std::vector<double>> reference;
    if (attention) {
        reference = {range(fixture.attention_q, 7168, 0, 6144),
                     range(fixture.attention_v, 7168, 0, 6144),
                     range(fixture.attention_q, 7168, 6144, 1024),
                     range(fixture.attention_v, 7168, 6144, 1024)};
    } else {
        reference.resize(2);
        reference[0].resize(static_cast<std::size_t>(10240) * kT);
        for (int token = 0; token < kT; ++token) {
            std::copy_n(fixture.gdn_q.data() + token * 4096, 4096,
                        reference[0].data() + token * 10240);
            std::copy_n(fixture.gdn_v.data() + token * 12288, 6144,
                        reference[0].data() + token * 10240 + 4096);
        }
        reference[1] = range(fixture.gdn_v, 12288, 6144, 6144);
    }
    const auto reset = [&](cudaStream_t stream) {
        for (auto* output : outputs) {
            const Tensor tensor = output->tensor();
            cuda_check(cudaMemsetAsync(output->data(), 0xff,
                static_cast<std::size_t>(tensor.ne[0]) * kT * 2, stream), "Q5 poison outputs");
        }
    };
    const auto launch = [&](cudaStream_t stream) {
        if (attention) {
            ops::attn_input_proj(x, fixture.aq.view(), fixture.av.view(), tq, tg, tk, tv, stream);
        } else {
            ops::gdn_input_proj(x, fixture.gq.view(), fixture.gv.view(), tq, tv, stream);
        }
    };
    const auto verify = [&](const std::string& label) {
        int errors = 0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            errors += outputs[index]->verify_guards(label);
            errors += outputs[index]->verify_fully_written(label);
            errors += verify_reduction(label, outputs[index]->values(), reference[index],
                                        attention ? kAttention : kGdn);
        }
        return errors;
    };
    const auto capture = [&] {
        std::vector<std::vector<std::uint16_t>> bits;
        for (auto* output : outputs) { bits.push_back(output->bits()); }
        return bits;
    };
    const std::string label = "Q5 T" + std::to_string(kT) +
        (attention ? " attention parent7168" : " GDN parent12288");
    return eager_and_graph(label,
                           launch, reset, verify, capture);
}

double silu(double x) {
    if (x >= 0.0) { return x / (1.0 + std::exp(-x)); }
    const double exponential = std::exp(x);
    return x * exponential / (1.0 + exponential);
}

struct FusedResult {
    int failures;
    std::vector<std::vector<std::uint16_t>> outputs;
    std::vector<std::uint16_t> published_projection;
};

FusedResult fused(Fixture& fixture, bool record, int valid, bool dense) {
    const auto conv_bits = random_bits(kChannels * 4, 2541U, 0.02F);
    const auto before = random_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots,
                                    2543U, 0.05F);
    const std::vector<std::uint16_t> record_before(kChannels * kT, 0x3c00U);
    DeviceBuffer conv_device = to_device(conv_bits);
    const std::vector<std::int32_t> initial_values{kInitial}, base_values{kBase}, valid_values{valid};
    DeviceBuffer initial_device = to_device(initial_values), base_device = to_device(base_values);
    DeviceBuffer valid_device = to_device(valid_values);
    GuardedBf16Tensor state(kChannels * 3, kSlots), history(kChannels, kT);
    GuardedBf16Tensor query(2048, kT), key(2048, kT), value(6144, kT), z(6144, kT);
    Tensor x(fixture.device_x.p, DType::BF16, {kK, kT});
    Tensor conv(conv_device.p, DType::BF16, {kChannels, 4});
    Tensor states(state.data(), DType::BF16, {kChannels, 3, kSlots});
    Tensor initial(initial_device.p, DType::I32, {1}), base(base_device.p, DType::I32, {1});
    Tensor valid_tensor;
    if (!dense) { valid_tensor = Tensor(valid_device.p, DType::I32, {1}); }
    Tensor q = query.tensor(), k = key.tensor(), v = value.tensor(), tz = z.tensor();
    Tensor projected = history.tensor();
    std::array<GuardedBf16Tensor*, 4> outputs{&query, &key, &value, &z};
    std::array<int, 4> rows{2048, 2048, 6144, 6144};
    std::array<std::vector<double>, 4> expected{
        std::vector<double>(2048 * kT), std::vector<double>(2048 * kT),
        std::vector<double>(6144 * kT), range(fixture.gdn_v, 12288, 6144, 6144)};
    std::vector<double> ideal_state(before.size()), ideal_record(kChannels * kT);
    for (std::size_t index = 0; index < before.size(); ++index) {
        ideal_state[index] = bf16_to_f32(before[index]);
    }
    for (int row = 0; row < kChannels; ++row) {
        const std::size_t first = static_cast<std::size_t>(kInitial) * 3 * kChannels + row;
        double s0 = bf16_to_f32(before[first]);
        double s1 = bf16_to_f32(before[first + kChannels]);
        double s2 = bf16_to_f32(before[first + 2 * kChannels]);
        for (int token = 0; token < valid; ++token) {
            const double p = row < 4096 ? fixture.gdn_q[token * 4096 + row]
                                       : fixture.gdn_v[token * 12288 + row - 4096];
            const double convolution = bf16_to_f32(conv_bits[row]) * s0 +
                bf16_to_f32(conv_bits[kChannels + row]) * s1 +
                bf16_to_f32(conv_bits[2 * kChannels + row]) * s2 +
                bf16_to_f32(conv_bits[3 * kChannels + row]) * p;
            const int part = row < 2048 ? 0 : row < 4096 ? 1 : 2;
            const int local = row - (part == 0 ? 0 : part == 1 ? 2048 : 4096);
            expected[part][token * rows[part] + local] = silu(convolution);
            const std::size_t destination = static_cast<std::size_t>(kBase + token) *
                                            3 * kChannels + row;
            ideal_state[destination] = s1;
            ideal_state[destination + kChannels] = s2;
            ideal_state[destination + 2 * kChannels] = p;
            ideal_record[token * kChannels + row] = p;
            s0 = s1; s1 = s2; s2 = p;
        }
    }
    const std::size_t capacity = record
        ? ops::gdn_input_proj_conv_record_workspace_capacity_bytes(2048, 2048, 6144, 1, kT, kT)
        : ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
              QType::Q5G64_F16S, 2048, 2048, 6144, 1, kT, kT);
    WorkspaceArena workspace(std::max<std::size_t>(256, capacity));
    const auto reset = [&](cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(state.data(), before.data(), before.size() * 2,
                                    cudaMemcpyHostToDevice, stream), "Q5 reset initial history");
        cuda_check(cudaMemcpyAsync(history.data(), record_before.data(), record_before.size() * 2,
                                    cudaMemcpyHostToDevice, stream), "Q5 reset record");
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            cuda_check(cudaMemsetAsync(outputs[index]->data(), 0xff,
                static_cast<std::size_t>(rows[index]) * kT * 2, stream), "Q5 poison fused output");
        }
    };
    const auto launch = [&](cudaStream_t stream) {
        if (record) {
            ops::gdn_input_proj_conv_record(x, fixture.gq.view(), fixture.gv.view(), conv, states,
                valid_tensor, initial, projected, q, k, v, tz, workspace, stream);
        } else {
            ops::gdn_input_proj_conv_snapshot(x, fixture.gq.view(), fixture.gv.view(), conv, states,
                valid_tensor, initial, base, q, k, v, tz, workspace, stream);
        }
    };
    const auto verify = [&](const std::string& label) {
        int errors = state.verify_guards(label) + history.verify_guards(label);
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            errors += outputs[index]->verify_guards(label);
            errors += outputs[index]->verify_fully_written(label);
            errors += verify_reduction(label, outputs[index]->values(), expected[index], kFused);
            if (index < 3) {
                const auto bits = outputs[index]->bits();
                for (int token = valid; token < kT; ++token) {
                    for (int row = 0; row < rows[index]; ++row) {
                        if (bits[token * rows[index] + row] != 0) { ++errors; }
                    }
                }
            }
        }
        const auto state_bits = state.bits();
        const auto record_bits = history.bits();
        std::vector<double> actual_history, reference_history;
        if (record) {
            errors += verify_exact((label + " immutable source history").c_str(), state_bits, before);
            for (int token = 0; token < kT; ++token) {
                for (int row = 0; row < kChannels; ++row) {
                    const int position = token * kChannels + row;
                    if (token < valid) {
                        actual_history.push_back(bf16_to_f32(record_bits[position]));
                        reference_history.push_back(ideal_record[position]);
                    }
                }
            }
        } else {
            for (int slot = 0; slot < kSlots; ++slot) {
                const bool published = slot >= kBase && slot < kBase + valid;
                for (int offset = 0; offset < 3 * kChannels; ++offset) {
                    const std::size_t position = static_cast<std::size_t>(slot) * 3 * kChannels + offset;
                    if (published) {
                        actual_history.push_back(bf16_to_f32(state_bits[position]));
                        reference_history.push_back(ideal_state[position]);
                    } else if (state_bits[position] != before[position]) { ++errors; }
                }
            }
        }
        errors += verify_reduction(label + " all published FP64 history", actual_history,
                                    reference_history, kFused);
        if (workspace.used() != 0 || workspace.peak_used() != capacity) { ++errors; }
        errors += verify_preserved(label + " conv", conv_device, conv_bits);
        errors += verify_preserved(label + " initial selector", initial_device, initial_values);
        errors += verify_preserved(label + " base selector", base_device, base_values);
        errors += verify_preserved(label + " valid selector", valid_device, valid_values);
        return errors;
    };
    const auto capture = [&] {
        std::vector<std::vector<std::uint16_t>> bits;
        for (auto* output : outputs) { bits.push_back(output->bits()); }
        bits.push_back(state.bits());
        if (record) {
            auto defined_record = history.bits();
            defined_record.resize(static_cast<std::size_t>(valid) * kChannels);
            bits.push_back(std::move(defined_record));
        }
        return bits;
    };
    const std::string label = "Q5 T" + std::to_string(kT) + " fused " +
        (record ? "Record" : "Snapshot") +
        " valid=" + std::to_string(valid) + (dense ? " dense" : " masked");
    int failures = eager_and_graph(label, launch, reset, verify, capture);
    FusedResult result{failures, {}, {}};
    for (auto* output : outputs) { result.outputs.push_back(output->bits()); }
    const auto published = record ? history.bits() : state.bits();
    for (int token = 0; token < valid; ++token) {
        for (int row = 0; row < kChannels; ++row) {
            const std::size_t position = record ? static_cast<std::size_t>(token) * kChannels + row
                : static_cast<std::size_t>(kBase + token) * 3 * kChannels + 2 * kChannels + row;
            result.published_projection.push_back(published[position]);
        }
    }
    return result;
}
}

int main() {
    using namespace ninfer::test;
    if (ninfer::test::cuda_unavailable()) { return 77; }
    try {
        int failures = verify_exact_q5_small_t_codec();
        Fixture fixture;
        failures += independent(fixture, true);
        failures += independent(fixture, false);
        for (const int valid : {1, kT - 1, kT}) {
            const bool dense = valid == kT;
            auto snapshot = fused(fixture, false, valid, dense);
            auto record = fused(fixture, true, valid, dense);
            failures += snapshot.failures + record.failures;
            for (std::size_t index = 0; index < snapshot.outputs.size(); ++index) {
                failures += verify_exact("Q5 Snapshot/Record exact BF16 output",
                                          snapshot.outputs[index], record.outputs[index]);
            }
            failures += verify_exact("Q5 Snapshot/Record exact BF16 projection publish",
                                      snapshot.published_projection, record.published_projection);
        }
        failures += fixture.aq.verify_preserved("Q5 attention Q4 weights");
        failures += fixture.av.verify_preserved("Q5 attention Q5 weights");
        failures += fixture.gq.verify_preserved("Q5 GDN Q4 weights");
        failures += fixture.gv.verify_preserved("Q5 GDN Q5 weights");
        failures += verify_preserved("Q5 represented BF16 input", fixture.device_x, fixture.x_bits);
        std::cout << (failures ? "FAIL" : "OK") << " exact Q5 T" << kT << " complete Ops\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Q5 exact T" << kT << " test: " << error.what() << '\n';
        return 1;
    }
}
