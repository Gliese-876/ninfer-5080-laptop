// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/speculative_round.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
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

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kFullRows  = 248320;
constexpr std::int32_t kValidRows = 248077;
constexpr std::int32_t kRows      = 131072;
constexpr std::int32_t kGroup     = 64;
constexpr std::int32_t kGroups    = kHidden / kGroup;
constexpr std::int32_t kTieFirst  = 511;
constexpr std::int32_t kTieSecond = 16384;
constexpr ReductionCriterion kA16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

struct RowSplitWeight {
    DeviceBuffer payload;
    Weight weight;
    std::size_t scale_offset;

    explicit RowSplitWeight(std::int32_t rows)
        : payload(static_cast<std::size_t>(rows) * (kHidden / 2 + kGroups * 2)),
          scale_offset(static_cast<std::size_t>(rows) * kHidden / 2) {
        payload.fill(0);
        weight.payload       = payload.p;
        weight.payload_bytes = payload.bytes;
        weight.qtype         = QType::Q4G64_F16S;
        weight.group_size    = kGroup;
        weight.shape[0] = weight.padded_shape[0] = rows;
        weight.shape[1] = weight.padded_shape[1] = kHidden;
        weight.ndim                              = 2;
        weight.qdata                             = payload.p;
        weight.scales      = static_cast<const std::uint8_t*>(payload.p) + scale_offset;
        weight.n           = rows;
        weight.k           = kHidden;
        weight.group       = kGroup;
        weight.layout      = QuantLayout::RowSplit;
        weight.scale_dtype = DType::FP16;
    }
};

struct SparseRow {
    std::int32_t local;
    std::int32_t physical;
    std::array<std::uint8_t, kHidden / 2> codes{};
    std::array<std::uint16_t, kGroups> scales{};
};

void put_row(RowSplitWeight& weight, std::int32_t row, const SparseRow& source) {
    weight.payload.copy_from_host(source.codes.data(), source.codes.size(),
                                  static_cast<std::size_t>(row) * source.codes.size());
    weight.payload.copy_from_host(
        source.scales.data(), source.scales.size() * sizeof(std::uint16_t),
        weight.scale_offset + static_cast<std::size_t>(row) * kGroups * 2);
}

std::vector<std::int32_t> make_ids() {
    std::vector<std::int32_t> ids(kValidRows);
    std::iota(ids.begin(), ids.end(), 0);
    std::mt19937 random(41387U);
    std::shuffle(ids.begin(), ids.end(), random);
    // Keep the domain endpoints and row/tile boundaries in a deliberately nonmonotonic map.
    constexpr std::array<std::pair<std::int32_t, std::int32_t>, 13> pinned{{
        {0, 131071},
        {1, 1},
        {7, 7},
        {8, 8},
        {15, 15},
        {16, 16},
        {31, 512},
        {32, 16383},
        {64, 16384},
        {kTieFirst, kValidRows - 1},
        {kTieSecond, 0},
        {kRows - 2, kValidRows - 2},
        {kRows - 1, 511},
    }};
    for (const auto& [local, physical] : pinned) {
        const auto found = std::find(ids.begin(), ids.end(), physical);
        std::iter_swap(ids.begin() + local, found);
    }
    ids.resize(kRows);
    auto sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= kValidRows ||
        std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::runtime_error("indexed fixture has an invalid row map");
    }
    return ids;
}

struct Fixture {
    RowSplitWeight full{kFullRows};
    RowSplitWeight compact{kRows};
    std::vector<std::int32_t> ids = make_ids();
    GuardedDeviceBuffer device_ids{ids.size() * sizeof(std::int32_t)};
    std::vector<SparseRow> rows;

    Fixture() { device_ids.copy_from_host(ids.data(), device_ids.bytes()); }

    Tensor id_tensor() { return Tensor(device_ids.data(), DType::I32, {kRows}); }

    void reset() {
        full.payload.fill(0);
        compact.payload.fill(0);
        rows.clear();
    }

    void add(SparseRow row) {
        row.physical = ids[row.local];
        put_row(full, row.physical, row);
        put_row(compact, row.local, row);
        rows.push_back(row);
    }

    int verify_preserved() const {
        int failures = verify_exact("indexed row ids preserved",
                                    from_device<std::int32_t>(device_ids.data(), ids.size()), ids);
        failures += device_ids.verify_guards("indexed row ids");
        for (const auto& row : rows) {
            const auto* codes  = static_cast<const std::uint8_t*>(full.weight.qdata) +
                                 static_cast<std::size_t>(row.physical) * row.codes.size();
            const auto* scales = static_cast<const std::uint16_t*>(full.weight.scales) +
                                 static_cast<std::size_t>(row.physical) * kGroups;
            failures += verify_exact("indexed source codes preserved",
                                     from_device<std::uint8_t>(codes, row.codes.size()),
                                     std::vector<std::uint8_t>(row.codes.begin(), row.codes.end()));
            failures +=
                verify_exact("indexed source scales preserved",
                             from_device<std::uint16_t>(scales, row.scales.size()),
                             std::vector<std::uint16_t>(row.scales.begin(), row.scales.end()));
        }
        return failures;
    }
};

SparseRow make_signed_row(std::int32_t local, std::int32_t seed) {
    SparseRow row{local, 0};
    for (std::int32_t k = 0; k < kHidden; ++k) {
        const std::int32_t code = ((k * 13 + seed * 7 + k / kGroup) % 16) - 8;
        const auto nibble       = static_cast<std::uint8_t>(code & 15);
        row.codes[k / 2] |= static_cast<std::uint8_t>(nibble << ((k & 1) * 4));
    }
    for (std::int32_t group = 0; group < kGroups; ++group) {
        const float value = 0.03125F + static_cast<float>((group * 3 + seed * 5) % 37) * 0.001F;
        row.scales[group] = quantized_weight::detail::f32_to_f16(value);
    }
    return row;
}

void fill_signed_fixture(Fixture& fixture) {
    fixture.reset();
    constexpr std::array<std::int32_t, 25> locals{
        0,   1,    7,    8,    15,    16,    31,    32,    63,    64,     127,       128,       511,
        512, 1023, 4095, 8191, 16383, 16384, 32767, 65535, 65536, 130560, kRows - 2, kRows - 1,
    };
    for (std::size_t i = 0; i < locals.size(); ++i) {
        fixture.add(make_signed_row(locals[i], static_cast<std::int32_t>(i + 1)));
    }
    // A projection through contiguous source rows, or through padded rows, must not leak these.
    SparseRow poison{0, 0};
    poison.codes.fill(0x77);
    poison.scales.fill(quantized_weight::detail::f32_to_f16(32.0F));
    std::vector<bool> selected(kValidRows, false);
    for (const auto id : fixture.ids) { selected[id] = true; }
    const auto unselected = std::find(selected.begin(), selected.end(), false) - selected.begin();
    put_row(fixture.full, static_cast<std::int32_t>(unselected), poison);
    put_row(fixture.full, kValidRows, poison);
    put_row(fixture.full, kFullRows - 1, poison);
}

std::vector<std::uint16_t> make_hidden(std::int32_t columns, bool impulse = false) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(kHidden) * columns, 0);
    for (std::int32_t column = 0; column < columns; ++column) {
        if (impulse) {
            hidden[static_cast<std::size_t>(column) * kHidden] = f32_to_bf16(1.0F);
            continue;
        }
        for (std::int32_t k = 0; k < kHidden; ++k) {
            float value = 0.25F + static_cast<float>((k * 11 + column * 7) % 29) * 0.03125F +
                          static_cast<float>(column) * 0.015625F;
            if (k % 17 == 0) { value *= -0.25F; }
            hidden[static_cast<std::size_t>(column) * kHidden + k] = f32_to_bf16(value);
        }
    }
    return hidden;
}

std::vector<double> oracle(const Fixture& fixture, const std::vector<std::uint16_t>& hidden,
                           std::int32_t columns) {
    std::vector<double> reference(static_cast<std::size_t>(kRows) * columns, 0.0);
    // Unpatched represented weights have zero codes and zero stored scales. Their complete dot
    // product is exactly zero. Only patched rows require the full 5120-term independent oracle.
    for (const auto& row : fixture.rows) {
        if (fixture.ids[row.local] != row.physical) {
            throw std::runtime_error("indexed fixture row address changed");
        }
        for (std::int32_t column = 0; column < columns; ++column) {
            double sum = 0.0;
            for (std::int32_t k = 0; k < kHidden; ++k) {
                const std::uint8_t packed = row.codes[k / 2];
                const std::int32_t raw    = (packed >> ((k & 1) * 4)) & 15;
                const std::int32_t code   = raw < 8 ? raw : raw - 16;
                const double scale = quantized_weight::detail::f16_to_f32(row.scales[k / kGroup]);
                const double input =
                    bf16_to_f32(hidden[static_cast<std::size_t>(column) * kHidden + k]);
                sum += static_cast<double>(code) * scale * input;
            }
            reference[static_cast<std::size_t>(column) * kRows + row.local] = sum;
        }
    }
    return reference;
}

int verify_projection(const std::string& label, const Fixture& fixture,
                      const std::vector<std::uint16_t>& actual,
                      const std::vector<double>& reference, bool exact_bf16) {
    std::vector<double> promoted(actual.size());
    std::vector<std::uint16_t> rounded;
    if (exact_bf16) { rounded.resize(reference.size()); }
    std::vector<bool> patched(kRows, false);
    for (const auto& row : fixture.rows) { patched[row.local] = true; }
    int failures = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        promoted[i] = bf16_to_f32(actual[i]);
        if (!patched[i % kRows] && promoted[i] != 0.0) {
            std::cerr << label << ": unselected zero row changed at " << i << '\n';
            ++failures;
            break;
        }
    }
    failures += verify_reduction(label + " independent FP64", promoted, reference, kA16Criterion);
    if (exact_bf16) {
        for (std::size_t i = 0; i < reference.size(); ++i) {
            rounded[i] = f32_to_bf16(static_cast<float>(reference[i]));
        }
        failures += verify_exact((label + " semantic BF16 cast").c_str(), actual, rounded);
    }
    return failures;
}

std::vector<std::int32_t> local_argmax(const std::vector<std::uint16_t>& logits,
                                       std::int32_t columns) {
    std::vector<std::int32_t> expected(columns, 0);
    for (std::int32_t column = 0; column < columns; ++column) {
        const auto base = static_cast<std::size_t>(column) * kRows;
        float best      = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < kRows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best) {
                best             = value;
                expected[column] = row;
            }
        }
    }
    return expected;
}

int verify_argmax(Fixture& fixture, Tensor& out, const std::vector<std::uint16_t>& logits,
                  std::int32_t columns, std::int32_t known_winner) {
    GuardedDeviceBuffer token_storage(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    Tensor tokens(token_storage.data(), DType::I32, {columns});
    auto expected = local_argmax(logits, columns);
    int failures  = 0;
    if (known_winner >= 0) {
        failures += verify_exact("indexed known local winner", expected,
                                 std::vector<std::int32_t>(columns, known_winner));
    }
    ops::argmax(out, tokens, kRows, nullptr);
    cuda_synchronize();
    failures += verify_exact("indexed local argmax",
                             from_device<std::int32_t>(tokens.data, columns), expected);
    ops::proposal_remap_token_ids(
        tokens, static_cast<const std::int32_t*>(fixture.device_ids.data()), kRows, nullptr);
    cuda_synchronize();
    for (auto& id : expected) { id = fixture.ids[id]; }
    failures += verify_exact("indexed global remap after local tie",
                             from_device<std::int32_t>(tokens.data, columns), expected);
    failures += token_storage.verify_guards("indexed proposal tokens");
    return failures;
}

int run_case(Fixture& fixture, std::int32_t columns, const std::vector<std::uint16_t>& host_hidden,
             bool exact_bf16 = false, std::int32_t known_winner = -1) {
    const std::string label = "linear_gather_rows T=" + std::to_string(columns);
    GuardedDeviceBuffer hidden_storage(host_hidden.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer indexed_storage(static_cast<std::size_t>(kRows) * columns * 2);
    hidden_storage.copy_from_host(host_hidden.data(), hidden_storage.bytes());
    indexed_storage.fill(0xcd);
    Tensor hidden(hidden_storage.data(), DType::BF16, {kHidden, columns});
    Tensor indexed(indexed_storage.data(), DType::BF16, {kRows, columns});
    Tensor ids = fixture.id_tensor();
    ops::linear_gather_rows(hidden, fixture.full.weight, ids, indexed, nullptr);
    cuda_synchronize();
    const auto actual = from_device<std::uint16_t>(indexed.data, indexed.numel());
    int failures = verify_projection(label, fixture, actual, oracle(fixture, host_hidden, columns),
                                     exact_bf16);
    // These calls have the same qualified GEMV / small-T schedule. For larger T the indexed
    // contract uses eight-column chunks, while the compact public Linear may choose another route.
    if (columns <= 8) {
        GuardedDeviceBuffer compact_storage(indexed_storage.bytes());
        compact_storage.fill(0xef);
        Tensor compact(compact_storage.data(), DType::BF16, {kRows, columns});
        ops::linear(hidden, fixture.compact.weight, compact, nullptr);
        cuda_synchronize();
        failures += verify_exact((label + " compact BF16 parity").c_str(), actual,
                                 from_device<std::uint16_t>(compact.data, compact.numel()));
        failures += compact_storage.verify_guards(label + " compact");
    }
    failures += verify_argmax(fixture, indexed, actual, columns, known_winner);
    failures +=
        verify_exact((label + " hidden preserved").c_str(),
                     from_device<std::uint16_t>(hidden.data, host_hidden.size()), host_hidden);
    failures += hidden_storage.verify_guards(label + " hidden");
    failures += indexed_storage.verify_guards(label + " output");
    return failures;
}

void fill_tie_fixture(Fixture& fixture, bool collapse) {
    fixture.reset();
    SparseRow first{kTieFirst, 0};
    first.codes[0] = 1;
    first.scales.fill(quantized_weight::detail::f32_to_f16(1.0F));
    fixture.add(first);
    SparseRow second = first;
    second.local     = kTieSecond;
    if (collapse) {
        second.scales[0] = quantized_weight::detail::f32_to_f16(1.0F + 1.0F / 1024.0F);
        const float a    = quantized_weight::detail::f16_to_f32(first.scales[0]);
        const float b    = quantized_weight::detail::f16_to_f32(second.scales[0]);
        if (!(b > a) || f32_to_bf16(a) != f32_to_bf16(b)) {
            throw std::runtime_error("indexed BF16-collapse fixture does not collapse");
        }
    }
    fixture.add(second);
    if (fixture.ids[kTieFirst] <= fixture.ids[kTieSecond]) {
        throw std::runtime_error("indexed tie fixture does not distinguish local/global ordering");
    }
}

int run_graph(Fixture& fixture, std::int32_t columns = 7) {
    auto host_hidden               = make_hidden(columns, true);
    GuardedDeviceBuffer hidden_storage(host_hidden.size() * 2);
    GuardedDeviceBuffer output_storage(static_cast<std::size_t>(kRows) * columns * 2);
    GuardedDeviceBuffer token_storage(columns * sizeof(std::int32_t));
    hidden_storage.copy_from_host(host_hidden.data(), hidden_storage.bytes());
    Tensor hidden(hidden_storage.data(), DType::BF16, {kHidden, columns});
    Tensor out(output_storage.data(), DType::BF16, {kRows, columns});
    Tensor tokens(token_storage.data(), DType::I32, {columns});
    Tensor ids                = fixture.id_tensor();
    cudaStream_t stream       = nullptr;
    cudaGraph_t graph         = nullptr;
    cudaGraphExec_t execution = nullptr;
    cuda_check(cudaStreamCreate(&stream), "indexed create stream");
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "indexed begin graph");
    ops::linear_gather_rows(hidden, fixture.full.weight, ids, out, stream);
    ops::argmax(out, tokens, kRows, stream);
    ops::proposal_remap_token_ids(tokens, static_cast<const std::int32_t*>(ids.data), kRows,
                                  stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "indexed end graph");
    cuda_check(cudaGraphInstantiate(&execution, graph, 0), "indexed instantiate graph");
    int failures = 0;
    for (int phase = 0; phase < 2; ++phase) {
        output_storage.fill(0xcd);
        token_storage.fill(0xcd);
        cuda_check(cudaGraphLaunch(execution, stream), "indexed first graph replay");
        cuda_check(cudaGraphLaunch(execution, stream), "indexed second graph replay");
        cuda_synchronize(stream);
        failures += verify_projection("indexed graph", fixture,
                                      from_device<std::uint16_t>(out.data, out.numel()),
                                      oracle(fixture, host_hidden, columns), true);
        const auto winner = phase == 0 ? kTieFirst : 0;
        failures += verify_exact("indexed graph local tie/remap",
                                 from_device<std::int32_t>(tokens.data, columns),
                                 std::vector<std::int32_t>(columns, fixture.ids[winner]));
        // Reuse captured addresses with changed input. Both output and argmax must be overwritten.
        std::fill(host_hidden.begin(), host_hidden.end(), 0);
        hidden_storage.copy_from_host(host_hidden.data(), hidden_storage.bytes());
    }
    cuda_check(cudaGraphExecDestroy(execution), "indexed destroy graph execution");
    cuda_check(cudaGraphDestroy(graph), "indexed destroy graph");
    cuda_check(cudaStreamDestroy(stream), "indexed destroy stream");
    failures += hidden_storage.verify_guards("indexed graph hidden");
    failures += output_storage.verify_guards("indexed graph output");
    failures += token_storage.verify_guards("indexed graph tokens");
    return failures;
}

template <class Call>
int expect_invalid(const char* label, Call&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "linear_gather_rows accepted " << label << '\n';
    return 1;
}

int invalid_io(Fixture& fixture) {
    DeviceBuffer hidden_storage(kHidden * 2);
    DeviceBuffer output_storage(kRows * 2);
    Tensor hidden(hidden_storage.p, DType::BF16, {kHidden, 1});
    Tensor out(output_storage.p, DType::BF16, {kRows, 1});
    Tensor ids      = fixture.id_tensor();
    const auto call = [&](const Tensor& x, const Weight& weight, const Tensor& row_ids, Tensor& y) {
        ops::linear_gather_rows(x, weight, row_ids, y, nullptr);
    };
    int failures  = 0;
    auto invalid  = hidden;
    invalid.dtype = DType::FP32;
    failures +=
        expect_invalid("non-BF16 hidden", [&] { call(invalid, fixture.full.weight, ids, out); });
    invalid = hidden;
    invalid.nb[0] += 2;
    failures += expect_invalid("noncontiguous hidden",
                               [&] { call(invalid, fixture.full.weight, ids, out); });
    invalid       = ids;
    invalid.dtype = DType::BF16;
    failures +=
        expect_invalid("non-I32 map", [&] { call(hidden, fixture.full.weight, invalid, out); });
    invalid = ids;
    invalid.ne[0] -= 1;
    failures += expect_invalid("wrong map length",
                               [&] { call(hidden, fixture.full.weight, invalid, out); });
    invalid = out;
    invalid.ne[0] -= 1;
    failures += expect_invalid("wrong output rows",
                               [&] { call(hidden, fixture.full.weight, ids, invalid); });
    auto weight  = fixture.full.weight;
    weight.qtype = QType::W8G32_F16S;
    failures += expect_invalid("non-Q4 source", [&] { call(hidden, weight, ids, out); });
    failures += expect_invalid("compact source geometry",
                               [&] { call(hidden, fixture.compact.weight, ids, out); });
    invalid      = out;
    invalid.data = hidden.data;
    failures += expect_invalid("overlapping input/output",
                               [&] { call(hidden, fixture.full.weight, ids, invalid); });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        Fixture fixture;
        fill_signed_fixture(fixture);
        int failures = invalid_io(fixture);
        for (const std::int32_t columns : {1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17}) {
            failures += run_case(fixture, columns, make_hidden(columns));
        }
        failures += fixture.verify_preserved();
        fill_tie_fixture(fixture, false);
        failures += run_case(fixture, 1, make_hidden(1, true), true, kTieFirst);
        failures += run_case(fixture, 7, make_hidden(7, true), true, kTieFirst);
        fill_tie_fixture(fixture, true);
        for (const std::int32_t columns : {1, 7, 17}) {
            failures += run_case(fixture, columns, make_hidden(columns, true), true, kTieFirst);
        }
        failures += run_graph(fixture, 1);
        failures += run_graph(fixture);
        failures += run_case(fixture, 9, std::vector<std::uint16_t>(kHidden * 9, 0), true, 0);
        failures += fixture.verify_preserved();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " linear_gather_rows\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "linear_gather_rows: " << error.what() << '\n';
        return 1;
    }
}
