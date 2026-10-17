// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
// Standalone CPU-only qualification driver. Compile TOKENIZER_REFERENCE_ONLY against the
// frozen original tokenizer objects/header first, then compare the candidate against its IDs.
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

std::string read(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct Resources {
    std::string tokenizer, config, generation;
    fi::TokenizerResources view() const { return {tokenizer, config, generation}; }
};

Resources resources(const fs::path& model) {
    std::ifstream stream(model, std::ios::binary);
    std::array<unsigned char, 16> prefix{};
    stream.read(reinterpret_cast<char*>(prefix.data()), prefix.size());
    if (!stream || std::string_view(reinterpret_cast<const char*>(prefix.data()), 8) !=
                       std::string_view("NINFER\0\2", 8)) {
        throw std::runtime_error("expected NInfer v2");
    }
    std::uint64_t size = 0;
    for (int index = 0; index < 8; ++index) { size |= std::uint64_t(prefix[8 + index]) << (8 * index); }
    if (size > 16 * 1024 * 1024) { throw std::runtime_error("invalid directory size"); }
    std::string metadata(size, '\0');
    stream.read(metadata.data(), metadata.size());
    const Json directory = Json::parse(metadata);
    const std::uint64_t base = (16 + size + 4095) / 4096 * 4096;
    Resources result;
    for (const auto& item : directory.at("objects")) {
        std::string* destination = nullptr;
        const std::string name = item.at("name");
        if (name == "frontend/tokenizer.json") { destination = &result.tokenizer; }
        if (name == "frontend/tokenizer_config.json") { destination = &result.config; }
        if (name == "frontend/generation_config.json") { destination = &result.generation; }
        if (!destination) { continue; }
        destination->resize(item.at("bytes").get<std::size_t>());
        stream.seekg(base + item.at("offset").get<std::uint64_t>());
        stream.read(destination->data(), destination->size());
        if (!stream) { throw std::runtime_error("truncated frontend resource"); }
    }
    return result;
}

struct Encoded {
    std::vector<int> tokens;
    std::vector<int> checkpoint_tokens;
    bool checkpoint_exact = false;
    std::string error;
    bool operator==(const Encoded&) const = default;
    Json json() const {
        if (!error.empty()) { return {{"error", error}}; }
        return {{"tokens", tokens}, {"checkpoint_tokens", checkpoint_tokens},
                {"checkpoint_exact", checkpoint_exact}};
    }
    static Encoded from_json(const Json& value) {
        if (value.contains("error")) { return {{}, {}, false, value.at("error")}; }
        return {value.at("tokens").get<std::vector<int>>(),
                value.at("checkpoint_tokens").get<std::vector<int>>(),
                value.at("checkpoint_exact").get<bool>(), {}};
    }
};

Encoded encode(const fi::Tokenizer& tokenizer, std::string_view text, std::size_t checkpoint,
                 bool cached) {
    Encoded result;
    try {
#ifdef TOKENIZER_REFERENCE_ONLY
        (void)cached;
        auto ids = tokenizer.encode(text);
        auto prefix = tokenizer.encode(text.substr(0, checkpoint));
#else
        auto ids = cached ? tokenizer.encode_cached(text) : tokenizer.encode(text);
        auto prefix = cached ? tokenizer.encode_cached(text.substr(0, checkpoint))
                             : tokenizer.encode(text.substr(0, checkpoint));
#endif
        const bool exact = prefix.size() <= ids.size() &&
                           std::equal(prefix.begin(), prefix.end(), ids.begin());
        result = {std::move(ids), std::move(prefix), exact, {}};
    } catch (const std::exception& error) {
        result.error = error.what();
    }
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::cerr << "usage: tokenizer_cpu_driver MODEL CASES_JSON OUTPUT_JSON [REFERENCE_JSON] [REPEATS]\n";
            return 2;
        }
        const Resources assets = resources(argv[1]);
        const fi::Tokenizer tokenizer(assets.view());
        const Json manifest = Json::parse(read(argv[2]));
        const fs::path parent = fs::path(argv[2]).parent_path();
        Json output = {{"cases", Json::array()}};
#ifdef TOKENIZER_REFERENCE_ONLY
        output["implementation"] = "frozen original CPU tokenizer objects";
#else
        if (argc < 5) { throw std::runtime_error("candidate requires frozen reference vectors"); }
        const Json reference = Json::parse(read(argv[4]));
        output["implementation"] = "candidate plain and bounded ordinary-run cache";
        const int repeats = argc > 5 ? std::stoi(argv[5]) : 5;
        if (repeats < 1) { throw std::runtime_error("repeats must be positive"); }
        bool passed = true;
#endif
        std::size_t index = 0;
        for (const auto& item : manifest.at("cases")) {
            const std::string text = read(parent / item.at("file").get<std::string>());
            const std::size_t checkpoint = item.at("checkpoint_byte_offset");
            const auto start = Clock::now();
            Encoded actual = encode(tokenizer, text, checkpoint, false);
            const double cold = std::chrono::duration<double>(Clock::now() - start).count();
            if (actual.error.empty() && item.contains("expected_prompt_tokens") &&
                actual.tokens.size() != item.at("expected_prompt_tokens").get<std::size_t>()) {
                throw std::runtime_error("rendered case/native prompt count mismatch: " +
                                         item.at("name").get<std::string>());
            }
            Json record = {{"name", item.at("name")}, {"reference", actual.json()},
                           {"plain_first_seconds", cold}};
#ifndef TOKENIZER_REFERENCE_ONLY
            if (reference.at("cases").at(index).at("name") != item.at("name")) {
                throw std::runtime_error("reference case order differs");
            }
            const Encoded expected = Encoded::from_json(reference.at("cases").at(index).at("reference"));
            const auto cached_began = Clock::now();
            const Encoded first_cached = encode(tokenizer, text, checkpoint, true);
            record["cached_first_seconds"] =
                std::chrono::duration<double>(Clock::now() - cached_began).count();
            std::vector<double> off, on;
            bool equal = actual == expected && first_cached == expected;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                for (const bool cached : {false, true}) {
                    const auto began = Clock::now();
                    Encoded value = encode(tokenizer, text, checkpoint, cached);
                    const double seconds = std::chrono::duration<double>(Clock::now() - began).count();
                    equal = equal && value == expected;
                    (cached ? on : off).push_back(seconds);
                }
            }
            std::sort(off.begin(), off.end());
            std::sort(on.begin(), on.end());
            record["exact_old_tokens_and_checkpoint"] = equal;
            record["uncached_median_seconds"] = off[off.size() / 2];
            record["cached_median_seconds"] = on[on.size() / 2];
            record["uncached_seconds"] = off;
            record["cached_seconds"] = on;
            passed = passed && equal;
#endif
            output["cases"].push_back(std::move(record));
            ++index;
        }
#ifndef TOKENIZER_REFERENCE_ONLY
        const auto stats = tokenizer.cache_stats();
        output["passed"] = passed;
        output["cache"] = {{"capacity_bytes", stats.capacity_bytes},
                             {"managed_bytes", stats.managed_bytes}, {"entries", stats.entries},
                             {"hits", stats.hits}, {"misses", stats.misses},
                             {"evictions", stats.evictions}, {"bypasses", stats.bypasses}};
#endif
        std::ofstream stream(argv[3]);
        stream << output.dump(2) << '\n';
        if (!stream) { throw std::runtime_error("cannot write result"); }
        std::cout << "qualified " << index << " CPU cases\n";
#ifndef TOKENIZER_REFERENCE_ONLY
        return passed ? 0 : 1;
#else
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
