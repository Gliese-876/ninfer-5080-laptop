// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "targets/qwen3_6/impl/frontend/tokenizer_test_access.h"

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fi = ninfer::targets::qwen3_6::frontend_internal;
using Json = nlohmann::json;

namespace {

std::string utf8(int cp) {
    if (cp < 128) { return std::string(1, static_cast<char>(cp)); }
    if (cp < 2048) {
        return {static_cast<char>(192 | (cp >> 6)), static_cast<char>(128 | (cp & 63))};
    }
    return {static_cast<char>(224 | (cp >> 12)), static_cast<char>(128 | ((cp >> 6) & 63)),
            static_cast<char>(128 | (cp & 63))};
}

// Complete byte alphabet with byte-value IDs, plus two explicit ranked BPE merges.
// Golden IDs below are hand specified, independent of both cached/plain encode paths.
struct Fixture {
    std::string tokenizer, config, generation = R"({"eos_token_id":[1001]})";
    explicit Fixture(bool long_first = false) {
        Json vocab = Json::object();
        int extra = 256;
        for (int byte = 0; byte < 256; ++byte) {
            const bool identity = (byte >= 33 && byte <= 126) ||
                                  (byte >= 161 && byte <= 172) || byte >= 174;
            vocab[utf8(identity ? byte : extra++)] = byte;
        }
        vocab["ab"] = 300;
        vocab["abc"] = 301;
        Json added = Json::array();
        const std::array<std::string, 6> content{
            "<|im_start|>", "<|im_end|>", "<think>", "</think>",
            long_first ? "<A>B" : "<A>", long_first ? "<A>" : "<A>B"};
        Json decoder = Json::object();
        for (std::size_t index = 0; index < content.size(); ++index) {
            Json item = {{"id", 1000 + index}, {"content", content[index]},
                         {"single_word", false}, {"lstrip", false}, {"rstrip", false},
                         {"normalized", false}, {"special", index < 2}};
            added.push_back(item);
            item.erase("id");
            decoder[std::to_string(1000 + index)] = item;
        }
        tokenizer = Json{{"model", {{"type", "BPE"}, {"vocab", vocab},
                                     {"merges", Json::array({Json::array({"a", "b"}),
                                                               Json::array({"ab", "c"})})}}},
                         {"added_tokens", added}}.dump();
        config = Json{{"added_tokens_decoder", decoder}}.dump();
    }
    fi::TokenizerResources resources() const { return {tokenizer, config, generation}; }
};

int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::cerr << "FAIL " << message << '\n'; ++failures; }
}

struct Result {
    std::vector<int> ids;
    std::string error;
    bool operator==(const Result&) const = default;
};

Result encode(const fi::Tokenizer& tokenizer, const std::string& input, bool cached,
                 bool parse_added = true) {
    try {
        return {cached ? tokenizer.encode_cached(input, {.parse_added_tokens = parse_added})
                       : tokenizer.encode(input, {.parse_added_tokens = parse_added}), {}};
    } catch (const std::invalid_argument& error) {
        return {{}, error.what()};
    }
}

void golden_and_boundary() {
    Fixture fixture;
    fi::Tokenizer tokenizer(fixture.resources(), {.min_run_bytes = 1});
    check(tokenizer.encode_cached("abc") == std::vector<int>{301}, "handwritten abc BPE golden");
    check(tokenizer.encode_cached("ab") == std::vector<int>{300}, "handwritten ab BPE golden");
    check(tokenizer.encode_cached("abc") == std::vector<int>{301}, "append changes earlier merge");
    check(tokenizer.encode_cached("e") == std::vector<int>{101}, "handwritten ASCII golden");
    check(tokenizer.encode_cached("e\xcc\x81") == std::vector<int>({195, 169}),
          "NFC starter must recompute when combining suffix arrives");
    check(tokenizer.encode_cached("<A>B") == std::vector<int>({1004, 66}),
          "added token first-candidate priority must remain short first");
    check(tokenizer.encode_cached("<think>") == std::vector<int>{1002},
          "special=false added token still resets ordinary BPE");
    Fixture other(true);
    fi::Tokenizer long_first(other.resources(), {.min_run_bytes = 1});
    check(long_first.encode_cached("<A>B") == std::vector<int>{1004},
          "other model instance has independent longer-first tokenizer");
    check(tokenizer.encode_cached(std::string("a\0b", 3)) == std::vector<int>({97, 0, 98}),
          "embedded NUL is part of the exact raw key");

    const std::vector<std::string> cases{
        "", "abc", "abcabc", " ab", "a  b", "a \t\r\n b", "'RE", "I'm", "12345",
        "e\xcc\x81", "\xc3\xa9", "a\xcc\x81\xcc\xa7", "a\xcc\xa7\xcc\x81",
        "\xe1\x84\x80\xe1\x85\xa1\xe1\x86\xa8", "\xe4\xb8\xad\xe6\x96\x87",
        "\xf0\x9f\x91\xa9\xe2\x80\x8d\xf0\x9f\x92\xbb", "<|im_start|>abc<|im_end|>",
        "<think>abc</think>", "<A>B<A>", "<|im_sta", "<|im_start|>",
        std::string("a\0b", 3), std::string("\xc3", 1), std::string("\x80", 1),
        std::string("\xc0\xaf", 2), std::string("\xed\xa0\x80", 3),
        std::string("\xf4\x90\x80\x80", 4), "abc<|im_end|>\xc3"};
    for (const auto& text : cases) {
        for (std::size_t cut = 0; cut <= text.size(); ++cut) {
            (void)encode(tokenizer, text.substr(0, cut), true);
            check(encode(tokenizer, text, true) == encode(tokenizer, text, false),
                  "warm arbitrary byte prefix must equal full uncached input/error");
        }
        check(encode(tokenizer, text, true, false) == encode(tokenizer, text, false, false),
              "parse_added_tokens=false cannot share incompatible marker encoding");
    }
    std::string long_marks = "e";
    for (int index = 0; index < 1024; ++index) { long_marks += "\xcc\x81"; }
    check(encode(tokenizer, long_marks, true) == encode(tokenizer, long_marks, false),
          "unbounded combining tail cannot use a fixed rollback heuristic");
    check(tokenizer.cache_stats().hits > 0, "boundary tests must actually exercise cache hits");
}

void collisions_eviction_and_limits() {
    Fixture fixture;
    fi::Tokenizer tokenizer(fixture.resources(),
                            {.capacity_bytes = 4096, .max_entries = 2,
                             .max_entry_bytes = 2048, .min_run_bytes = 1});
    fi::TokenizerCacheTestAccess::force_hash_collisions(tokenizer);
    const std::array<std::string, 3> keys{"ab", "ac", std::string("a\0", 2)};
    for (int pass = 0; pass < 5; ++pass) {
        for (const auto& text : keys) {
            check(encode(tokenizer, text, true) == encode(tokenizer, text, false),
                  "equal hash and equal key length require exact raw byte equality");
        }
    }
    auto stats = tokenizer.cache_stats();
    check(stats.entries <= 2 && stats.evictions > 0 && stats.managed_bytes <= 4096,
          "entry eviction and allocator budget must hold");
    const std::string oversized(3000, 'a');
    check(encode(tokenizer, oversized, true) == encode(tokenizer, oversized, false),
          "oversize cache bypass preserves valid encoding");
    check(tokenizer.cache_stats().bypasses > 0, "oversized run must actually bypass admission");
    fi::Tokenizer byte_limited(fixture.resources(),
                               {.capacity_bytes = 4096, .max_entries = 64,
                                .max_entry_bytes = 2048, .min_run_bytes = 1});
    for (int key = 0; key < 10; ++key) {
        const std::string text(128, static_cast<char>('a' + key));
        check(encode(byte_limited, text, true) == encode(byte_limited, text, false),
              "byte-pressure eviction preserves exact output");
        check(byte_limited.cache_stats().managed_bytes <= 4096, "actual allocator budget exceeded");
    }
    check(byte_limited.cache_stats().evictions > 0 && byte_limited.cache_stats().entries < 64,
          "byte limit must evict before entry limit is reached");
    std::string mutable_input(128, 'x');
    (void)byte_limited.encode_cached(mutable_input);
    mutable_input.assign(128, 'y');
    check(encode(byte_limited, mutable_input, true) == encode(byte_limited, mutable_input, false),
          "cache keys must own their bytes after caller storage changes");
    fi::Tokenizer disabled(fixture.resources(), {.capacity_bytes = 0, .min_run_bytes = 1});
    check(encode(disabled, "abc", true) == encode(disabled, "abc", false) &&
              disabled.cache_stats().managed_bytes == 0 && disabled.cache_stats().entries == 0,
          "disabled internal cache has exact off semantics without product options");
    fi::Tokenizer tiny(fixture.resources(),
                       {.capacity_bytes = 1, .max_entries = 1, .max_entry_bytes = 1, .min_run_bytes = 1});
    check(encode(tiny, "abc", true) == encode(tiny, "abc", false) &&
              tiny.cache_stats().managed_bytes == 0,
          "failed cache admission must not reject or leak memory");
}

void concurrency() {
    Fixture fixture;
    fi::Tokenizer tokenizer(fixture.resources(),
                            {.capacity_bytes = 4096, .max_entries = 3,
                             .max_entry_bytes = 2048, .min_run_bytes = 1});
    const std::array<std::string, 8> texts{
        "abc", "ab", " ac", "e\xcc\x81", "<think>abc</think>", "\xe4\xb8\xad\xe6\x96\x87",
        std::string("a\0b", 3), std::string("\xc3", 1)};
    std::array<Result, 8> expected;
    for (std::size_t index = 0; index < texts.size(); ++index) {
        expected[index] = encode(tokenizer, texts[index], false);
    }
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 8; ++worker) {
        threads.emplace_back([&, worker] {
            for (int repeat = 0; repeat < 96; ++repeat) {
                const std::size_t index = (repeat + worker) % texts.size();
                if (encode(tokenizer, texts[index], true) != expected[index]) { ++errors; }
                const auto stats = tokenizer.cache_stats();
                if (stats.entries > 3 || stats.managed_bytes > 4096) { ++errors; }
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    check(errors == 0, "concurrent hits/misses/eviction preserve exact outputs and budget");
}

} // namespace

int main() {
    golden_and_boundary();
    collisions_eviction_and_limits();
    concurrency();
    if (failures == 0) { std::cout << "exact tokenization cache CPU contracts passed\n"; }
    return failures == 0 ? 0 : 1;
}
