// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
};

// Internal host encoding policy. No request/CLI option changes model input semantics.
struct TokenizationCacheOptions {
    std::size_t capacity_bytes = 8 * 1024 * 1024;
    std::size_t max_entries = 512;
    std::size_t max_entry_bytes = 2 * 1024 * 1024;
    std::size_t min_run_bytes = 256;
};

struct TokenizationCacheStats {
    std::size_t capacity_bytes = 0;
    std::size_t managed_bytes = 0;
    std::size_t entries = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t evictions = 0;
    std::size_t bypasses = 0;
};

struct TokenizerCacheTestAccess;

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources, TokenizationCacheOptions cache = {});
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Uncached exact reference route remains available for decisions, media and CPU qualification.
    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    std::vector<int> encode_cached(std::string_view text, EncodeOptions options = {}) const;
    [[nodiscard]] TokenizationCacheStats cache_stats() const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    std::string decode_token_bytes(int id, bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
    class OrdinaryRunCache;
    std::vector<int> encode_impl(std::string_view text, EncodeOptions options, bool cache) const;

    std::vector<std::string> id_to_token_;
    std::vector<bool> valid_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    std::unordered_map<std::string, int> bpe_merge_ranks_;
    bool has_bpe_merges_ = true;
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_;
    std::vector<int> default_stop_token_ids_;
    std::unique_ptr<OrdinaryRunCache> ordinary_run_cache_;

    friend struct TokenizerCacheTestAccess;
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
