// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#pragma once

#include "targets/qwen3_6/impl/frontend/tokenizer.h"

namespace ninfer::targets::qwen3_6::frontend_internal {

// Private CPU-test hook: exercise exact key comparisons under deliberate hash collisions.
struct TokenizerCacheTestAccess {
    static void force_hash_collisions(Tokenizer& tokenizer);
};

} // namespace ninfer::targets::qwen3_6::frontend_internal
