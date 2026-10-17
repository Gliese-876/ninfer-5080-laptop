// Modified for the RTX 5080 Laptop fork; see CHANGELOG.md.
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        // Decode and the four-token MTP verifier use distinct A16 routes at the
        // model's registered [34816,5120] gate/up shape.
        constexpr std::array<std::int32_t, 8> kTokenCases{
            1, 2, 3, 4, 5, 6, 7, 8,
        };

        const int a16_failures = run_profile(
            "LinearSwiGLU Q3_A16",
            {
                QType::Q3G64_F16S,
                34816,
                5120,
                17408,
                1403U,
                ActivationCompute::A16
            },
            kTokenCases);

        std::cout
            << (a16_failures == 0 ? "OK" : "FAIL")
            << " LinearSwiGLU Q3_A16 correctness T=1..8\n";

        // Every K position of every token contributes, covering both existing
        // MMA instances and all newly active columns. Hashed Q3 rows/scales and
        // BF16 inputs feed the same FP64 formula and unchanged A16 criterion.
        constexpr std::array<std::int32_t, 6> kDenseTokenCases{2, 4, 5, 6, 7, 8};
        const int dense_failures = run_profile(
            "LinearSwiGLU Q3_A16 dense",
            {QType::Q3G64_F16S, 34816, 5120, 17408, 1409U, ActivationCompute::A16,
             InputPattern::AllDense},
            kDenseTokenCases);
        std::cout
            << (dense_failures == 0 ? "OK" : "FAIL")
            << " LinearSwiGLU Q3_A16 dense correctness T=2,4,5,6,7,8\n";

        // The production AllowA8 policy starts activation quantization at T=257;
        // T=1792 is the release prefill chunk. The shared independent FP64
        // oracle checks every output, with one dense token and up to four
        // nonzeros per later token. A changed A8 kernel additionally needs real dense-input
        // numerical qualification; these cases establish route/shape correctness.
        constexpr std::array<std::int32_t, 2> kA8TokenCases{
            257, 1792,
        };
        const int a8_failures = run_profile(
            "LinearSwiGLU Q3_A8",
            {QType::Q3G64_F16S, 34816, 5120, 17408, 1403U, ActivationCompute::A8},
            kA8TokenCases);

        std::cout
            << (a8_failures == 0 ? "OK" : "FAIL")
            << " LinearSwiGLU Q3_A8 correctness T=257,1792\n";

        return a16_failures == 0 && dense_failures == 0 && a8_failures == 0 ? 0 : 1;

    } catch (const std::exception& error) {

        std::cerr
            << "LinearSwiGLU Q3 test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}
