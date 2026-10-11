#include "runtime/engine/issue58_kv_denial.h"

#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using namespace ninfer::runtime;
    // Argument order: (old, reclaimable, new, entitled, logical, physical).

    int failures = 0;
    // Issue #58 counterexample (100/80/10/30): a lane's retained old entitlement
    // equals the pool entitlement (100), reclaimable accounting claims 80 more,
    // the candidate requests 10, logical capacity 300, the allocator holds 30
    // physical pages. the reclaimable-entitlement guard rejects 80 > 100 - 100 before any unsigned subtraction;
    // the first failing guard is InvalidReclaimableEntitlement (80 > 100 - 100):
    // a fitting 10-page request is denied by reservation-vs-allocator
    // accounting, the demonstrated denial class.
    failures += check(
        issue58_kv_denial(100u, 80u, 10u, 100u, 300u, 30u) ==
            Issue58KvDenial::InvalidReclaimableEntitlement,
        "100/80/10/30 counterexample did not reproduce the demonstrated denial");

    // Allocator discriminant: identical reservation accounting (old 10,
    // entitled 100, reclaimable 20 -> committed 70) against different allocator
    // bounds: a 69-page allocator denies the 10-page request (committed 70 exceeds physical 69); a 300-page allocator accepts it. Same reservation, different
    // allocator bound -> denied, then accepted: the reservation-vs-allocator
    // identity the telemetry must be able to separate.
    failures += check(
        issue58_kv_denial(10u, 20u, 10u, 100u, 300u, 69u) ==
            Issue58KvDenial::InsufficientPhysicalPages,
        "69-page allocator did not deny the fitting-by-entitlement request");
    failures += check(
        issue58_kv_denial(10u, 20u, 10u, 100u, 300u, 300u) == Issue58KvDenial::None,
        "300-page allocator did not accept the fitting-by-entitlement request");

    // Pool boundary (reclaimable may not exceed the uncommitted entitlement):
    // old 10, entitled 10 -> 11 reclaimable > 0 entitlement -> InvalidReclaimable.
    failures += check(
        issue58_kv_denial(10u, 11u, 0u, 10u, 300u, 300u) ==
            Issue58KvDenial::InvalidReclaimableEntitlement,
        "pool entitlement boundary did not flag reclaimable exceeding entitled-old");

    // Exact-fit pool boundary: committed (entitled - old - reclaimable) ==
    // physical (no slack), requested 0 fits.
    failures += check(
        issue58_kv_denial(9u, 0u, 0u, 10u, 300u, 10u) == Issue58KvDenial::None,
        "exact-fit pool boundary (committed==physical, new=0) denied a fitting plan");
    // One page past the pool exact-fit boundary: old 11 > entitled 10 fires the
    // InvalidOldEntitlement guard (first in chain).
    failures += check(
        issue58_kv_denial(11u, 0u, 1u, 10u, 300u, 10u) ==
            Issue58KvDenial::InvalidOldEntitlement,
        "one page past the pool exact-fit boundary was not denied");

    // Exact-fit allocator (reservation-vs-allocator) boundary: entitled 10
    // (== physical 10), old 0 -> committed 10 == physical; requested 0 fits
    // exactly; 1 does not.
    failures += check(
        issue58_kv_denial(0u, 0u, 0u, 10u, 300u, 10u) == Issue58KvDenial::None,
        "exact-fit allocator boundary (entitled==physical) denied a fitting plan");
    // One page past the allocator exact-fit boundary: physical 10, requested
    // 1 > 10 - committed(10) = 0 fires the new_pages bound.
    failures += check(
        issue58_kv_denial(0u, 0u, 1u, 10u, 300u, 10u) ==
            Issue58KvDenial::InsufficientPhysicalPages,
        "one page past the allocator exact-fit boundary was not denied");

    // Pool boundary: requested exceeds logical capacity.
    failures += check(
        issue58_kv_denial(0u, 0u, 301u, 10u, 300u, 300u) ==
            Issue58KvDenial::ExceedsLogicalCapacity,
        "logical-capacity boundary did not flag an over-capacity request");

    // Pool boundary: old entitlement above the pool entitlement.
    failures += check(
        issue58_kv_denial(11u, 0u, 1u, 10u, 300u, 300u) ==
            Issue58KvDenial::InvalidOldEntitlement,
        "old-entitlement boundary did not flag an over-entitled lane");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}