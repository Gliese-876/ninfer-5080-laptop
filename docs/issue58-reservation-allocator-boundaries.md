# Issue #58 — reservation vs allocator: KV admission boundary facts

Task `ISSUE58_SCHEDULER_ROOT_CAUSE`. Telemetry/evidence instrumentation only;
scheduling, reservations, allocator, FIFO/protection, and live ownership are
**unchanged**. This document is evidence, not a root-cause verdict.

## 1. Evidence identity correction (demonstrated correlation gap)

In the protected-backfill path, `admit_planned_request`'s retained-eviction
re-check loop and final feasibility re-check previously emitted
`[ADMISSION-KV-DENIAL]` evidence with the **protected head's** request id
(`protection_->head_request_id`) whenever `protection_` was active. The
admission under probe, however, was the **backfill candidate's** own request.
Consequence: a denial attributed to the head id is causally unjoinable to the
head's admission events (the head was already admitted) and mis-joins the
candidate's denials onto the head's chronology.

Fix (smallest possible, telemetry-only): those two call sites now pass
`request->id` — the probing candidate's id when true. `can_admit_lane` /
`can_admit_lane_after_retained_eviction` treat `head_request_id == 0` as the
legacy case: the `head=` field is **omitted** and the line is explicit
unjoinable/legacy (byte-identical to pre-head-id output). Cross-file and
chronology join guards in `issue58_c2_trace_audit.py` are preserved:
- legacy lane-only lines are never joined (explicit unjoinable/legacy);
- a head-id-bearing denial joins a selected-admission line only when BOTH
  share the same request id AND the same lane (different id or lane never joins);
- a join is a shared-id association, never a causal proof (sampling/throttling
  applies).

No scheduling, reservation, eviction, or capacity behavior depends on the id
value: `head_request_id` is read only inside `issue58_log_kv_denial`
(`program_impl.h`), which formats the diagnostic line and never feeds the
predicate.

## 2. Reservation vs allocator boundary facts (host tests)

`issue58_kv_denial(old, reclaimable, new, entitled, logical, physical)`
(`src/runtime/engine/issue58_kv_denial.h`) is a `constexpr` pure predicate —
the first-failed-guard classifier of the admission numeric guards. The new
host test `tests/test_issue58_kv_reservation_boundary.cpp`
(ctest `ninfer_issue58_kv_reservation_boundary_test`) pins:

- **100/80/10/30 counterexample** — the demonstrated denial class: old
  entitlement 100 == pool entitlement, reclaimable 80 > entitled-old (0),
  candidate requests 10, logical 300, allocator 30 physical pages. The
  reclaimable-entitlement guard fires first; a fitting 10-page request is
  denied by reservation-vs-allocator accounting.
- **Allocator discriminant** — identical reservation (old 10, entitled 100,
  reclaimable 20 → committed 70) against a 69-page allocator is denied
  (`insufficient_physical_pages`); the same reservation against a 300-page
  allocator is accepted (`none`). Same reservation, different allocator bound:
  the telemetry must be able to separate the two denial identities.
- **Pool boundaries** — reclaimable > entitled-old →
  `invalid_reclaimable_entitlement`; old > entitled →
  `invalid_old_entitlement`; requested > logical →
  `exceeds_logical_capacity`; exact fit (committed == physical, new 0) fits;
  one page past exact fit fires `insufficient_physical_pages`.
- **Allocator boundaries** — exact fit `entitled == physical` accepts; one
  page past the allocator bound is denied (`insufficient_physical_pages`).

Caveat recorded as evidence, not fixed (out of scope): the
`entitled - old - reclaimable` committed computation is `std::uint32_t`, so an
over-claiming reservation silently wraps instead of clamping, and the
first-failed guard then names `insufficient_physical_pages` (on the wrapped
value) instead of the true reclaimable-entitlement defect. The telemetry
identity fix in §1 is what lets future captures attribute such denials to the
probing request; changing the predicate or clamping behavior is
**not** authorized by this task.

## 3. What this does NOT prove

- The Issue #58 **scheduling** root cause (starvation/tail-latency mechanism)
  **remains unproven**. This PR fixes evidence identity and pins the
  reservation/allocator boundary semantics; it does not identify, and does not
  claim to identify, the scheduling defect.
- No scheduling, reservation, allocator, FIFO/protection, or ownership change
  is made; the telemetry parameter is consumed only by the diagnostic logger.