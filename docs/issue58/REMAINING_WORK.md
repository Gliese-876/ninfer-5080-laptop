# Issue #58 remaining work

C2 credit refresh is closed as rejected. This does **not** close Issue #58.

## Still-open engineering objectives

1. **Reversible KV/MTP lane yield.** Implement only at a proven safe boundary; preserve request lifecycle, cancellation, deadline, deterministic sampling, MTP/recurrent/hidden state, and prefix identity. Do not use abort/truncation as yield.
2. **Physical KV export/import.** Establish page ownership, lifetime, stream synchronization, allocation/release/reacquire and restoration correctness before claiming transfer works.
3. **Shared-executor validation.** Exercise real executor ownership, cancellation, drain, retained-state eviction, and pressure/lane transitions with discriminating host fixtures.
4. **GPU qualification.** Only after host evidence and explicit authorization: validate C2 pressure, MTP output equivalence, cancellation, capacity, and service restoration.
5. **Telemetry writer/lifetime work.** Separate scope. Existing stalled-sink/lifecycle concerns are retained as risks, not accepted fixes.

## Current evidence boundaries

- KV entitlement denial is not the same as physical-lane occupancy. Small requests can fit a vacant lane while a larger entitlement does not.
- The rejected C2 credit candidate is not evidence of a scheduler fix.
- Existing GPU evidence documents lane-occupancy behavior and mitigation experiments; it does not validate true reversible preemption.

## Handoff rule

Read [C2 decision record](C2_DECISION_RECORD.md) and [evidence manifest](EVIDENCE_MANIFEST.md) before any new Issue #58 scheduler work. Start from an exact current repository HEAD, preserve all archival worktrees, and define a real workload-level acceptance predicate before changing admission code.
