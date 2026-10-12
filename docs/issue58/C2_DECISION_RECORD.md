# Issue #58 C2 decision record — protected-head credit candidate

**Status:** archived decision, 2026-10-12.  **Baseline:** `2c3510cd74dea0983e146dc2c3cf650d58ddd4fc`.

## Decision

`C2_CANDIDATE=REJECTED`. The dirty `refresh_temporal_credit` experiment is preserved in its original worktree and must not be merged, revived, or treated as a scheduler fix without new contradictory evidence.

`BASELINE_STARVATION=NOT_REPRODUCED` means only that the available evidence did not contain an authentic original failing request sequence. It is **not** proof that the full serving system is starvation-free.

## Candidate and evidence

The candidate intended to refresh temporal admission credit as a donor progressed. Review found that it reconstructs the donor-excluded future ledger already required to fit when protection is created and during persistent admission. The ledger therefore fits in normal valid states, so the refresh returns zero and overwrites valid epoch credit.

The retained synthetic policy probes are discriminating against this candidate, but are **not** reproductions of a user-visible failure:

| Probe | Result | Meaning |
|---|---:|---|
| original policy | RC 0; `credit=20`, `temporal_admit=1` | legal temporal backfill accepted |
| candidate policy | RC 1; `credit=0`, `temporal_admit=0` | same legal backfill rejected |

The candidate is more restrictive; that does not establish a capacity or KV-overcommit improvement.

## Assumptions and uncertainty

The admission-path analysis assumes donor progress through decode rounds. Decode scheduling, prefill interleave, GPU execution, and stalls outside admission were not tested by the synthetic probes and remain unproven. No authentic old-baseline head-starvation request sequence was present in the permitted concise artifacts, so no real-executor fixture was run.

## Reopen C2 only when

1. An authoritative old-baseline request sequence/trace demonstrates protected-head delay beyond the claimed donor-frontier bound; or
2. A new, approved temporal-credit invariant specifies whether credit may replenish and proves borrowers cannot extend protected-head delay.

Then run one bounded real-executor CPU fixture at the exact baseline before altering scheduler code.

## Lessons and avoided approaches

- Do not infer a user-visible scheduler failure from an arithmetic/policy probe.
- Do not revive credit refresh/replenishment merely because it seems fairer; it changes the anti-delay budget and needs a stated invariant plus a failing baseline trace.
- Do not expand telemetry to compensate for a missing reproduction. Telemetry writer/lifecycle work is separate and has its own blocking and lifetime risks.
- Preserve failed candidates and their hashes; do not “repair” their tests to create a success narrative.

See [evidence manifest](EVIDENCE_MANIFEST.md) and [remaining work](REMAINING_WORK.md).
