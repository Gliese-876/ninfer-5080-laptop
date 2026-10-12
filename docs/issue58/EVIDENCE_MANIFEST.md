# Issue #58 evidence manifest

This manifest identifies the preserved external archive as **ISSUE58-C2-ARCHIVE-2026-10-12**. Current host location is an operational locator only; durable references below use artifact names, revisions, and SHA-256 values. Archive publication/storage is still required before these artifacts are portable.

## Worktree identities

| Artifact | Revision/state | Preservation status |
|---|---|---|
| immutable review baseline | `2c3510cd74dea0983e146dc2c3cf650d58ddd4fc`, clean | retained |
| C2 candidate worktree | same HEAD; 4 modified files + starvation test | retained, rejected, not reverted |
| C2 attempt-1 worktree | same HEAD; dirty source plus harness | retained |

## Decision artifacts

| Artifact ID | SHA-256 | Classification |
|---|---|---|
| `reports/issue58-c2-final-baseline-verification.md` | `7e9214b5bf7bb3da23e73035c8f45a4cd0e1f8dce6ea6b9320b02f10bc05b7a1` | verified decision report |
| `reports/issue58-c2-fix-run-blocker.md` | `8aaf5a7145ad96cd35f3a60c0f540709a1d340ab855a4cf1f5cc0a2df45789c5` | verified candidate falsification |
| `reports/issue58-c2-starvation-handoff.md` | sidecar SHA-256 verified | exploration handoff; not production qualification |
| `reports/sol-credit-validation.md` | `5332616e793f9b3b1abf1d34bdb9cc5931498c5ae30a4854946f62f70af71527` | verified host validation/blocker |
| `reviews/opus-c2-substantive-review.md` | sidecar SHA-256 verified | independent analysis; advisory |

## Reproduction commands

These commands are archival references, not a substitute for an authentic workload trace:

`timeout 20s bin/probe-old` — expected RC 0 and `temporal_admit=1`.  
`timeout 20s bin/probe-new` — expected RC 1 and `temporal_admit=0`.

The probe binaries and compiler command lines are recorded in the archive evidence bundle. A future verifier must bind them to the exact archive manifest and baseline revision before treating results as evidence.

## Other retained findings

- Admission policy and KV reservation-boundary checks passed in the SOL host bundle, but do not qualify a repaired scheduler.
- Build recovery material, CMake/test manifests, and ccache command captures are retained as recovery evidence.
- Telemetry writer experiments document blocking/stalled-sink and resource-lifetime risks; they are separate from C2 and include unaccepted implementation material.

## Preservation gaps

The external archive is not repository-tracked and remains dependent on host retention. Before moving or deleting any worktree, publish/store a content-addressed archive containing the named reports, evidence manifests, relevant patches, and this manifest; verify all sidecars from their artifact directories.
