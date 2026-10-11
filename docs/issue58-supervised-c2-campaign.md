# Issue #58 — C2 diagnostic campaign (draft)

This branch proposes a staged, supervised C2 test. No live execution is authorised by merging or validating this branch. Production C1 port 8080 must remain the supported endpoint.

## Phase A: safe host validation

OpenClaw MAIN must fetch exact HEAD; inspect all scripts for subprocess, signal, timeout, watchdog, lock and failure-path correctness. Run `bash -n`, `python3 -m py_compile`, and `git diff --check`. Add deterministic mocked tests for abort-before-interruption and C1 restore behaviour. Build a distinct `ninfer-serve` from pinned main with ccache and existing CUDA toolchain into a separate worktree/build directory; record commit, binary SHA256, compiler versions and CMake launchers. No model or GPU execution, service changes or port usage during Phase A. Do not use the historical candidate binary.

## Phase B: separately approved outage

After Phase A passes, request **explicit consent for the production interruption**. MAIN may then invoke `tools/bench/issue58_c2_supervised_campaign.sh` with `ISSUE58_APPROVED_OUTAGE=YES`, an independently staged binary path and a verified SHA256. The runner performs C1 health and port preflight, arms independent root watchdog, stops C1 using approved wrapper, launches C2 only at localhost:18080, bounds startup and request durations, captures raw stderr and per-request outcomes, stops C2, starts C1 and verifies HTTP200 before disarming the watchdog. Preserve all artifacts and report restoration failures immediately.

The client uses a bounded synthetic long/long/short sequence. Offline parser reports selected-admission queue residence, but sampled KV logs lack request identity; no causal join or C2 production qualification is claimed. Preserve failures and cancellations rather than reporting admitted-only statistics as overall latency. Do not deploy, change scheduler policy, close Issue #58 or conflate this short probe with the previous 75-job corpus. If any guard fails, stop with evidence and leave C1 alone.
