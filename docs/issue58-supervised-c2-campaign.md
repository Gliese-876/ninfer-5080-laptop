# Issue #58 — C2 diagnostic campaign (draft)

This branch proposes a staged, supervised C2 test. No live execution is authorised by merging or validating this branch. Production C1 port 8080 must remain the supported endpoint.

## Phase A: safe host validation

OpenClaw MAIN must fetch exact HEAD; inspect all scripts for subprocess, signal, timeout, watchdog, lock and failure-path correctness. Run `bash -n`, `python3 -m py_compile`, and `git diff --check`. Add deterministic mocked tests for abort-before-interruption and C1 restore behaviour. Build a distinct `ninfer-serve` from pinned main with ccache and existing CUDA toolchain into a separate worktree/build directory; record commit, binary SHA256, compiler versions and CMake launchers. No model or GPU execution, service changes or port usage during Phase A. Do not use the historical candidate binary.

## Phase B: separately approved outage

After Phase A passes, request **explicit consent for the production interruption**. MAIN may then invoke `tools/bench/issue58_c2_supervised_campaign.sh` with `ISSUE58_APPROVED_OUTAGE=YES`, an independently staged binary path and a verified SHA256. The runner performs C1 health and port preflight, arms independent root watchdog, stops C1 using approved wrapper, launches C2 only at localhost:18080, bounds startup and request durations, captures raw stderr and per-request outcomes, stops C2, starts C1 and verifies HTTP200 before disarming the watchdog. Preserve all artifacts and report restoration failures immediately.

The client uses a bounded synthetic long/long/short sequence. Offline parser reports selected-admission queue residence, but sampled KV logs lack request identity; no causal join or C2 production qualification is claimed. Preserve failures and cancellations rather than reporting admitted-only statistics as overall latency. Do not deploy, change scheduler policy, close Issue #58 or conflate this short probe with the previous 75-job corpus. If any guard fails, stop with evidence and leave C1 alone.

## 2026-10-11 supervised Phase B result (executed, diagnostic only)

At runner commit `d080e691e2d603eb141bda8455f928d272ef9932`, the owner ran one supervised campaign on Brain with staged binary SHA-256 `93f3402bf4e33eccedcc503d55e30ddc50aa99454d96893fe635ff99157f0f5b`. Evidence directory: `/tmp/issue58-c2-campaign.ZF8cGD9S`; the authoritative issue log is [Issue #58 execution comment](https://github.com/toddballinger/ninfer-5080/issues/58#issuecomment-6108225182).

| Case | Max-token budget | TTFT (s) | Total (s) | Stream chunks | Finish | HTTP |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| long_a | 1536 | 0.1785 | 18.38 | 672 | length | 200 |
| unhinted_budget | 16384 | 18.176 | 86.0547 | 2514 | stop | 200 |
| short_after_long | 256 | 0.2172 | 0.6358 | 12 | stop | 200 |

All three streams completed and `TRACE_AUDIT_RC=0`. The production service was stopped temporarily and restored with `PRODUCTION_RESTORED=PASS`; the root watchdog was armed and the run disarmed. These are observed client timings, not admission-delay measurements: stream TTFT includes admission, prefill and initial decode. `short_after_long` had low TTFT during outstanding work, but the unhinted request experienced 18.176s TTFT. Neither the historical two >600s C2 outliers nor C3 900s timeout are resolved by a three-case test. C1 remains the production configuration. No additional interruption is authorised solely by this result.

### Required read-only follow-up before scheduler changes

1. On Brain, verify C1 health and that the root watchdog exited; preserve the entire capture directory unchanged. Collect `client.jsonl`, `raw.stderr.log`, `trace_audit.json`, `report.log` with file SHA-256 and source/binary provenance.
2. Parse the raw scheduler trace for each observed admission selection, queue-to-selected interval, lane eligibility/deferral and KV-denial reason. Distinguish elapsed client TTFT from server admission waits. Logs lacking shared request IDs permit aggregate/temporal alignment only; **do not manufacture causal per-request joins**.
3. Reconcile each completed stream (including scheduled start offsets and finish reason) with the admission timeline; explain exactly what can and cannot be inferred about the 18.176s case. Keep incomplete/canceled events in denominators.
4. Prepare an independently reviewable **separate draft PR** for the smallest justified instrumentation or scheduling fix, with deterministic tests and an explicit proof plan against the 75-request starvation corpus. If evidence cannot isolate cause, draft telemetry/test improvements rather than guessing at scheduler changes.
5. Do not run more GPU experiments, stop production C1, change model/backend, merge PRs or close Issue #58 without a new explicit gate.

### Cross-branch dependency: strict evidence gate

The PR #78 campaign invocation now opts into `--require-evidence`, which is **not yet available in the `main` version** of `issue58_c2_trace_audit.py`. It depends on the separate Issue #58 forensic trace-audit draft PR. Until that parser option is integrated into the campaign checkout and validated together, **do not execute the campaign**: the audit would fail argument parsing even after a successful model run. No additional GPU test is authorised. Keep both PRs draft and reconcile the dependency during review.
