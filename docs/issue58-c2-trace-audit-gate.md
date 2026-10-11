# Issue #58 — C2 evidence audit and supervised candidate run gate

## Scope
This PR prepares an **offline, CPU-only** audit of diagnostic stderr records added by merged PRs #73–#76 and numerical KV-denial trace added by #75. It **does not start a C2 server or run workloads**. Production C1 port 8080 stays unchanged. This work does not establish scheduler causal root cause or C2 release readiness.

Use `tools/bench/issue58_c2_trace_audit.py` on saved **raw** stderr captures. Report retains source/line identifiers for selected-admission timing, counts exact lane/policy/KV records and reports p50/p95/max queue residence among **admitted requests only**. It does not pretend the sampled trace records share a reliable request key: KV-denial records lack request IDs; lane observations and KV causes are different probes, potentially throttled independently. Never infer causal relationships from proximity alone. Rejected, timed-out, cancelled, and never-admitted requests are excluded from queue-time distribution (survivor bias). Existing `issue58_admission_forensics.py` separately reports completed TTFT only; do not merge its samples without verified identity and capture scope.

## Host-only validation
Verify exact PR HEAD, clean worktree, Python version, and three-file scope:
```bash
python3 -m unittest tests/test_issue58_c2_trace_audit.py -v
python3 -m py_compile tools/bench/issue58_c2_trace_audit.py tests/test_issue58_c2_trace_audit.py
git diff --check main...HEAD
git diff --name-only main...HEAD
```
Report all RCs. Review parsers against actual source formats in `concurrent_executor.h` and `program_impl.h`. For a saved capture:
```bash
python3 tools/bench/issue58_c2_trace_audit.py /path/to/saved/raw.stderr.log > /tmp/issue58-trace-audit.json
```
Do not invent a capture path. No service access is required to validate the parser.

## Observed failure evidence (supervised C2 campaign, 2026-10-11T05:30:10Z)

A candidate staged-binary C2 run (capture directory `/tmp/issue58-c2-campaign.8qEZ82hD`)
passed preflight and the independent root watchdog, stopped C1, then the staged
`ninfer-serve` process died before its health endpoint came up inside the 80-second
startup window: `C2_STARTUP=FAIL`, `TEST_RC=54`, `PRODUCTION_RESTORED=PASS`. Its
`raw.stderr.log` was **0 bytes**, so the campaign gate ran
`issue58_c2_trace_audit.py` on an empty file, obtained `TRACE_AUDIT_RC=0`, and wrote a
**0-byte `trace_audit.json`** — the gate reported a clean audit while the candidate
server had produced no diagnostic evidence at all.

**Bounded inference.** The gate's only pass condition is "no unrecognised/invalid
admission lines"; that condition is vacuously true for a zero-record file. A zero-record
capture is therefore indistinguishable from a healthy run that admitted nothing (the
admission traces are sampled at a 10-second throttle while the FIFO head is blocked, so
short healthy runs also emit nothing), and the gate asserted success in a case where the
process was dead. **No causal join between request and KV state is claimed or implied**;
the failure below is a gate/evidence defect, not a scheduler-starvation verdict, and
nothing here establishes a scheduler root cause or C2 release readiness.

**Deficiency and fix.** The audit tool failed to distinguish *no evidence* from
evidence of health, which masked the startup failure (this forensic pass confirms the
masking by re-running the audit on the 0-byte log: `AUDIT_RC=0`). The tool now offers an
opt-in `--require-evidence` mode: any input file with zero recognized admission records
is listed under `no_admission_evidence_files` in the report and the process exits `7`
(fail-closed, distinct from parser-malformed `1` and zero-record `0` default mode). The
flag is off by default so every existing offline audit invocation is unchanged.

**Cross-branch dependency.** This PR adds `--require-evidence` to
`tools/bench/issue58_c2_trace_audit.py` on top of `main` (PR #77 base), while the
gated campaign runner `tools/bench/issue58_c2_supervised_campaign.sh` already exists
on the separate **PR #78** branch (head `2f73a8b6`), where `1f713648` wires that runner
to invoke the audit *with* `--require-evidence` so a capture that produced no admission
records can never be reported as a clean audit. **Do not merge PR #78 before this PR
lands on `main`**: until the flag exists in the campaign checkout, its audit step fails
argument parsing, and the campaign must not be executed.

Zero-evidence never diagnoses the process: it requires re-examination (startup, crash,
short healthy run) using the capture's other artifacts.

## Separate, explicitly authorised candidate C2 run (NOT authorised by this PR)
Before ANY candidate GPU run, require owner approval of exact separate staging binary/model build, 18080-only port (never production 8080), isolation and rollback strategy, root watchdog and C1 health checks before/after, CPU/GPU resource budgets, timeout ceilings, 128K context expectation, and specific short/long interleaved workload. Historical supervised tests stopped C1 and restored it; **do not repeat such interruption without separate approval**. No auto-run, no use of `ninfer-local-model.service` control commands here.

Candidate goals (only after approval): capture raw stderr for the same staged server instance with `NINFER_ADMISSION_TRACE=1`, full server JSONL with known instance boundary, client request schedule and exact per-request arrival/TTFT/failure outcomes. Compare C1 and C2 only with paired workloads and count all failures. Report the old 600+ second TTFT tail and false short-hint starvation separately. Diagnostic traces are sampled, do not assert a KV shortage from `not_admittable` alone.

## Next
Open a separate supervised experiment protocol PR if needed; only after one run establishes specific, reproducible denial behavior should scheduler modifications be proposed. Keep Issue #58 OPEN.
