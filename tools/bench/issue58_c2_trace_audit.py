#!/usr/bin/env python3
"""Offline audit of Issue #58 diagnostic stderr captures.

Legacy lane-only KV denial lines (no shared request id) are NOT causally joined; a causal
join to a selected-admission event is claimed only when BOTH events carry the SAME head
request id (and same lane, when the denial line names a lane). Head-id-bearing denial
lines may still be sampled/throttled, so a join is an association, not a proof of the
denial's originating request.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

PATTERNS = {
    "queue": re.compile(r"^\[ADMISSION-QUEUE-TIMING\] id=(\d+) lane=(\d+) queue_wait_ms=(\d+) event=selected_for_admission$"),
    "lane": re.compile(r"^\[ADMISSION-LANE\] head=(\d+) lane=(\d+) result=([a-z_]+)$"),
    "deferral": re.compile(r"^\[ADMISSION-DEFERRAL\] head=(\d+) queued=(\d+) reason=([a-z_]+)$"),
    "kv": re.compile(r"^\[ADMISSION-KV-DENIAL\](?: head=(\d+))? lane=(\d+) path=([a-z_]+) pool=(main|backend) cause=([a-z_]+) old=(\d+) reclaimable=(\d+) requested=(\d+) entitled=(\d+) logical=(\d+) physical=(\d+)$"),
}
QUEUE = re.compile(
    r"^\[ADMISSION-TRACE\] head=(\d+) queued=(\d+) active=(\d+) reason=([a-z_]+) "
    r"head_pages_main=(\d+) head_pages_backend=(\d+) "
    r"used_pages_main=(\d+) used_pages_backend=(\d+) used_lanes=(\d+) "
    r"capacity_pages_main=(\d+) capacity_pages_backend=(\d+) capacity_lanes=(\d+) "
    r"deadline_remaining_ms=(-?\d+) protection_epoch=(\d+) protection_phase=(none|drain|open)$"
)
VALID_LANES = {"occupied", "plan_unavailable", "direct_admittable", "retained_eviction_admittable", "not_admittable"}
VALID_REASONS = {"short_lane_policy_hold", "no_vacant_lane", "vacant_lane_not_admittable"}
VALID_CAUSES = {"invalid_old_entitlement", "invalid_reclaimable_entitlement", "exceeds_logical_capacity", "insufficient_physical_pages"}

def audit(paths, require_evidence=False):
    counts = Counter()
    errors = []
    observed = []
    kv_denials = []
    queue_values = []
    gate = []
    for path in paths:
        records = 0
        with path.open(encoding="utf-8", errors="replace") as stream:
            for index, raw in enumerate(stream, 1):
                line = raw.rstrip("\r\n")
                # journald prefixes are not silently stripped: only exact raw stderr is accepted.
                if not line.startswith("[ADMISSION-"):
                    continue
                found = False
                for name, pattern in PATTERNS.items():
                    match = pattern.fullmatch(line)
                    if match is None:
                        continue
                    found = True
                    fields = match.groups()
                    if name == "queue":
                        rid, lane, ms = map(int, fields)
                        queue_values.append(ms)
                        observed.append({"source": str(path), "line": index, "kind": "selected_admission",
                                         "request_id": rid, "lane": lane, "queue_wait_ms": ms})
                        records += 1
                    elif name == "lane":
                        if fields[2] not in VALID_LANES:
                            errors.append(f"{path}:{index}: invalid lane result")
                        counts["lane_" + fields[2]] += 1
                    elif name == "deferral":
                        if fields[2] not in VALID_REASONS:
                            errors.append(f"{path}:{index}: invalid policy reason")
                        counts["deferral_" + fields[2]] += 1
                    elif name == "kv":
                        has_head = fields[0] is not None
                        head = fields[0]
                        fields = fields[1:]  # strip optional head field; lane first
                        if head is None: head = 0
                        if fields[3] not in VALID_CAUSES:
                            errors.append(f"{path}:{index}: invalid KV denial cause")
                        counts["kv_" + fields[3]] += 1
                        counts["kv_with_head_id" if has_head else "kv_lane_only"] += 1
                        kv_denials.append({"source": str(path), "line": index, "kind": "kv_denial",
                                           "request_id": int(head), "lane": int(fields[0]),
                                           "lane_id_joinable": has_head,
                                           "path": fields[1], "pool": fields[2], "cause": fields[3]})
                    counts[name] += 1
                    records += 1
                    break
                if not found:
                    match = QUEUE.fullmatch(line)
                    if match:
                        if match.group(4) not in VALID_REASONS:
                            errors.append(f"{path}:{index}: invalid summary reason")
                        counts["summary_" + match.group(4)] += 1
                        counts["summary"] += 1
                        records += 1
                    else:
                        errors.append(f"{path}:{index}: unrecognised admission trace format")
        if require_evidence and records == 0:
            gate.append(str(path))
    queue_values.sort()
    def percentile(p):
        if not queue_values:
            return None
        pos = (len(queue_values) - 1) * p
        low = int(pos)
        high = min(low + 1, len(queue_values) - 1)
        return queue_values[low] + (queue_values[high] - queue_values[low]) * (pos - low)
    return {
        "schema": "issue58-c2-offline-trace-audit-v1",
        "files": [str(p) for p in paths],
        "counts": dict(sorted(counts.items())),
        "selected_admission_queue_ms": {
            "n": len(queue_values), "p50": percentile(.5), "p95": percentile(.95),
            "max": max(queue_values) if queue_values else None,
        },
        "selected_admissions": observed,
        "kv_denials": kv_denials,
        "kv_denial_joins": [
            {"kv_line": d["source"] + ":" + str(d["line"]),
             "request_id": a["request_id"],
             "admission_line": a["source"] + ":" + str(a["line"]),
             "lane": d["lane"],
             "note": "shared request id + lane; sampling/throttling applies, no causal proof",
            }
            for d in kv_denials
            if d["lane_id_joinable"]
            for a in observed
            if a["request_id"] == d["request_id"]
            and (d["lane"] == a["lane"] or a["lane"] is None)
        ],
        "unrecognised_or_invalid_lines": errors,
        "no_admission_evidence_files": gate,
        "limitations": [
            "Request id in selected_admission logs is scheduler-local; never join across sessions/restarts without a verified common namespace.",
            "KV denial logs have lane but NO request id; per-lane and admission traces are sampled; no causal join is claimed for lane-only denials.",
            "A kv_denial_joins entry joins a head-id-bearing KV denial line to a selected-admission line that share the SAME request id and lane; this is a shared-id association, not causal proof (sampling/throttling applies).",
            "A head id never proves the denial's true originating request; denials are logged from the probing request's perspective at the admission lane.",
            "Selected admission durations exclude rejected, cancelled, expired and never-admitted requests (survivor bias).",
            "No GPU-start time, first-token time, GPU schedule ownership or end-to-end TTFT is measured here.",
            "The report cannot establish starvation root cause or authorize C2 rollout.",
            "Admission traces are emitted only while a FIFO head is blocked (10s throttle); zero records also occur for healthy short runs and for dead/crashed processes; the no-evidence gate never diagnoses the process, it only demands re-examination.",
        ],
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stderr", type=Path, nargs="+", help="Previously saved raw stderr files; never a live service")
    parser.add_argument("--require-evidence", action="store_true",
        help="fail closed (rc 7) when an input file contains no recognized admission records")
    args = parser.parse_args(argv)
    report = audit(args.stderr, require_evidence=args.require_evidence)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_evidence and report["no_admission_evidence_files"]:
        return 7
    return int(bool(report["unrecognised_or_invalid_lines"]))

if __name__ == "__main__":
    raise SystemExit(main())
