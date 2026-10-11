import tempfile
import unittest
from pathlib import Path
from tools.bench.issue58_c2_trace_audit import audit

class TestIssue58C2TraceAudit(unittest.TestCase):
    def capture(self, content):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stderr.log"
            path.write_text(content, encoding="utf-8")
            return audit([path])

    def test_empty(self):
        r = self.capture("")
        self.assertEqual(r["selected_admission_queue_ms"]["n"], 0)
        self.assertIsNone(r["selected_admission_queue_ms"]["p95"])

    def test_selected_admission_only(self):
        r = self.capture("[ADMISSION-QUEUE-TIMING] id=7 lane=0 queue_wait_ms=125 event=selected_for_admission\n")
        self.assertEqual(r["selected_admission_queue_ms"]["p95"], 125)
        self.assertEqual(r["selected_admissions"][0]["request_id"], 7)
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_multi_trace_with_no_causal_join(self):
        data = "\n".join((
            "[ADMISSION-TRACE] head=7 queued=2 active=1 reason=vacant_lane_not_admittable "
            "head_pages_main=2 head_pages_backend=1 used_pages_main=10 used_pages_backend=3 used_lanes=1 "
            "capacity_pages_main=18 capacity_pages_backend=8 capacity_lanes=2 "
            "deadline_remaining_ms=600000 protection_epoch=0 protection_phase=none",
            "[ADMISSION-LANE] head=7 lane=0 result=occupied",
            "[ADMISSION-LANE] head=7 lane=1 result=not_admittable",
            "[ADMISSION-KV-DENIAL] lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18",
            "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission",
            "[ADMISSION-QUEUE-TIMING] id=8 lane=0 queue_wait_ms=300 event=selected_for_admission",
        )) + "\n"
        r = self.capture(data)
        self.assertEqual(r["selected_admission_queue_ms"]["p50"], 200)
        self.assertEqual(r["selected_admission_queue_ms"]["p95"], 290)
        self.assertEqual(r["counts"]["kv_insufficient_physical_pages"], 1)
        self.assertEqual(len(r["selected_admissions"]), 2)
        self.assertEqual(r["counts"]["kv_lane_only"], 1)
        self.assertEqual(r["counts"].get("kv_with_head_id", 0), 0)
        self.assertFalse(r["kv_denials"][0]["lane_id_joinable"])
        # Legacy lane-only denial: no causal join is ever claimed, even though
        # the same-lane admission event exists in the capture.
        self.assertEqual(r["kv_denial_joins"], [])
        self.assertNotIn("cause", r["selected_admissions"][0])
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_head_id_kv_denial_joins_when_shared_id_and_lane(self):
        data = "\n".join((
            "[ADMISSION-KV-DENIAL] head=7 lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18",
            "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission",
        )) + "\n"
        r = self.capture(data)
        self.assertEqual(r["counts"]["kv_with_head_id"], 1)
        self.assertTrue(r["kv_denials"][0]["lane_id_joinable"])
        self.assertEqual(len(r["kv_denial_joins"]), 1)
        self.assertEqual(r["kv_denial_joins"][0]["request_id"], 7)
        self.assertEqual(r["kv_denial_joins"][0]["lane"], 1)
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_head_id_kv_denial_no_join_on_id_mismatch(self):
        data = "\n".join((
            "[ADMISSION-KV-DENIAL] head=9 lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18",
            "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission",
        )) + "\n"
        r = self.capture(data)
        self.assertEqual(r["kv_denial_joins"], [])
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_head_id_kv_denial_no_join_on_lane_mismatch(self):
        data = "\n".join((
            "[ADMISSION-KV-DENIAL] head=7 lane=0 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18",
            "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission",
        )) + "\n"
        r = self.capture(data)
        self.assertEqual(r["kv_denial_joins"], [])
        self.assertFalse(r["unrecognised_or_invalid_lines"])

    def test_no_join_across_files_with_reused_id(self):
        denial = "[ADMISSION-KV-DENIAL] head=7 lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18\n"
        admission = "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission\n"
        with tempfile.TemporaryDirectory() as td:
            first, second = Path(td) / "run1.log", Path(td) / "run2.log"
            first.write_text(denial, encoding="utf-8")
            second.write_text(admission, encoding="utf-8")
            self.assertEqual(audit([first, second])["kv_denial_joins"], [])

    def test_no_join_when_denial_occurs_after_admission(self):
        admission = "[ADMISSION-QUEUE-TIMING] id=7 lane=1 queue_wait_ms=100 event=selected_for_admission"
        denial = "[ADMISSION-KV-DENIAL] head=7 lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18"
        r = self.capture(admission + "\n" + denial + "\n")
        self.assertEqual(r["kv_denial_joins"], [])

    def test_zero_head_is_not_joinable(self):
        denial = "[ADMISSION-KV-DENIAL] head=0 lane=1 path=direct pool=main cause=insufficient_physical_pages old=2 reclaimable=0 requested=12 entitled=10 logical=100 physical=18"
        admission = "[ADMISSION-QUEUE-TIMING] id=0 lane=1 queue_wait_ms=100 event=selected_for_admission"
        self.assertEqual(self.capture(denial + "\n" + admission + "\n")["kv_denial_joins"], [])

    def test_truncated_summary_rejected(self):
        r = self.capture(
            "[ADMISSION-TRACE] head=7 queued=2 active=1 reason=vacant_lane_not_admittable "
            "head_pages_main=2\n"
        )
        self.assertEqual(r["counts"].get("summary", 0), 0)
        self.assertEqual(len(r["unrecognised_or_invalid_lines"]), 1)

    def test_summary_trailing_data_rejected(self):
        r = self.capture(
            "[ADMISSION-TRACE] head=7 queued=2 active=1 reason=no_vacant_lane "
            "head_pages_main=2 head_pages_backend=1 used_pages_main=10 used_pages_backend=3 used_lanes=1 "
            "capacity_pages_main=18 capacity_pages_backend=8 capacity_lanes=2 "
            "deadline_remaining_ms=600000 protection_epoch=0 protection_phase=none extra=1\n"
        )
        self.assertEqual(r["counts"].get("summary", 0), 0)
        self.assertEqual(len(r["unrecognised_or_invalid_lines"]), 1)

    def test_reject_malformed(self):
        r = self.capture("[ADMISSION-QUEUE-TIMING] id=1 lane=0 queue_wait_ms=-1 event=selected_for_admission\n")
        self.assertEqual(len(r["unrecognised_or_invalid_lines"]), 1)

    def test_no_assumed_journal_prefix(self):
        r = self.capture("Oct 10 host [ADMISSION-QUEUE-TIMING] id=1 lane=0 queue_wait_ms=42 event=selected_for_admission\n")
        self.assertEqual(r["selected_admission_queue_ms"]["n"], 0)

    def test_sampling_and_policy(self):
        r = self.capture("[ADMISSION-DEFERRAL] head=3 queued=2 reason=short_lane_policy_hold\n")
        self.assertEqual(r["counts"]["deferral_short_lane_policy_hold"], 1)

    def test_zero_evidence_gate_flags_dead_capture(self):
        # Reproduces the PR78 defect: a startup-FAIL/crashed C2 server leaves a
        # zero-record raw.stderr.log that the old gate audited as success (rc 0),
        # masking the failure. With the gate, zero evidence must fail closed (rc 7).
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "dead.stderr.log"
            path.write_text("", encoding="utf-8")
            r = self.audit(path, require_evidence=True)
            self.assertEqual(len(r["no_admission_evidence_files"]), 1)
            self.assertEqual(r["no_admission_evidence_files"][0], str(path))
            self.assertEqual(r["selected_admission_queue_ms"]["n"], 0)
            self.assertEqual(self.exit_code(r, require_evidence=True), 7)

    def test_zero_evidence_gate_disabled_by_default(self):
        # Backward compatibility: without the flag, an empty capture is still rc 0.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "empty.stderr.log"
            path.write_text("", encoding="utf-8")
            r = self.audit(path, require_evidence=False)
            self.assertEqual(len(r["no_admission_evidence_files"]), 0)
            self.assertEqual(self.exit_code(r, require_evidence=False), 0)

    def test_gate_passes_when_evidence_present(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ok.stderr.log"
            path.write_text("[ADMISSION-QUEUE-TIMING] id=7 lane=0 queue_wait_ms=125 event=selected_for_admission\n",
                            encoding="utf-8")
            r = self.audit(path, require_evidence=True)
            self.assertEqual(len(r["no_admission_evidence_files"]), 0)
            self.assertEqual(self.exit_code(r, require_evidence=True), 0)

    def audit(self, path, require_evidence):
        return audit([path], require_evidence=require_evidence)

    def exit_code(self, report, require_evidence):
        if require_evidence and report["no_admission_evidence_files"]:
            return 7
        return int(bool(report["unrecognised_or_invalid_lines"]))

if __name__ == "__main__":
    unittest.main()
