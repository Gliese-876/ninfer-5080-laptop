# SPDX-License-Identifier: Apache-2.0
"""Protocol/measurement tests. Mock output is never a hardware benchmark."""

import argparse
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "long_context_acceptance.py"
spec = importlib.util.spec_from_file_location("acceptance", MODULE_PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class SSETests(unittest.TestCase):
    def test_multiline_crlf_unicode_and_comments(self):
        wire = ': heartbeat\r\nevent: message\r\ndata: {"a":\r\ndata: "中文"}\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
        frames = list(bench.sse_events(io.BytesIO(wire)))
        self.assertEqual(json.loads(frames[0][1]), {"a": "中文"})
        self.assertEqual(frames[1], ("message", "[DONE]"))

    def test_role_usage_and_empty_chunks_do_not_become_first_token(self):
        result = bench.ChatAccumulator()
        self.assertFalse(result.accept("message", json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]})))
        self.assertFalse(result.accept("message", json.dumps({"choices": [], "usage": {"completion_tokens": 7}})))
        self.assertTrue(result.accept("message", json.dumps({"choices": [{"delta": {"reasoning_content": "核对"}}]})))
        self.assertEqual(result.response()["reasoning_content"], "核对")
        self.assertEqual(result.response()["usage"]["completion_tokens"], 7)

    def test_fragmented_tool_json_and_stream_error(self):
        result = bench.ChatAccumulator()
        for fragment in [{"index": 0, "id": "call_x", "function": {"name": "read_", "arguments": '{"path":'}},
                         {"index": 0, "function": {"name": "source", "arguments": '"x.cpp"}'}}]:
            result.accept("message", json.dumps({"choices": [{"delta": {"tool_calls": [fragment]}}]}))
        call = result.response()["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "read_source")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "x.cpp"})
        with self.assertRaises(bench.AcceptanceError):
            result.accept("error", '{"error":{"message":"CUDA failed"}}')

    def test_http_ttft_excludes_immediate_role_and_uses_delayed_visible_data(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def send(obj):
                    wire = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
                    # Force arbitrary transport fragments, including multibyte UTF-8.
                    for offset in range(0, len(wire), 7):
                        self.wfile.write(wire[offset:offset+7])
                        self.wfile.flush()

                send({"id": "mock", "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
                time.sleep(.12)
                send({"id": "mock", "choices": [{"index": 0, "delta": {"content": "中文答案"}}]})
                time.sleep(.04)
                send({"id": "mock", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                send({"id": "mock", "choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 7}})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                result = bench.Client(f"http://127.0.0.1:{server.server_port}").stream({"messages": []}, Path(directory))
                self.assertEqual(result["content"], "中文答案")
                self.assertEqual(result["usage"]["completion_tokens"], 7)
                self.assertGreaterEqual(result["client_timings_seconds"]["ttft"], .1)
                self.assertGreaterEqual(result["client_timings_seconds"]["wall"], .14)
                self.assertTrue((Path(directory) / "sse.jsonl").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class CorrectnessTests(unittest.TestCase):
    def test_native_reasoning_omits_forcing_and_preserves_thinking(self):
        payload = bench.base_payload("model", [{"role": "user", "content": "source review"}], "reasoning", 4096,
                                     reasoning_budget=None)
        self.assertNotIn("reasoning_budget", payload)
        self.assertTrue(payload["enable_thinking"])
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(bench.chat_to_count_request(payload)["thinking"], {"type": "enabled"})
        self.assertIsNone(bench.parser().parse_args(["prepare", "--trace-log", "real.jsonl", "--output", "new-fixtures",
                                                    "--reasoning-budget", "native"]).reasoning_budget)
        self.assertIsNone(bench.parser().parse_args(["prepare", "--trace-log", "real.jsonl", "--output", "new-fixtures"]).reasoning_budget)

    def test_single_turn_4K_output_does_not_inherit_agent_history_reservation(self):
        class ReachedClient(Exception):
            pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            args = argparse.Namespace(output=str(Path(directory) / "new-fixtures"), target_input_tokens=126976,
                output_tokens=4096, max_context=131072, cache_input_tokens=102400, agent_headroom=2048,
                url="http://127.0.0.1:1", api_key_env="NO_SUCH_TEST_KEY", timeout=1)
            with mock.patch.object(bench, "Client", side_effect=ReachedClient):
                with self.assertRaises(ReachedClient):
                    bench.command_prepare(args)
                args.output_tokens = 4097
                with self.assertRaisesRegex(bench.AcceptanceError, "single-turn"):
                    bench.command_prepare(args)

    def test_native_reasoning_rejects_inherited_server_forcing_before_requests(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            log = root / "server.jsonl"
            log.write_text(json.dumps({"artifact_type": "ninfer_serve_request_log", "schema_version": 11,
                "event": "server_start", "server_instance_id": "mock", "server": {"default_thinking_budget": 2048}}) + "\n")
            args = argparse.Namespace(output=str(root / "run"), url="http://127.0.0.1:1", api_key_env="NO_SUCH_TEST_KEY",
                                      timeout=1, server_log=str(log))
            with self.assertRaisesRegex(bench.AcceptanceError, "without --default-thinking-budget"):
                bench.Runner(args, {"reasoning_policy": {"mode": "native"}})

    def test_earlier_correct_json_does_not_mask_wrong_final_answer(self):
        self.assertEqual(bench.json_answer('{"correct":true}\n最后：\n{"correct":false}'), {"correct": False})
        self.assertIsNone(bench.json_answer('{"correct":true}\n结果见上。'))

    def test_reasoning_accepts_only_closed_source_aliases_and_formula_whitespace(self):
        evidence = [{"path": path, "start_line": begin, "end_line": end} for path, begin, end in bench.SOURCE_RANGES]
        answer = dict(bench.EXPECTED_REASONING, references=[{"path": item["path"], "line": item["start_line"]} for item in evidence])
        answer["completion_source"] = "result.generated_token_ids"
        answer["preserve_thinking_true_checkpoint"] = "RewriteCheckpointKind::ResponseReplay"
        answer["computed_prefill_formula"] = "max (0, prompt_tokens - prefix_cache_hit_tokens)"
        answer["decode_numerator"] = "max( 0, completion_tokens - 1 )"
        checked = bench.check_reasoning("中文解释。\n" + json.dumps(answer), evidence)
        self.assertTrue(checked["passed"])
        self.assertEqual(len(checked["normalized_aliases"]), 4)
        for field, bad in [("completion_source", "fake.generated_token_ids"),
                           ("preserve_thinking_true_checkpoint", "OtherKind::ResponseReplay"),
                           ("computed_prefill_formula", "max(0,prompt_tokens+prefix_cache_hit_tokens)"),
                           ("decode_numerator", "max(0,completion_tokens-2)"),
                           ("schema_version", 11.0),
                           ("drafted_tokens_are_output", 0)]:
            changed = dict(answer, **{field: bad})
            self.assertFalse(bench.check_reasoning(json.dumps(changed), evidence)["passed"], (field, bad))
        invalid_refs = dict(answer, references=[{"path": item["path"], "line": 1} for item in evidence])
        self.assertFalse(bench.check_reasoning(json.dumps(invalid_refs), evidence)["passed"])

    def test_generated_code_rejects_side_effects_and_allows_builtin_annotations(self):
        with self.assertRaises(bench.AcceptanceError):
            bench.validate_pure_code("import os\ndef analyze_request_logs(records): return {}")
        with self.assertRaises(bench.AcceptanceError):
            bench.validate_pure_code("def analyze_request_logs(records):\n    return open('secret').read()")
        bench.validate_pure_code("def analyze_request_logs(records: list[dict]) -> dict:\n    return {}")

    def test_wrong_generated_function_fails_independent_execution_cases(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            result = bench.check_code("```python\ndef analyze_request_logs(records):\n    return {}\n```", Path(directory))
            self.assertFalse(result["passed"])

    def test_valid_annotated_generated_function_executes_in_child(self):
        code = '''def analyze_request_logs(records: list[dict]) -> dict:
    original = repr(records)
    result = {"completed_requests": 0, "failed_requests": 0, "ignored_records": 0,
              "prompt_tokens": 0, "cached_tokens": 0, "computed_prefill_tokens": 0,
              "decode_tokens": 0, "decode_seconds": 0.0, "decode_tok_s": None,
              "drafted_tokens": 0, "accepted_draft_tokens": 0}
    for record in records:
        if not isinstance(record, dict) or record.get("schema_version") != 11:
            result["ignored_records"] += 1
            continue
        if record.get("event") in ("request_error", "request_rejected"):
            result["failed_requests"] += 1
            continue
        try:
            if record.get("event") != "request_done":
                raise ValueError()
            values = [record["result"][key] for key in ("prompt_tokens", "completion_tokens", "prefix_cache_hit_tokens", "computed_prefill_tokens")]
            values += [record["speculative"]["drafted_tokens"], record["speculative"]["accepted_tokens"]]
            duration = record["timings_seconds"]["decode"]
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError()
            if type(duration) not in (int, float) or not duration >= 0:
                raise ValueError()
            if values[2] > values[0] or values[3] != values[0] - values[2]:
                raise ValueError()
        except (TypeError, KeyError, ValueError):
            result["ignored_records"] += 1
            continue
        result["completed_requests"] += 1
        for key, value in zip(("prompt_tokens", "cached_tokens", "computed_prefill_tokens", "drafted_tokens", "accepted_draft_tokens"), (values[0], values[2], values[3], values[4], values[5])):
            result[key] += value
        result["decode_tokens"] += max(values[1] - 1, 0)
        result["decode_seconds"] += duration
    if result["decode_seconds"]:
        result["decode_tok_s"] = result["decode_tokens"] / result["decode_seconds"]
    assert repr(records) == original
    return result
'''
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            result = bench.check_code("```python\n" + code + "```", Path(directory))
            self.assertTrue(result["passed"], result)

    def test_count_translation_keeps_tools_reasoning_and_actual_history(self):
        payload = bench.base_payload("local-model", [
            {"role": "system", "content": "system"}, {"role": "user", "content": "question"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_x", "function": {"name": "read_source", "arguments": '{"path":"src/serve/request_log.cpp"}'}}]},
            {"role": "tool", "tool_call_id": "call_x", "content": "actual source"}], "agent", 768)
        counted = bench.chat_to_count_request(payload)
        self.assertEqual(counted["thinking"], {"type": "disabled"})
        self.assertFalse(counted["preserve_thinking"])
        self.assertEqual(counted["messages"][1]["content"][0]["input"], {"path": "src/serve/request_log.cpp"})
        self.assertEqual(counted["messages"][2]["content"][0]["tool_use_id"], "call_x")
        self.assertEqual(len(counted["tools"]), 3)

    def test_preserve_true_flows_to_actual_payload_count_and_summary(self):
        payload = bench.base_payload("model", [{"role": "user", "content": "source review"}], "agent", 768,
                                     preserve_thinking=True)
        self.assertTrue(payload["preserve_thinking"])
        self.assertTrue(bench.chat_to_count_request(payload)["preserve_thinking"])
        artifact = {"max_context": 131072, "prompt_policy": {"preserve_thinking": True},
                    "fixtures": {"agent": {"payload": payload}}}
        args = argparse.Namespace(profile="optimized", repeats=20, suites=["long", "cached"])
        startup = {"schema_version": 11, "argv": [], "engine": {
            "max_context": 131072, "kv_capacity": 131072, "max_concurrency": 1}}
        summary = bench.summarize([], args, artifact, startup)
        self.assertTrue(summary["launch_capacity_verified"])
        self.assertEqual(summary["cache_policy"]["effective_checkpoint_mode"], "ResponseReplay")
        self.assertEqual(summary["cache_policy"]["configured_prefix_checkpoint_policy"], "stable-turn")
        self.assertTrue(bench.parser().parse_args(["prepare", "--trace-log", "real.jsonl", "--output", "new",
                                                  "--preserve-thinking", "true"]).preserve_thinking)

    def test_mixed_payload_history_policies_are_rejected(self):
        with self.assertRaises(bench.AcceptanceError):
            bench.fixture_preserve_thinking({"fixtures": {"a": {"payload": {"preserve_thinking": False}},
                                                         "b": {"payload": {"preserve_thinking": True}}}})

    def test_comparison_rejects_mixed_request_policies_even_if_hashes_match(self):
        first = {"metric_revision": bench.METRIC_REVISION, "minimum_verified_cached_hit_tokens": bench.MIN_VERIFIED_CACHE_HIT_TOKENS,
                 "assessment_revision": bench.ASSESSMENT_REVISION,
                 "sampling_revision": bench.SAMPLING_REVISION,
                 "minimum_long_input_tokens": 126000,
                 "workload_sha256": "unit-only", "artifact_identity": {"target": "target", "weights_id": "weights", "size_bytes": 1},
                 "reasoning_policy": {"mode": "native", "reasoning_budget": None}, "cache_policy": {"preserve_thinking": True}}
        second = dict(first, cache_policy={"preserve_thinking": False})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            for name, value in [("a.json", first), ("b.json", second)]:
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            args = argparse.Namespace(summary=[str(root / "a.json"), str(root / "b.json")], output=str(root / "comparison.json"))
            with self.assertRaisesRegex(bench.AcceptanceError, "same explicit reasoning budget"):
                bench.command_compare(args)
            self.assertFalse((root / "comparison.json").exists())

    def test_missing_cache_samples_and_slow_cache_misses_are_reported(self):
        args = argparse.Namespace(profile="optimized", repeats=20, suites=["long", "cached"])
        records = []
        for trial in range(20):
            for turn in range(4):
                miss = trial < 5 and turn == 0
                records.append({"warmup": False, "status": "ok", "correctness": {"passed": True},
                    "suite": "cached", "kind": "agent", "trial": trial, "turn": turn,
                    "finish_reason": "stop" if turn == 3 else "tool_calls",
                    "metrics": {"cache_hit_tokens": 0 if miss else 102400,
                                "client_ttft_seconds": 3 if miss else .5}})
        summary = bench.summarize(records, args, {"max_context": 131072}, None)
        self.assertEqual(summary["cached_agent_attempted_samples"], 80)
        self.assertEqual(summary["cached_agent_cache_miss_or_missing_hit_samples"], 5)
        self.assertEqual(summary["cached_agent_client_visible_ttft_p95_seconds"], 3)
        self.assertEqual(summary["qualification"], "NOT_QUALIFIED")
        self.assertFalse(summary["launch_capacity_verified"])

    def test_code_only_diagnostic_never_qualifies_or_reports_full_mixed_tps(self):
        args = argparse.Namespace(profile="baseline", repeats=20, suites=["long"], long_kinds=["code"])
        records = []
        for trial in range(22):
            records.append({"warmup": trial < 2, "status": "ok", "correctness": {"passed": True},
                "metric_revision": bench.METRIC_REVISION,
                "suite": "long", "kind": "code", "trial": trial, "turn": 0, "finish_reason": "stop",
                "metrics": {"prompt_tokens": 126976, "completion_tokens": 1001,
                            "server_decode_tokens": 1000, "server_decode_seconds": 5,
                            "server_reported_decode_phase_tok_s": 200,
                            "server_postfirst_wall_seconds": 6, "server_postfirst_wall_tok_s": 1000/6,
                            "server_postfirst_c1_scope_verified": True}})
        startup = {"schema_version": 11, "argv": [], "engine": {
            "max_context": 131072, "kv_capacity": 131072, "max_concurrency": 1}}
        summary = bench.summarize(records, args, {"max_context": 131072}, startup)
        self.assertEqual(summary["expected_measured_trials"], 20)
        self.assertTrue(summary["correctness_all_trials_passed"])
        self.assertEqual(summary["selected_postfirst_server_wall_tok_s"], 1000/6)
        self.assertEqual(summary["selected_reported_decode_phase_tok_s_diagnostic"], 200)
        self.assertIsNone(summary["mixed_postfirst_server_wall_tok_s"])
        self.assertFalse(summary["full_mixed_workload_requested"])
        self.assertFalse(summary["requested_repeats_succeeded_per_long_category"])
        self.assertEqual(summary["qualification"], "NOT_QUALIFIED")

    def test_actual_cached_hit_six_tokens_short_fails_otherwise_qualified_gate(self):
        args = argparse.Namespace(profile="optimized", repeats=20, suites=["long", "cached"],
                                  long_kinds=["reasoning", "code", "agent"])
        records = []

        def add(suite, kind, trial, warmup):
            for turn in range(4 if kind == "agent" else 1):
                records.append({"metric_revision": bench.METRIC_REVISION, "suite": suite, "kind": kind,
                    "trial": trial, "turn": turn, "warmup": warmup, "status": "ok", "correctness": {"passed": True},
                    "finish_reason": "tool_calls" if kind == "agent" and turn < 3 else "stop",
                    "tool_flow": {"valid": True, "completed_stages": ["read_source", "read_request_log", "summarize_request_log"][:min(turn+1, 3)]} if kind == "agent" else None,
                    "metrics": {"prompt_tokens": 126976, "completion_tokens": 101, "server_decode_tokens": 100,
                                "server_postfirst_wall_seconds": .8, "server_postfirst_wall_tok_s": 125,
                                "server_postfirst_c1_scope_verified": True, "server_decode_seconds": .7,
                                "server_reported_decode_phase_tok_s": 100/.7, "cache_hit_tokens": 102400,
                                "client_ttft_seconds": .5, "prefix_reuse_path": "append_frontier"}})

        for kind in args.long_kinds:
            for trial in range(20):
                add("long", kind, trial, False)
            for trial in (-2, -1):
                add("long", kind, trial, True)
        for trial in range(20):
            add("cached", "agent", trial, False)
        startup = {"schema_version": 11, "argv": [], "engine": {
            "max_context": 131072, "kv_capacity": 131072, "max_concurrency": 1}}
        artifact = {"max_context": 131072, "prompt_policy": {"preserve_thinking": True}}
        self.assertEqual(bench.summarize(records, args, artifact, startup)["qualification"], "PASS")
        next(r for r in records if r["suite"] == "cached")["metrics"]["cache_hit_tokens"] = 102394
        summary = bench.summarize(records, args, artifact, startup)
        self.assertEqual(summary["minimum_verified_cached_hit_tokens"], 102400)
        self.assertEqual(summary["cached_agent_samples_with_100K_verified_hit"], 79)
        self.assertEqual(summary["cached_agent_cache_miss_or_missing_hit_samples"], 1)
        self.assertEqual(summary["qualification"], "NOT_QUALIFIED")


class CircuitAndContextTests(unittest.TestCase):
    def test_absolute_stream_budget_closes_transport_and_keeps_partial_evidence(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(b'data: {"id":"budget-unit","choices":[{"delta":{"role":"assistant"}}]}\n\n')
                    self.wfile.flush()
                    time.sleep(.4)
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                except OSError:
                    pass
                self.close_connection = True

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                started = time.perf_counter()
                client = bench.Client(f"http://127.0.0.1:{server.server_port}", timeout=5, deadline=started+.15)
                with self.assertRaises(bench.RunBudgetExceeded):
                    client.stream({"messages": []}, Path(directory))
                self.assertLess(time.perf_counter()-started, .35)
                partial = json.loads((Path(directory) / "partial_response.json").read_text(encoding="utf-8"))
                self.assertTrue(partial["run_budget_exceeded"])
                self.assertFalse(partial["stream_done"])
                self.assertIn("budget-unit", (Path(directory) / "sse.jsonl").read_text(encoding="utf-8"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_short_defaults_and_small_sample_cached_observation(self):
        parsed = bench.parser().parse_args(["run", "--fixtures", "fixture", "--profile", "baseline", "--output", "out"])
        self.assertEqual((parsed.repeats, parsed.warmups, parsed.max_run_seconds, parsed.suites), (1, 0, 600, ["long"]))
        args = argparse.Namespace(profile="baseline", repeats=5, warmups=0, suites=["cached"], long_kinds=["reasoning", "code", "agent"])
        args.max_cached_ttft_seconds = 2.0
        records = [{"metric_revision": bench.METRIC_REVISION, "suite": "cached", "kind": "agent", "trial": trial,
            "turn": turn, "warmup": False, "status": "ok", "correctness": {"passed": True},
            "finish_reason": "tool_calls" if turn < 3 else "stop", "metrics": {
                "client_ttft_seconds": .5, "cache_hit_tokens": 102400, "prefix_reuse_path": "append_frontier"}}
            for trial in range(5) for turn in range(4)]
        for record in records:
            record["tool_flow"] = {"valid": True, "completed_stages": ["read_source", "read_request_log", "summarize_request_log"][:min(record["turn"]+1, 3)]}
        startup = {"schema_version": 11, "engine": {"max_context": 131072, "kv_capacity": 131072, "max_concurrency": 1}}
        summary = bench.summarize(records, args, {"max_context": 131072}, startup)
        self.assertTrue(summary["cached_goal_observed_passed"])
        self.assertEqual(summary["cached_ttft_sample_count"], 20)
        self.assertTrue(summary["cached_ttft_small_sample"])
        self.assertEqual(summary["cached_agent_expected_samples"], 20)
        self.assertIsNone(summary["mixed_postfirst_server_wall_tok_s"])

    def test_offline_user_stop_preserves_raw_and_selects_first_not_fastest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            source = root / "old"
            source.mkdir()
            artifact = {"max_context": 131072, "minimum_long_input_tokens": 124519, "workload_sha256": "unit-only"}
            bench.write_json(root / "fixtures.json", artifact)
            command = ["script.py", "run", "--fixtures", str(root / "fixtures.json"), "--profile", "baseline",
                       "--output", str(source), "--repeats", "20", "--warmups", "4"]
            bench.write_json(source / "run_manifest.json", {"command": command, "workload_sha256": "unit-only", "server_start": None})
            rows = [{"metric_revision": bench.METRIC_REVISION, "suite": "long", "kind": "reasoning", "trial": trial,
                "turn": 0, "warmup": False, "status": "ok", "correctness": {"passed": True}, "finish_reason": "stop",
                "evidence": f"evidence/{trial}", "metrics": {"prompt_tokens": 124928, "completion_tokens": 101,
                    "server_decode_tokens": 100, "server_decode_seconds": .9, "server_postfirst_wall_seconds": seconds,
                    "server_postfirst_wall_tok_s": 100/seconds, "server_postfirst_c1_scope_verified": True}}
                for trial, seconds in ((4, 1), (5, .5))]
            raw = "\n".join(bench.canonical(row) for row in rows)+"\n"
            (source / "requests.jsonl").write_text(raw, encoding="utf-8")
            args = argparse.Namespace(run_dir=str(source), output=str(root / "closure"))
            with mock.patch("builtins.print"):
                bench.command_close_run(args)
            summary = json.loads((root / "closure" / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["user_stopped"])
            self.assertEqual(summary["predefined_first_measured_reasoning"]["trial"], 4)
            self.assertEqual(summary["measured_requests"], 2)
            self.assertEqual(summary["expected_measured_trials"], 80)
            self.assertEqual(summary["unattempted_measured_trials"], 78)
            self.assertFalse(summary["full_mixed_workload_observed"])
            self.assertIsNone(summary["mixed_postfirst_server_wall_tok_s"])
            self.assertEqual((source / "requests.jsonl").read_text(encoding="utf-8"), raw)
            self.assertFalse((source / "summary.json").exists())

    def test_confirmed_unavailable_stops_remaining_suites_without_synthetic_samples(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            payloads = {kind: {"payload": bench.base_payload("model", [{"role": "user", "content": "unit fixture"}], kind,
                                6144 if kind != "agent" else 768, preserve_thinking=True)} for kind in ("reasoning", "code", "agent")}
            payloads["cached_prime"] = payloads["agent"]
            artifact = {"max_context": 131072, "minimum_long_input_tokens": 124519,
                "fixtures": payloads, "source_evidence": [{"path": p, "start_line": a, "end_line": b} for p, a, b in bench.SOURCE_RANGES],
                "reasoning_policy": {"mode": "native", "reasoning_budget": None}}
            artifact["workload_sha256"] = bench.digest(artifact)
            bench.write_json(root / "fixtures.json", artifact)
            client = mock.Mock()
            unavailable = bench.HttpFailure(503, '{"error":{"code":"service_unavailable","message":"inference engine is unavailable"}}')
            client.count.side_effect = [124928, 124928, unavailable]
            client.stream.side_effect = [{"usage": {"prompt_tokens": 124928, "completion_tokens": 3},
                "client_timings_seconds": {"wall": 2.1, "ttft": 1.1, "first_output_to_done": 1},
                "content": "truncated response", "finish_reason": "length"},
                bench.StreamFailure({"error": {"type": "internal_error", "message": "slice range out of bounds"}})]
            log = mock.Mock()
            log.startup = {"schema_version": 11, "server": {"default_thinking_budget": None}, "engine": {
                "max_context": 131072, "kv_capacity": 131072, "max_concurrency": 1}}
            log.done.return_value = {"request": {"reasoning_budget": None},
                "result": {"prompt_tokens": 124928, "completion_tokens": 3, "prefix_cache_hit_tokens": 0,
                           "computed_prefill_tokens": 124928, "prefix_reuse_path": "full_reset"},
                "timings_seconds": {"prepare": .1, "vision": 0, "prefill": .8, "ttft": 1, "decode": .9, "total": 2},
                "speculative": {"backend": "mtp"}}
            args = bench.parser().parse_args(["run", "--fixtures", str(root / "fixtures.json"), "--profile", "baseline",
                "--server-log", str(root / "unit.log"), "--output", str(root / "assessment"), "--repeats", "20", "--warmups", "3", "--suites", "long", "cached"])
            with mock.patch.object(bench, "Client", return_value=client), mock.patch.object(bench, "ServerLog", return_value=log), mock.patch("builtins.print"):
                self.assertEqual(bench.command_run(args), 2)
            summary = json.loads((root / "assessment" / "summary.json").read_text(encoding="utf-8"))
            rows = (root / "assessment" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertEqual(client.count.call_count, 3)
            self.assertEqual(client.stream.call_count, 2)
            self.assertTrue(summary["campaign_stopped_for_fatal_engine"])
            self.assertEqual(summary["unattempted_measured_trials"], 80)
            self.assertEqual(summary["unattempted_measured_request_slots"], 200)
            self.assertEqual(summary["cached_agent_unattempted_request_slots"], 80)
            self.assertEqual(summary["attempted_warmup_request_slots"], 3)
            self.assertEqual(summary["qualification"], "NOT_QUALIFIED")
            self.assertTrue((root / "assessment" / "fatal_circuit.json").exists())

    def test_queue_timeout_is_not_misclassified_as_fatal_engine(self):
        failure = bench.HttpFailure(503, '{"error":{"code":"request_queue_timeout","message":"inference request expired while waiting for admission"}}')
        self.assertFalse(failure.engine_unavailable())

    def test_explicit_95_percent_input_and_6144_output_fit_full_capacity(self):
        class ReachedClient(Exception):
            pass
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            args = argparse.Namespace(output=str(Path(directory) / "new"), target_input_tokens=124928,
                output_tokens=6144, max_context=131072, cache_input_tokens=102432, agent_headroom=2048,
                minimum_long_input_tokens=124519, url="http://127.0.0.1:1", api_key_env="NO_SUCH_TEST_KEY", timeout=1)
            with mock.patch.object(bench, "Client", side_effect=ReachedClient):
                with self.assertRaises(ReachedClient):
                    bench.command_prepare(args)
                args.minimum_long_input_tokens = 124518
                with self.assertRaisesRegex(bench.AcceptanceError, "ceil"):
                    bench.command_prepare(args)

    def test_summary_uses_declared_input_floor_and_does_not_claim_full_input(self):
        args = argparse.Namespace(profile="baseline", repeats=1, warmups=0, suites=["long"], long_kinds=["code"])
        records = [{"warmup": False, "status": "ok", "correctness": {"passed": True}, "suite": "long", "kind": "code",
            "trial": 0, "turn": 0, "finish_reason": "stop", "metric_revision": bench.METRIC_REVISION,
            "metrics": {"prompt_tokens": 124928, "completion_tokens": 101, "server_decode_tokens": 100,
                        "server_postfirst_wall_seconds": 1, "server_postfirst_wall_tok_s": 100,
                        "server_postfirst_c1_scope_verified": True, "server_decode_seconds": .9}}]
        artifact = {"max_context": 131072, "minimum_long_input_tokens": 124519}
        summary = bench.summarize(records, args, artifact, None)
        self.assertTrue(summary["near_128K_actual_usage_verified"])
        self.assertEqual(summary["measured_long_actual_input_fraction_min"], .953125)
        self.assertIn("131072 input tokens are not claimed", summary["input_claim"])
        records[0]["metrics"]["prompt_tokens"] = 124518
        self.assertFalse(bench.summarize(records, args, artifact, None)["near_128K_actual_usage_verified"])

    def test_comparison_rejects_mixed_126K_and_95_percent_contracts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            for name, minimum in (("a.json", 126000), ("b.json", 124519)):
                (root / name).write_text(json.dumps({"metric_revision": bench.METRIC_REVISION,
                    "sampling_revision": bench.SAMPLING_REVISION,
                    "minimum_verified_cached_hit_tokens": 102400, "minimum_long_input_tokens": minimum}))
            args = argparse.Namespace(summary=[str(root / "a.json"), str(root / "b.json")], output=str(root / "comparison.json"))
            with self.assertRaisesRegex(bench.AcceptanceError, "minimum_long_input_tokens"):
                bench.command_compare(args)


class NumericalPerformanceScopeTests(unittest.TestCase):
    def observed(self):
        args = argparse.Namespace(profile="optimized", repeats=1, warmups=0, suites=["long"],
                                  long_kinds=["reasoning", "code", "agent"])
        records = []
        stages = ["read_source", "read_request_log", "summarize_request_log"]
        for kind in args.long_kinds:
            for turn in range(4 if kind == "agent" else 1):
                records.append({"suite": "long", "kind": kind, "trial": 0, "turn": turn, "warmup": False,
                    "status": "ok", "metric_revision": bench.METRIC_REVISION,
                    "correctness": {"passed": kind != "code", "reason": "unit content QA"},
                    "finish_reason": "tool_calls" if kind == "agent" and turn < 3 else "stop",
                    "tool_flow": {"valid": True, "completed_stages": stages[:min(turn+1, 3)]} if kind == "agent" else None,
                    "metrics": {"prompt_tokens": 124928, "completion_tokens": 101, "server_decode_tokens": 100,
                        "server_decode_seconds": .4, "server_postfirst_wall_seconds": .5,
                        "server_postfirst_wall_tok_s": 200, "server_postfirst_c1_scope_verified": True}})
        artifact = {"max_context": 131072, "minimum_long_input_tokens": 124519}
        startup = {"schema_version": 11, "engine": {"max_context": 131072, "kv_capacity": 131072, "max_concurrency": 1}}
        return args, records, artifact, startup

    def test_content_failure_keeps_complete_performance_and_qa_separate_without_receipt(self):
        args, records, artifact, startup = self.observed()
        summary = bench.summarize(records, args, artifact, startup)
        self.assertEqual(summary["mixed_postfirst_server_wall_tok_s"], 200)
        self.assertTrue(summary["measurement_complete"])
        self.assertTrue(summary["throughput_reportable"])
        self.assertFalse(summary["performance_targets_observed_passed"])
        self.assertFalse(summary["long_goal_observed_passed"])
        self.assertEqual(summary["requested_performance_thresholds"], {"min_decode_tps": None, "max_cached_ttft_seconds": None})
        self.assertEqual(summary["qualification"], "PASS")
        self.assertFalse(summary["correctness_all_trials_passed"])
        self.assertEqual(summary["content_correctness"]["failed_requests"], 1)
        self.assertEqual(summary["numerical_validation"]["status"], "NOT_PROVIDED")
        self.assertFalse(summary["numerically_validated_performance_targets_observed"])
        records[-1]["correctness"]["passed"] = False
        self.assertTrue(bench.summarize(records, args, artifact, startup)["measurement_complete"])

    def test_missing_actual_agent_stage_never_becomes_complete_mixed_measurement(self):
        args, records, artifact, startup = self.observed()
        del records[4]
        summary = bench.summarize(records, args, artifact, startup)
        self.assertFalse(summary["measurement_complete"])
        self.assertFalse(summary["throughput_reportable"])
        self.assertIsNone(summary["mixed_postfirst_server_wall_tok_s"])
        self.assertEqual(summary["qualification"], "NOT_QUALIFIED")

    def test_tool_flow_missing_or_invalid_is_not_replaced_by_correct_sounding_answer(self):
        args, records, artifact, startup = self.observed()
        records[-1]["tool_flow"]["completed_stages"] = ["read_source"]
        records[-1]["correctness"]["passed"] = True
        summary = bench.summarize(records, args, artifact, startup)
        self.assertFalse(summary["measurement_complete"])
        self.assertIsNone(summary["mixed_postfirst_server_wall_tok_s"])

    def test_independent_receipt_is_bound_to_running_binary_and_never_inferred_from_content(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            bench.write_json(root/"launch.json", {"binary_sha256": "binary-a", "model_verification": {"verified": True, "sha256": "model-a"}})
            bench.write_json(root/"receipt.json", {"status": "PASS", "binary_sha256": "binary-a", "tests": [{"status": "PASS"}]})
            validation = bench.numerical_validation_sources([str(root/"receipt.json")], root/"launch.json")
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["model_verification"]["sha256"], "model-a")
            args, records, artifact, startup = self.observed()
            args.numerical_validation = validation
            args.min_decode_tps = 110
            summary = bench.summarize(records, args, artifact, startup)
            self.assertTrue(summary["numerically_validated_performance_targets_observed"])
            self.assertFalse(summary["correctness_all_trials_passed"])
            bench.write_json(root/"receipt.json", {"status": "PASS", "binary_sha256": "binary-b"})
            self.assertEqual(bench.numerical_validation_sources([str(root/"receipt.json")], root/"launch.json")["status"], "UNVERIFIED_OR_FAILED")

    def test_old_cli_stays_valid_and_receipt_is_optional(self):
        parsed = bench.parser().parse_args(["run", "--fixtures", "f", "--profile", "optimized", "--output", "o"])
        self.assertEqual(parsed.numerical_validation_receipt, [])
        self.assertEqual((parsed.min_decode_tps, parsed.max_cached_ttft_seconds), (0, None))
        parsed = bench.parser().parse_args(["run", "--fixtures", "f", "--profile", "optimized", "--output", "o",
                                           "--numerical-validation-receipt", "receipt.json"])
        self.assertEqual(parsed.numerical_validation_receipt, ["receipt.json"])

    def test_explicit_decode_threshold_controls_gate(self):
        args, records, artifact, startup = self.observed()
        args.min_decode_tps = 110
        summary = bench.summarize(records, args, artifact, startup)
        self.assertTrue(summary["long_goal_observed_passed"])
        self.assertEqual(summary["qualification"], "PASS")
        args.min_decode_tps = 210
        summary = bench.summarize(records, args, artifact, startup)
        self.assertTrue(summary["throughput_reportable"])
        self.assertFalse(summary["long_goal_observed_passed"])
        self.assertEqual(summary["qualification"], "NOT_QUALIFIED")

    def test_explicit_cached_latency_threshold_controls_gate(self):
        args, records, artifact, startup = self.observed()
        args.suites = ["cached"]
        records = records[2:]
        for record in records:
            record["suite"] = "cached"
            record["metrics"].update(cache_hit_tokens=102400, client_ttft_seconds=1.0)
        args.max_cached_ttft_seconds = 2.0
        summary = bench.summarize(records, args, artifact, startup)
        self.assertTrue(summary["cached_goal_observed_passed"])
        self.assertEqual(summary["qualification"], "PASS")
        args.max_cached_ttft_seconds = .8
        summary = bench.summarize(records, args, artifact, startup)
        self.assertTrue(summary["measurement_complete"])
        self.assertFalse(summary["cached_goal_observed_passed"])
        self.assertEqual(summary["qualification"], "NOT_QUALIFIED")

    def test_wrong_final_answer_preserves_actual_tool_history_for_next_cached_trial(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            runner = bench.Runner.__new__(bench.Runner)
            runner.output = Path(directory)
            runner.args = argparse.Namespace(max_agent_turns=4)
            runner.stopping = lambda: False
            trace = [{"schema_version": 11, "event": "request_done", "request": {"request_id": i},
                "result": {"prompt_tokens": 10, "completion_tokens": 5, "prefix_cache_hit_tokens": 0, "computed_prefill_tokens": 10},
                "timings_seconds": {"decode": 1}, "speculative": {"drafted_tokens": 0, "accepted_tokens": 0}} for i in (1, 2)]
            runner.artifact = {"actual_request_trace": trace, "source_evidence": [{"path": "source"}], "trace_log_origin": "unit-only"}
            calls = [("read_source", {"path": "src/serve/request_log.cpp"}), ("read_request_log", {"limit": 2}),
                     ("summarize_request_log", {"request_ids": [1, 2]})]
            replies = []
            for turn in range(4):
                record = {"evidence": str(turn), "correctness": {"passed": False}}
                tool = [{"id": "call"+str(turn), "function": {"name": calls[turn][0], "arguments": json.dumps(calls[turn][1])}}] if turn < 3 else []
                replies.append((record, {"tool_calls": tool, "content": "wrong final answer" if turn == 3 else "", "reasoning_content": ""}))
            runner.request = mock.Mock(side_effect=replies)
            runner.save = mock.Mock()
            result = runner.exercise({"messages": [{"role": "user", "content": "unit-only"}]}, "agent", 0, False, "cached")
            self.assertEqual(runner.request.call_count, 4)
            self.assertEqual(len(result), 8)
            self.assertEqual(result[-1]["content"], "wrong final answer")
            final = runner.save.call_args.args[0]
            self.assertFalse(final["correctness"]["passed"])
            self.assertTrue(final["tool_flow"]["valid"])
            self.assertEqual(final["tool_flow"]["completed_stages"], [name for name, _ in calls])

    def test_comparison_refuses_historical_content_gated_assessment(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            args, records, artifact, startup = self.observed()
            current = bench.summarize(records, args, artifact, startup)
            historical = dict(current)
            historical.pop("assessment_revision")
            bench.write_json(root/"current.json", current)
            bench.write_json(root/"historical.json", historical)
            with self.assertRaisesRegex(bench.AcceptanceError, "assessment_revision"):
                bench.command_compare(argparse.Namespace(summary=[str(root/"current.json"),str(root/"historical.json")], output=str(root/"out.json")))


class TimingContractTests(unittest.TestCase):
    def server(self):
        return {"result": {"prompt_tokens": 100, "completion_tokens": 101, "prefix_cache_hit_tokens": 80,
                           "computed_prefill_tokens": 20, "prefix_reuse_path": "append_frontier"},
                "timings_seconds": {"prepare": .2, "vision": 0, "prefill": 1.8, "ttft": 5., "decode": 8., "total": 15.},
                "speculative": {"backend": "mtp", "rounds": 30, "drafted_tokens": 90, "accepted_tokens": 60}}

    def test_missing_forced_bridge_and_host_phase_time_is_in_primary_wall(self):
        server = self.server()
        metrics = bench.server_timing_metrics(server, 101, 1)
        self.assertEqual(metrics["server_postfirst_wall_seconds"], 10)
        self.assertEqual(metrics["server_postfirst_wall_tok_s"], 10)
        self.assertEqual(metrics["server_reported_decode_phase_tok_s"], 12.5)
        self.assertEqual(metrics["server_postfirst_minus_reported_decode_seconds"], 2)
        self.assertEqual(metrics["server_timings_seconds"], server["timings_seconds"])

    def test_inconsistent_or_nonfinite_timings_are_rejected(self):
        for changed in [{"total": 4.}, {"total": 5.}, {"total": 12.}, {"ttft": float("nan")}, {"decode": float("inf")}]:
            server = self.server()
            server["timings_seconds"].update(changed)
            with self.assertRaises(bench.AcceptanceError, msg=changed):
                bench.server_timing_metrics(server, 101, 1)

    def test_prefill_initial_mtp_cost_stays_in_ttft_without_false_isolated_estimate(self):
        metrics = bench.server_timing_metrics(self.server(), 101, 1)
        self.assertEqual(metrics["server_ttft_seconds"], 5)
        self.assertEqual(metrics["server_prefill_seconds"], 1.8)
        self.assertIn("inside TTFT/prefill", metrics["initial_mtp_preparation_timing_scope"])
        self.assertIsNone(metrics["initial_mtp_preparation_separate_seconds"])
        self.assertIsNone(bench.server_timing_metrics(self.server(), 101, 2)["server_postfirst_wall_tok_s"])

    def test_buffered_tool_visible_to_done_interval_never_sets_decode_denominator(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            runner = bench.Runner.__new__(bench.Runner)
            runner.output, runner.counter, runner.records = Path(directory), 0, []
            runner.args = argparse.Namespace(profile="baseline")
            runner.artifact = {"workload_sha256": "unit-only", "max_context": 131072}
            runner.client = mock.Mock()
            runner.client.count.return_value = 100
            runner.client.stream.return_value = {"usage": {"prompt_tokens": 100, "completion_tokens": 101},
                "client_timings_seconds": {"wall": 15.1, "ttft": 15.099, "first_output_to_done": .001},
                "finish_reason": "tool_calls"}
            runner.server_log = mock.Mock()
            runner.server_log.startup = {"engine": {"max_concurrency": 1}}
            runner.server_log.done.return_value = self.server()
            payload = bench.base_payload("model", [{"role": "user", "content": "tool question"}], "agent", 768)
            record, _ = runner.request(payload, "agent", 0, 0, False, "long")
            self.assertEqual(record["metrics"]["server_postfirst_wall_tok_s"], 10)
            self.assertEqual(record["metrics"]["client_first_output_to_done_seconds"], .001)
            self.assertAlmostEqual(record["metrics"]["client_request_tok_s"], 101/15.1)
            self.assertEqual(record["metric_revision"], bench.METRIC_REVISION)

    def test_legacy_phase_summary_cannot_be_compared_with_postfirst_metric(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / "new.json").write_text(json.dumps({"metric_revision": bench.METRIC_REVISION}))
            (root / "old.json").write_text(json.dumps({"metric_revision": "legacy-reported-phase"}))
            args = argparse.Namespace(summary=[str(root / "new.json"), str(root / "old.json")], output=str(root / "comparison.json"))
            with self.assertRaisesRegex(bench.AcceptanceError, "metric_revision"):
                bench.command_compare(args)
            self.assertFalse((root / "comparison.json").exists())

    def test_comparison_rejects_legacy_decimal_100K_cache_threshold(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            (root / "new.json").write_text(json.dumps({"metric_revision": bench.METRIC_REVISION,
                "sampling_revision": bench.SAMPLING_REVISION,
                "minimum_verified_cached_hit_tokens": 102400}))
            (root / "old.json").write_text(json.dumps({"metric_revision": bench.METRIC_REVISION,
                "sampling_revision": bench.SAMPLING_REVISION,
                "minimum_verified_cached_hit_tokens": 100000}))
            args = argparse.Namespace(summary=[str(root / "new.json"), str(root / "old.json")], output=str(root / "comparison.json"))
            with self.assertRaisesRegex(bench.AcceptanceError, "minimum_verified_cached_hit_tokens"):
                bench.command_compare(args)


if __name__ == "__main__":
    unittest.main()
