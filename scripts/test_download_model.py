#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Black-box downloader tests using only a local, deliberately faulty HTTP server.

Run: python -m unittest discover -s scripts -p test_download_model.py -v
No Hugging Face URL, real model file, package installation, or GPU is used.
"""

import concurrent.futures
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
import unittest

try:
    from . import download_model as downloader
except ImportError:
    import download_model as downloader


class MockModelServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, payload):
        super().__init__(("127.0.0.1", 0), MockModelHandler)
        self.payload = payload
        self.etag = '"local-mock-model-v1"'
        self.mode = "ok"
        self.requests = []
        self.lock = threading.Lock()
        self.active_ranges = 0
        self.max_active_ranges = 0
        self.disconnected = False
        self.disconnected_range = None
        self.block_data = False
        self.saw_data_range = threading.Event()
        self.release_data = threading.Event()
        self.thread = threading.Thread(target=self.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:{}/model.bin".format(self.server_address[1])

    def close(self):
        self.release_data.set()
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=5)

    def snapshot_requests(self):
        with self.lock:
            return [dict(request) for request in self.requests]


class MockModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, _format, *_args):
        pass

    def do_HEAD(self):
        with self.server.lock:
            self.server.requests.append({"method": "HEAD", "range": self.headers.get("Range"),
                                         "if_range": self.headers.get("If-Range")})
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", self.server.etag)
        self.send_header("Last-Modified", "Wed, 01 Oct 2025 00:00:00 GMT")
        self.end_headers()

    def do_GET(self):
        payload = self.server.payload
        range_header = self.headers.get("Range")
        match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header or "")
        if range_header and not match:
            self.send_error(416)
            return
        start = int(match.group(1)) if match else 0
        end = int(match.group(2)) if match and match.group(2) else len(payload) - 1
        if start > end or end >= len(payload):
            self.send_response(416)
            self.send_header("Content-Range", "bytes */{}".format(len(payload)))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data_range = match is not None and end - start + 1 > 1024
        with self.server.lock:
            self.server.requests.append({"method": "GET", "range": range_header,
                                         "start": start, "end": end,
                                         "if_range": self.headers.get("If-Range")})
            if data_range:
                self.server.active_ranges += 1
                self.server.max_active_ranges = max(self.server.max_active_ranges,
                                                    self.server.active_ranges)
        try:
            if data_range:
                self.server.saw_data_range.set()
                if self.server.block_data:
                    self.server.release_data.wait(timeout=10)
                # Overlap makes the parallel-range assertion deterministic even
                # though this model is only 100 KiB and localhost is very fast.
                time.sleep(0.04)
            mode = self.server.mode
            body = payload[start:end + 1]
            status = 206 if match else 200
            if mode == "ignore_range" and match:
                status = 200
                body = payload
            self.send_response(status)
            if status == 206:
                reported_start = start + 1 if mode == "bad_content_range" else start
                self.send_header("Content-Range", "bytes {}-{}/{}".format(reported_start, end, len(payload)))
            length = len(body) + (7 if mode == "bad_content_length" and match else 0)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", self.server.etag)
            self.send_header("Last-Modified", "Wed, 01 Oct 2025 00:00:00 GMT")
            self.end_headers()
            disconnect = mode == "truncate" and data_range
            with self.server.lock:
                if mode == "disconnect_once" and data_range and not self.server.disconnected:
                    self.server.disconnected = True
                    self.server.disconnected_range = (start, end)
                    disconnect = True
            if disconnect:
                body = body[:max(1, len(body) // 2)]
            try:
                self.wfile.write(body)
                self.wfile.flush()
                if disconnect:
                    self.connection.shutdown(socket.SHUT_WR)
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Rejection before reading the body is expected in several tests.
                pass
            self.close_connection = True
        finally:
            if data_range:
                with self.server.lock:
                    self.server.active_ranges -= 1


class DownloadModelTests(unittest.TestCase):
    def setUp(self):
        # Keep disposable files beside this test, inside the writable workspace.
        self.temporary = tempfile.TemporaryDirectory(prefix="download-model-test-",
                                                      dir=str(Path(__file__).resolve().parent))
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.payload = bytes((index * 37 + 11) % 256 for index in range(100 * 1024))
        self.prefix = self.payload[:13 * 1024]
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.server = MockModelServer(self.payload)
        self.addCleanup(self.server.close)
        self.model = self.directory / "mock-model.bin"
        self.model.write_bytes(self.prefix)
        self.manifest = self.directory / "manifest.json"
        self.revision = "0123456789abcdef0123456789abcdef01234567"
        self.manifest.write_text(json.dumps({
            "download_url": self.server.url, "size_bytes": len(self.payload),
            "sha256": self.digest, "huggingface_revision": self.revision,
        }), encoding="utf-8")
        self.receipt = self.directory / "receipt.json"

    def download(self, retries=0):
        return downloader.download_model(self.model, self.manifest, workers=4,
                                         progress_interval=0.01, retries=retries,
                                         receipt=self.receipt)

    def assert_prefix_preserved(self):
        self.assertEqual(self.model.read_bytes(), self.prefix)
        self.assertFalse(self.receipt.exists(), "a failed download must not have a success receipt")

    def test_contiguous_prefix_parallel_ranges_and_receipt(self):
        result = self.download()
        self.assertIsInstance(result, dict)
        self.assertEqual(self.model.read_bytes(), self.payload)
        self.assertEqual(hashlib.sha256(self.model.read_bytes()).hexdigest(), self.digest)
        receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
        self.assertEqual(receipt["sha256"], self.digest)
        self.assertEqual(receipt["size_bytes"], len(self.payload))
        self.assertEqual(receipt["huggingface_revision"], self.revision)
        data_requests = [request for request in self.server.snapshot_requests()
                         if request["method"] == "GET" and request.get("range")
                         and request["end"] - request["start"] + 1 > 1024]
        self.assertEqual(len({(request["start"], request["end"]) for request in data_requests}), 4)
        self.assertGreaterEqual(self.server.max_active_ranges, 2,
                                "four ranges must execute concurrently")
        for request in data_requests:
            self.assertGreaterEqual(request["start"], len(self.prefix))

    def test_http_200_for_range_is_rejected_without_changing_prefix(self):
        self.server.mode = "ignore_range"
        with self.assertRaises((ValueError, OSError)):
            self.download()
        self.assert_prefix_preserved()

    def test_wrong_content_range_is_rejected_without_changing_prefix(self):
        self.server.mode = "bad_content_range"
        with self.assertRaises((ValueError, OSError)):
            self.download()
        self.assert_prefix_preserved()

    def test_interrupted_part_is_saved_then_resumed_with_if_range(self):
        self.server.mode = "disconnect_once"
        with self.assertRaises((ValueError, OSError)):
            self.download(retries=0)
        self.assert_prefix_preserved()
        parts = self.model.with_name(self.model.name + ".parts")
        self.assertTrue(parts.is_dir())
        self.assertTrue(any(part.stat().st_size > 0 for part in parts.glob("part-*.bin")),
                        "interrupted downloads must retain downloaded part bytes")
        self.assertTrue(list(parts.glob("part-*.json")), "persist range metadata beside the parts")
        interrupted_start, interrupted_end = self.server.disconnected_range
        previous_requests = len(self.server.snapshot_requests())
        self.server.mode = "ok"
        self.download()
        self.assertEqual(self.model.read_bytes(), self.payload)
        new_requests = self.server.snapshot_requests()[previous_requests:]
        resumed_requests = [request for request in new_requests
                            if request["method"] == "GET" and request.get("range")
                            and interrupted_start < request["start"] <= interrupted_end
                            and request["end"] == interrupted_end]
        self.assertTrue(resumed_requests, "resume the interrupted range after its retained bytes")
        self.assertTrue(all(request["if_range"] == self.server.etag for request in resumed_requests),
                        "use the persisted strong ETag in If-Range when resuming")

    def test_truncated_body_is_rejected_without_changing_prefix(self):
        self.server.mode = "truncate"
        with self.assertRaises((ValueError, OSError)):
            self.download()
        self.assert_prefix_preserved()

    def test_changed_etag_on_resume_is_rejected_without_changing_prefix(self):
        self.server.mode = "disconnect_once"
        with self.assertRaises((ValueError, OSError)):
            self.download(retries=0)
        self.assert_prefix_preserved()
        original_etag = self.server.etag
        previous_requests = len(self.server.snapshot_requests())
        self.server.etag = '"different-remote-artifact"'
        self.server.mode = "ok"
        with self.assertRaises((ValueError, OSError)):
            self.download(retries=0)
        self.assert_prefix_preserved()
        new_requests = self.server.snapshot_requests()[previous_requests:]
        self.assertTrue(any(request.get("if_range") == original_etag for request in new_requests),
                        "a resumed range must keep the saved validator")

    def test_wrong_content_length_is_rejected_without_changing_prefix(self):
        self.server.mode = "bad_content_length"
        with self.assertRaises((ValueError, OSError)):
            self.download()
        self.assert_prefix_preserved()

    def test_external_model_change_prevents_append(self):
        self.server.block_data = True
        changed_model = self.prefix + b"externally appended bytes"
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.download)
            try:
                self.assertTrue(self.server.saw_data_range.wait(timeout=5),
                                "downloader did not start a data range")
                self.model.write_bytes(changed_model)
            finally:
                self.server.release_data.set()
            with self.assertRaises((ValueError, OSError)):
                future.result(timeout=10)
        self.assertEqual(self.model.read_bytes(), changed_model)
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
