#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resume the pinned official artifact with validated, parallel HTTP byte ranges."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request

from verify_model import verify_model

WORKSPACE = Path(__file__).resolve().parent.parent
CHUNK_BYTES = 1024 * 1024
HTTP_TIMEOUT_SECONDS = 30


class DownloadError(ValueError):
    """A response or local state cannot safely extend this artifact."""


class InterruptedTransfer(OSError):
    """A validated response ended before its requested range was complete."""


def _write_all(stream, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise OSError("Disk write did not make progress")
        remaining = remaining[written:]


def _snapshot(path: Path) -> tuple | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        raise DownloadError(f"Model path is not a regular file: {path}")
    return (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino)


def _check_prefix(model: Path, snapshot: tuple | None) -> None:
    if _snapshot(model) != snapshot:
        raise DownloadError("Model prefix changed during download; stop the other writer "
                            "before resuming. No downloaded parts have been appended.")


@contextmanager
def _download_lock(model: Path):
    """OS locks release after a crash; the lock file can safely remain on disk."""
    path = model.with_name(model.name + ".download.lock")
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            _write_all(handle, b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise DownloadError(f"Another downloader holds {path}") from error
        yield
    finally:
        # Closing the descriptor releases either operating system's lock.
        handle.close()


def _save_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _part_state(folder: Path, expected: dict, start: int, end: int) -> tuple[Path, Path, dict]:
    part = folder / f"part-{start:012d}-{end:012d}.bin"
    metadata_path = part.with_suffix(".json")
    identity = {
        "schema_version": 1, "download_url": expected["download_url"],
        "size_bytes": expected["size_bytes"], "sha256": expected["sha256"],
        "huggingface_revision": expected["huggingface_revision"],
        "start": start, "end": end,
    }
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DownloadError(f"Unreadable part metadata: {metadata_path}") from error
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise DownloadError(f"Part metadata does not match the pinned artifact: {metadata_path}")
    else:
        if part.exists() and part.stat().st_size:
            raise DownloadError(f"Part has no provenance metadata: {part}")
        metadata = {**identity, "etag": None, "last_modified": None}
        _save_metadata(metadata_path, metadata)
    if part.exists() and part.stat().st_size > end - start + 1:
        raise DownloadError(f"Part exceeds its assigned byte range: {part}")
    return part, metadata_path, metadata


def _validator(metadata: dict) -> str | None:
    etag = metadata.get("etag")
    if isinstance(etag, str) and etag and not etag.startswith("W/"):
        return etag
    modified = metadata.get("last_modified")
    return modified if isinstance(modified, str) and modified else None


def _validate_response(response, start: int, end: int, total: int, metadata: dict) -> None:
    if response.status != 206:
        raise DownloadError(f"Range request requires HTTP 206; received HTTP {response.status}. "
                            "Refusing to append a full-body response.")
    content_range = response.headers.get("Content-Range", "").strip()
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
    if match is None or tuple(map(int, match.groups())) != (start, end, total):
        raise DownloadError(f"Incorrect Content-Range: {content_range!r}; "
                            f"expected bytes {start}-{end}/{total}")
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            valid_length = int(length) == end - start + 1
        except ValueError:
            valid_length = False
        if not valid_length:
            raise DownloadError(f"Incorrect Content-Length: {length!r}; "
                                f"expected {end - start + 1}")
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    if encoding != "identity":
        raise DownloadError(f"Range response unexpectedly uses Content-Encoding {encoding!r}")
    for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
        previous, current = metadata.get(key), response.headers.get(header)
        if previous and current and previous != current:
            raise DownloadError(f"Remote {header} changed for an existing part")


def _download_part(part: Path, metadata_path: Path, metadata: dict,
                   stopped: threading.Event, retries: int) -> None:
    start, end = metadata["start"], metadata["end"]
    required_size = end - start + 1
    for attempt in range(retries + 1):
        if stopped.is_set():
            raise DownloadError("Range transfer cancelled")
        already = part.stat().st_size if part.exists() else 0
        if already == required_size:
            return
        request_start = start + already
        headers = {
            "Range": f"bytes={request_start}-{end}",
            "Accept-Encoding": "identity", "User-Agent": "ninfer-laptop-downloader/1",
        }
        validator = _validator(metadata)
        if validator:
            headers["If-Range"] = validator
        request = urllib.request.Request(metadata["download_url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                _validate_response(response, request_start, end, metadata["size_bytes"], metadata)
                for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
                    value = response.headers.get(header)
                    if value:
                        metadata[key] = value
                _save_metadata(metadata_path, metadata)
                remaining = end - request_start + 1
                # Flush each received chunk so interruption leaves a reusable contiguous prefix.
                with part.open("ab", buffering=0) as output:
                    while remaining:
                        if stopped.is_set():
                            raise DownloadError("Range transfer cancelled")
                        chunk = response.read(min(CHUNK_BYTES, remaining))
                        if not chunk:
                            raise InterruptedTransfer(f"HTTP range ended {remaining} bytes early")
                        _write_all(output, chunk)
                        remaining -= len(chunk)
                    if response.read(1):
                        raise DownloadError("HTTP range body contains bytes beyond Content-Range")
            if part.stat().st_size != required_size:
                raise DownloadError(f"Completed part has incorrect size: {part}")
            return
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429) and error.code < 500:
                raise DownloadError(f"HTTP {error.code} for range {request_start}-{end}") from error
            failure = error
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            failure = error
        if attempt == retries:
            raise DownloadError(f"Range {start}-{end} failed after {attempt + 1} attempts: {failure}") from failure
        print(f"Retrying range {start}-{end}: {failure}", file=sys.stderr, flush=True)
        if stopped.wait(min(2 ** attempt, 8)):
            raise DownloadError("Range transfer cancelled")


def _received(parts: list[tuple[Path, Path, dict]]) -> int:
    return sum(part.stat().st_size for part, _, _ in parts if part.exists())


def _append_parts(model: Path, snapshot: tuple | None,
                  parts: list[tuple[Path, Path, dict]], total: int) -> None:
    _check_prefix(model, snapshot)
    expected_size = snapshot[0] if snapshot is not None else 0
    for part, _, metadata in parts:
        if not part.is_file() or part.stat().st_size != metadata["end"] - metadata["start"] + 1:
            raise DownloadError(f"Cannot merge an incomplete part: {part}")
    # Existing model bytes are untouched until every range is complete.
    mode = "ab" if snapshot is not None else "xb"
    with model.open(mode, buffering=0) as output:
        for part, _, _ in parts:
            with part.open("rb") as source:
                while chunk := source.read(CHUNK_BYTES):
                    if os.fstat(output.fileno()).st_size != expected_size:
                        raise DownloadError("Another writer changed the model while appending parts")
                    _write_all(output, chunk)
                    expected_size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    if expected_size != total or model.stat().st_size != total:
        raise DownloadError("Merged artifact has an unexpected byte length")


def download_model(model: Path, manifest: Path, workers: int = 4,
                   progress_interval: float = 5.0, retries: int = 4,
                   receipt: Path | None = None) -> dict:
    """Download remaining ranges, merge only complete parts, then run full SHA-256."""
    if workers < 1 or progress_interval <= 0 or retries < 0:
        raise DownloadError("workers/interval must be positive and retries nonnegative")
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))
    total = expected["size_bytes"]
    if not isinstance(total, int) or total <= 0:
        raise DownloadError("Manifest size_bytes must be a positive integer")
    model = Path(model).expanduser().resolve()
    model.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(model):
        snapshot = _snapshot(model)
        prefix = snapshot[0] if snapshot is not None else 0
        if prefix > total:
            raise DownloadError(f"Existing model exceeds manifest size: {prefix} > {total}")
        if prefix == total:
            return verify_model(model, Path(manifest), receipt, cached=False)
        folder = model.with_name(model.name + ".parts")
        folder.mkdir(parents=True, exist_ok=True)
        count = min(workers, total - prefix)
        width = (total - prefix + count - 1) // count
        parts = [_part_state(folder, expected, start, min(start + width, total) - 1)
                 for start in range(prefix, total, width)]
        stopped = threading.Event()
        started = time.monotonic()
        initial_received = _received(parts)
        print(f"Pinned artifact: {expected['huggingface_revision']}; "
              f"existing prefix {prefix:,}/{total:,} bytes; "
              f"{len(parts)} ranges; {initial_received:,} bytes reusable", flush=True)
        pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="model-range")
        try:
            pending = {pool.submit(_download_part, *part, stopped, retries) for part in parts}
            while pending:
                done, pending = wait(pending, timeout=progress_interval, return_when=FIRST_EXCEPTION)
                for future in done:
                    future.result()
                _check_prefix(model, snapshot)
                received = _received(parts)
                elapsed = time.monotonic() - started
                speed = (received - initial_received) / max(elapsed, 0.001) / (1024 * 1024)
                print(f"Download {(prefix + received) / total:.2%}: "
                      f"{prefix + received:,}/{total:,} bytes; "
                      f"{speed:.2f} MiB/s; {elapsed:.1f}s", flush=True)
        finally:
            stopped.set()
            pool.shutdown(wait=True, cancel_futures=True)
        _check_prefix(model, snapshot)
        print("All byte ranges complete; appending in order, then verifying full SHA-256...", flush=True)
        _append_parts(model, snapshot, parts, total)
        result = verify_model(model, Path(manifest), receipt, cached=False)
        summary = {
            "utc": datetime.now(timezone.utc).isoformat(), "download_url": expected["download_url"],
            "original_prefix_bytes": prefix, "range_workers": count,
            "elapsed_seconds": time.monotonic() - started, "verification": result,
        }
        _save_metadata(folder / "completed.json", summary)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path,
                        default=WORKSPACE / "models/qwen3_8_27b.ninfer")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    options = parser.parse_args()
    try:
        report = download_model(options.model, WORKSPACE / "configs/model_manifest.json",
                                options.workers, options.progress_interval, options.retries,
                                WORKSPACE / "logs/model-verification.json")
    except (OSError, ValueError, KeyError) as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download interrupted; completed bytes in each part are retained for resume.", file=sys.stderr)
        return 130
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
