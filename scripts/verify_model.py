#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify the immutable official .ninfer artifact without external packages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parent.parent


def verify_model(model: Path, manifest: Path, receipt: Path | None = None,
                 cached: bool = False) -> dict:
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    model = model.resolve(strict=True)
    stat = model.stat()
    if stat.st_size != expected["size_bytes"]:
        raise ValueError(f"Artifact size mismatch: {stat.st_size} bytes; "
                         f"expected {expected['size_bytes']} (download incomplete or wrong file)")
    identity = {
        "model": str(model), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": expected["sha256"],
        "huggingface_revision": expected["huggingface_revision"],
    }
    if cached and receipt is not None and receipt.is_file():
        try:
            previous = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("verified") is True and all(previous.get(k) == v
                                                       for k, v in identity.items()):
            return {**previous, "verification_mode": "cached_stat_receipt"}
    started = time.monotonic()
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    after = model.stat()
    if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
        raise ValueError("Artifact changed during SHA-256 verification")
    actual = digest.hexdigest()
    if actual != expected["sha256"]:
        raise ValueError(f"Artifact SHA-256 mismatch: {actual}; expected {expected['sha256']}")
    report = {
        **identity, "verified": True, "verification_mode": "full_sha256",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "verification_seconds": time.monotonic() - started,
    }
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path,
                        default=WORKSPACE / "models/qwen3_8_27b.ninfer")
    parser.add_argument("--manifest", type=Path, default=WORKSPACE / "configs/model_manifest.json")
    parser.add_argument("--receipt", type=Path, default=WORKSPACE / "logs/model-verification.json")
    parser.add_argument("--cached", action="store_true",
                        help="Reuse a trusted local receipt only if path, size and mtime match. "
                             "Omit to always read and hash the complete artifact.")
    args = parser.parse_args()
    try:
        result = verify_model(args.model, args.manifest, args.receipt, args.cached)
    except (OSError, ValueError, KeyError) as error:
        print(f"Model verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
