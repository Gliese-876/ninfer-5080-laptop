#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch the selected 128K profile and retain exact command/binary provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from verify_model import verify_model

WORKSPACE = Path(__file__).resolve().parent.parent


def runtime_environment(cuda_dir: Path) -> dict[str, str]:
    """Select one toolkit runtime while preserving unrelated library paths."""
    environment = os.environ.copy()
    selected = str(cuda_dir.expanduser().resolve())
    previous = environment.get("NINFER_CUDA", "")
    known_roots = {selected}
    if previous:
        known_roots.add(str(Path(previous).expanduser()))
        known_roots.add(str(Path(previous).expanduser().resolve()))
    kept = []
    for entry in environment.get("LD_LIBRARY_PATH", "").split(":"):
        conventional_cuda = re.fullmatch(
            r"/usr/local/cuda(?:-[^/]+)?/(?:lib64|targets/[^/]+/lib)/?", entry)
        known_cuda = any(entry.rstrip("/") == root + "/lib64" or
                         (entry.startswith(root + "/targets/") and entry.rstrip("/").endswith("/lib"))
                         for root in known_roots)
        if entry and not conventional_cuda and not known_cuda and entry not in kept:
            kept.append(entry)
    environment["NINFER_CUDA"] = selected
    environment["LD_LIBRARY_PATH"] = ":".join([selected + "/lib64", *kept])
    return environment


def runtime_evidence(binary: Path, cuda_dir: Path, environment: dict[str, str]) -> dict:
    result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True,
                            env=environment, check=True)
    output = result.stdout + result.stderr
    resolved_cudart = []
    selected = cuda_dir.expanduser().resolve()
    for line in output.splitlines():
        if "libcudart" not in line:
            continue
        if "not found" in line:
            raise ValueError(f"CUDA runtime unavailable: {line.strip()}; select --cuda-dir")
        if "=>" in line:
            library, location = (value.strip() for value in line.split("=>", 1))
        else:
            location = line.strip()
            library = Path(location.rsplit(" (", 1)[0]).name
        location = location.rsplit(" (", 1)[0].strip()
        if not location.startswith("/"):
            raise ValueError(f"Cannot validate CUDA runtime from ldd: {line.strip()}")
        resolved = Path(location).resolve()
        if not resolved.is_relative_to(selected):
            raise ValueError(f"CUDA runtime mismatch: selected {selected}, but "
                             f"{library} resolves to {resolved}; select matching --cuda-dir")
        resolved_cudart.append({"library": library, "resolved_path": str(resolved)})
    return {"command": ["ldd", str(binary)], "exit_code": result.returncode,
            "output": output, "resolved_cudart": resolved_cudart}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=WORKSPACE / "configs/laptop-128k-mtp4.json")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--cuda-dir", type=Path,
                        help="Toolkit runtime directory; overrides NINFER_CUDA (default /usr/local/cuda-13.4)")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--draft-tokens", type=int)
    parser.add_argument("--full-model-hash", action="store_true",
                        help="Read/hash all model bytes even when a matching local receipt exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print argv; do not verify model, create logs or start inference")
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Additional server arguments after --; recorded in launch.json")
    options = parser.parse_args()
    profile = json.loads(options.profile.read_text(encoding="utf-8"))
    binary = (options.binary or WORKSPACE / profile["binary_relative_to_workspace"]).expanduser().resolve()
    cuda_dir = (options.cuda_dir or Path(os.environ.get("NINFER_CUDA", "/usr/local/cuda-13.4"))).expanduser().resolve()
    environment = runtime_environment(cuda_dir)
    model = (options.model or WORKSPACE / profile["model_relative_to_workspace"]).expanduser().resolve()
    run_dir = (options.run_dir or WORKSPACE / "logs" / (
        profile["name"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    arguments = list(profile["server_arguments"])
    for flag, value in (("--port", options.port), ("--prefill-chunk", options.prefill_chunk),
                        ("--draft-tokens", options.draft_tokens)):
        if value is not None:
            arguments[arguments.index(flag) + 1] = str(value)
    extra = options.extra[1:] if options.extra[:1] == ["--"] else options.extra
    arguments += ["--request-log-jsonl", str(run_dir / "server.requests.jsonl"), *extra]
    argv = [str(binary), str(model), *arguments]
    command = shlex.join(["env", f"NINFER_CUDA={environment['NINFER_CUDA']}",
                          f"LD_LIBRARY_PATH={environment['LD_LIBRARY_PATH']}", *argv])
    print(command, flush=True)
    if options.dry_run:
        return 0
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Server executable unavailable: {binary}; build it or pass --binary")
    if not (cuda_dir / "lib64").is_dir():
        raise ValueError(f"CUDA runtime directory unavailable: {cuda_dir / 'lib64'}")
    runtime = runtime_evidence(binary, cuda_dir, environment)
    run_dir.mkdir(parents=True, exist_ok=True)
    verified = verify_model(model, WORKSPACE / "configs/model_manifest.json",
                            WORKSPACE / "logs/model-verification.json", not options.full_model_hash)
    help_result = subprocess.run([str(binary), "--help"], capture_output=True, text=True,
                                 env=environment, check=True)
    help_text = help_result.stdout + help_result.stderr
    flags = {value for value in arguments if value.startswith("--")}
    missing = sorted(flag for flag in flags if flag not in help_text)
    if missing:
        raise ValueError(f"Binary does not advertise requested options: {', '.join(missing)}")
    (run_dir / "ninfer-serve-help.txt").write_text(help_text, encoding="utf-8")
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    launch = {
        "schema_version": 1, "profile": profile, "profile_file": str(options.profile.resolve()),
        "utc": datetime.now(timezone.utc).isoformat(), "argv": argv,
        "command": command, "binary_sha256": digest,
        "cuda_dir": str(cuda_dir), "runtime_environment": {
            key: environment[key] for key in ("NINFER_CUDA", "LD_LIBRARY_PATH")},
        "ldd": runtime,
        "model_verification": verified, "run_dir": str(run_dir),
    }
    (run_dir / "launch.json").write_text(json.dumps(launch, indent=2) + "\n", encoding="utf-8")
    print(f"Run evidence: {run_dir}", flush=True)
    os.execve(str(binary), argv, environment)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Launch failed: {error}", file=sys.stderr)
        sys.exit(1)
