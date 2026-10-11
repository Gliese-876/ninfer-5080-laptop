#!/usr/bin/env python3
"""Independent privileged C1 restoration watchdog; only staged runner starts this."""
import os
import pathlib
import signal
import subprocess
import sys
import time

def main():
    if len(sys.argv) != 4 or os.geteuid() != 0:
        return 21
    root, wrapper, staged = map(pathlib.Path, sys.argv[1:])
    if not root.is_dir() or str(root).startswith("/tmp/issue58-c2-campaign.") is False:
        return 22
    if str(wrapper) != "/usr/local/sbin/openclaw-ninfer-user-service":
        return 23
    (root / "watchdog.pid").write_text(str(os.getpid()) + "\n")
    (root / "watchdog_ready").write_text("root watchdog live\n")
    time.sleep(330)
    if (root / "disarmed").exists():
        return 0
    with (root / "watchdog.log").open("a") as log:
        print("WATCHDOG_FIRED=YES", file=log, flush=True)
        pid_path = root / "test.pid"
        if pid_path.exists():
            try:
                pid = int(pid_path.read_text().strip())
                if pid > 1:
                    cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    # Only terminate exact staged executable, not an unrelated reused PID.
                    if cmdline and os.path.realpath(os.fsdecode(cmdline[0])) == os.path.realpath(staged):
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(4)
                        try:
                            if pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0] == cmdline[0]:
                                os.kill(pid, signal.SIGKILL)
                        except (ProcessLookupError, FileNotFoundError):
                            pass
                    else:
                        print("WATCHDOG_PID_IDENTITY_MISMATCH=YES", file=log, flush=True)
            except (ValueError, OSError, ProcessLookupError) as exc:
                print(f"WATCHDOG_KILL_WARNING={type(exc).__name__}", file=log, flush=True)
        try:
            restored = subprocess.run([str(wrapper), "start"], stdout=log, stderr=log,
                                      timeout=45, check=False)
            print(f"WATCHDOG_START_RC={restored.returncode}", file=log, flush=True)
        except Exception as exc:
            print(f"WATCHDOG_START_ERROR={type(exc).__name__}", file=log, flush=True)
        # Health is the true restoration gate, not systemctl's successful exit code.
        for _ in range(20):
            probe = subprocess.run(
                ["curl", "-fsS", "--max-time", "2", "http://127.0.0.1:8080/health"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if probe.returncode == 0:
                print("WATCHDOG_C1_HTTP200=PASS", file=log, flush=True)
                return 0
            time.sleep(2)
        print("WATCHDOG_C1_HTTP200=FAIL", file=log, flush=True)
        return 30

if __name__ == "__main__":
    raise SystemExit(main())
