#!/usr/bin/env bash
# Candidate C2 diagnostic: deliberately interrupts C1 only after explicit approval.
set -Eeuo pipefail
umask 077
WRAPPER=/usr/local/sbin/openclaw-ninfer-user-service
MODEL=/models/ninfer-custom/qwen3_8_27b_5080_128k_24vz_7gv_473dade.ninfer
: "$ISSUE58_STAGED_BIN" "$ISSUE58_STAGED_SHA256"
STAGED="$ISSUE58_STAGED_BIN"
[[ "$ISSUE58_APPROVED_OUTAGE" == YES ]] || { echo "BLOCKED: explicit approved outage flag required"; exit 40; }
[[ "$(id -un)" == toddballinger && "$STAGED" == /* ]] || { echo "BLOCKED: user or stage path"; exit 41; }
[[ "$ISSUE58_STAGED_SHA256" =~ ^[a-fA-F0-9]{64}$ ]] || { echo "BLOCKED: SHA syntax"; exit 42; }
[[ -x "$STAGED" && -r "$MODEL" && -x "$WRAPPER" ]] || { echo "BLOCKED: binary/model/wrapper missing"; exit 43; }
[[ "$(sha256sum "$STAGED" | cut -d' ' -f1)" == "$ISSUE58_STAGED_SHA256" ]] || { echo "BLOCKED: SHA mismatch"; exit 44; }
PROD=/models/ninfer-builds/build-production-current-main-d5ee1bf/apps/ninfer-serve
[[ "$(readlink -f "$STAGED")" != "$(readlink -f "$PROD")" ]] || { echo "BLOCKED: production binary"; exit 45; }
exec 9>/tmp/issue58-c2-supervision.lock
flock -n 9 || { echo "BLOCKED: overlapping campaign"; exit 46; }
[[ "$(systemctl --user show ninfer-local-model.service -p ActiveState --value)" == active ]] || { echo "BLOCKED: inactive C1"; exit 47; }
curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null || { echo "BLOCKED: C1 unhealthy"; exit 48; }
! ss -lnt | grep -q ':18080 ' || { echo "BLOCKED: 18080 busy"; exit 49; }
sudo -n true || { echo "BLOCKED: sudo required"; exit 50; }
ROOT="$(mktemp -d /tmp/issue58-c2-campaign.XXXXXXXX)"
REPORT="$ROOT/report.log"
SERVERLOG="$ROOT/raw.stderr.log"
TEST_PID=""
STOP_ATTEMPTED=0
RESTORED=0
finish() {
    local rc=$?
    trap - EXIT INT TERM HUP
    set +e
    {
        echo "TEST_RC=$rc"
        if [[ -n "$TEST_PID" ]]; then
            kill -TERM "$TEST_PID" 2>/dev/null
            for _ in $(seq 1 12); do kill -0 "$TEST_PID" 2>/dev/null || break; sleep 1; done
            kill -KILL "$TEST_PID" 2>/dev/null
            wait "$TEST_PID" 2>/dev/null
        fi
        if (( STOP_ATTEMPTED )); then
            sudo -n "$WRAPPER" start
            for _ in $(seq 1 24); do
                if curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
                    RESTORED=1
                    break
                fi
                sleep 2
            done
            if (( RESTORED )); then
                echo "PRODUCTION_RESTORED=PASS"
                touch "$ROOT/disarmed"
            else
                echo "PRODUCTION_RESTORED=FAIL; WATCHDOG REMAINS ARMED"
            fi
        fi
        if [[ -f "$SERVERLOG" ]]; then
            python3 tools/bench/issue58_c2_trace_audit.py "$SERVERLOG" > "$ROOT/trace_audit.json"
            echo "TRACE_AUDIT_RC=$?"
        fi
        echo "CAPTURE_DIR=$ROOT"
    } >>"$REPORT" 2>&1
    tail -75 "$REPORT"
    python3 - "$REPORT" <<'PY'
import base64, pathlib, sys
lines=pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines()
print("OSC52_CLIPBOARD=ISSUE58_CAMPAIGN")
print("\033]52;c;"+base64.b64encode("\n".join(lines[-75:]).encode()).decode()+"\a",end="")
PY
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
{
 echo "=== ISSUE58 C2 candidate ==="
 date -u -Is
 echo "STAGED_SHA256=$ISSUE58_STAGED_SHA256"
 echo "PRODUCTION_HTTP_BEFORE=200"
 echo "CAPTURE_DIR=$ROOT"
 sudo -n bash -c 'nohup python3 "$1" "$2" "$3" "$4" >/dev/null 2>&1 </dev/null &' _ \
     "$(pwd)/tools/bench/issue58_c2_watchdog.py" "$ROOT" "$WRAPPER" "$STAGED"
 sleep 2
 [[ -f "$ROOT/watchdog_ready" ]] || { echo "BLOCKED: root watchdog not armed"; exit 51; }
 sudo -n kill -0 "$(cat "$ROOT/watchdog.pid")" || { echo "BLOCKED: watchdog missing"; exit 52; }
 echo "ROOT_WATCHDOG=ARMED"
 STOP_ATTEMPTED=1
 sudo -n "$WRAPPER" stop
 for _ in $(seq 1 30); do
     if ! curl -fsS --max-time 1 http://127.0.0.1:8080/health >/dev/null 2>&1; then break; fi
     sleep 1
 done
 ! curl -fsS --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1 || { echo "BLOCKED: C1 still serving"; exit 53; }
 echo "C1_STOPPED=YES"
 NINFER_ADMISSION_TRACE=1 NINFER_SHORT_LANE_RESERVE=1 NINFER_LONG_OUTPUT_THRESHOLD=1024 \
     "$STAGED" "$MODEL" \
     --host 127.0.0.1 --port 18080 --model-id local-model \
     --max-context 131072 --kv-capacity 131072 --prefill-chunk 1792 \
     --kv-dtype q4 --spec mtp --draft-tokens 3 \
     --max-concurrency 2 --max-pending-requests 16 --pending-timeout-ms 180000 \
     --vision --vision-max-tokens 2048 --default-thinking-budget 2048 \
     --prefix-checkpoint-policy rolling-tool --embedding-host \
     >"$SERVERLOG" 2>&1 &
 TEST_PID=$!
 echo "$TEST_PID" >"$ROOT/test.pid"
 ready=0
 for _ in $(seq 1 40); do
     if curl -fsS --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then ready=1; break; fi
     kill -0 "$TEST_PID" 2>/dev/null || break
     sleep 2
 done
 ((ready)) || { echo "C2_STARTUP=FAIL"; exit 54; }
 echo "C2_STARTUP=PASS"
 timeout -s TERM -k 5 120 python3 tools/bench/issue58_c2_capture.py "$ROOT/client.jsonl" 95
 echo "WORKLOAD_COMPLETED=YES"
} >>"$REPORT" 2>&1
