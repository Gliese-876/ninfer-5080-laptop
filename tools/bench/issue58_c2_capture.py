#!/usr/bin/env python3
"""Issue58 candidate C2 request capture. LOCALHOST ONLY; hard client timeout."""
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request

CASES = (
    ("long_a", 1536, None, 0.0),
    ("unhinted_budget", 16384, None, 0.25),
    ("short_after_long", 256, False, 2.0),
)
PROMPT_LONG = "Give a detailed numbered technical account of scheduling fairness, progressing through as many examples as permitted."
PROMPT_SHORT = "Say OK and stop."

def request_case(name, tokens, hint, delay, url, limit):
    time.sleep(delay)
    start = time.monotonic()
    payload = {"model": "local-model", "messages": [{"role": "user",
               "content": PROMPT_SHORT if name.startswith("short") else PROMPT_LONG}],
               "max_tokens": tokens, "stream": True, "temperature": 0.1}
    if hint is not None:
        payload["ninfer_short_operation"] = hint
    result = {"case": name, "budget": tokens, "hint": hint, "scheduled_delay_s": delay,
              "started_monotonic_ns": time.monotonic_ns()}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local-model"})
    try:
        with urllib.request.urlopen(req, timeout=limit) as resp:
            result["http"] = resp.status
            chunks = 0
            for line in resp:
                if time.monotonic() - start > limit:
                    raise TimeoutError("per-request monotonic deadline exceeded")
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    result["done_marker"] = True
                    break
                try:
                    packet = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    continue
                for choice in packet.get("choices", []):
                    delta = choice.get("delta") or {}
                    if "ttft_s" not in result and (delta.get("content") or delta.get("reasoning_content")):
                        result["ttft_s"] = round(time.monotonic() - start, 4)
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
                    chunks += 1
            result["chunks"] = chunks
    except (Exception) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - start, 4)
    result["ended_monotonic_ns"] = time.monotonic_ns()
    return result

def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: issue58_c2_capture.py OUTPUT.jsonl TIMEOUT_SECONDS")
    path, seconds = sys.argv[1], float(sys.argv[2])
    if not (10 <= seconds <= 100):
        raise SystemExit("timeout must be between 10 and 100s")
    url = "http://127.0.0.1:18080/v1/chat/completions"
    # No request is ever sent to production port 8080.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(request_case, *case, url, seconds) for case in CASES]
        results = [future.result(timeout=seconds + 6) for future in futures]
    with open(path, "x", encoding="utf-8") as dest:
        for result in results:
            dest.write(json.dumps(result, sort_keys=True) + "\n")
            print(json.dumps(result, sort_keys=True), flush=True)
    failures = sum(x.get("http") != 200 or "error" in x for x in results)
    print(f"CAPTURE_COMPLETE=1 CASES={len(results)} FAILURES={failures}", flush=True)
    return 0 if failures == 0 else 4

if __name__ == "__main__":
    raise SystemExit(main())
