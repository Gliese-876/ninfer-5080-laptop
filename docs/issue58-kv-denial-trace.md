# Issue #58 — specific KV admission denial reasons (diagnostic only)

This patch follows merged #72–#74, and reports numerical denial causes **at the source** in Qwen3.8/Qwen3.6 runtime's `ProgramImplCore::can_admit_lane` and `can_admit_lane_after_retained_eviction`.

When `NINFER_ADMISSION_TRACE=1` (otherwise OFF), `[ADMISSION-KV-DENIAL]` records the lane number, direct or retained-eviction path, main/backend pool, first failed numerical guard, and old/reclaimable/new/entitled/logical/physical page counts. Cause categories: `invalid_old_entitlement`, `invalid_reclaimable_entitlement`, `exceeds_logical_capacity`, `insufficient_physical_pages`; `none` represents no numerical denial. The logger is throttled **per thread** to one line per 10 seconds. A first failure in main pool means no backend check was reached. Other reasons not represented here include invalid plan/lane or active lifecycle; **absence of these logs never proves a request was admitted**. No request content is emitted.

**Optional head request id (this patch).** When the admitting call supplies a shared request id, the line emits an optional **head** field (the request id) after the tag, e.g. `head=7`; when no shared id is supplied the field is **omitted entirely** and the line keeps the exact historical lane-only format. The audit tool (`tools/bench/issue58_c2_trace_audit.py`) accepts both formats backward-compatibly and reports `kv_lane_only` / `kv_with_head_id` plus `kv_denial_joins`: a head-id-bearing denial line is joined to a selected-admission line **only when both share the same request id (and the same lane)**. Legacy lane-only denial lines are **never** joined; they carry no request id, so the prior no-causal-join behavior is preserved. A join is a **shared-id association, not causal proof** — denials are logged from the probing request's perspective and are 10s-throttled/sampled, so a head id never proves the denial's true originating request, and the 18.176s TTFT remains unresolved.

The existing direct and eviction admission boolean predicates are preserved verbatim, independently of the diagnostic classifier; the classifier is used **only for reporting**. No eligibility probes, eviction, policy, queue ordering, device operations or capacity allocations have been added. Trace observations do not establish actual queue-enter/admission wait duration, and are not yet a C2 fix.

Base: main commit `f1830427d6ba23152a821dcf57bd799b4066adcc`. Exactly four files:
- `src/runtime/engine/issue58_kv_denial.h` (pure reason classifier)
- `src/targets/qwen3_6/impl/runtime/program_impl.h` (opt-in source of numerical denials)
- `tools/tests/issue58_kv_denial_test.cpp` (host classifier cases)
- `docs/issue58-kv-denial-trace.md` (this document)

## OpenClaw MAIN one-shot host gate

Fetch exact draft PR HEAD, confirm clean checkout, and validate:
```bash
ccache c++ -std=c++20 -Wall -Wextra -Werror -Isrc tools/tests/issue58_kv_denial_test.cpp -o /tmp/issue58_kv_denial_test
/tmp/issue58_kv_denial_test
git diff --check main...HEAD
git diff --name-only main...HEAD
```
Run the **actual source translation unit** for `program_impl.h` via GCC 15.2 C++20 `-fsyntax-only` with the existing CUDA 13.4 header paths and project definitions (find its source inclusion or current build compile command; do **not** assume `engine.cpp` includes this target header). No nvcc, GPU execution, full CUDA rebuild or production action. Record exact command/RC.

Review predicate equivalence line-by-line including short-circuit order and unsigned arithmetic, conditionally compiled target namespace alias, header availability, and diagnostics side effects. Specifically verify `runtime::` qualification resolves under the target namespace. If compilation or predicate equivalence fails, perform no more than two bounded fixes within the four files, rerun all host gates and publish final HEAD/command/RC.

**Do not deploy or enable tracing on Brain's production C1**, restart any service, run the GPU, reload models, test C2, or merge. Leave issue #58 OPEN. Post evidence to draft PR, return `ISSUE58_KV_VALIDATED_DRAFT` or `ISSUE58_KV_BLOCKED`. Subsequent independent ChatGPT review and explicit merge approval required.

## Limitations
The emitted numerical reason may be sampled and may not represent the same head request as a contemporaneous 10-second lane trace, especially under concurrent workers. The initial trace reports the first observed rejection in a predicate chain, not every possible constraint. The optional head request id is scheduler-local per process/session; never join across sessions or restarts without a verified common namespace. A controlled candidate C2 reproduction and admission timestamps remain necessary before proposing changes to scheduler behavior. The 18.176s TTFT remains unresolved; this telemetry does not claim to resolve it.

## Offline join identity safeguards (independent web review)

A KV-denial/admission association requires **the same source capture file**, identical non-zero request ID and lane, and a denial log line preceding the selected-admission line. IDs can be reused after process restart, so never join across files or independent processes; even within one file a combined multi-process capture may need further provenance inspection. Records lacking an ID remain unjoined. These safeguards prevent several false matches but do **not** establish causal proof or explain TTFT. Tests explicitly cover reused IDs across files, reverse chronology and head=0.
