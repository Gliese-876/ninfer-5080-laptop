# NInfer preservation reconciliation — 2026-10-12

**Status:** Source exports archived; production `main` unchanged. This is a record of inspection, not permission to delete unique evidence.

## Verified remote archives

- Issue #58 source + decisions: commit [00de582](https://github.com/toddballinger/ninfer-5080/commit/00de582e8e83684108bad200647b39e78f7f36a4), `docs/issue58/artifacts/issue58-source-export-20261012.tar.gz`, SHA256 `7cfe460dcc9557f5092fa50e5ddccd9abc2a198cd75fafe7318b85eae535e377`.
- Multi-repo source/status reconciliation: commit [6a0c32b](https://github.com/toddballinger/ninfer-5080/commit/6a0c32b809f40254f5f4569e034305675a57e01e), `docs/preservation/reconciliation-20261012T140953Z.tar.gz`, SHA256 `da97ff62dc1dd913867958d2ce7ae7f951f61df251bdde3a29e839f9c5246446`.
- Four original `docs/issue58/` decision/manifest documents were separately published and verified on this branch.
- The initial 4.1 MB inventory and 1.7 MB extension inventory remain separate scan artifacts; neither is asserted to be published in this branch.

## Source reconciliation

### Issue #45: independent sampling oracle

Local checkout `/home/toddballinger/ninfer-5080-issue45-sampling-oracle` HEAD `318bc8a1378d3d02302a2708831235db13520b75` has:
- modified `tests/CMakeLists.txt` (registration), and
- untracked `tests/ops/test_sampling_oracle.cpp` (21,305 bytes; SHA256 `88bba839bc0b4cefa5f68ac65a9258b9109df482ffdfded41be0347fa33fe6ae`).

Both originals are included in the reconciliation export. This is candidate independent CPU sampling coverage, **not** a demonstrated production sampling bug. Compare with existing [PR #68](https://github.com/toddballinger/ninfer-5080/pull/68) and [Issue #45](https://github.com/toddballinger/ninfer-5080/issues/45); do not merge the older checkout blindly.

### Issue #58: KV-denial identity / boundary facts

Local checkout `/home/toddballinger/ninfer-5080-issue58-kv-denial-request-id` HEAD `464a687d9ed29a9436510ea6a9bac77bd18efb95` has 3 modified tracked files and 2 untracked source/docs files. Main includes the same destination paths through [PR #81](https://github.com/toddballinger/ninfer-5080/pull/81), and current main tracked file identity for `src/runtime/engine/concurrent_executor.h` and `tests/CMakeLists.txt` matches the resulting versions shown in the local patch headers. However, the local untracked test/document blobs differ from main: local docs `d7ba782a25bc9bfbd0cf4437ad335d74b2c88f19` vs main `dc4df84d048ec590ed90c88ba1f2cf3946d18745`; local test `b28868ccb2787caa0e94f287ee49a486b45e091a` vs main `3ccdef5d472a6b722d9a3e9ec01a7a6c7084908d`. Treat local versions as preserved variants, **not identical duplicates**. Main already documents their technical subject, so do not open another root-cause issue solely because blobs differ. Source variants are in the export.

### Other repositories

- Nine previously ownership-blocked Git statuses were re-run: eight clean; `/home/toddballinger/Strata-5080` contains three untracked runtime/log/config files outside NInfer; preserve privately, do not publish indiscriminately.
- `/home/openclaw/ninfer-issue58-transfer` at `7f4e3fc14dba0b40` has 1,245 staged and working-tree deletions, not evidence of 1,245 engineering innovations. The deletion patch/status is preserved in the reconciliation export. **Do not run reset, clean, or remove this checkout until a separate ownership/unique-objects check.**
- Five experimental Issue #58 worktrees and their patches/tests are preserved in the source export. The rejected temporal-credit change is an archival artifact, **not a scheduler fix**.
- The Opus workspace contained five untracked agent identity/instruction files. Those files are not suitable for publication in this NInfer repository and remain a separate private-backup responsibility.

## Outstanding gates before destructive cleanup

1. Confirm any unique large sealed evidence and benchmark campaigns on Brain have at least one **off-host, checksum-verified** recoverable copy. The GitHub source export does **not** itself include all binary/campaign data.
2. Inspect any worktree about to be deleted for later changes since the 2026-10-12 inventory.
3. Keep protected production build/model/services, credentials, OpenClaw sessions, and any active artifacts intact.
4. Only clean individually identified reproducible build intermediates after checking open use, provenance and dependency.
5. Issue #58 remains P0/open: real workload >600s C2 TTFT and C3 timeout are not explained by a reproduced scheduler bug. The synthetic `refresh_temporal_credit` candidate is rejected.

**Disposition:** GitHub source-evidence reconciliation documented on this archival branch; large-evidence backup and disk-cleanup safety clearance are NOT complete.
