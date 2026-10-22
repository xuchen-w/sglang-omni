# AuK / Voxtral shared-runtime compatibility review

Reviewed on 2026-10-23 (Asia/Shanghai): upstream #2444 at `69b1651e7a935e6592945b4b8b0cef94ebd71394` and #2926 at `fa25721a5294299cf66e36ea23f125229d68555d`.

The compatibility patch is now on #2444 at [cc662c38](https://github.com/xuchen-w/sglang-omni/commit/cc662c380db67bd4fdaab2c95b8f082ddd948e69). The integration code tree is [4cb42330](https://github.com/xuchen-w/sglang-omni/commit/4cb42330b6d080891429a3f6f9f4f3d4f88f7015), combining that patch with #2926. This branch is a reproduction/reference branch, not a request to merge another upstream PR.

## Findings and resolutions

| Area | Finding | Resolution |
|---|---|---|
| Shared service / ASR adapter | Keeping #2444's migrated ASR implementation while merging #2926's header compiles, but the model never receives its name/provenance. Voxtral's explicit model-name mismatch no longer returns 404. | Preserve the final nonempty CLI-name flag before generating a default. Call optional Service::SetModelName before Register; forward through AsrService to ServedModel before AddHandlers. Accessors match #2926; existing Register signatures remain unchanged. |
| Golden dispatch | check_golden.py merges automatically. report() exits, so AuK does not fall through into ASR scoring. | Retain AuK's early checker and Voxtral's synthesis/final-checker branches. Add CLI dispatch regression coverage rather than a new registry. |
| Mac CI prerequisites | #2926's M4 Pro run failed its disk gate, then still downloaded models and ran missing binaries. #2444 also had unconditional downstream tests. | Gate downloads/tests on successful setup/build/provisioning. Continue independent tests after golden failures. Failed AuK preparation skips its own goldens with failure status, while other models still run. Always upload available artifacts. |
| CMake / model pins / test file | Additive edits conflict in the same insertion regions. | Preserve both sets of targets, model pins and tests. No evidence of incompatible library behavior. |
| Workflow disk budget | Choosing either fixed 20 GiB or AuK's unchanged 17 GiB baseline loses part of the combined budget. | Integration keeps the AuK dynamic missing-source/preparation/replacement formula and raises its baseline to 20 GiB. |
| README | Additive documentation merges automatically. | Retain both model entries. |

A trial merge of the reviewed heads has five textual conflicts: workflow, CMakeLists.txt, model_pins.json, test_check_golden.py and asr_service.cpp. Of the seven shared changed files, README and check_golden.py merge automatically. The model-name issue is a semantic migration risk beyond resolving conflict markers.

The patch does not change the ASR transcription/realtime interfaces, loading thread, supervisor shutdown protocol, RequestStates forwarding or cancellation path. Voxtral's decoder options preserve existing ASR defaults. Voxtral retains its bounded queue, 503 and timeout behavior; it should not be switched to AuK's single-request/429 policy as part of this compatibility work.

## Regression coverage and validation

The patch adds a test-only asr_service_probe target, excluded from default builds and not installed as a production executable. Six parameterized HTTP cases inspect name/provenance at handler registration, /v1/models consistency and Transcribes=false, covering generated names, explicit names, empty names, repeated flags, and explicit names resembling generated names. Mac CI builds and runs the probe.

- #2444 golden harness: 13 passed.
- Combined AuK / AuK-Flash / Voxtral golden harness: 18 passed.
- Actual common/service.cpp and common/http.cpp linked to pinned CivetWeb on Linux: six HTTP cases passed; the same cases all failed with the pre-fix common/service.cpp.
- ASR adapter and native probe: C++20 syntax checks passed against MLX 0.32.3 headers. Full adapter execution was not performed on Linux.
- Workflow expressions and the actual golden shell loop were checked for disk/setup/build failure, cancellation, independent golden failure and failed AuK preparation.
- Full pre-commit passed except the unrelated Rust hook, which could not run because the host lacks cargo. Changed-file formatting/lint checks passed.
- Full Apple Silicon build, native ASR-probe execution and combined model API/numerical tests remain required.

## Current review and CI state

#2926 remains open at fa25721a5. Its seven inline threads are resolved, but there is no human approval. Copilot's latest review is a comment, not an approval. The [M4 Pro run](https://github.com/sgl-project/sglang-omni/actions/runs/39812656958/job/119942900534) failed with 16 GiB free against a 20 GiB gate; the build was skipped, so subsequent missing-binary errors are not numerical regressions.

Before this compatibility patch, #2444 had four unresolved review threads whose fixes had been submitted, including produced-length resampling, interrupted artifact recovery/tests, and shared TTS responses. Unresolved threads should not be described as proof that those bugs remain. Its run at 69b1651e was awaiting fork workflow approval. Both PRs already have run-mac-ci.

The combined disk formula requests approximately 72 GiB free for cold AuK sources/artifacts, or 34 GiB when both prepared artifacts and sources are cached. These are code-budget estimates, not measured total peak usage. The observed 16 GiB runner capacity is insufficient.

AuK/AuK-Flash currently only record Apple M5 Pro goldens and fail closed on an unrecorded chip after exporting results. The M4 Pro bootstrap therefore requires a run, review/import of the exported outputs, and a rerun. Voxtral currently records Apple M4 and permits an unrecorded chip to check prompt IDs only; import reviewed M4 Pro codes for full seeded audio-code regression. Do not weaken either policy to obtain a green run.

The model golden loop and API steps are sequential across the two TTS families. The combined 60-minute job budget and peak memory still need runner measurement. The AuK server's 1 GiB MLX cache cap does not establish the memory bound of its separate synthesis CLI.

## Suggested ownership and coordination

Keep the shared-runtime compatibility change in #2444, which owns the lifecycle extraction and already coordinates shared components with #2870. #2926 should rebase onto that interface when maintainers decide the merge order, retaining both model-specific golden and API paths. If #2926 lands first, #2444 should adapt its extraction during rebase using the same name-propagation contract; there is no need for both authors to redo the shared runtime.

The Voxtral CLI-dispatch test is included in this integration branch and can be carried with #2926's rebase. The name-propagation fix, generic probe, AuK dispatch tests and prerequisite guards have already been pushed to #2444.

Posting this review to the upstream PR conversation was attempted but GitHub returned HTTP 403, Resource not accessible by integration. No upstream coordination comment was published; this note is ready to share with the two authors and maintainers.
