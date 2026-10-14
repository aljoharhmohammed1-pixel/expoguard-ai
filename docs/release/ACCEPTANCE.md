# Integration release acceptance

Version: **v0.1.0-integration**. Target: native Windows x64 / Python 3.11.9 / CPU. This is local integration readiness, not production certification or backend end-to-end approval.

## Executed verification

| Check | Result | Evidence |
|---|---|---|
| Existing environment full suite | 148 passed / 0 failed / 0 skipped | TESTS.json; real weights and legacy regressions included |
| Clean environment full suite | 148 passed / 0 failed / 0 skipped | TESTS_CLEAN.json |
| Clean install | PASS | New `.venv-integration-test`, installed only requirements.txt; pip check: no broken requirements |
| Real HTTP service acceptance | 17 checks PASS | ACCEPTANCE.json and API_RESPONSES.json |
| Health/version/device/models | PASS | Running Uvicorn from clean environment, CPU; both models ready with example camera config |
| Fire, smoke, normal, corrupt images | PASS | Real weights; normal has empty findings; corrupt returns HTTP 200 / UNUSABLE |
| ROI-configured and unconfigured camera | PASS | Configured camera reports visible people; unknown camera omits crowdMetrics with limitation |
| Arabic text, correlation, summary | PASS | Actual local implementations via HTTP |
| Recommendations | PASS for current behavior | NO_APPROVED_PLAYBOOK; no actions invented; approved-fixture path covered by unit/API tests |
| Validation and size/type errors | PASS | HTTP 422, 413 and 415 |
| Swagger | PASS for HTML delivery | `/docs` returns Swagger page; browser CDN availability is not certified |
| OpenAPI | PASS | Exported from running service, parsed as JSON; service version checked |
| Docker | NOT VERIFIED | `docker` executable is not available; build/run could not be attempted |
| GitHub | NOT PUSHED | `gh` executable is not available; no remote repository URL configured |

The temporary clean environment is ignored and retained locally for repeat verification. Installation used available package-download cache in a new isolated venv, not system-site-packages. No dependency was added to the user's unrelated global Python environment.

Only release metadata/default version and packaging/documentation changed in application delivery preparation. No AI weights, thresholds or inference algorithms changed. Existing Starlette emits a TestClient httpx deprecation warning; tests pass with the pinned versions, and no new dependency was introduced just to suppress it.

## Model and data integrity

Both existing weights have recorded SHA256 and size in MODEL_MANIFEST.json and are tracked with Git LFS. No model downloads or training were performed. Only six public-source regression images and the existing expected-output baseline are included; provenance is in their sources.json. Large datasets, videos, demos, logs, caches and virtual environments remain outside Git.

The commit candidate set was scanned for common credential/token patterns, private keys, credential-bearing URLs and personal absolute filesystem paths. No matches were found. `.env` is excluded, `.env.example` has safe configuration, and only the intended two LFS weight files are eligible. Pattern scanning is not an external security certification. Final staged-file checks are recorded in SECURITY_CHECK.json.

## Remaining external acceptance

- Publish the local repository and tag to a **private** GitHub remote, including LFS objects; see PRIVATE_GITHUB.md. No repository URL or shared Swagger host has been invented.
- Deploy to the agreed private backend environment, verify Docker there if desired, and test backend IDs/enums, gateway access, retry/deduplication and persistence behavior.
- Supply real approved camera ROI/capacity and operational playbooks. Current examples are synthetic configuration, not Expo Saudi measurements.
- External speech/general-vision providers remain unconnected. Accuracy metrics, concurrency capacity, CUDA and production behavior remain unverified.

The final Git commit and tag IDs are reported outside this committed file to avoid self-referential commit hashes. The tag is created only after documentation and verification are finalized and the tracked worktree is clean.
