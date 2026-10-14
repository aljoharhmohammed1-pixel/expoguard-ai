# Initial delivery audit

Inspected the current source, configuration, local models, tests and environment before delivery edits. Historical reports are not verification for this release.

| Item | Actual initial state |
|---|---|
| Structure | `app/` HTTP service; `ai/` schemas, inference, quality, crowd, text, provider ports, operations, evaluation and tests; `config/`, `playbooks/`, historical `docs/ai/` |
| Endpoints | GET health/models; POST image/text/correlation/summary/recommend (see integration guide for exact paths) |
| Models | Ultralytics PyTorch fire/smoke and YOLO26n person detector; local Arabic rules |
| Weight paths | `ai/models/infernoguard_best.pt`, `ai/models/yolo26n.pt` |
| Tests | unittest runner `python -m ai.run_tests --integration`; unit, contract, API, real inference and legacy regression checks |
| Dependencies | Python 3.11 project venv; direct pins existed but Torch required a separate installation; transitive lock absent |
| FastAPI | Configured, startup model validation, bounded request bodies, structured errors and Swagger |
| Dockerfile | Exists; Docker executable absent on this host |
| .gitignore | Absent |
| .env.example | Exists, safe local defaults; release version needed updating |
| README | Exists, accumulated instructions; stale claim that ROI is unsupported |
| Crowd | Person detection + bottom-center ROI test + visible count/capacity/occupancy/level; default camera catalog empty |
| Text | Local Arabic keyword/negation rules; nullable confidence; human review |
| Correlation | Time/category/location and lexical Jaccard heuristic; suggestions only |
| Recommendation | Approved, active, non-test host playbooks only; no approved production playbooks supplied |
| Audio | Provider interface only; not connected to production provider; no HTTP endpoint |
| General vision | Provider interface only; not connected to production provider; no HTTP endpoint |
| Git | Not a repository; Git and Git LFS available; GitHub CLI absent |

No architecture, weights, detection thresholds or AI features are changed for delivery. Historical generated galleries and private/unrelated datasets are excluded from the delivery commit. The six public-source regression images and their original provenance manifest are retained for reproducibility; they are not an independent accuracy dataset.
