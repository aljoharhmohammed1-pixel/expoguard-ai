# ExpoGuard AI

Local AI evidence and human-review suggestions for event operations, exposed through FastAPI. The backend manages alerts and incidents.

## Current Integration Version

**v0.1.0-integration** — integration release, not production certification.

## Available Features

| Feature | Status |
|---|---|
| Fire/Smoke Detection | Real local weights tested |
| Image Analysis API | Implemented; unified v2 contract |
| ROI Support | Supported through camera configuration |
| Crowd V1 Occupancy Analysis | MVP / Limited — Basic Crowd V1 occupancy analysis |
| Text Classification | Local Arabic rules; limited vocabulary and negation handling |
| Alert Correlation Assistance | Local heuristic review proposals |
| Summary Assistance | Deterministic summary of supplied facts |
| Playbook Recommendations | Engine tested; no approved operational playbooks supplied |
| General Multimodal Vision | Not connected to production provider |
| Speech-to-Text | Not connected to production provider |
| Docker | NOT VERIFIED — Docker executable unavailable |

Release evidence: [acceptance](docs/release/ACCEPTANCE.md). Real HTTP responses: [API examples](docs/API_EXAMPLES.md).

## Architecture and scope

`Backend → FastAPI → input validation → local AI analysis → structured evidence/suggestions → Backend`

- `app/`: HTTP routes, model lifecycle, safe logs, camera/playbook loading.
- `ai/core/`, `ai/contracts/`: validated analysis and advisory schemas.
- `ai/inference/`, `ai/quality/`, `ai/crowd/`, `ai/text/`: local analysis.
- `ai/operations/`: correlation, summaries and approved playbook retrieval.
- `ai/providers/`, `ai/audio/`, `ai/vision/`: unconnected provider ports.
- `ai/tests/`, `ai/evaluation/`: tests and evaluation tools.
- `config/`, `playbooks/`: operator configuration; default cameras empty and no operational playbooks.

The AI service provides structured AI analysis. The backend is responsible for:

- Alert persistence and Incident persistence.
- Final merge decision and incident lifecycle.
- Notification delivery and staff/task assignment.
- Final operational authorization and human review.

No Flutter, Firebase, WhatsApp transport, backend domain changes, tracking or crowd prediction are included. Legacy standalone experiments are not API features.

## Installation

Use Python **3.11**, Git and Git LFS. Substitute the actual private repository URL supplied by the owner:

```powershell
git clone <PRIVATE_REPOSITORY_URL> expoguard-ai
cd expoguard-ai
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

requirements.txt pins the dependency closure from the working project environment, including CPU Torch/torchvision, using PyPI and the PyTorch CPU index. It excludes unrelated machine packages. Release verification targets Windows Python 3.11. Linux/Docker and CUDA need separate validation. `ai/requirements-contract.txt` is schema-only, not the service installer.

## Model setup

Required: `ai/models/infernoguard_best.pt`. Crowd-enabled cameras also require `ai/models/yolo26n.pt`. Both are tracked with Git LFS. Check sizes and SHA256 in [MODELS.md](docs/MODELS.md). Before the private repository is published, obtain the exact files from the owner through approved private transfer. Models are not auto-downloaded or retrained.

## Environment variables

.env.example lists actual supported variables. Local startup reads process environment variables; it does **not** automatically load .env. Docker supports --env-file.

| Variable | Default | Purpose |
|---|---|---|
| APP_ENV | development | development / production / test |
| AI_SERVICE_VERSION | 0.1.0-integration | Service release label |
| FIRE_MODEL_PATH | ai/models/infernoguard_best.pt | Trusted local weights |
| PERSON_MODEL_PATH | ai/models/yolo26n.pt | Person weights |
| MODEL_DEVICE | auto | CPU unless CUDA available; cpu/cuda accepted |
| MAX_IMAGE_SIZE_MB | 10 | Upload limit, 1–20 MiB |
| CAMERA_CONFIG_PATH | config/cameras.json | Host camera catalog |
| INCLUDE_PERSON_DETECTIONS | false | Optional person evidence boxes |
| PLAYBOOK_DIR | playbooks | Host approved catalog |

Fire confidence **0.35**, image size **640**, policy **fire-review-v0.3.0** remain unchanged; unused threshold/policy environment variables are not advertised. Runtime dependency auto-install is disabled.

ROI is supported through normalized camera polygons. The bottom-center of a person's box determines membership. Capacity must be configured, not inferred. [Example camera catalog](docs/cameras.example.json) is synthetic, not approved Expo Saudi site geometry. Enable it only for integration demonstration:

```powershell
$env:CAMERA_CONFIG_PATH = "docs/cameras.example.json"
```

## Local startup

From repository root:

```powershell
$env:MODEL_DEVICE = "cpu"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --limit-concurrency 8 --no-access-log
```

Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
OpenAPI: [http://127.0.0.1:8000/openapi.json](http://127.0.0.1:8000/openapi.json); exported [schema](docs/openapi.json).

Localhost works only on the running machine. Share a deployed private host URL ending in /docs, or have teammates run their clone. GitHub stores code and does not run this API. No authentication is implemented: use the team's private network/gateway; do not expose the bare service publicly.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Service version, device and readiness |
| GET | /v1/models | Model IDs, versions, task, classes and readiness |
| POST | /v1/analyze/image | Multipart image and metadata → UnifiedAnalysis v2 |
| POST | /v1/analyze/text | Arabic text JSON → UnifiedAnalysis v2 |
| POST | /v1/correlation/analyze | Alert pair → suggestion and score explanation |
| POST | /v1/summary | Supplied facts → Arabic summary |
| POST | /v1/recommend | Facts → approved proposals or explicit absence |

No video-upload, audio or general-vision HTTP routes. Legacy Python envelopes differ from HTTP v2; use exported OpenAPI for this release.

## Request/response examples

```powershell
curl.exe http://127.0.0.1:8000/v1/analyze/image -F "file=@ai/datasets/fire_smoke_samples/fire_1.jpg" -F "requestId=demo-1" -F "imageId=image-1" -F "source=VISITOR_APP" -F "capturedAt=2026-10-14T09:00:00Z"
```

Image requires file, requestId, imageId, source, capturedAt; CCTV also requires cameraId. A normal-image response includes this actual field subset (full envelope in the examples):

```json
{"schemaVersion":"2.0.0","analysisStatus":"OK","detections":[],"candidates":[]}
```

Full captured request/response examples for fire, smoke, normal, unusable, configured/unconfigured crowd, text, correlation, summary, recommendation and errors: [API_EXAMPLES.md](docs/API_EXAMPLES.md).

HTTP 200 does **not** mean danger detected. Inspect analysisStatus, detections, candidates and crowdMetrics when present. OK with empty findings is completed negative detection, not guaranteed safety. UNUSABLE is insufficient input quality. ERROR/timeout/unavailability must never mean “no danger”. Null values are unknown. [INTEGRATION.md](docs/INTEGRATION.md) documents fields, errors, retries and boundaries.

## Testing

```powershell
.\.venv\Scripts\python.exe -m ai.run_tests --report docs/release/FAST_TESTS.json
.\.venv\Scripts\python.exe -m ai.run_tests --integration --report docs/release/TESTS.json
.\.venv\Scripts\python.exe scripts/verify_integration.py
```

Fast tests intentionally skip real-weight cases. Full integration uses local weights. The last command starts/stops its own Uvicorn server, sends real HTTP requests and exports examples/OpenAPI with a synthetic camera catalog only for that subprocess. Default production configuration is unchanged. Six public-source regression images plus their expected detections are included for reproducibility, not accuracy measurement.

## Docker

**NOT VERIFIED:** Docker executable is unavailable on the verification host. No build/run success is claimed. Intended commands:

```powershell
docker build -t expoguard-ai:0.1.0-integration .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env.example -e MODEL_DEVICE=cpu --mount "type=bind,source=$((Get-Location).Path)/ai/models,target=/service/ai/models,readonly" expoguard-ai:0.1.0-integration
```

Weights are excluded from build context and mounted read-only. For configured cameras/playbooks, mount approved configuration and set relevant environment paths. UID 10001 needs read access to mounts. Verify build, health, models, Swagger and image analysis on the target host before deployment.

## Known limitations and handoff

Read [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md), [MODELS.md](docs/MODELS.md), [INTEGRATION.md](docs/INTEGRATION.md), [release acceptance](docs/release/ACCEPTANCE.md) and [CHANGELOG](CHANGELOG.md).

Crowd V1 measures visible persons/ROI/capacity/occupancy/level. It does not reliably count occluded crowds, track unique people, measure flow or predict congestion. Recommendations need approved playbooks and backend/human authorization. Independent accuracy metrics and production capacity are not measured.
