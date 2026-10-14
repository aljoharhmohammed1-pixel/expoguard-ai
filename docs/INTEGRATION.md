# Backend integration — v0.1.0-integration

Backend → ExpoGuard AI API → structured analysis → backend creates/manages Alert/Incident.

Base URL locally: `http://127.0.0.1:8000`. Swagger: `/docs`. OpenAPI: `/openapi.json`; exported schema: [openapi.json](openapi.json). A localhost link only works on the machine running the service. For remote teammates, deploy on an agreed private host and share `https://<private-host>/docs` through the team's gateway/VPN. GitHub stores code; it does not run this Python service. Do not advertise a placeholder as a deployed URL.

## Responsibility and security

The AI service provides structured AI analysis. It does NOT create or persist Incident records, send Firebase notifications, assign staff or tasks, or merge alerts. It returns AI evidence and suggestions only. The backend owns alert/incident persistence, final merge decision, notification delivery, staff/task assignment, incident lifecycle and final operational authorization.

Authentication is not implemented in this service. Use private network access and the backend's existing gateway controls. No API key is required by the current service. Do not send user media to external providers; none are connected.

## Actual endpoints

| Method/path | Content type | Required fields | Response |
|---|---|---|---|
| GET /health | none | none | status, version, device, modelsReady |
| GET /v1/models | none | none | models array with IDs/versions/task/classes or purpose and readiness; no local paths |
| POST /v1/analyze/image | multipart/form-data | file, requestId, imageId, source, capturedAt; cameraId for CCTV | UnifiedAnalysis v2 |
| POST /v1/analyze/text | application/json | requestId, textId, text | UnifiedAnalysis v2 |
| POST /v1/correlation/analyze | application/json | left, right (each alertId, category, observedAt) | proposal, scoreBreakdown |
| POST /v1/summary | application/json | alerts (1–100 unique alertId facts) | summaryAr, sourceCount, alertIds, policyVersion, limitations |
| POST /v1/recommend | application/json | referenceId, category, evidenceIds (nonempty) | status, proposals, reasons, policyVersion, requiresHumanReview |

Image source: CCTV or VISITOR_APP. capturedAt/observedAt require timezone. Image optional metadata: cameraId, zoneId, assetId, eventId. Catalog zone must agree with supplied zone, otherwise 422. Text source defaults VISITOR_APP, language defaults ar; optional zoneId/assetId. Text max 20,000 characters. Correlation optional zoneId/assetId/eventId/summary/aiConfidence and policy (default 300-second window, 0.7 review threshold). Recommendation optional zoneId/assetId; callers cannot supply playbooks in the HTTP request. Unknown JSON fields are rejected. Read the exported schema for all exact enums, ranges and nullable fields.

## Unified analysis contract

`schemaVersion` is 2.0.0, independent of service release version. requestId and the appropriate imageId/textId identify the input. `modelVersion`, `policyVersion` and `componentVersions` identify analysis components; preserve them with the evidence. Image top-level model version remains the fire model; crowd versions/configuration digest appear in componentVersions when applicable. Source/capture metadata is not all repeated in the canonical response: retain the original request in the backend.

`detections` contains evidence IDs, labels, nullable confidence, normalized [x1,y1,x2,y2] boxes where available, evidence text and nullable textSpan. Text offsets are Python Unicode code points [start,end), not UTF-16 positions. `candidates` reference evidence IDs and require review. Nullable confidence/severity/location is unknown, not zero or an inferred location. imageQuality is applicable to images. Non-applicable imageId/textId/audioId/imageQuality/transcript/error can be null. `crowdMetrics` is omitted when not applicable; do not require it or replace absence with zero.

| analysisStatus | Meaning |
|---|---|
| OK | Analysis completed. Empty detections and candidates means no supported evidence was detected, not a safety guarantee. |
| UNUSABLE | Input cannot be analyzed reliably, e.g. corrupt or too dark. Check imageQuality/limitations; request another input. |
| ERROR | Analysis failed; inspect error code. This is never a negative detection result. |

## HTTP and retries

| HTTP | Actual meaning |
|---|---|
| 200 | Processed response; image can be OK or UNUSABLE. Inspect analysisStatus and evidence. Operational endpoints use their own status/decision fields. |
| 400 | Possible malformed transport/multipart request; not the normal schema-validation code. |
| 413 | File/request body too large (image default 10 MiB; multipart overhead allowance 64 KiB; JSON 256 KiB). |
| 415 | Unsupported image MIME type; supported JPEG, PNG, WebP, BMP. |
| 422 | Invalid schema/form/metadata or camera-zone conflict. |
| 500 | Unexpected service failure, redacted error. |
| 503 | Image AI ERROR or unavailable health/concurrency. Required model startup failure may instead prevent the service from accepting connections at all. |

Transport errors use `{"error":{"code":"INVALID_REQUEST","message":"Request could not be completed."}}` with relevant code. Image inference failures use the full UnifiedAnalysis ERROR envelope. Uvicorn/proxies can emit non-JSON errors (e.g. concurrency limits); handle connection failure and non-JSON as service unavailability too.

Start integration with a 5-second connection and 120-second response timeout on CPU, then tune using deployment measurements; this is a client recommendation, not a server execution deadline/SLA. Retry only transient network/503 failures with bounded exponential backoff (e.g. two attempts), retaining requestId. Requests are analysis-only, but service-side deduplication is absent; backend must prevent duplicate alerts. Do not repeatedly retry corrupt files, 413, 415 or 422 unchanged. Never convert timeout/unavailability into “no danger detected”.

## Crowd configuration

ROI is supported through the host camera catalog selected by CAMERA_CONFIG_PATH. Coordinates are normalized to [0,1]; each detected person's bottom-center is tested against the polygon (boundary included). Alternative full-frame counting requires useFullFrame=true. Default `config/cameras.json` contains no real cameras. See [camera example](cameras.example.json), which is only a test configuration.

crowdMetrics: peopleCount = visible detections in the ROI; capacity = configured capacity or null; occupancyRatio = peopleCount/capacity or null (can exceed 1); crowdLevel = configured classification or null; cameraId/zoneId = camera metadata; roiApplied indicates polygon use. Verify exact field names against OpenAPI. Missing capacity does not invent a density percentage. HIGH/CRITICAL produce review candidates; MEDIUM does so only if reviewMedium is enabled. Default thresholds: 0.5/0.7/0.9. Larger inferenceImageSize can cost substantial CPU time. These are not certified safety thresholds.

## Handoff acceptance

The receiving backend team must test its own authentication, IDs/enums, persistence, retries, lifecycle and human approval rules against these responses. Supply approved camera geometry/capacity and approved operational playbooks separately. Integration examples are captured responses, not accuracy certification or instructions to execute actions.
