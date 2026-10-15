---
title: ExpoGuard AI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_path: /docs
pinned: false
---

# ExpoGuard AI v0.1.0-integration

Private FastAPI integration service. Open `/docs` for Swagger, `/openapi.json` for the contract, `/health` for readiness and `/v1/models` for model metadata.

The service.zip archive contains the existing Python service and both exact local weights from the tested integration release. The Docker build extracts it without downloading substitute models. Source commit: d16e553ed41a6d3cf51cb2968eed1fd1ce0a584d. Source repository: https://github.com/aljoharhmohammed1-pixel/expoguard-ai (private).

Image and Arabic text analysis, correlation, fact summaries and approved-playbook recommendation assistance are supported. Basic Crowd V1 requires approved camera ROI/capacity configuration; the default camera catalog is empty. Person weights are included but remain not_configured until a valid camera is enabled. No approved operational playbooks are supplied. Audio and general multimodal providers are not connected.

AI returns evidence/suggestions only. Backend owns incidents, persistence, merging, staff/tasks and notifications. HTTP 200 is not a safety decision: inspect analysisStatus. UNUSABLE/ERROR/timeouts never mean no danger. This is an integration release, not production certification.

The Space must remain PRIVATE. Backend access must use authorized Hugging Face credentials kept server-side. Do not expose tokens in a browser or mobile client. Runtime verification is pending until the Space builds successfully; local Windows verification is not proof of Linux deployment.
