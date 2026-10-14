# Known limitations — v0.1.0-integration

- This is an integration release, not certified production readiness.
- Basic Crowd V1 occupancy analysis counts visible detected people inside a configured ROI (or explicitly selected full frame). It uses configured capacity; it does not estimate safe capacity.
- Dense/occluded crowds are undercounted. No tracking, unique-person counting over time, entry/exit flow, speed estimation or future congestion prediction is exposed by this API. Historical standalone tracking experiments are not delivered API features.
- MVP crowd thresholds are configurable operational assumptions, not certified safety limits. Default camera catalog is empty. Missing camera configuration means crowd analysis was skipped, not that a crowd is absent.
- ROI and capacity must be approved for each real camera. Example camera configuration is synthetic and not an Expo Saudi site measurement.
- Fire/smoke thresholds and model weights are unchanged. Smoke and flame evidence both produce FIRE review candidates. Detection does not confirm an incident.
- Local Arabic classification is limited keyword/negation logic, not general language understanding. Confidence can be null. Correlation is a heuristic score, not a calibrated probability. Summaries use only supplied alert facts.
- General Multimodal Vision: **Not connected to production provider**.
- Speech-to-Text: **Not connected to production provider**. No audio HTTP route is published.
- Recommendations require approved host-managed playbooks. None are supplied; the default response is NO_APPROVED_PLAYBOOK. The algorithm is tested with synthetic fixtures, which are not operational instructions.
- No authentication or rate limiter is implemented. Deploy on a private network behind the backend/gateway; do not expose the bare service publicly. Swagger uses external CDN assets by default and may need internet in the viewing browser.
- Request-size limits exist; there is no hard model-execution timeout or cancellation. Model calls serialize per worker. Each worker loads its own weights and consumes memory. CPU load/concurrency capacity and CUDA execution are not validated for production.
- HTTP timeout, 503, ERROR and UNUSABLE must never be treated as “no danger”. No automatic incident creation or persistence occurs.
- Docker: **NOT VERIFIED**; Docker executable is unavailable on this host. Linux dependency resolution/runtime remain unverified.
- Precision, recall, F1, mAP50, mAP50-95, false-positive/false-negative rates and production error rate: **NOT MEASURED** on an independent reviewed dataset. Regression images verify reproducibility, not accuracy. Per-request latency is diagnostic, not a throughput benchmark.
- Model redistribution/deployment licensing and provenance must be confirmed by the project owner; weights are the exact existing artifacts, not newly downloaded replacements. Public-source regression image URLs are retained in their sources.json.
