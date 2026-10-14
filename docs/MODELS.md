# Model weights handoff

These are the existing project artifacts; no weights were changed or downloaded during delivery preparation. The two exact files are tracked with Git LFS through .gitattributes. A normal source ZIP without LFS objects is insufficient.

## Obtain and verify

The source is this project’s private repository LFS storage once the owner pushes it. Until a remote is published, obtain the two existing files directly from the project owner by the team’s approved private file transfer, retaining the paths and hashes below. No hosted download URL has yet been provisioned. Do not substitute another model with the same filename.

```powershell
git lfs install
git lfs pull
git lfs fsck
Get-FileHash ai/models/infernoguard_best.pt -Algorithm SHA256
Get-FileHash ai/models/yolo26n.pt -Algorithm SHA256
```

A Git LFS pointer is not a model file. The service never downloads missing weights. Startup fails for missing/invalid required weights. /v1/models exposes versions, readiness and checksums without filesystem paths. Both artifacts are smaller than 100 MiB; LFS is used deliberately for binary handoff and versioning. Ensure the private remote has LFS storage enabled and adequate quota before sharing a clone URL.

## fire-smoke

- **modelId**: `fire-smoke`
- **modelVersion**: `fire-smoke-infernoguard-v1.0.0`
- **purpose**: `Fire and smoke object detection`
- **framework**: `Ultralytics / PyTorch`
- **path**: `ai/models/infernoguard_best.pt`
- **required**: `Yes, service startup`
- **sizeBytes**: `18346049`
- **sha256**: `5fe6eb54fc9006e6f31c178fa2efe04524e2faf6b6955e8c6967d57ffd008ab5`

## person-detector

- **modelId**: `person-detector`
- **modelVersion**: `yolo26n-person-v1.0.0`
- **purpose**: `Person detection for Basic Crowd V1 occupancy analysis`
- **framework**: `Ultralytics / PyTorch`
- **path**: `ai/models/yolo26n.pt`
- **required**: `When a camera enables valid crowd ROI/full-frame analysis`
- **sizeBytes**: `5544453`
- **sha256**: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`

## Non-weight components

Arabic classification uses `arabic-rules-v0.1.0`; no weight file is required. Correlation, summary and recommendations use local deterministic rules. General vision and speech have no connected production provider. Camera configuration is separate from model weights.

Fire labels map to visible_flame/suspected_smoke. Fire confidence remains 0.35 and image size 640. Crowd person confidence remains 0.25; bounded image size is configured per camera. This delivery changes no model thresholds.

Review the original models’ provenance and licensing before wider redistribution or production deployment; this handoff records artifacts, not a new license grant. The fire model’s upstream training provenance is not established by a checksum.
