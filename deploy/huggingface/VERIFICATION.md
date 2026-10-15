# Hugging Face deployment verification

Verified 2026-10-15 (Asia/Riyadh).

- Space: https://huggingface.co/spaces/Basmaalmuntaser/expoguard-ai
- Swagger: https://basmaalmuntaser-expoguard-ai.hf.space/docs
- OpenAPI: https://basmaalmuntaser-expoguard-ai.hf.space/openapi.json
- API base: https://basmaalmuntaser-expoguard-ai.hf.space
- Deployed Space commit: ee3b242442a8ea3072075fcf38f4f193be33a5ec
- Application source commit: d16e553ed41a6d3cf51cb2968eed1fd1ce0a584d
- Visibility: PRIVATE. Hardware: CPU Basic (2 vCPU, 16 GB), displayed as Free. No upgrade was purchased.

## Observed results

Docker built successfully after correcting ownership of the application directory for UID 1000. Startup completed and the Space reported Running. These observations verify this Space-specific Dockerfile, not the unchanged original local Dockerfile.

Health returned status=ok, version=0.1.0-integration, device=cpu, modelsReady=true. Swagger loaded the seven current endpoints and the OpenAPI schema.

/v1/models reported the fire model ready with SHA256 5fe6eb54fc9006e6f31c178fa2efe04524e2faf6b6955e8c6967d57ffd008ab5. Person model status is not_configured because the camera catalog is empty. Both exact model artifacts were included and their hashes checked before upload.

A real image POST through authenticated Swagger returned HTTP 200, analysisStatus=OK, 3 detections (one suspected_smoke and two visible_flame) and 2 FIRE review candidates. requestId=hf-fire-test, imageId=fire-1. Reported server processing time was 1741.204 ms for this single request; this is not a benchmark. The fixture fire_1.jpg was verified byte-for-byte against its public source at https://raw.githubusercontent.com/DeepQuestAI/Fire-Smoke-Dataset/master/Assets/fire_1.jpg (SHA256 8932fe12d0781e3bc8650cbe81a9597866c5f14b56379f3ba73a74829c6e688b). No private visitor image was used.

An unauthenticated external GET /health returned HTTP 404 from the private Space. Browser-authenticated success does not establish backend-token access: server-to-server authentication still requires the receiving team to configure and test its authorized Hugging Face token.

## Backend connection

Store an authorized token in the backend secret store as HF_TOKEN; never in source control, Swagger examples, client-side JavaScript or a mobile app. Use a token with access to this private Space. Do not make the Space public to work around authentication. No token was created or collected during deployment.

Example health request after the team configures its secret:

```powershell
curl.exe -H "Authorization: Bearer $env:HF_TOKEN" https://basmaalmuntaser-expoguard-ai.hf.space/health
```

Example image request (retain the current multipart contract):

```powershell
curl.exe -H "Authorization: Bearer $env:HF_TOKEN" -F "file=@image.jpg" -F "requestId=backend-test-1" -F "imageId=image-1" -F "source=VISITOR_APP" -F "capturedAt=2026-10-15T00:00:00+03:00" https://basmaalmuntaser-expoguard-ai.hf.space/v1/analyze/image
```

These token-authenticated commands are handoff examples, not requests executed with a team token in this verification. HTTP 404 can mean missing private-Space authorization, not an absent API route. Root / has no route; open /docs instead.

## Scope and remaining checks

Remote health/models/Swagger and the fire/smoke inference smoke test passed. The complete 148-test suite was previously executed locally, not rerun inside this Space. Remote text/correlation/recommendation regressions and normal/corrupt images have not been independently retested in this deployment pass. Crowd needs approved camera ROI/capacity; playbook recommendations need approved playbooks. Speech/general-vision providers remain unconnected. CPU Basic can sleep after inactivity; startup delay is not a negative detection. Production load, independent accuracy and backend persistence integration are not certified.
