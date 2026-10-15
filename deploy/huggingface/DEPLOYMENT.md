# Hugging Face deployment

Target private Space: https://huggingface.co/spaces/Basmaalmuntaser/expoguard-ai

The deployment uses Docker and CPU Basic as selected by the owner. No hardware upgrade, new subscription, provider connection or public access was configured.

The Dockerfile in this directory differs from the original local Dockerfile only in deployment packaging, port 7860, user UID 1000, writable cache locations and inclusion of the two existing weights. Source code and model policies are unchanged. The original Dockerfile and integration tag are preserved.

Staged upload is in ignored dist/huggingface: Dockerfile, README.md, requirements.txt copied from the integration lock, service.zip and PACKAGE.json. The archive contains only selected tracked Python sources, camera/playbook defaults and the two original weights, not environments, secrets, training datasets or user uploads. PACKAGE.json records source paths, source commit and archive SHA256.

Default camera configuration is empty. Crowd analysis remains disabled until actual approved camera configuration is provided. Person weights are included. No production playbooks have been supplied.

Remote build and inference smoke verification completed; see VERIFICATION.md for evidence and final URLs. Browser login alone does not configure server-to-server credentials; use an authorized Hugging Face token only in the backend's secret store. Do not make the Space public to avoid configuring access.
