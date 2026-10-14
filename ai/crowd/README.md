# Crowd Detection V1

Uses the existing shared `ai/models/yolo26n.pt` pretrained model to detect only
the COCO `person` class and ByteTrack to assign tracking IDs. No training is run.
The overlay shows the current frame's person count, not a cumulative visitor count.

Put the input video at `ai/crowd/videos/crowd_test.mp4`. Run `crowd_detection.py`
with the AI project's Python environment (Ultralytics and OpenCV are required).
The annotated video is written to `ai/crowd/outputs/crowd_tracked.mp4`;
running the script again replaces that output.

All paths are resolved from the script location using `pathlib`, independent of
the working directory. When installed at `C:\ExpoGuard\ai`, the model resolves to
`C:\ExpoGuard\ai\models\yolo26n.pt`. Keep all AI models in that shared directory.
