"""Crowd Detection V1: pretrained person detection with ByteTrack IDs."""

from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "yolo26n.pt"
VIDEO_PATH = BASE_DIR / "crowd" / "videos" / "crowd_test.mp4"
OUTPUT_DIR = BASE_DIR / "crowd" / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "crowd_tracked.mp4"


def main():
    for path in (MODEL_PATH, VIDEO_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = YOLO(str(MODEL_PATH))
    if model.names.get(0) != "person":
        raise ValueError("Expected a pretrained COCO model with person at class 0.")

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    writer = None
    frame_count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open input video: {VIDEO_PATH}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < float("inf"):
            raise ValueError(f"Input video has an invalid frame rate: {fps}")

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {OUTPUT_PATH}")

            result = model.track(
                source=frame,
                classes=[0],  # COCO person class only.
                tracker="bytetrack.yaml",
                persist=True,  # Keep IDs between successive video frames.
                conf=0.1,  # Retain low-confidence detections for ByteTrack association.
                device="cpu",
                verbose=False,
            )[0]
            annotated = result.plot()
            person_count = len(result.boxes) if result.boxes is not None else 0
            cv2.putText(
                annotated, f"People: {person_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
            )
            writer.write(annotated)
            frame_count += 1

        if frame_count == 0:
            raise RuntimeError(f"No readable frames in input video: {VIDEO_PATH}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Saved {frame_count} tracked frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
