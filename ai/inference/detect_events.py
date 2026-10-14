"""Emit JSON events from an image, video or local camera on CPU."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from detection_events import build_detection_events


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Image/video path or camera index, e.g. 0')
    parser.add_argument('--zone', default='B1')
    parser.add_argument('--camera-id', default='CAM-B1-01')
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--output', type=Path, default=base/'inference'/'runs'/'events.jsonl')
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error('--conf must be between 0 and 1')
    if not args.zone or not args.camera_id:
        parser.error('--zone and --camera-id must not be empty')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base/'models'/'infernoguard_best.pt'))
    source = int(args.source) if args.source.isdecimal() else args.source
    # Append so earlier events survive subsequent runs.
    with args.output.open('a', encoding='utf-8', buffering=1) as log:
        for result in model.predict(source=source, device='cpu', conf=args.conf,
                                    imgsz=640, stream=True, verbose=False):
            detections = [{'class': result.names[int(b.cls.item())],
                           'confidence': float(b.conf.item())} for b in result.boxes]
            for event in build_detection_events(detections, args.zone, args.camera_id):
                line = json.dumps(event, ensure_ascii=False)
                log.write(line+'\n')
                print(line, flush=True)


if __name__ == '__main__':
    main()
