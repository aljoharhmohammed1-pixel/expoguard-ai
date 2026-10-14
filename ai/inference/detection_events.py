"""Convert frame detections to the camera event schema."""
from datetime import datetime, timedelta, timezone
import math

RIYADH = timezone(timedelta(hours=3))


def build_detection_events(detections, zone='B1', camera_id='CAM-B1-01', timestamp=None):
    """One event per detected class per frame, using its highest confidence.

    Timestamp is processing/detection time in Riyadh, without an offset to match
    the requested schema. No event is produced for an empty frame. This function
    does not suppress repeated detections across consecutive frames.
    """
    if not zone or not camera_id:
        raise ValueError('zone and camera_id must not be empty')
    observed = {}
    for detection in detections:
        kind = str(detection['class']).upper()
        if kind not in ('FIRE', 'SMOKE'):
            continue
        confidence = float(detection['confidence'])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('confidence must be a finite number between 0 and 1')
        observed[kind] = max(observed.get(kind, 0), confidence)
    at = timestamp or datetime.now(RIYADH)
    if at.tzinfo is not None:
        at = at.astimezone(RIYADH)
    return [
        {'type': kind, 'zone': zone, 'camera_id': camera_id,
         'confidence': round(observed[kind], 4),
         'timestamp': at.replace(tzinfo=None).isoformat(timespec='seconds'), 'status': 'DETECTED'}
        for kind in ('FIRE', 'SMOKE') if kind in observed
    ]
