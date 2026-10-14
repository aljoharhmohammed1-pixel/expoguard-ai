"""Compatibility guarantees for the original JSONL event interface."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ai.inference.detection_events import build_detection_events
from test_analysis import box


class LegacyTests(unittest.TestCase):
    def test_exact_legacy_shape_grouping_rounding_and_timezone(self):
        events = build_detection_events([
            {'class': 'smoke', 'confidence': .6}, {'class': 'fire', 'confidence': .712345},
            {'class': 'FIRE', 'confidence': .4}, {'class': 'person', 'confidence': .9}],
            zone='Z9', camera_id='C9', timestamp=datetime(2026, 10, 13, 10, tzinfo=timezone.utc))
        self.assertEqual(events, [
            dict(type='FIRE', zone='Z9', camera_id='C9', confidence=.7123, timestamp='2026-10-13T13:00:00', status='DETECTED'),
            dict(type='SMOKE', zone='Z9', camera_id='C9', confidence=.6, timestamp='2026-10-13T13:00:00', status='DETECTED')])

    def test_empty_frame_and_legacy_defaults(self):
        self.assertEqual(build_detection_events([]), [])
        event = build_detection_events([dict(**{'class': 'fire'}, confidence=.5)], timestamp=datetime(2026, 1, 1))[0]
        self.assertEqual((event['zone'], event['camera_id']), ('B1', 'CAM-B1-01'))
        self.assertEqual(event['timestamp'], '2026-01-01T00:00:00')

    def test_invalid_confidence_and_empty_metadata(self):
        for value in (-1, 2, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                build_detection_events([{'class': 'fire', 'confidence': value}])
        for args in (dict(zone=''), dict(camera_id='')):
            with self.assertRaises(ValueError):
                build_detection_events([], **args)

    def test_script_appends_jsonl_and_preserves_camera_options(self):
        script = Path(__file__).resolve().parents[1] / 'inference/detect_events.py'
        calls = []
        class FakeYOLO:
            def __init__(self, path):
                self.path = path
            def predict(self, **kwargs):
                calls.append(kwargs)
                yield SimpleNamespace(names={0: 'smoke', 1: 'fire'}, boxes=[box(1), box(0)])
                yield SimpleNamespace(names={0: 'smoke', 1: 'fire'}, boxes=[])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'events.jsonl'
            output.write_text('{"existing":true}\n', encoding='utf-8')
            args = [str(script), '--source', '0', '--zone', 'Z9', '--camera-id', 'C9', '--output', str(output)]
            capture = io.StringIO()
            with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=FakeYOLO)}), \
                 patch.object(sys, 'path', [str(script.parent), *sys.path]), patch.object(sys, 'argv', args), redirect_stdout(capture):
                runpy.run_path(str(script), run_name='__main__')
            rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0], {'existing': True})
            self.assertEqual([r['type'] for r in rows[1:]], ['FIRE', 'SMOKE'])
            self.assertEqual([json.loads(line) for line in capture.getvalue().splitlines()], rows[1:])
            self.assertTrue(all(r['zone'] == 'Z9' and r['camera_id'] == 'C9' for r in rows[1:]))
            self.assertEqual(calls[0], dict(source=0, device='cpu', conf=.35, imgsz=640, stream=True, verbose=False))


if __name__ == '__main__':
    unittest.main()
