"""Opt-in offline integration with real local weights; no accuracy claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image, ImageOps
from ai.core.analysis import VisionAnalysis
from ai.inference.fire_adapter import FireAnalyzer, MODEL_PATH
from test_analysis import request


@unittest.skipUnless(os.environ.get('EXPOGUARD_REAL_TESTS') == '1', 'Enable with --integration')
class RealIntegrationTests(unittest.TestCase):
    def test_corrupt_after_model_import_does_not_install_codecs(self):
        import app
        from ultralytics.utils import checks
        from ai.quality.image_quality import quality_check
        self.assertFalse(checks.AUTOINSTALL)
        with patch.object(checks, 'check_requirements', side_effect=AssertionError('Unexpected dependency lookup')):
            _, quality = quality_check(b'broken image after model import')
        self.assertEqual(quality.issues, ['DECODE_FAILED'])

    def test_six_images_match_historical_boxes_confidence_and_labels(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root/'inference/runs/fire_smoke_samples/report.json').read_text(encoding='utf-8'))
        self.assertEqual(len(baseline['results']), 6)
        before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        analyzer = FireAnalyzer()
        mapping = {'fire': 'visible_flame', 'smoke': 'suspected_smoke'}
        for row in baseline['results']:
            with self.subTest(image=row['file']):
                path = root/'datasets/fire_smoke_samples'/row['file']
                with Image.open(path) as image:
                    width, height = ImageOps.exif_transpose(image).size
                result = analyzer.analyze(request(imageId=row['file']), path.read_bytes())
                self.assertEqual(result.analysisStatus, 'OK')
                self.assertEqual(VisionAnalysis.model_validate_json(result.model_dump_json()), result)
                expected = sorted((mapping[d['class']], tuple(v/s for v, s in zip(d['xyxy'], (width,height,width,height))), d['confidence']) for d in row['detections'])
                actual = sorted((d.label, d.bbox, d.confidence) for d in result.detections)
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected):
                    self.assertEqual(a[0], e[0])
                    self.assertAlmostEqual(a[2], e[2], delta=.0001)
                    for x, y in zip(a[1], e[1]):
                        self.assertAlmostEqual(x, y, delta=.0001)
        self.assertEqual(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), before)

    def test_legacy_script_with_actual_weights(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'events.jsonl'
            output.write_text('{"existing":true}\n', encoding='utf-8')
            completed = subprocess.run([sys.executable, str(root/'inference/detect_events.py'),
                '--source', str(root/'datasets/fire_smoke_samples/fire_1.jpg'), '--zone', 'TEST',
                '--camera-id', 'CAM-TEST', '--output', str(output)], capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0], {'existing': True})
            self.assertEqual([r['type'] for r in rows[1:]], ['FIRE', 'SMOKE'])
            self.assertTrue(all(set(r) == {'type','zone','camera_id','confidence','timestamp','status'} for r in rows[1:]))
            self.assertTrue(all(r['zone'] == 'TEST' and r['camera_id'] == 'CAM-TEST' for r in rows[1:]))


if __name__ == '__main__':
    unittest.main()
