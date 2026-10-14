from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pydantic import ValidationError
from ai.evaluation.metrics import summarize, percentile
from ai.evaluation.benchmark import Dataset, Sample, run
from ai.inference.fire_adapter import FireAnalyzer
from test_analysis import Model, picture, box


def row(expected, predicted, status='OK', latency=10):
    return dict(expectedFire=expected, predictedFire=predicted, status=status, latencyMs=latency)


class EvaluationTests(unittest.TestCase):
    def test_known_confusion_matrix(self):
        result = summarize([row(True, True), row(False, True), row(True, False), row(False, False)])
        for key in ('truePositives','falsePositives','falseNegatives','trueNegatives'):
            self.assertEqual(result[key], 1)
        for key in ('accuracy','precision','recall','f1','falsePositiveRate'):
            self.assertEqual(result[key], .5)
        self.assertEqual(result['falseAlertsPer100ScoredImages'], 25)

    def test_failures_are_abstentions_and_end_to_end_misses(self):
        result = summarize([row(True, True), row(True, None, 'ERROR'), row(True, None, 'UNUSABLE'), row(False, None, 'ERROR')])
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['endToEndRecall'], 1/3)
        self.assertEqual(result['reviewedAbstentions'], 3)
        self.assertEqual(result['missedPositiveImagesIncludingAbstentions'], 2)
        self.assertEqual(result['trueNegatives'], 0)

    def test_unreviewed_and_undefined_metrics_are_null(self):
        for rows in ([], [row(None, True)]):
            result = summarize(rows)
            for key in ('accuracy','precision','recall','f1','mAP50','mAP50_95','falseAlertsPerCameraHour'):
                self.assertIsNone(result[key])

    def test_percentiles(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([10], .95), 10)
        self.assertEqual(percentile([0,100], .95), 95)

    def test_reviews_require_attribution_and_real_boolean(self):
        for values in [dict(reviewStatus='REVIEWED'), dict(expectedFire=True),
                       dict(reviewStatus='REVIEWED', expectedFire='true', reviewer='R', reviewedAt='2026-10-13T00:00:00Z')]:
            with self.assertRaises(ValidationError):
                Sample(imageId='i', path='x.png', **values)
        Sample(imageId='i', path='x.png', reviewStatus='REVIEWED', expectedFire=True,
               reviewer='R', reviewedAt='2026-10-13T00:00:00Z')

    def test_duplicate_ids_rejected(self):
        sample = dict(imageId='i', path='x.png')
        with self.assertRaises(ValidationError):
            Dataset(datasetVersion='v1', provenance='test', samples=[sample,sample])

    def test_benchmark_produces_review_queue_and_keeps_missing_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'image.png').write_bytes(picture())
            manifest = dict(datasetVersion='test-v1', provenance='synthetic test', samples=[
                dict(imageId='i1', path='image.png'), dict(imageId='i2', path='missing.png')])
            (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with patch('ai.evaluation.benchmark.FireAnalyzer', return_value=FireAnalyzer(Model([box()]))), redirect_stdout(io.StringIO()):
                report = run(root/'manifest.json', root/'output')
            self.assertEqual(report['summary']['pendingReview'], 2)
            self.assertEqual(report['summary']['errorRate'], .5)
            self.assertIsNone(report['summary']['precision'])
            self.assertEqual(report['rows'][1]['error'], 'INPUT_READ_FAILED')
            queue = Dataset.model_validate_json((root/'output/human-review.json').read_text(encoding='utf-8'))
            self.assertIsNone(queue.samples[0].expectedFire)
            self.assertTrue(Path(queue.samples[0].path).is_absolute())
            with self.assertRaises(FileExistsError):
                run(root/'manifest.json', root/'output')


if __name__ == '__main__':
    unittest.main()
