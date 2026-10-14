import io
import unittest
from types import SimpleNamespace
from PIL import Image
from pydantic import ValidationError
from ai.core.analysis import ImageRequest, Detection, VisionAnalysis
from ai.inference.fire_adapter import FireAnalyzer


def request(**changes):
    values = dict(requestId='req-1', imageId='img-1', source='CCTV', cameraId='cam-1', capturedAt='2026-10-13T10:00:00Z')
    values.update(changes)
    return ImageRequest(**values)


def picture(color=100, size=(100, 80)):
    stream = io.BytesIO()
    Image.new('RGB', size, (color, color, color)).save(stream, format='PNG')
    return stream.getvalue()


def box(kind=0, confidence=.8, coords=(10, 8, 50, 40)):
    return SimpleNamespace(cls=SimpleNamespace(item=lambda: kind), conf=SimpleNamespace(item=lambda: confidence),
                           xyxy=[SimpleNamespace(tolist=lambda: list(coords))])


class Model:
    def __init__(self, boxes=(), fail=False):
        self.boxes, self.fail, self.calls = boxes, fail, 0

    def predict(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError('test failure')
        image = kwargs['source']
        return [SimpleNamespace(names={0: 'fire', 1: 'smoke'}, boxes=self.boxes, orig_shape=(image.height, image.width))]


class AnalysisTests(unittest.TestCase):
    def test_normal_and_roundtrip(self):
        result = FireAnalyzer(Model()).analyze(request(), picture())
        self.assertEqual(result.analysisStatus, 'OK')
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.detections, [])
        self.assertEqual(VisionAnalysis.model_validate_json(result.model_dump_json()), result)

    def test_fire_identity_coordinates_and_unknown_zone(self):
        analyzer = FireAnalyzer(Model([box()]))
        result = analyzer.analyze(request(), picture())
        self.assertEqual(result.requestId, 'req-1')
        self.assertEqual(result.imageId, 'img-1')
        self.assertEqual(result.detections[0].bbox, (.1, .1, .5, .5))
        candidate = result.candidates[0]
        self.assertEqual(candidate.category, 'FIRE')
        self.assertIsNone(candidate.zoneId)
        self.assertIsNone(candidate.severity)
        self.assertEqual(candidate.decision, 'REVIEW')
        self.assertEqual(result, analyzer.analyze(request(), picture()))

    def test_smoke(self):
        result = FireAnalyzer(Model([box(1)])).analyze(request(zoneId='B1'), picture())
        self.assertEqual(result.detections[0].label, 'suspected_smoke')
        self.assertEqual(result.candidates[0].zoneId, 'B1')

    def test_grouping(self):
        result = FireAnalyzer(Model([box(), box(1), box(coords=(70, 60, 90, 75))])).analyze(request(), picture())
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.candidates[0].detectionIds, ['det-1', 'det-2'])

    def test_quality_does_not_infer(self):
        for data, issue in [(b'', 'EMPTY_IMAGE'), (b'bad', 'DECODE_FAILED'), (picture(0), 'TOO_DARK'),
                            (picture(255), 'TOO_BRIGHT'), (picture(size=(10, 10)), 'TOO_SMALL')]:
            with self.subTest(issue=issue):
                model = Model()
                result = FireAnalyzer(model).analyze(request(), data)
                self.assertEqual(result.analysisStatus, 'UNUSABLE')
                self.assertEqual(result.imageQuality.issues, [issue])
                self.assertEqual(model.calls, 0)

    def test_model_failures(self):
        def broken_loader():
            raise RuntimeError('cannot load')
        for analyzer, code in [(FireAnalyzer(model_loader=broken_loader), 'MODEL_LOAD_FAILED'),
                               (FireAnalyzer(Model(fail=True)), 'INFERENCE_FAILED'),
                               (FireAnalyzer(Model([box(confidence=float('nan'))])), 'INVALID_MODEL_OUTPUT'),
                               (FireAnalyzer(Model([box(coords=(-1, 0, 20, 20))])), 'INVALID_MODEL_OUTPUT')]:
            result = analyzer.analyze(request(), picture())
            self.assertEqual(result.analysisStatus, 'ERROR')
            self.assertEqual(result.error.code, code)
            self.assertEqual(result.detections, [])

    def test_request_validation(self):
        for changes in [dict(requestId=' '), dict(cameraId=None), dict(capturedAt='2026-10-13T10:00:00'),
                        dict(source='invented'), dict(regionOfInterest=(0, 0, 1, 1))]:
            with self.assertRaises(ValidationError):
                request(**changes)

    def test_box_validation(self):
        for coords in [(0, 0, 0, 1), (0, .8, 1, .2), (0, 0, 2, 1)]:
            with self.assertRaises(ValidationError):
                Detection(detectionId='d', label='visible_flame', confidence=.8, bbox=coords, evidence='test')

    def test_references_and_status_validation(self):
        result = FireAnalyzer(Model([box()])).analyze(request(), picture()).model_dump()
        result['candidates'][0]['detectionIds'] = ['missing']
        with self.assertRaises(ValidationError):
            VisionAnalysis.model_validate(result)
        result['candidates'] = []
        result['analysisStatus'] = 'ERROR'
        with self.assertRaises(ValidationError):
            VisionAnalysis.model_validate(result)


if __name__ == '__main__':
    unittest.main()
