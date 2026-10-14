import io
import unittest
from types import SimpleNamespace
from PIL import Image
from ai.core.analysis import ImageQuality
from ai.inference.fire_adapter import adapt_yolo_result, FireAnalyzer
from test_analysis import box, request, picture, Model


def result(boxes=(), names=None, shape=(80, 100)):
    return SimpleNamespace(boxes=boxes, names=names if names is not None else {0: 'smoke', 1: 'fire'}, orig_shape=shape)


class AdapterTests(unittest.TestCase):
    def adapt(self, value):
        return adapt_yolo_result(value, request(), (100, 80), ImageQuality(usable=True))

    def test_real_class_order_and_precision(self):
        analysis = self.adapt(result([box(0, .87654321)]))
        self.assertEqual(analysis.detections[0].label, 'suspected_smoke')
        self.assertEqual(analysis.detections[0].confidence, .87654321)
        self.assertNotIn('لهب', analysis.candidates[0].summaryAr)
        self.assertIn('لا يثبت', analysis.candidates[0].summaryAr)
        self.assertEqual(analysis.candidates[0].category, 'FIRE')

    def test_order_independent_ids(self):
        boxes = [box(1), box(0), box(1, coords=(60, 50, 90, 70))]
        self.assertEqual(self.adapt(result(boxes)), self.adapt(result(list(reversed(boxes)))))

    def test_empty_result_and_list_names(self):
        analysis = self.adapt(result(names=['smoke', 'fire']))
        self.assertEqual(analysis.analysisStatus, 'OK')
        self.assertEqual(analysis.detections, [])
        self.assertEqual(analysis.candidates, [])

    def test_invalid_results_never_become_normal(self):
        invalid = [result(names={0: 'person'}), result(shape=(640, 640)),
                   result([box(.5)]), result([box(5)]), result(boxes=None),
                   result([box(0), box(1, coords=(-1, 0, 10, 10))])]
        for value in invalid:
            with self.subTest(value=value):
                analysis = self.adapt(value)
                self.assertEqual(analysis.analysisStatus, 'ERROR')
                self.assertEqual(analysis.error.code, 'INVALID_MODEL_OUTPUT')
                self.assertEqual(analysis.detections, [])

    def test_flame_summary_and_no_operational_severity(self):
        analysis = self.adapt(result([box(1, .999)]))
        candidate = analysis.candidates[0]
        self.assertNotIn('دخان', candidate.summaryAr)
        self.assertIsNone(candidate.severity)
        self.assertEqual(candidate.decision, 'REVIEW')

    def test_exif_orientation_before_inference(self):
        image = Image.new('RGB', (100, 80), (100, 100, 100))
        exif = Image.Exif()
        exif[274] = 6
        data = io.BytesIO()
        image.save(data, format='JPEG', exif=exif)
        class InspectModel(Model):
            def predict(self, **kwargs):
                self.size = kwargs['source'].size
                return super().predict(**kwargs)
        model = InspectModel([box(coords=(8, 10, 40, 50))])
        analysis = FireAnalyzer(model).analyze(request(), data.getvalue())
        self.assertEqual(model.size, (80, 100))
        self.assertEqual(analysis.detections[0].bbox, (.1, .1, .5, .5))

    def test_model_loaded_once_and_settings_preserved(self):
        loads = []
        class InspectModel(Model):
            def predict(self, **kwargs):
                self.settings = kwargs
                return super().predict(**kwargs)
        model = InspectModel()
        def loader():
            loads.append(True)
            return model
        analyzer = FireAnalyzer(model_loader=loader)
        for _ in range(2):
            self.assertEqual(analyzer.analyze(request(), picture()).analysisStatus, 'OK')
        self.assertEqual(len(loads), 1)
        self.assertEqual(model.settings['conf'], .35)
        self.assertEqual(model.settings['imgsz'], 640)
        self.assertEqual(model.settings['device'], 'cpu')


if __name__ == '__main__':
    unittest.main()
