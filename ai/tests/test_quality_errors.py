import io
import unittest
from unittest.mock import patch
from PIL import Image
from ai.quality.image_quality import quality_check
from ai.inference.coordinates import normalize_box, validate_image_size, CoordinateError
from ai.inference.fire_adapter import FireAnalyzer
from test_analysis import request, picture, Model, box


def encode(image, fmt='PNG'):
    stream = io.BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


class QualityErrorTests(unittest.TestCase):
    def test_corrupt_and_truncated_images(self):
        for data in (b'not an image', picture()[:40]):
            image, quality = quality_check(data)
            self.assertIsNone(image)
            self.assertEqual(quality.issues, ['DECODE_FAILED'])

    def test_supported_formats_and_bytes_like_input(self):
        for fmt in ('PNG', 'JPEG', 'BMP', 'WEBP'):
            for wrap in (bytes, bytearray, memoryview):
                with self.subTest(fmt=fmt, wrap=wrap):
                    image, quality = quality_check(wrap(encode(Image.new('RGB', (32, 32), 'gray'), fmt)))
                    self.assertTrue(quality.usable)
                    self.assertEqual(image.mode, 'RGB')

    def test_unsupported_format(self):
        _, quality = quality_check(encode(Image.new('RGB', (32, 32), 'gray'), 'GIF'))
        self.assertEqual(quality.issues, ['UNSUPPORTED_FORMAT'])

    def test_animated_png_rejected(self):
        stream = io.BytesIO()
        a = Image.new('RGB', (32, 32), 'gray')
        a.save(stream, format='PNG', save_all=True, append_images=[Image.new('RGB', (32, 32), 'red')])
        self.assertEqual(quality_check(stream.getvalue())[1].issues, ['UNSUPPORTED_FORMAT'])

    def test_size_limits_before_inference(self):
        for limit, value in [('MAX_BYTES', 10), ('MAX_PIXELS', 100)]:
            with patch('ai.quality.image_quality.' + limit, value):
                model = Model()
                result = FireAnalyzer(model).analyze(request(), picture())
                self.assertEqual(result.analysisStatus, 'UNUSABLE')
                self.assertEqual(result.imageQuality.issues, ['TOO_LARGE'])
                self.assertEqual(model.calls, 0)

    def test_decompression_warning_is_controlled(self):
        with patch.object(Image, 'MAX_IMAGE_PIXELS', 5000):
            self.assertEqual(quality_check(picture())[1].issues, ['TOO_LARGE'])

    def test_transparency_cannot_reveal_hidden_content(self):
        image = Image.new('RGBA', (32, 32), (255, 100, 0, 0))
        self.assertEqual(quality_check(encode(image))[1].issues, ['EMPTY_IMAGE'])
        image.putpixel((0, 0), (100, 100, 100, 255))
        decoded, quality = quality_check(encode(image))
        self.assertTrue(quality.usable)
        self.assertEqual(decoded.getpixel((1, 1)), (0, 0, 0))

    def test_dark_scene_with_small_bright_region_preserved(self):
        image = Image.new('RGB', (100, 100), 'black')
        image.putpixel((50, 50), (255, 100, 0))
        self.assertTrue(quality_check(encode(image))[1].usable)

    def test_brightness_boundary(self):
        for level, usable in [(0, False), (2, False), (3, True), (252, True), (253, False), (255, False)]:
            with self.subTest(level=level):
                self.assertEqual(quality_check(picture(level))[1].usable, usable)

    def test_wrong_input_type_is_caller_error(self):
        for data in (None, '', 123):
            with self.assertRaises(TypeError):
                quality_check(data)

    def test_bad_dimensions(self):
        for size in ((0, 80), (float('nan'), 80), (float('inf'), 80), (True, 80), (1.5, 80)):
            with self.assertRaises(CoordinateError):
                validate_image_size(size, (80, size[0]))
        with self.assertRaises(CoordinateError):
            validate_image_size((100, 80), (640, 640))

    def test_bad_boxes_are_not_clamped(self):
        for coords in ((-1, 0, 10, 10), (0, 0, 101, 80), (10, 0, 5, 80),
                       (0, 0, 0, 80), (0, 0, float('nan'), 80), (True, 0, 10, 80)):
            with self.assertRaises(CoordinateError):
                normalize_box(coords, (100, 80))
        self.assertEqual(normalize_box((0, 0, 100, 80), (100, 80)), (0, 0, 1, 1))

    def test_coordinate_error_has_specific_reason(self):
        analysis = FireAnalyzer(Model([box(coords=(-1, 0, 10, 10))])).analyze(request(), picture())
        self.assertEqual(analysis.analysisStatus, 'ERROR')
        self.assertIn('coordinates', analysis.error.message)
        self.assertEqual(analysis.candidates, [])

    def test_inference_recovers_after_failure(self):
        model = Model(fail=True)
        analyzer = FireAnalyzer(model)
        failed = analyzer.analyze(request(), picture())
        self.assertEqual(failed.error.code, 'INFERENCE_FAILED')
        model.fail = False
        self.assertEqual(analyzer.analyze(request(), picture()).analysisStatus, 'OK')

    def test_wrong_result_count_is_explicit_error(self):
        for output in ([], [None, None], None):
            with patch.object(Model, 'predict', return_value=output):
                result = FireAnalyzer(Model()).analyze(request(), picture())
                self.assertEqual(result.error.code, 'INVALID_MODEL_OUTPUT')
                self.assertEqual(result.detections, [])


if __name__ == '__main__':
    unittest.main()
