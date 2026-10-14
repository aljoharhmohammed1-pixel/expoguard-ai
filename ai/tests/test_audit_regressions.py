"""Regressions for final-audit P0-01/P0-02; no external providers."""
import unittest
from ai.core.multimodal import AudioRequest, UnifiedAnalysis, validate_unified_for_request
from ai.text.classifier import ArabicTextClassifier, normalize
from ai.audio.speech import SpeechAnalyzer
from ai.vision.general import GeneralVisionAnalyzer
from test_multimodal import text_request, SpeechStub, VisionStub
from test_analysis import request, picture


class AuditRegressionTests(unittest.TestCase):
    def test_unicode_expansion_preserves_source_offsets(self):
        for text in ['İ دخان', 'İİ دُخَان', '😀 İ دخان ثم حريق', 'دخان İ']:
            with self.subTest(text=text):
                normalized, indices = normalize(text)
                self.assertEqual(len(normalized), len(indices))
                req = text_request(text)
                result = ArabicTextClassifier().analyze(req)
                self.assertTrue(result.detections)
                self.assertIn(result.detections[0].evidence, ['دخان', 'دُخَان'])
                for d in result.detections:
                    self.assertEqual(text[slice(*d.textSpan)], d.evidence)
                    self.assertLessEqual(d.textSpan[1], len(text))
                validate_unified_for_request(UnifiedAnalysis.model_validate_json(result.model_dump_json()), req)

    def test_invalid_provider_versions_rejected_before_calls(self):
        class Provider:
            def classify(self, **kwargs):
                raise AssertionError('Must not call invalid provider')
            analyze = classify
            transcribe = classify
        for factory in [ArabicTextClassifier, GeneralVisionAnalyzer, SpeechAnalyzer]:
            for version in ['', '   ', None, 12, True]:
                with self.subTest(factory=factory.__name__, version=version):
                    provider = Provider()
                    provider.model_version = version
                    with self.assertRaises(ValueError):
                        factory(provider)
            with self.assertRaises(ValueError):
                factory(Provider())

    def test_provider_version_snapshot_keeps_failure_envelope_valid(self):
        class Provider:
            model_version = 'test-v1'
            def classify(self, **kwargs):
                raise RuntimeError('private provider detail')
        provider = Provider()
        classifier = ArabicTextClassifier(provider)
        provider.model_version = ''
        result = classifier.analyze(text_request('دخان'))
        self.assertEqual(result.error.code, 'TEXT_PROVIDER_FAILED')
        self.assertEqual(result.modelVersion, 'test-v1')
        self.assertNotIn('private provider detail', result.model_dump_json())

    def test_vision_and_speech_versions_snapshot_on_failure(self):
        vision = VisionStub(fail=True)
        analyzer = GeneralVisionAnalyzer(vision)
        vision.model_version = ''
        result = analyzer.analyze(request(), picture())
        self.assertEqual(result.error.code, 'VISION_PROVIDER_FAILED')
        self.assertEqual(result.modelVersion, 'test-vision-v1')
        speech = SpeechStub(fail=True)
        analyzer = SpeechAnalyzer(speech)
        speech.model_version = None
        result = analyzer.analyze(AudioRequest(requestId='r', audioId='a', mimeType='audio/wav'), b'test')
        self.assertEqual(result.error.code, 'TRANSCRIPTION_FAILED')
        self.assertEqual(result.modelVersion, 'test-speech-v1')

    def test_text_and_audio_reject_invalid_source_spans(self):
        req = text_request('دخان')
        text = ArabicTextClassifier().analyze(req).model_dump()
        audio_req = AudioRequest(requestId='r', audioId='a', mimeType='audio/wav')
        audio = SpeechAnalyzer(SpeechStub('دخان')).analyze(audio_req, b'test').model_dump()
        for original, request in [(text, req), (audio, audio_req)]:
            for span in [(0, 999999), (0, 5), (False, 4), ('0', 4), (0.0, 4), (-1, 4), (4, 4), (4, 0)]:
                with self.subTest(kind=original['inputType'], span=span):
                    import copy
                    data = copy.deepcopy(original)
                    data['detections'][0]['textSpan'] = span
                    with self.assertRaises(ValueError):
                        validate_unified_for_request(data, request)

    def test_audio_rejects_out_of_bounds_classifier_output(self):
        class BadClassifier:
            def analyze(self, req):
                result = ArabicTextClassifier().analyze(req)
                result.detections[0].textSpan = (0, 999999)
                return result
        req = AudioRequest(requestId='r', audioId='a', mimeType='audio/wav')
        result = SpeechAnalyzer(SpeechStub('دخان'), BadClassifier()).analyze(req, b'test')
        self.assertEqual(result.error.code, 'INVALID_CLASSIFICATION')

    def test_valid_unicode_audio_roundtrip(self):
        req = AudioRequest(requestId='r', audioId='a', mimeType='audio/wav')
        result = SpeechAnalyzer(SpeechStub('😀 İ دُخان')).analyze(req, b'test')
        self.assertEqual(result.analysisStatus, 'OK')
        validate_unified_for_request(UnifiedAnalysis.model_validate_json(result.model_dump_json()), req)
