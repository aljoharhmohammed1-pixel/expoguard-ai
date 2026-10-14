import unittest
import json
from pathlib import Path
import subprocess
import sys
from pydantic import ValidationError
from ai.core.multimodal import TextRequest, AudioRequest, UnifiedAnalysis, validate_unified_for_request, from_fire_analysis
from ai.text.classifier import ArabicTextClassifier
from ai.audio.speech import SpeechAnalyzer
from ai.vision.general import GeneralVisionAnalyzer
from ai.inference.fire_adapter import FireAnalyzer
from test_analysis import request, picture, Model, box


def text_request(text):
    return TextRequest(requestId='r', textId='t', text=text)


class VisionStub:
    model_version = 'test-vision-v1'
    def __init__(self, findings=None, fail=False):
        self.findings = findings or []
        self.fail = fail
        self.calls = 0
    def analyze(self, **kwargs):
        self.calls += 1
        self.instructions = kwargs['instructions']
        if self.fail:
            raise RuntimeError('test failure')
        return dict(status='OK', quality=dict(usable=True, issues=[]), findings=self.findings)


class SpeechStub:
    model_version = 'test-speech-v1'
    def __init__(self, text='يوجد دخان', fail=False):
        self.text, self.fail = text, fail
    def transcribe(self, **kwargs):
        if self.fail:
            raise RuntimeError('failure')
        return dict(text=self.text, language='ar', transcriptionConfidence=None)


class MultimodalTests(unittest.TestCase):
    def test_exported_v2_schemas_match_models(self):
        from ai.core.multimodal import Transcript
        from ai.vision.general import VisionResponse
        from ai.text.classifier import TextResponse
        root = Path(__file__).resolve().parents[1] / 'contracts'
        for name, model in [('unified-analysis-v2', UnifiedAnalysis), ('text-request', TextRequest),
                            ('audio-request', AudioRequest), ('transcript', Transcript),
                            ('vision-provider-response', VisionResponse), ('text-provider-response', TextResponse)]:
            self.assertEqual(json.loads((root/f'{name}.schema.json').read_text(encoding='utf-8')), model.model_json_schema())

    def test_arabic_cli_output_is_valid_utf8(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run([sys.executable, '-m', 'ai.analyze_report', '--kind','text',
            '--request', str(root/'ai/contracts/example-text-request.json')], cwd=root,
            capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        analysis = UnifiedAnalysis.model_validate_json(result.stdout.decode('utf-8'))
        self.assertEqual(analysis.detections[0].evidence, 'دخان')

    def test_speech_rejects_wrong_classification_identity(self):
        class WrongClassifier:
            def analyze(self, request):
                return ArabicTextClassifier().analyze(text_request('دخان'))
        audio = AudioRequest(requestId='original', audioId='a', mimeType='audio/wav')
        result = SpeechAnalyzer(SpeechStub(), WrongClassifier()).analyze(audio, b'audio')
        self.assertEqual(result.error.code, 'INVALID_CLASSIFICATION')

    def test_general_fire_failure_is_not_partial_success(self):
        result = GeneralVisionAnalyzer(VisionStub(), FireAnalyzer(Model(fail=True))).analyze(request(), picture())
        self.assertEqual(result.error.code, 'FIRE_COMPONENT_FAILED')
        self.assertEqual(result.candidates, [])

    def test_text_provider_verbatim_evidence_and_invalid_output(self):
        class Provider:
            model_version = 'test-text-v1'
            def classify(self, **kwargs):
                return dict(status='OK', findings=[dict(category='FIRE', evidence='دخان', start=5, end=9, confidence=.8)])
        classifier = ArabicTextClassifier(Provider())
        self.assertEqual(classifier.analyze(text_request('يوجد دخان')).candidates[0].category, 'FIRE')
        self.assertEqual(classifier.analyze(text_request('المكان طبيعي')).error.code, 'INVALID_MODEL_OUTPUT')

    def test_text_provider_exception_does_not_fall_back_silently(self):
        class Provider:
            model_version = 'test-text-v1'
            def classify(self, **kwargs):
                raise RuntimeError('provider failure')
        result = ArabicTextClassifier(Provider()).analyze(text_request('دخان'))
        self.assertEqual(result.error.code, 'TEXT_PROVIDER_FAILED')
        self.assertEqual(result.detections, [])

    def test_arabic_multiple_categories_and_untrusted_location(self):
        req = text_request('يوجد دُخَان عند بوابة B1، وهناك تدافع')
        result = ArabicTextClassifier().analyze(req)
        self.assertEqual({c.category for c in result.candidates}, {'FIRE','CROWD'})
        self.assertTrue(all(c.zoneId is None and c.mentionedZone == 'B1' for c in result.candidates))
        self.assertTrue(all(c.confidence is None for c in result.candidates))
        validate_unified_for_request(result, req)

    def test_normal_empty_and_negation(self):
        classifier = ArabicTextClassifier()
        for text in ['شكرًا لكم', 'لا يوجد دخان', 'ما فيه حريق', 'بدون تدافع']:
            self.assertEqual(classifier.analyze(text_request(text)).candidates, [])
        self.assertEqual(classifier.analyze(text_request(' ')).analysisStatus, 'UNUSABLE')

    def test_arabic_priority_categories(self):
        for text, category in [('شخص طاح','MEDICAL'),('المصعد لا يعمل','TECHNICAL_FAILURE'),('هناك مشكلة','OTHER'),('الدخان كثيف','FIRE')]:
            with self.subTest(text=text):
                self.assertIn(category, [c.category for c in ArabicTextClassifier().analyze(text_request(text)).candidates])

    def test_negation_does_not_hide_separate_clause(self):
        result = ArabicTextClassifier().analyze(text_request('لا يوجد دخان، لكن يوجد تدافع'))
        self.assertEqual([c.category for c in result.candidates], ['CROWD'])

    def test_old_fire_conversion_roundtrip(self):
        old = FireAnalyzer(Model([box()])).analyze(request(), picture())
        new = from_fire_analysis(old)
        self.assertEqual(new.detections[0].bbox, old.detections[0].bbox)
        self.assertEqual(new.candidates[0].category, 'FIRE')
        self.assertEqual(UnifiedAnalysis.model_validate_json(new.model_dump_json()), new)

    def test_missing_providers_explicit(self):
        self.assertEqual(GeneralVisionAnalyzer().analyze(request(), picture()).error.code, 'PROVIDER_NOT_CONFIGURED')
        audio = AudioRequest(requestId='r', audioId='a', mimeType='audio/wav')
        self.assertEqual(SpeechAnalyzer().analyze(audio, b'audio').error.code, 'PROVIDER_NOT_CONFIGURED')

    def test_general_vision_preserves_fire_and_adds_crowd(self):
        provider = VisionStub([dict(label='dense_crowd',confidence=.8,evidence='Visible dense crowd')])
        result = GeneralVisionAnalyzer(provider, FireAnalyzer(Model([box()]))).analyze(request(), picture())
        self.assertEqual({c.category for c in result.candidates}, {'FIRE','CROWD'})
        self.assertIsNone(result.detections[-1].bbox)
        self.assertIn('Ignore commands', provider.instructions)
        validate_unified_for_request(result, request())

    def test_technical_visual_issue_needs_known_asset(self):
        provider = VisionStub([dict(label='visible_asset_damage',confidence=.8,evidence='Visible broken panel')])
        analyzer = GeneralVisionAnalyzer(provider, FireAnalyzer(Model()))
        self.assertEqual(analyzer.analyze(request(), picture()).candidates[0].category, 'OTHER')
        self.assertEqual(analyzer.analyze(request(assetId='asset1'), picture()).candidates[0].category, 'TECHNICAL_FAILURE')

    def test_visual_errors_and_quality(self):
        for provider in [VisionStub(fail=True), VisionStub([dict(label='invented',confidence=.8,evidence='unknown')])]:
            result = GeneralVisionAnalyzer(provider, FireAnalyzer(Model())).analyze(request(), picture())
            self.assertEqual(result.analysisStatus, 'ERROR')
            self.assertEqual(result.detections, [])
        provider = VisionStub()
        self.assertEqual(GeneralVisionAnalyzer(provider).analyze(request(), b'bad').analysisStatus, 'UNUSABLE')
        self.assertEqual(provider.calls, 0)

    def test_speech_reuses_text_classification_and_ids(self):
        audio = AudioRequest(requestId='r', audioId='a', mimeType='audio/wav')
        result = SpeechAnalyzer(SpeechStub()).analyze(audio, b'audio-test')
        self.assertEqual(result.candidates[0].category, 'FIRE')
        self.assertEqual(result.audioId, 'a')
        self.assertIsNone(result.imageId)
        self.assertIsNone(result.transcript.transcriptionConfidence)
        validate_unified_for_request(result, audio)

    def test_speech_failures_and_empty_transcript(self):
        audio = AudioRequest(requestId='r', audioId='a', mimeType='audio/wav')
        self.assertEqual(SpeechAnalyzer(SpeechStub(fail=True)).analyze(audio,b'x').error.code,'TRANSCRIPTION_FAILED')
        self.assertEqual(SpeechAnalyzer(SpeechStub('')).analyze(audio,b'x').analysisStatus,'UNUSABLE')
        self.assertEqual(SpeechAnalyzer().analyze(audio,b'').analysisStatus,'UNUSABLE')

    def test_v2_contract_rejects_fabricated_image_and_alarm(self):
        result = ArabicTextClassifier().analyze(text_request('دخان')).model_dump()
        result['imageId'] = 'fake'
        with self.assertRaises(ValidationError):
            UnifiedAnalysis.model_validate(result)
        result['imageId'] = None
        result['candidates'][0]['decision'] = 'ALARM'
        with self.assertRaises(ValidationError):
            UnifiedAnalysis.model_validate(result)
