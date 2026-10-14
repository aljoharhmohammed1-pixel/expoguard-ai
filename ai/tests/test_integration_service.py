import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from ai.integration.service import FireIntegrationService, IntegrationRequest, IntegrationResponse
from ai.integration.cli import read_bounded
from ai.inference.fire_adapter import FireAnalyzer
from test_analysis import request, picture, Model, box

ROOT = Path(__file__).resolve().parents[2]


def integration_request(**changes):
    data = dict(inputType='IMAGE', analysisMode='FIRE_SMOKE', metadata=request(eventId='event-1').model_dump())
    data.update(changes)
    return data


class IntegrationServiceTests(unittest.TestCase):
    def test_source_metadata_roundtrip_and_review_only(self):
        service = FireIntegrationService(FireAnalyzer(Model([box(), box(1)])))
        result = service.analyze(integration_request(), picture())
        self.assertIsNone(result.error)
        self.assertEqual(result.analysis.schemaVersion, '2.0.0')
        self.assertEqual(result.request.metadata.eventId, 'event-1')
        self.assertEqual(result.request.metadata.cameraId, 'cam-1')
        self.assertEqual(result.analysis.candidates[0].decision, 'REVIEW')
        self.assertIsNone(result.analysis.candidates[0].severity)
        self.assertEqual(IntegrationResponse.model_validate_json(result.model_dump_json()), result)

    def test_invalid_request_never_runs_inference(self):
        model = Model()
        service = FireIntegrationService(FireAnalyzer(model))
        for changes in [dict(contractVersion='9.0.0'), dict(inputType='AUDIO'), dict(analysisMode='GENERAL'),
                        dict(metadata={'requestId': 'private-input'})]:
            result = service.analyze(integration_request(**changes), picture())
            self.assertEqual(result.error.code, 'INVALID_REQUEST')
            self.assertNotIn('private-input', result.model_dump_json())
        self.assertEqual(model.calls, 0)

    def test_no_detection_is_not_a_safety_decision(self):
        result = FireIntegrationService(FireAnalyzer(Model())).analyze(integration_request(), picture())
        self.assertEqual(result.analysis.analysisStatus, 'OK')
        self.assertEqual(result.analysis.candidates, [])
        self.assertFalse(hasattr(result, 'safe'))

    def test_quality_and_model_errors_keep_the_analysis_contract(self):
        for analyzer, media, status in [(FireAnalyzer(Model()), b'bad', 'UNUSABLE'),
                                        (FireAnalyzer(Model(fail=True)), picture(), 'ERROR')]:
            result = FireIntegrationService(analyzer).analyze(integration_request(), media)
            self.assertIsNone(result.error)
            self.assertEqual(result.analysis.analysisStatus, status)
            self.assertEqual(result.analysis.candidates, [])

    def test_unexpected_exception_is_redacted(self):
        class Broken:
            def analyze(self, *args):
                raise RuntimeError('secret-file-path')
        result = FireIntegrationService(Broken()).analyze(integration_request(), picture())
        self.assertEqual(result.error.code, 'INTERNAL_ERROR')
        self.assertNotIn('secret-file-path', result.model_dump_json())

    def test_wrong_result_identity_is_rejected(self):
        class Wrong:
            def analyze(self, req, media):
                return FireAnalyzer(Model()).analyze(request(requestId='wrong'), media)
        result = FireIntegrationService(Wrong()).analyze(integration_request(), picture())
        self.assertEqual(result.error.code, 'INVALID_ANALYSIS')

    def test_retries_preserve_candidate_ids_without_backend_side_effects(self):
        service = FireIntegrationService(FireAnalyzer(Model([box()])))
        first = service.analyze(integration_request(), picture())
        self.assertEqual(first, service.analyze(integration_request(), picture()))

    def test_media_type_is_explicit(self):
        result = FireIntegrationService(FireAnalyzer(Model())).analyze(integration_request(), 'https://example.invalid')
        self.assertEqual(result.error.code, 'INVALID_MEDIA')

    def test_bounded_reader_rejects_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input'
            path.write_bytes(b'12345')
            self.assertEqual(read_bounded(path, 5), b'12345')
            with self.assertRaises(ValueError):
                read_bounded(path, 4)

    def test_exported_integration_schemas(self):
        for name, model in [('integration-request', IntegrationRequest), ('integration-response', IntegrationResponse)]:
            schema = json.loads((ROOT/'ai/contracts'/f'{name}.schema.json').read_text(encoding='utf-8'))
            self.assertEqual(schema, model.model_json_schema())

    def test_cli_missing_media_and_invalid_request_are_json(self):
        with tempfile.TemporaryDirectory() as directory:
            req = Path(directory)/'request.json'
            for data, code in [(integration_request(), 'MEDIA_READ_FAILED'), ({}, 'INVALID_REQUEST')]:
                # Pydantic serializes aware datetime in metadata for the JSON request.
                raw = IntegrationRequest.model_validate(data).model_dump_json() if data else '{}'
                req.write_text(raw, encoding='utf-8')
                run = subprocess.run([sys.executable, '-m', 'ai.integration.cli', '--request', str(req),
                                      '--media', str(Path(directory)/'absent')], cwd=ROOT, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 2)
                result = IntegrationResponse.model_validate_json(run.stdout.decode('utf-8'))
                self.assertEqual(result.error.code, code)

    @unittest.skipUnless(os.environ.get('EXPOGUARD_REAL_TESTS') == '1', 'Enable with --integration')
    def test_actual_weights_to_backend_json(self):
        run = subprocess.run([sys.executable, '-m', 'ai.integration.cli',
            '--request', str(ROOT/'ai/contracts/example-integration-request.json'),
            '--media', str(ROOT/'ai/datasets/fire_smoke_samples/fire_1.jpg')], cwd=ROOT,
            capture_output=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = IntegrationResponse.model_validate_json(run.stdout.decode('utf-8'))
        self.assertEqual(result.analysis.analysisStatus, 'OK')
        self.assertTrue(result.analysis.candidates)
        self.assertEqual(result.analysis.requestId, result.request.metadata.requestId)
