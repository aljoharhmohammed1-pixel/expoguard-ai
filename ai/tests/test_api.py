import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from app.main import create_app
from app.settings import Settings
from app.registry import ModelRegistry, select_device
from test_analysis import Model, box, picture


def registry(boxes=(), fail=False):
    model = Model(boxes, fail)
    model.names = {0:'fire',1:'smoke'}
    loader = Mock(return_value=model)
    return ModelRegistry(Settings(app_env='test'), loader=loader, cuda_available=False)


class APIFoundationTests(unittest.TestCase):
    def test_openapi_error_contract_matches_runtime(self):
        with TestClient(create_app(Settings(), registry())) as client:
            schema = client.get('/openapi.json').json()
            responses = schema['paths']['/v1/analyze/image']['post']['responses']
            self.assertEqual(responses['422']['content']['application/json']['schema']['$ref'], '#/components/schemas/APIErrorResponse')
            self.assertEqual(responses['503']['content']['application/json']['schema']['$ref'], '#/components/schemas/UnifiedAnalysis')

    def test_structured_logs_do_not_contain_report_or_headers(self):
        with TestClient(create_app(Settings(), registry())) as client:
            with self.assertLogs('expoguard.api', level='INFO') as logs:
                client.post('/v1/analyze/text', json={'requestId':'r','textId':'t','text':'private-body-123'},
                            headers={'Authorization':'private-key-456'})
                self.post_image(client)
                client.post('/v1/analyze/text', json={'text':'private-invalid-789'})
            combined = '\n'.join(logs.output)
            for secret in ['private-body-123','private-key-456','private-invalid-789']:
                self.assertNotIn(secret, combined)
            self.assertIn('inferenceDurationMs', combined)
            self.assertIn('modelVersion', combined)
            self.assertIn('INVALID_REQUEST', combined)

    def test_recommendation_empty_and_trusted_catalog(self):
        facts = {'referenceId':'review-1','category':'FIRE','evidenceIds':['det-1']}
        with TestClient(create_app(Settings(), registry())) as client:
            self.assertEqual(client.post('/v1/recommend', json=facts).json()['status'], 'NO_APPROVED_PLAYBOOK')
            self.assertEqual(client.post('/v1/recommend', json=dict(facts, playbooks=[])).status_code, 422)
        with tempfile.TemporaryDirectory() as directory:
            # Synthetic fixture only; non-operational text, never shipped in the host catalog.
            book = {'playbookId':'fixture-only','version':'1','title':'TEST ONLY', 'sourceRef':'test-fixture',
                'approvalStatus':'APPROVED','approvedBy':'test-suite','approvedAt':'2020-01-01T00:00:00Z',
                'effectiveFrom':'2020-01-01T00:00:00Z','categories':['FIRE'],
                'steps':[{'stepId':'s1','textAr':'نص اصطناعي للاختبار فقط وليس إجراءً تشغيليًا.','sourceSection':'test'}]}
            path = Path(directory)/'fixture.json'
            path.write_text(json.dumps(book), encoding='utf-8')
            with TestClient(create_app(Settings(playbook_dir=Path(directory)), registry())) as client:
                result = client.post('/v1/recommend', json=facts).json()
                self.assertEqual(result['status'], 'PROPOSED')
                self.assertEqual(result['proposals'][0]['playbookId'], 'fixture-only')
                self.assertTrue(result['proposals'][0]['requiresHumanApproval'])
                other = client.post('/v1/recommend', json=dict(facts, category='CROWD')).json()
                self.assertEqual(other['status'], 'NO_APPLICABLE_PLAYBOOK')

    def test_summary_only_supplied_facts(self):
        fact = {'alertId':'A1','category':'FIRE','observedAt':'2026-10-13T10:00:00Z'}
        with TestClient(create_app(Settings(), registry())) as client:
            response = client.post('/v1/summary', json={'alerts':[fact]})
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result['sourceCount'], 1)
            self.assertEqual(result['alertIds'], ['A1'])
            self.assertNotIn('B1', result['summaryAr'])
            self.assertIn('التحقق البشري', result['summaryAr'])
            self.assertEqual(client.post('/v1/summary', json={'alerts':[fact,fact]}).status_code, 422)

    def test_correlation_breakdown_and_conflicts(self):
        left = {'alertId':'A1','category':'FIRE','zoneId':'B1','observedAt':'2026-10-13T10:00:00Z'}
        right = dict(left, alertId='A2', observedAt='2026-10-13T10:00:08Z')
        with TestClient(create_app(Settings(), registry())) as client:
            payload = client.post('/v1/correlation/analyze', json={'left':left,'right':right}).json()
            self.assertEqual(payload['proposal']['decision'], 'REVIEW_LINK')
            parts = payload['scoreBreakdown']
            self.assertAlmostEqual(sum(parts[k] for k in ['categoryContribution','locationContribution','timeContribution','lexicalContribution']), parts['finalScore'], places=5)
            for change in [dict(zoneId='B7'), dict(observedAt='2026-10-13T12:00:00Z')]:
                payload = client.post('/v1/correlation/analyze', json={'left':left,'right':dict(right, **change)}).json()
                self.assertEqual(payload['proposal']['decision'], 'NO_SUGGESTION')
                self.assertEqual(payload['proposal']['correlationScore'], 0)
                self.assertTrue(payload['scoreBreakdown']['conflictOrTimeGate'])

    def test_arabic_text_categories_and_location(self):
        with TestClient(create_app(Settings(), registry())) as client:
            for text, category in [('في دخان كثيف عند بوابة B1','FIRE'), ('يوجد تدافع','CROWD'),
                                   ('شخص طاح','MEDICAL'), ('المصعد لا يعمل','TECHNICAL_FAILURE'), ('هناك مشكلة','OTHER')]:
                response = client.post('/v1/analyze/text', json={'requestId':'r','textId':'t','text':text})
                self.assertEqual(response.status_code, 200)
                candidate = response.json()['candidates'][0]
                self.assertEqual(candidate['category'], category)
                self.assertIsNone(candidate['zoneId'])
                self.assertIsNone(candidate['confidence'])
                self.assertTrue(candidate['requiresHumanReview'])
                if category == 'FIRE':
                    self.assertEqual(candidate['mentionedZone'], 'B1')
            response = client.post('/v1/analyze/text', json={'requestId':'r','text':'secret private content'})
            self.assertEqual(response.status_code, 422)
            self.assertNotIn('secret private content', response.text)

    def post_image(self, client, data=None, media=None, mime='image/png'):
        return client.post('/v1/analyze/image', data=data if data is not None else {
            'requestId':'r', 'imageId':'i', 'source':'CCTV', 'cameraId':'c', 'capturedAt':'2026-10-13T10:00:00Z'},
            files={'file':('input.png', picture() if media is None else media, mime)})

    def test_image_cases(self):
        for boxes, media, status in [([box()], picture(), 'OK'), ([box(1)], picture(), 'OK'),
                                      ([], picture(), 'OK'), ([], b'broken', 'UNUSABLE')]:
            with self.subTest(status=status, boxes=len(boxes)):
                with TestClient(create_app(Settings(), registry(boxes))) as client:
                    response = self.post_image(client, media=media)
                    self.assertEqual(response.status_code, 200)
                    result = response.json()
                    self.assertEqual(result['analysisStatus'], status)
                    self.assertEqual(len(result['candidates']), 1 if boxes else 0)
                    if boxes:
                        self.assertIsNone(result['candidates'][0]['zoneId'])
                        self.assertEqual(result['candidates'][0]['category'], 'FIRE')
                    self.assertIn('x-process-time-ms', response.headers)

    def test_image_invalid_request_and_media(self):
        with TestClient(create_app(Settings(max_image_mb=1), registry())) as client:
            self.assertEqual(self.post_image(client, data={}).status_code, 422)
            self.assertEqual(self.post_image(client, mime='text/plain').status_code, 415)
            self.assertEqual(self.post_image(client, media=b'x'*(1024*1024+1)).status_code, 413)
            self.assertEqual(client.post('/v1/analyze/image', content=b'x'*(2*1024*1024)).status_code, 413)

    def test_image_failure_controlled(self):
        with TestClient(create_app(Settings(), registry(fail=True))) as client:
            response = self.post_image(client)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()['error']['code'], 'INFERENCE_FAILED')
            self.assertNotIn('Traceback', response.text)

    def test_models(self):
        with TestClient(create_app(Settings(app_env='test'), registry())) as client:
            response = client.get('/v1/models')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['models'][0]['status'], 'ready')
            self.assertNotIn(str(Path.cwd()), response.text)
            self.assertEqual(len(response.json()['models'][0]['sha256']), 64)

    def test_health(self):
        with TestClient(create_app(Settings(app_env='test'), registry())) as client:
            response = client.get('/health')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['device'], 'cpu')
            self.assertTrue(response.json()['modelsReady'])

    def test_lifespan_loads_once_and_swagger(self):
        reg = registry()
        with TestClient(create_app(Settings(app_env='test'), reg)) as client:
            self.assertTrue(reg.ready)
            self.assertEqual(client.get('/docs').status_code, 200)
            self.assertEqual(client.get('/openapi.json').status_code, 200)
            self.assertEqual(reg.loader.call_count, 1)
        self.assertFalse(reg.ready)

    def test_startup_failure_explicit(self):
        reg = ModelRegistry(Settings(model_path=Path('absent.pt')), cuda_available=False)
        with self.assertRaisesRegex(RuntimeError, 'startup validation'):
            with TestClient(create_app(Settings(), reg)):
                pass

    def test_device_selection(self):
        self.assertEqual(select_device('auto', False), 'cpu')
        self.assertEqual(select_device('auto', True), 'cuda')
        self.assertEqual(select_device('cpu', True), 'cpu')
        with self.assertRaises(RuntimeError):
            select_device('cuda', False)
