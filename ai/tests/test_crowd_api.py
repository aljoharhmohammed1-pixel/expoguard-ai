import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from fastapi.testclient import TestClient
from app.main import create_app
from app.registry import ModelRegistry
from app.settings import Settings
from ai.core.multimodal import UnifiedAnalysis
from test_analysis import Model, box, picture


class CrowdAPITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'cameras.json'

    def client(self, crowd=None, fire=(), people=(), fail=False, debug=False, zone=None):
        config = {'enabled':True, 'useFullFrame':True, 'capacity':1} if crowd is None else crowd
        self.path.write_text(json.dumps({'cameras':[{'cameraId':'test-cam','zoneId':zone,'crowd':config}]}))
        settings = Settings(camera_config_path=self.path, include_person_detections=debug)
        self.fire = Model(fire)
        self.fire.names = {0:'fire',1:'smoke'}
        self.people = Model(people, fail)
        self.people.names = {0:'person'}
        self.loader = Mock(return_value=self.people)
        self.registry = ModelRegistry(settings, loader=Mock(return_value=self.fire),
                                      cuda_available=False, person_loader=self.loader)
        return TestClient(create_app(settings,self.registry))

    def post(self, client, camera='test-cam', zone=None, media=None):
        data = {'requestId':'r','imageId':'i','source':'CCTV','cameraId':camera,'capturedAt':'2026-10-13T10:00:00Z'}
        if zone is not None:
            data['zoneId'] = zone
        return client.post('/v1/analyze/image', data=data, files={'file':('x.png',picture() if media is None else media,'image/png')})

    def test_combined_candidates_and_wire_contract(self):
        with self.client(fire=[box()],people=[box()],zone='verified-zone') as client:
            response = self.post(client)
            self.assertEqual(response.status_code,200)
            result = UnifiedAnalysis.model_validate(response.json())
            self.assertEqual([c.category for c in result.candidates],['FIRE','CROWD'])
            self.assertTrue(all(c.zoneId=='verified-zone' for c in result.candidates))
            self.assertEqual(result.crowdMetrics.peopleCount,1)
            self.assertEqual(result.crowdMetrics.crowdLevel,'CRITICAL')
            self.assertFalse(any(d.label=='person' for d in result.detections))
            self.assertEqual(len(result.componentVersions['crowdConfig']),64)
            self.post(client)
            self.assertEqual(self.loader.call_count,1)
            self.assertEqual(self.people.calls,2)

    def test_no_people_and_crowd_only(self):
        for people, categories in [((),[]),([box()],['CROWD'])]:
            with self.client(people=people) as client:
                payload = self.post(client).json()
                self.assertEqual([c['category'] for c in payload['candidates']],categories)
                self.assertEqual(payload['crowdMetrics']['peopleCount'],len(people))

    def test_camera_resolution_reaches_person_only(self):
        with self.client(crowd={'enabled':True,'useFullFrame':True,'capacity':1,'inferenceImageSize':1600},people=[box()]) as client:
            self.people.predict=Mock(wraps=self.people.predict)
            self.fire.predict=Mock(wraps=self.fire.predict)
            payload=self.post(client).json()
            self.assertEqual(payload['analysisStatus'],'OK')
            self.assertEqual(self.people.predict.call_args.kwargs['imgsz'],1600)
            self.assertEqual(self.fire.predict.call_args.kwargs['imgsz'],640)
            self.assertEqual(self.fire.predict.call_args.kwargs['conf'],.35)
            self.assertIn('crowdInference',payload['componentVersions'])

    def test_disabled_missing_roi_and_unknown_camera_preserve_fire(self):
        for config,camera in [({'enabled':False},'test-cam'),({'enabled':True},'test-cam'),
                              ({'enabled':True,'useFullFrame':True},'unknown')]:
            with self.client(crowd=config,fire=[box()]) as client:
                payload = self.post(client,camera=camera).json()
                self.assertEqual(payload['candidates'][0]['category'],'FIRE')
                self.assertNotIn('crowdMetrics',payload)
                self.assertTrue(any('skipped' in item for item in payload['limitations']))
                self.assertEqual(self.people.calls,0)

    def test_missing_capacity_and_debug(self):
        with self.client(crowd={'enabled':True,'useFullFrame':True},people=[box()],debug=True) as client:
            payload = self.post(client).json()
            self.assertEqual(payload['crowdMetrics']['peopleCount'],1)
            self.assertIsNone(payload['crowdMetrics']['crowdLevel'])
            self.assertIsNone(payload['crowdMetrics']['occupancyRatio'])
            self.assertEqual(payload['candidates'],[])
            self.assertEqual(payload['detections'][0]['label'],'person')

    def test_camera_zone_conflict_before_inference(self):
        for zone in [None,'verified']:
            with self.client(zone=zone) as client:
                response = self.post(client,zone='caller-zone')
                self.assertEqual(response.status_code,422)
                self.assertEqual(response.json()['error']['code'],'CAMERA_ZONE_CONFLICT')
                self.assertEqual(self.fire.calls,0)
                self.assertEqual(self.people.calls,0)

    def test_quality_gate_before_both_models(self):
        with self.client() as client:
            for media in [b'broken',picture(0)]:
                response = self.post(client,media=media)
                self.assertEqual(response.json()['analysisStatus'],'UNUSABLE')
                self.assertNotIn('crowdMetrics',response.json())
            self.assertEqual(self.people.calls,0)
            self.assertEqual(self.fire.calls,0)

    def test_failure_discards_partial_fire(self):
        with self.client(fire=[box()],fail=True) as client:
            response = self.post(client)
            self.assertEqual(response.status_code,503)
            payload = response.json()
            self.assertEqual(payload['error']['code'],'PERSON_INFERENCE_FAILED')
            self.assertEqual(payload['detections'],[])
            self.assertEqual(payload['candidates'],[])
            self.assertNotIn('crowdMetrics',payload)

    def test_registry_and_openapi(self):
        with self.client() as client:
            models = client.get('/v1/models').json()['models']
            self.assertEqual([m['id'] for m in models],['fire-smoke','person-detector'])
            self.assertEqual(models[1]['usedFor'],'crowd-analysis')
            self.assertEqual(models[1]['status'],'ready')
            self.assertNotIn(str(Path.cwd()),json.dumps(models))
            schema = client.get('/openapi.json').json()
            self.assertIn('crowdMetrics',schema['components']['schemas']['UnifiedAnalysis']['properties'])
            self.assertNotIn('/analyze/crowd',schema['paths'])
            for code in ['200','503']:
                example = schema['paths']['/v1/analyze/image']['post']['responses'][code]['content']['application/json']['example']
                UnifiedAnalysis.model_validate(example)

    def test_invalid_catalog_blocks_startup(self):
        client = self.client()
        self.path.write_text('{"cameras":[{"cameraId":"bad","crowd":{"capacity":0}}]}')
        with self.assertRaisesRegex(RuntimeError,'startup validation'):
            with client:
                pass
