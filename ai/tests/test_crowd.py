import unittest
import tempfile
import os
from unittest.mock import Mock
from PIL import Image
from pathlib import Path
from ai.crowd.config import CameraCatalog, CameraConfig, CrowdConfig, load_cameras
from ai.crowd.person import PersonDetector, PersonDetection, CrowdFailure
from ai.crowd.analysis import measure, map_crowd
from ai.crowd.geometry import bottom_center, point_in_polygon
from test_analysis import Model, box

MODEL_PATH=Path(__file__).resolve().parents[1]/'models/yolo26n.pt'


class PersonModel(Model):
    names={0:'person',1:'bicycle'}
    def predict(self,**kwargs):
        self.arguments=kwargs
        return super().predict(**kwargs)
from ai.crowd.schemas import CrowdMetrics


class CrowdSchemaTests(unittest.TestCase):
    def test_metrics_roundtrip_and_missing_capacity(self):
        for kwargs in [{},dict(capacity=100,occupancyRatio=.83,crowdLevel='HIGH')]:
            result = CrowdMetrics(peopleCount=83,cameraId='test',roiApplied=True,**kwargs)
            self.assertEqual(CrowdMetrics.model_validate_json(result.model_dump_json()), result)

    def test_invalid_metrics(self):
        for change in [dict(peopleCount=-1),dict(peopleCount=True),dict(capacity=0),dict(capacity=-2),
                       dict(occupancyRatio=-1),dict(occupancyRatio=float('nan')),dict(crowdLevel='HIGH'),
                       dict(capacity=10,occupancyRatio=.5,crowdLevel='MEDIUM')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                data = dict(peopleCount=1,cameraId='test',roiApplied=False)
                data.update(change)
                CrowdMetrics.model_validate(data)


class CrowdConfigTests(unittest.TestCase):
    def test_catalog_loading_and_missing_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cameras.json'
            path.write_text('{"cameras":[{"cameraId":"test","crowd":{"enabled":true,"useFullFrame":true}}]}')
            catalog=load_cameras(path)
            self.assertTrue(catalog.resolve('test').crowd.enabled)
            self.assertIsNone(catalog.resolve('unknown'))
            path.write_text('bad')
            with self.assertRaises(ValueError): load_cameras(path)

    def test_invalid_roi_and_thresholds(self):
        for change in [dict(roi=[(0,0),(1,1)]),dict(roi=[(0,0),(2,0),(1,1)]),
                       dict(roi=[(0,0),(1,0),(1,0)]),dict(roi=[(0,0),(.5,.5),(1,1)]),
                       dict(roi=[(0,0),(1,1),(0,1),(1,0)]),
                       dict(thresholds=dict(medium=.7,high=.5,critical=.9)),
                       dict(roi=[(0,0),(1,0),(1,1)],useFullFrame=True)]:
            with self.subTest(change=change),self.assertRaises(ValueError): CrowdConfig(**change)
        with self.assertRaises(ValueError):
            CameraCatalog(cameras=[{'cameraId':'x'},{'cameraId':'x'}])


class PersonTests(unittest.TestCase):
    def test_resolution_is_bounded_and_preserves_coordinates(self):
        model=PersonModel([box()])
        detector=PersonDetector(MODEL_PATH,loader=lambda p:model)
        detector.start()
        image=Image.new('RGB',(100,80),'gray')
        baseline=detector.detect(image)
        for size in (960,1280,1600):
            self.assertEqual(detector.detect(image,image_size=size),baseline)
            self.assertEqual(model.arguments['imgsz'],size)
            self.assertEqual(model.arguments['conf'],.25)
            self.assertEqual(CrowdConfig(inferenceImageSize=size).inferenceImageSize,size)
        for size in (0,6400,True,1600.0,'1600'):
            with self.assertRaises(ValueError): detector.detect(image,image_size=size)
        for size in (0,2048,'1600',True):
            with self.assertRaises(ValueError): CrowdConfig(inferenceImageSize=size)

    def test_person_only_and_load_once(self):
        loader=Mock(return_value=PersonModel([box(),box(1)]))
        detector=PersonDetector(MODEL_PATH,loader=loader)
        detector.start(); detector.start()
        people=detector.detect(Image.new('RGB',(100,80),'gray'))
        self.assertEqual(len(people),1)
        self.assertEqual(people[0].bbox,(.1,.1,.5,.5))
        self.assertEqual(people[0].confidence,.8)
        self.assertEqual(loader.call_count,1)
        self.assertEqual(loader.return_value.arguments['classes'],[0])

    def test_failures_and_no_people(self):
        for model,code in [(PersonModel(fail=True),'PERSON_INFERENCE_FAILED'),
                           (PersonModel([box(coords=(-10,0,40,40))]),'INVALID_PERSON_OUTPUT')]:
            detector=PersonDetector(MODEL_PATH,loader=lambda p:model); detector.start()
            with self.assertRaises(CrowdFailure) as raised: detector.detect(Image.new('RGB',(100,80),'gray'))
            self.assertEqual(raised.exception.code,code)
        detector=PersonDetector(MODEL_PATH,loader=lambda p:PersonModel()); detector.start()
        self.assertEqual(detector.detect(Image.new('RGB',(100,80),'gray')),[])
        with self.assertRaises(CrowdFailure): PersonDetector('absent.pt').start()

    @unittest.skipUnless(os.environ.get('EXPOGUARD_REAL_TESTS')=='1','Enable real local weights')
    def test_real_local_people_and_empty_image(self):
        import importlib.util
        assets=Path(importlib.util.find_spec('ultralytics').origin).parent/'assets'
        detector=PersonDetector(MODEL_PATH); detector.start()
        with Image.open(assets/'bus.jpg') as source:
            people=detector.detect(source.convert('RGB'))
        self.assertGreater(len(people),0)
        self.assertEqual(detector.detect(Image.new('RGB',(640,640),'gray')),[])


class GeometryTests(unittest.TestCase):
    def test_bottom_point_and_boundary_inclusive(self):
        roi=[(.2,.2),(.8,.2),(.8,.8),(.2,.8)]
        self.assertEqual(bottom_center((.2,.1,.4,.7)),(.30000000000000004,.7))
        for p in [(.3,.7),(.2,.2),(.8,.5)]: self.assertTrue(point_in_polygon(p,roi))
        for p in [(.1,.5),(.5,.9)]: self.assertFalse(point_in_polygon(p,roi))
        self.assertFalse(point_in_polygon(bottom_center((.3,.3,.5,.9)),roi))

    def test_concave_polygon_and_invalid_overlap(self):
        roi=[(0,0),(1,0),(1,.4),(.4,.4),(.4,1),(0,1)]
        self.assertTrue(point_in_polygon((.2,.8),roi))
        self.assertFalse(point_in_polygon((.8,.8),roi))
        with self.assertRaises(ValueError):
            CrowdConfig(roi=[(0,0),(1,0),(.5,0),(.5,1),(0,1)])


class MetricsTests(unittest.TestCase):
    def test_candidates_are_review_only_with_real_evidence(self):
        camera=CameraConfig(cameraId='x',crowd={'enabled':True,'useFullFrame':True,'capacity':10})
        people=[PersonDetection(bbox=(.1,.1,.2,.2),confidence=.9)]*10
        for count,expected in [(0,0),(4,0),(5,0),(7,1),(9,1)]:
            metrics,inside,_=measure(people[:count],camera)
            detections,candidates=map_crowd(metrics,inside,camera)
            self.assertEqual(len(candidates),expected)
            if candidates:
                self.assertEqual(candidates[0].detectionIds,[detections[0].detectionId])
                self.assertIsNone(candidates[0].severity)
                self.assertIsNone(candidates[0].confidence)
                self.assertIsNone(candidates[0].zoneId)
                self.assertEqual(candidates[0].decision,'REVIEW')
        camera.crowd.reviewMedium=True
        metrics,inside,_=measure(people[:5],camera)
        detections,candidates=map_crowd(metrics,inside,camera,include_people=True)
        self.assertEqual(len(detections),6)
        self.assertEqual(len(candidates),1)

    def test_threshold_boundaries_and_missing_capacity(self):
        camera=CameraConfig(cameraId='test',crowd={'enabled':True,'useFullFrame':True,'capacity':100})
        person=PersonDetection(bbox=(.1,.1,.2,.2),confidence=.8)
        for count,level in [(0,'NORMAL'),(49,'NORMAL'),(50,'MEDIUM'),(70,'HIGH'),(90,'CRITICAL'),(150,'CRITICAL')]:
            result,_,_=measure([person]*count,camera)
            self.assertEqual(result.crowdLevel,level)
            self.assertEqual(result.occupancyRatio,count/100)
        camera.crowd.capacity=None
        result,_,_=measure([person],camera)
        self.assertEqual(result.peopleCount,1)
        self.assertIsNone(result.occupancyRatio)
        self.assertIsNone(result.crowdLevel)

    def test_debug_preserves_input_and_metrics(self):
        from PIL import Image
        from ai.crowd.debug import render_debug
        camera=CameraConfig(cameraId='debug',crowd={'enabled':True,'capacity':10,'roi':[(0,0),(.5,0),(.5,1),(0,1)]})
        people=[PersonDetection(bbox=(.1,.1,.3,.9),confidence=.8),PersonDetection(bbox=(.7,.1,.9,.9),confidence=.8)]
        metrics,_,_=measure(people,camera)
        before=metrics.model_dump()
        image=Image.new('RGB',(640,480),'white')
        original=image.tobytes()
        result=render_debug(image,people,camera,metrics)
        self.assertEqual(result.size,image.size)
        self.assertNotEqual(result.tobytes(),original)
        self.assertEqual(image.tobytes(),original)
        self.assertEqual(metrics.model_dump(),before)

    def test_missing_disabled_roi_and_custom_thresholds(self):
        for camera in [None,CameraConfig(cameraId='x'),CameraConfig(cameraId='x',crowd={'enabled':True})]:
            result,inside,limits=measure([],camera)
            self.assertIsNone(result)
            self.assertEqual(inside,[])
            self.assertTrue(limits)
        camera=CameraConfig(cameraId='x',crowd={'enabled':True,'capacity':10,'roi':[(0,0),(.5,0),(.5,1),(0,1)],
            'thresholds':{'medium':.1,'high':.2,'critical':.3}})
        a=PersonDetection(bbox=(.1,.1,.3,.9),confidence=.9)
        b=PersonDetection(bbox=(.7,.1,.9,.9),confidence=.9)
        result,inside,_=measure([a,b],camera)
        self.assertEqual(result.peopleCount,1)
        self.assertEqual(result.crowdLevel,'MEDIUM')
        self.assertEqual(inside,[a])
