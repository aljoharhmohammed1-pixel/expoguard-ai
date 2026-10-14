import hashlib
from pathlib import Path
from threading import Lock
from time import perf_counter
from ai.inference.fire_adapter import FireAnalyzer, MODEL_VERSION, POLICY_VERSION
from ai.core.multimodal import UnifiedAnalysis, from_fire_analysis, validate_unified_for_request
from ai.crowd.config import load_cameras
from ai.crowd.person import PersonDetector, CrowdFailure, MODEL_VERSION as PERSON_VERSION
from ai.crowd.person import INFERENCE_VERSION
from ai.crowd.analysis import measure, map_crowd, skip_reason, POLICY_VERSION as CROWD_POLICY
from ai.quality.image_quality import quality_check


class CameraZoneConflict(ValueError):
    pass


def select_device(preference, cuda_available):
    if preference == 'cuda' and not cuda_available:
        raise RuntimeError('Configured CUDA device is unavailable')
    return 'cuda' if preference != 'cpu' and cuda_available else 'cpu'


class DeviceModel:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.inference_ms = None

    def predict(self, **kwargs):
        kwargs['device'] = self.device
        started = perf_counter()
        try:
            return self.model.predict(**kwargs)
        finally:
            self.inference_ms = (perf_counter()-started)*1000


class ModelRegistry:
    def __init__(self, settings, loader=None, cuda_available=None, person_loader=None):
        self.settings, self.loader, self.cuda_available = settings, loader, cuda_available
        self.ready = False
        self.lock = Lock()
        self.device = None
        self.person_loader = person_loader
        self.person = None

    def start(self):
        try:
            path = Path(self.settings.model_path)
            if not path.is_file():
                raise ValueError('Missing local weights')
            if self.cuda_available is None:
                import torch
                available = torch.cuda.is_available()
            else:
                available = self.cuda_available
            self.device = select_device(self.settings.device, available)
            loader = self.loader
            if loader is None:
                from ultralytics import YOLO
                loader = YOLO
            model = loader(str(path))
            names = model.names
            names = list(names.values()) if isinstance(names, dict) else list(names)
            if sorted(names) != ['fire', 'smoke']:
                raise ValueError('Unexpected model classes')
            self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.proxy = DeviceModel(model, self.device)
            self.analyzer = FireAnalyzer(self.proxy)
            self.cameras = load_cameras(self.settings.camera_config_path)
            if any(skip_reason(camera) is None for camera in self.cameras.cameras):
                self.person = PersonDetector(self.settings.person_model_path, self.device, self.person_loader)
                self.person.start()
                self.person_sha256 = hashlib.sha256(self.settings.person_model_path.read_bytes()).hexdigest()
            self.ready = True
        except Exception:
            self.ready = False
            raise RuntimeError('Required model or camera configuration failed startup validation') from None

    def stop(self):
        self.ready = False
        self.analyzer = None
        self.proxy = None
        self.person = None

    def analyze(self, request, media):
        with self.lock:
            if not self.ready:
                raise RuntimeError('Model is not ready')
            camera = self.cameras.resolve(request.cameraId)
            if camera is not None:
                if request.zoneId is not None and request.zoneId != camera.zoneId:
                    raise CameraZoneConflict('Camera zone conflicts with configured mapping')
                request = request.model_copy(update={'zoneId':camera.zoneId})
            self.proxy.inference_ms = None
            result = from_fire_analysis(self.analyzer.analyze(request, media))
            timings = {'fireInferenceMs':self.proxy.inference_ms, 'crowdInferenceMs':None}
            if result.analysisStatus == 'OK':
                reason = skip_reason(camera)
                if reason:
                    result.limitations.append(reason)
                else:
                    data = result.model_dump()
                    data['componentVersions'].update(crowd=PERSON_VERSION, crowdPolicy=CROWD_POLICY,
                                                     crowdConfig=camera.digest, crowdInference=INFERENCE_VERSION)
                    try:
                        image, quality = quality_check(media)
                        if not quality.usable:
                            raise CrowdFailure('CROWD_INPUT_UNUSABLE')
                        started = perf_counter()
                        try:
                            people = self.person.detect(image, image_size=camera.crowd.inferenceImageSize)
                        finally:
                            timings['crowdInferenceMs'] = (perf_counter()-started)*1000
                        metrics, inside, limits = measure(people, camera)
                        detections, candidates = map_crowd(metrics, inside, camera, request.assetId,
                                                          self.settings.include_person_detections)
                        data['crowdMetrics'] = metrics
                        data['detections'].extend(detections)
                        data['candidates'].extend(candidates)
                        data['limitations'].extend(limits)
                    except CrowdFailure as exc:
                        data.update(analysisStatus='ERROR', detections=[], candidates=[], crowdMetrics=None,
                                    error={'code':exc.code, 'message':'Crowd analysis could not be completed.'})
                    result = UnifiedAnalysis.model_validate(data)
            return validate_unified_for_request(result, request), timings

    def person_metadata(self):
        return {'id':'person-detector', 'version':PERSON_VERSION, 'policyVersion':CROWD_POLICY,
                'usedFor':'crowd-analysis', 'inferenceVersion':INFERENCE_VERSION,
                'supportedImageSizes':[640,960,1280,1600], 'task':'object_detection', 'framework':'ultralytics',
                'status':'ready' if self.ready and self.person is not None else 'not_configured' if self.ready else 'unavailable',
                'device':self.device, 'classes':['person'], 'sha256':getattr(self, 'person_sha256', None)}

    def metadata(self):
        return {'id':'fire-smoke', 'version':MODEL_VERSION, 'policyVersion':POLICY_VERSION,
                'task':'object_detection', 'framework':'ultralytics', 'status':'ready' if self.ready else 'unavailable',
                'device':self.device, 'classes':['visible_flame', 'suspected_smoke'],
                'sha256':getattr(self, 'sha256', None)}
