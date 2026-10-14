"""Synthetic contract examples, not measured detector results."""
from ai.core.multimodal import UnifiedAnalysis
from ai.crowd.config import CameraConfig
from ai.crowd.analysis import measure, map_crowd
from ai.crowd.person import PersonDetection, MODEL_VERSION


def crowd_example():
    camera = CameraConfig(cameraId='DEMO-CROWD', crowd={'enabled':True,'useFullFrame':True,'capacity':4})
    people = [PersonDetection(bbox=(.1,.1,.2,.8),confidence=.8)]*4
    metrics, inside, limits = measure(people,camera)
    detections,candidates = map_crowd(metrics,inside,camera)
    return UnifiedAnalysis(requestId='example-only',imageId='example-image',inputType='IMAGE',
        analysisStatus='OK',modelVersion='fire-smoke-infernoguard-v1.0.0',policyVersion='fire-review-v0.3.0',
        imageQuality={'usable':True,'issues':[]},componentVersions={'crowd':MODEL_VERSION},
        crowdMetrics=metrics,detections=detections,candidates=candidates,
        limitations=['Synthetic documentation example; not measured results.']+limits).model_dump(mode='json')


def failure_example():
    return UnifiedAnalysis(requestId='example-only',imageId='example-image',inputType='IMAGE',
        analysisStatus='ERROR',modelVersion='fire-smoke-infernoguard-v1.0.0',policyVersion='fire-review-v0.3.0',
        imageQuality={'usable':True,'issues':[]},
        error={'code':'PERSON_INFERENCE_FAILED','message':'Crowd analysis could not be completed.'}).model_dump(mode='json',exclude={'crowdMetrics'})
