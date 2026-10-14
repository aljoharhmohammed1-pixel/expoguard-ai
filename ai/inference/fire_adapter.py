"""Thin image adapter; accepts trusted metadata and image bytes, never remote URLs."""
import math
from pathlib import Path
from ai.quality.image_quality import quality_check
from ai.inference.coordinates import CoordinateError, normalize_box, validate_image_size
from ai.core.analysis import Candidate, Detection, ImageQuality, VisionAnalysis, AnalysisError, ImageRequest, validate_analysis_for_request

MODEL_VERSION = 'fire-smoke-infernoguard-v1.0.0'
POLICY_VERSION = 'fire-review-v0.3.0'
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'infernoguard_best.pt'


def map_result(result, request, image_size):
    validate_image_size(image_size, result.orig_shape)
    detections = []
    mapping = {'fire': 'visible_flame', 'smoke': 'suspected_smoke'}
    names = result.names
    names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else names
    if not isinstance(names, dict) or len(names) != 2 or set(names.values()) != set(mapping):
        raise ValueError('Expected a dedicated fire/smoke detector class map')
    for box in result.boxes:
        class_id = box.cls.item()
        if isinstance(class_id, bool) or not math.isfinite(class_id) or class_id != int(class_id):
            raise ValueError('Invalid class index')
        label = mapping[names[int(class_id)]]
        normalized = normalize_box(box.xyxy[0].tolist(), image_size)
        detections.append(Detection(
            detectionId=f'det-{len(detections)+1}', label=label,
            confidence=box.conf.item(), bbox=normalized,
            evidence='Visible flame indicator.' if label == 'visible_flame' else 'Suspected smoke; this does not confirm fire.'))
    # Keep local IDs stable when YOLO returns the same boxes in a different order.
    detections.sort(key=lambda d: (d.label, d.bbox, -d.confidence))
    detections = [d.model_copy(update={'detectionId': f'det-{i+1}'}) for i, d in enumerate(detections)]
    # Connected components of overlapping boxes: grouping evidence, not merging incidents.
    groups = []
    remaining = set(range(len(detections)))
    while remaining:
        group = {min(remaining)}
        remaining -= group
        changed = True
        while changed:
            changed = False
            for index in sorted(remaining):
                a = detections[index].bbox
                if any(min(a[2], detections[j].bbox[2]) > max(a[0], detections[j].bbox[0]) and
                       min(a[3], detections[j].bbox[3]) > max(a[1], detections[j].bbox[1]) for j in group):
                    group.add(index)
                    remaining.remove(index)
                    changed = True
        groups.append([detections[i] for i in sorted(group)])
    candidates = [Candidate(
        candidateId=f'cand-{i+1}', detectionIds=[d.detectionId for d in group], category='FIRE',
        confidence=max(d.confidence for d in group),
        summaryAr=_summary(group),
        zoneId=request.zoneId, assetId=request.assetId,
        decisionReason='Visual evidence requires backend policy evaluation; confidence is the maximum evidence score, not a calibrated incident probability.')
        for i, group in enumerate(groups)]
    return detections, candidates


def _summary(group):
    labels = {d.label for d in group}
    if labels == {'suspected_smoke'}:
        return 'رصد دخان محتمل؛ لا يثبت وجود حريق ويحتاج إلى مراجعة.'
    if labels == {'visible_flame'}:
        return 'رصد مؤشر بصري محتمل على وجود لهب؛ يحتاج إلى مراجعة.'
    return 'رصد مؤشرات بصرية على لهب ودخان محتمل؛ تحتاج إلى مراجعة.'


def adapt_yolo_result(result, request, image_size, image_quality):
    """Convert an existing YOLO result without loading or rerunning the model.

    Caller supplies the actual oriented source image dimensions and assessed quality.
    Resized/cropped external inputs are unsupported unless already restored to that frame.
    Invalid request/quality is a caller error; malformed model output is a controlled ERROR.
    """
    request = ImageRequest.model_validate(request)
    quality = ImageQuality.model_validate(image_quality)
    if not quality.usable:
        raise ValueError('Only adapt results from usable images')
    base = dict(requestId=request.requestId, imageId=request.imageId, imageQuality=quality,
                modelVersion=MODEL_VERSION, policyVersion=POLICY_VERSION)
    try:
        detections, candidates = map_result(result, request, image_size)
        analysis = VisionAnalysis(**base, analysisStatus='OK', detections=detections, candidates=candidates,
            limitations=['Single-image visual indicators only.',
                         'Candidates group overlapping boxes only; spatially separate evidence may belong to the same incident.',
                         'Candidate confidence is maximum detection confidence, not a calibrated probability.'])
        return validate_analysis_for_request(analysis, request)
    except CoordinateError:
        return VisionAnalysis(**base, analysisStatus='ERROR', error=AnalysisError(
            code='INVALID_MODEL_OUTPUT', message='Invalid original-image dimensions or bounding-box coordinates.'))
    except Exception:
        return VisionAnalysis(**base, analysisStatus='ERROR', error=AnalysisError(
            code='INVALID_MODEL_OUTPUT', message='Detector output failed contract validation.'))


class FireAnalyzer:
    def __init__(self, model=None, model_loader=None):
        self.model = model
        self.model_loader = model_loader or self._load

    @staticmethod
    def _load():
        import os
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        from ultralytics import YOLO
        return YOLO(str(MODEL_PATH))

    def analyze(self, request, image_bytes):
        request = ImageRequest.model_validate(request)
        image, quality = quality_check(image_bytes)
        base = dict(requestId=request.requestId, imageId=request.imageId,
                    modelVersion=MODEL_VERSION, policyVersion=POLICY_VERSION, imageQuality=quality)
        if not quality.usable:
            return VisionAnalysis(**base, analysisStatus='UNUSABLE')
        def failure(code, message):
            return VisionAnalysis(**base, analysisStatus='ERROR', error=AnalysisError(code=code, message=message))
        if self.model is None:
            try:
                self.model = self.model_loader()
            except Exception:
                return failure('MODEL_LOAD_FAILED', 'Could not load the configured detector.')
        try:
            results = self.model.predict(source=image, device='cpu', conf=0.35, imgsz=640, verbose=False)
        except Exception:
            return failure('INFERENCE_FAILED', 'Detector inference failed.')
        try:
            if len(results) != 1:
                raise ValueError('Expected one image result')
            analysis = adapt_yolo_result(results[0], request, image.size, quality)
            analysis.limitations.append('Quality gate checks decode, format, size and extreme brightness; blur and occlusion are not assessed.')
            return analysis
        except Exception:
            return failure('INVALID_MODEL_OUTPUT', 'Detector output failed contract validation.')
