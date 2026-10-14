"""Contract tests independent of model weights, image libraries and network."""
import copy
import json
from pathlib import Path
import unittest
from pydantic import ValidationError
from ai.core.analysis import (
    Candidate, Detection, ImageQuality, ImageRequest, VisionAnalysis,
    validate_analysis_for_request,
)


def payload():
    return dict(
        requestId='r1', imageId='i1', analysisStatus='OK',
        modelVersion='fire-v1', policyVersion='review-v1',
        imageQuality=dict(usable=True, issues=[]),
        detections=[dict(detectionId='d1', label='visible_flame', confidence=.9,
                         bbox=[0, 0, 1, 1], evidence='Visible flame indicator')],
        candidates=[dict(candidateId='c1', detectionIds=['d1'], category='FIRE',
                         confidence=.9, summaryAr='اشتباه لهب', decisionReason='Review visual evidence')],
    )


def metadata():
    return ImageRequest(requestId='r1', imageId='i1', source='CCTV', cameraId='cam1',
                        capturedAt='2026-10-13T12:00:00+03:00')


class ContractTests(unittest.TestCase):
    def test_valid_json_roundtrip(self):
        result = validate_analysis_for_request(payload(), metadata())
        self.assertEqual(VisionAnalysis.model_validate_json(result.model_dump_json()), result)
        self.assertEqual(result.detections[0].bbox, (0, 0, 1, 1))

    def test_required_fields(self):
        for key in ('requestId', 'imageId', 'analysisStatus', 'modelVersion', 'policyVersion', 'imageQuality'):
            with self.subTest(key=key):
                data = payload()
                del data[key]
                with self.assertRaises(ValidationError):
                    VisionAnalysis.model_validate(data)

    def test_unknown_fields_rejected_at_every_level(self):
        for section in ('root', 'quality', 'detection', 'candidate'):
            data = payload()
            target = {'root': data, 'quality': data['imageQuality'],
                      'detection': data['detections'][0], 'candidate': data['candidates'][0]}[section]
            target['unexpected'] = 'value'
            with self.subTest(section=section), self.assertRaises(ValidationError):
                VisionAnalysis.model_validate(data)

    def test_confidence_rejects_coercion_and_nonfinite(self):
        for value in (True, False, '0.9', -.01, 1.01, float('nan'), float('inf')):
            for section in ('detections', 'candidates'):
                data = payload()
                data[section][0]['confidence'] = value
                with self.subTest(value=value, section=section), self.assertRaises(ValidationError):
                    VisionAnalysis.model_validate(data)

    def test_box_length_range_order_and_type(self):
        for bbox in ([0, 0, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0],
                     [-.1, 0, 1, 1], [0, 0, 1.1, 1], ['0', 0, 1, 1], [False, 0, 1, 1]):
            data = payload()
            data['detections'][0]['bbox'] = bbox
            with self.subTest(bbox=bbox), self.assertRaises(ValidationError):
                VisionAnalysis.model_validate(data)

    def test_duplicate_ids_and_references(self):
        for section in ('detections', 'candidates'):
            data = payload()
            data[section].append(copy.deepcopy(data[section][0]))
            with self.assertRaises(ValidationError):
                VisionAnalysis.model_validate(data)
        for refs in ([], ['missing'], ['d1', 'd1']):
            data = payload()
            data['candidates'][0]['detectionIds'] = refs
            with self.assertRaises(ValidationError):
                VisionAnalysis.model_validate(data)

    def test_status_matrix(self):
        for status in ('OK', 'UNUSABLE', 'ERROR'):
            for usable in (False, True):
                for has_error in (False, True):
                    data = payload()
                    data.update(analysisStatus=status, detections=[], candidates=[],
                                imageQuality=dict(usable=usable, issues=[] if usable else ['DECODE_FAILED']))
                    if has_error:
                        data['error'] = dict(code='INFERENCE_FAILED', message='Detector failed')
                    valid = (status == 'OK' and usable and not has_error or
                             status == 'UNUSABLE' and not usable and not has_error or
                             status == 'ERROR' and has_error)
                    with self.subTest(status=status, usable=usable, error=has_error):
                        if valid:
                            VisionAnalysis.model_validate(data)
                        else:
                            with self.assertRaises(ValidationError):
                                VisionAnalysis.model_validate(data)

    def test_quality_consistency_and_strict_boolean(self):
        for value in (dict(usable='true'), dict(usable=1), dict(usable=False),
                      dict(usable=True, issues=['TOO_DARK']),
                      dict(usable=False, issues=['TOO_DARK', 'TOO_DARK'])):
            with self.assertRaises(ValidationError):
                ImageQuality.model_validate(value)
        ImageQuality(usable=False, issues=['TOO_BLURRY'])

    def test_failed_analysis_has_no_partial_output(self):
        for status in ('UNUSABLE', 'ERROR'):
            data = payload()
            data['analysisStatus'] = status
            if status == 'ERROR':
                data['error'] = dict(code='INFERENCE_FAILED', message='Failed')
            else:
                data['imageQuality'] = dict(usable=False, issues=['DECODE_FAILED'])
            with self.assertRaises(ValidationError):
                VisionAnalysis.model_validate(data)

    def test_review_and_evidence_category_consistency(self):
        data = payload()
        data['candidates'][0]['requiresHumanReview'] = False
        with self.assertRaises(ValidationError):
            VisionAnalysis.model_validate(data)
        data = payload()
        data['candidates'][0]['category'] = 'MEDICAL'
        with self.assertRaises(ValidationError):
            VisionAnalysis.model_validate(data)

    def test_metadata_identity_and_provenance(self):
        for key in ('requestId', 'imageId'):
            data = payload()
            data[key] = 'different'
            with self.assertRaises(ValueError):
                validate_analysis_for_request(data, metadata())
        for key in ('zoneId', 'assetId'):
            data = payload()
            data['candidates'][0][key] = 'invented'
            with self.assertRaises(ValueError):
                validate_analysis_for_request(data, metadata())

    def test_alarm_needs_backend_authority(self):
        data = payload()
        data['candidates'][0]['decision'] = 'ALARM'
        with self.assertRaises(ValueError):
            validate_analysis_for_request(data, metadata())

    def test_mutated_nested_instances_revalidated_at_boundary(self):
        result = VisionAnalysis.model_validate(payload())
        result.candidates[0].detectionIds.append('missing')
        with self.assertRaises(ValidationError):
            validate_analysis_for_request(result, metadata())

    def test_exported_schemas_match_models(self):
        root = Path(__file__).resolve().parents[1] / 'contracts'
        for name, model in [('image-request', ImageRequest), ('vision-analysis', VisionAnalysis)]:
            self.assertEqual(json.loads((root / f'{name}.schema.json').read_text(encoding='utf-8')),
                             model.model_json_schema())


if __name__ == '__main__':
    unittest.main()
