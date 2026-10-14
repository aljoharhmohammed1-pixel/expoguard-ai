"""Configured visible-person occupancy, not certified crowd safety."""
from ai.crowd.schemas import CrowdMetrics
from ai.crowd.geometry import point_in_polygon, bottom_center
from ai.core.multimodal import Evidence, UnifiedCandidate

POLICY_VERSION='crowd-occupancy-v1.0.0'
LIMITATIONS=[
    'Single-frame visible people only; occlusion and camera angle affect counts.',
    'Configured MVP operational thresholds are not certified crowd-safety limits.',
    'No identity, tracking, entry/exit flow, speed or prediction.']


def skip_reason(camera):
    if camera is None:
        return 'Crowd analysis skipped: no camera configuration.'
    if not camera.crowd.enabled:
        return 'Crowd analysis skipped: disabled for this camera.'
    if camera.crowd.roi is None and not camera.crowd.useFullFrame:
        return 'Crowd analysis skipped: no ROI and full-frame mode is not enabled.'
    return None


def measure(people,camera):
    reason=skip_reason(camera)
    if reason:
        return None,[],[reason]
    config=camera.crowd
    inside=[p for p in people if config.useFullFrame or point_in_polygon(bottom_center(p.bbox),config.roi)]
    count=len(inside)
    ratio=None if config.capacity is None else count/config.capacity
    level=None
    if ratio is not None:
        level='CRITICAL' if ratio>=config.thresholds.critical else 'HIGH' if ratio>=config.thresholds.high else 'MEDIUM' if ratio>=config.thresholds.medium else 'NORMAL'
    limitations=list(LIMITATIONS)
    if ratio is None:
        limitations.append('No configured capacity: count only, no occupancy ratio or crowd level.')
    return CrowdMetrics(peopleCount=count,capacity=config.capacity,occupancyRatio=ratio,crowdLevel=level,
                        cameraId=camera.cameraId,zoneId=camera.zoneId,roiApplied=config.roi is not None),inside,limitations


def map_crowd(metrics,people,camera,asset_id=None,include_people=False):
    if metrics is None:
        return [],[]
    detections=[]
    if include_people:
        detections=[Evidence(detectionId=f'crowd-person-{i+1}',modality='IMAGE',label='person',
            confidence=p.confidence,bbox=p.bbox,evidence='Visible person inside configured monitoring area.')
            for i,p in enumerate(people)]
    requires_review=metrics.crowdLevel in ('HIGH','CRITICAL') or (metrics.crowdLevel=='MEDIUM' and camera.crowd.reviewMedium)
    if not requires_review:
        return detections,[]
    evidence_id='crowd-count-1'
    detections.append(Evidence(detectionId=evidence_id,modality='IMAGE',label='visible_people_count',
        confidence=None,evidence=f'{metrics.peopleCount} visible people inside the configured monitoring area; capacity {metrics.capacity}.'))
    candidate=UnifiedCandidate(candidateId='crowd-cand-1',detectionIds=[evidence_id],category='CROWD',
        confidence=None,summaryAr=f'رُصد {metrics.peopleCount} شخصًا ضمن نطاق المراقبة؛ مستوى الإشغال التشغيلي يستدعي المراجعة.',
        zoneId=camera.zoneId,assetId=asset_id,decisionReason='Visible count meets a configured review threshold; this is not a certified safety assessment.')
    return detections,[candidate]
