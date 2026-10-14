"""Version 1 image-analysis contract. Operational decisions belong to the server."""
from typing import Annotated, Literal, Any
from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, StrictBool, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"\S")]
Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Box = tuple[Score, Score, Score, Score]
AnalysisStatus = Literal['OK', 'UNUSABLE', 'ERROR']
Decision = Literal['NONE', 'REVIEW', 'ALERT', 'ALARM']
Category = Literal['FIRE', 'CROWD', 'MEDICAL', 'ACCESSIBILITY', 'TECHNICAL_FAILURE', 'CLEANLINESS', 'SECURITY', 'OTHER']


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True, revalidate_instances='always')


class ImageRequest(Contract):
    requestId: Identifier
    imageId: Identifier
    source: Literal['CCTV', 'VISITOR_APP']
    capturedAt: AwareDatetime
    cameraId: Identifier | None = None
    zoneId: Identifier | None = None
    eventId: Identifier | None = None
    assetId: Identifier | None = None
    regionOfInterest: Box | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def valid_source(self):
        if self.source == 'CCTV' and self.cameraId is None:
            raise ValueError('CCTV requires cameraId')
        if self.regionOfInterest is not None:
            raise ValueError('ROI is not supported in this adapter version')
        return self


class ImageQuality(Contract):
    usable: StrictBool
    issues: list[Literal['DECODE_FAILED', 'TOO_SMALL', 'TOO_LARGE', 'TOO_DARK', 'TOO_BRIGHT', 'TOO_BLURRY', 'EMPTY_IMAGE', 'UNSUPPORTED_FORMAT']] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent(self):
        if self.usable == bool(self.issues):
            raise ValueError('Usable images have no issues; unusable images require a reason')
        if len(self.issues) != len(set(self.issues)):
            raise ValueError('Duplicate quality issues')
        return self


class Detection(Contract):
    detectionId: Identifier
    label: Literal['visible_flame', 'suspected_smoke']
    confidence: Score
    bbox: Box
    evidence: Identifier

    @model_validator(mode='after')
    def ordered_box(self):
        x1, y1, x2, y2 = self.bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError('bbox must have positive width and height')
        return self


class Candidate(Contract):
    candidateId: Identifier
    detectionIds: list[Identifier] = Field(min_length=1)
    category: Category
    severity: Literal['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] | None = None
    confidence: Score
    summaryAr: Identifier
    zoneId: Identifier | None = None
    assetId: Identifier | None = None
    affectsAccessibility: StrictBool | None = None
    decision: Decision = 'REVIEW'
    requiresHumanReview: StrictBool = True
    decisionReason: Identifier

    @model_validator(mode='after')
    def consistent(self):
        if len(self.detectionIds) != len(set(self.detectionIds)):
            raise ValueError('Duplicate detection references')
        if self.decision == 'REVIEW' and not self.requiresHumanReview:
            raise ValueError('REVIEW requires human review')
        return self


class AnalysisError(Contract):
    code: Literal['MODEL_LOAD_FAILED', 'INFERENCE_FAILED', 'INVALID_MODEL_OUTPUT']
    message: Identifier


class VisionAnalysis(Contract):
    requestId: Identifier
    imageId: Identifier
    analysisStatus: AnalysisStatus
    modelVersion: Identifier
    policyVersion: Identifier
    imageQuality: ImageQuality
    detections: list[Detection] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    limitations: list[Identifier] = Field(default_factory=list)
    error: AnalysisError | None = None

    @model_validator(mode='after')
    def consistent(self):
        if (self.analysisStatus == 'ERROR') != (self.error is not None):
            raise ValueError('ERROR requires an error; other statuses cannot have one')
        if self.analysisStatus == 'UNUSABLE' and self.imageQuality.usable:
            raise ValueError('UNUSABLE requires unusable image quality')
        if self.analysisStatus == 'OK' and not self.imageQuality.usable:
            raise ValueError('OK requires usable image quality')
        if self.analysisStatus != 'OK' and (self.detections or self.candidates):
            raise ValueError('Failed analysis cannot contain partial detections or candidates')
        ids = [d.detectionId for d in self.detections]
        candidate_ids = [c.candidateId for c in self.candidates]
        if len(ids) != len(set(ids)) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError('Duplicate IDs')
        for c in self.candidates:
            if len(c.detectionIds) != len(set(c.detectionIds)) or not set(c.detectionIds) <= set(ids):
                raise ValueError('Invalid detection references')
            # This schema version contains only flame/smoke visual evidence.
            if c.category != 'FIRE':
                raise ValueError('This version only supports FIRE candidates from flame/smoke evidence')
        return self


def validate_analysis_for_request(analysis, request):
    """Validate at the integration boundary, including trusted metadata provenance.

    Revalidation is intentional: mutable lists may have changed after construction.
    It returns a fresh validated result; it does not authorize operational alarms.
    """
    request = ImageRequest.model_validate(request)
    analysis = VisionAnalysis.model_validate(analysis)
    if analysis.requestId != request.requestId or analysis.imageId != request.imageId:
        raise ValueError('Analysis identity does not match request')
    for candidate in analysis.candidates:
        if candidate.zoneId != request.zoneId or candidate.assetId != request.assetId:
            raise ValueError('Candidate location/asset does not match trusted request metadata')
        if candidate.decision == 'ALARM':
            raise ValueError('AI boundary does not authorize ALARM; backend policy must decide')
    return analysis
