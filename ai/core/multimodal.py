"""Canonical v2 envelope for image, text and audio; legacy v1 remains supported."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from ai.core.analysis import Contract, Identifier, Score, Box, Category, ImageQuality, Candidate
from ai.crowd.schemas import CrowdMetrics

SourceOffset = Annotated[int, Field(strict=True, ge=0)]


def validate_source_span(source, span, evidence):
    """Offsets count Python Unicode code points, end exclusive (not UTF-16)."""
    if span is None or not 0 <= span[0] < span[1] <= len(source) or source[slice(*span)] != evidence:
        raise ValueError('Evidence must be an exact in-bounds source span')


class ReportRequest(Contract):
    requestId: Identifier
    source: Literal['VISITOR_APP'] = 'VISITOR_APP'
    zoneId: Identifier | None = None
    assetId: Identifier | None = None


class TextRequest(ReportRequest):
    textId: Identifier
    text: str = Field(max_length=20000)
    language: Literal['ar'] = 'ar'


class AudioRequest(ReportRequest):
    audioId: Identifier
    mimeType: Literal['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/webm', 'audio/ogg']
    language: Literal['ar'] = 'ar'


class Evidence(Contract):
    detectionId: Identifier
    modality: Literal['IMAGE', 'TEXT']
    label: Identifier
    confidence: Score | None
    bbox: Box | None = None
    evidence: Identifier
    textSpan: tuple[SourceOffset, SourceOffset] | None = Field(
        default=None, description='Python Unicode code point offsets [start, end), with 0 <= start < end <= source length. Not UTF-16 code units.')

    @model_validator(mode='after')
    def geometry(self):
        if self.bbox is not None:
            a,b,c,d = self.bbox
            if self.modality != 'IMAGE' or a >= c or b >= d:
                raise ValueError('Invalid image box')
        if self.modality == 'TEXT':
            if self.textSpan is None or not 0 <= self.textSpan[0] < self.textSpan[1]:
                raise ValueError('Text evidence requires a positive source span')
        elif self.textSpan is not None:
            raise ValueError('Image evidence cannot have text spans')
        return self


class UnifiedCandidate(Candidate):
    confidence: Score | None
    mentionedZone: Identifier | None = None


class UnifiedError(Contract):
    code: Identifier
    message: Identifier


class Transcript(Contract):
    text: str = Field(max_length=20000)
    language: Literal['ar']
    transcriptionConfidence: Score | None = None


class UnifiedAnalysis(Contract):
    schemaVersion: Literal['2.0.0'] = '2.0.0'
    requestId: Identifier
    inputType: Literal['IMAGE', 'TEXT', 'AUDIO']
    imageId: Identifier | None = None
    textId: Identifier | None = None
    audioId: Identifier | None = None
    analysisStatus: Literal['OK', 'UNUSABLE', 'ERROR']
    modelVersion: Identifier
    policyVersion: Identifier
    componentVersions: dict[str, Identifier] = Field(default_factory=dict)
    imageQuality: ImageQuality | None = None
    crowdMetrics: CrowdMetrics | None = None
    transcript: Transcript | None = None
    detections: list[Evidence] = Field(default_factory=list)
    candidates: list[UnifiedCandidate] = Field(default_factory=list)
    limitations: list[Identifier] = Field(default_factory=list)
    error: UnifiedError | None = None

    @model_validator(mode='after')
    def consistent(self):
        if self.crowdMetrics is not None and (self.inputType != 'IMAGE' or self.analysisStatus != 'OK'):
            raise ValueError('Crowd metrics require successful image analysis')
        identities = {'IMAGE': self.imageId, 'TEXT': self.textId, 'AUDIO': self.audioId}
        if identities[self.inputType] is None or sum(v is not None for v in identities.values()) != 1:
            raise ValueError('Exactly the appropriate input identity is required')
        if self.inputType == 'IMAGE':
            if self.imageQuality is None or self.transcript is not None:
                raise ValueError('Image result needs quality and no transcript')
            if self.analysisStatus == 'OK' and not self.imageQuality.usable:
                raise ValueError('OK image requires usable quality')
            if self.analysisStatus == 'UNUSABLE' and self.imageQuality.usable:
                raise ValueError('UNUSABLE image requires quality issues')
        elif self.imageQuality is not None:
            raise ValueError('Non-image results cannot invent image quality')
        if self.inputType != 'AUDIO' and self.transcript is not None:
            raise ValueError('Transcript belongs to audio only')
        if self.inputType == 'AUDIO' and self.analysisStatus == 'OK' and (self.transcript is None or not self.transcript.text.strip()):
            raise ValueError('Successful audio analysis requires a transcript')
        if (self.analysisStatus == 'ERROR') != (self.error is not None):
            raise ValueError('ERROR needs error details exclusively')
        if self.analysisStatus != 'OK' and (self.detections or self.candidates):
            raise ValueError('No partial evidence on failed analysis')
        ids = [d.detectionId for d in self.detections]
        if len(ids) != len(set(ids)) or len({c.candidateId for c in self.candidates}) != len(self.candidates):
            raise ValueError('Duplicate IDs')
        for d in self.detections:
            if d.modality != ('IMAGE' if self.inputType == 'IMAGE' else 'TEXT'):
                raise ValueError('Evidence modality differs from input')
            if self.inputType == 'AUDIO' and d.textSpan is not None and self.transcript is not None:
                validate_source_span(self.transcript.text, d.textSpan, d.evidence)
        for c in self.candidates:
            if not set(c.detectionIds) <= set(ids):
                raise ValueError('Missing evidence references')
            if c.decision != 'REVIEW' or not c.requiresHumanReview or c.severity is not None:
                raise ValueError('v2 shadow policy requires REVIEW and no operational severity')
        return self


def from_fire_analysis(result):
    """Lossless evidence conversion from the existing fire v1 result to v2."""
    data = result.model_dump()
    data['inputType'] = 'IMAGE'
    data['componentVersions'] = {'fire': result.modelVersion}
    data['detections'] = [dict(d.model_dump(), modality='IMAGE') for d in result.detections]
    return UnifiedAnalysis.model_validate(data)


def validate_unified_for_request(result, request):
    result = UnifiedAnalysis.model_validate(result)
    identity = {'IMAGE':'imageId', 'TEXT':'textId', 'AUDIO':'audioId'}[result.inputType]
    if result.requestId != request.requestId or getattr(result, identity) != getattr(request, identity, None):
        raise ValueError('Request identity mismatch')
    if result.crowdMetrics is not None and (result.crowdMetrics.cameraId != getattr(request, 'cameraId', None) or result.crowdMetrics.zoneId != request.zoneId):
        raise ValueError('Crowd camera or location mismatch')
    for c in result.candidates:
        if c.zoneId != request.zoneId or c.assetId != request.assetId:
            raise ValueError('Untrusted location or asset')
    if result.inputType == 'TEXT':
        for d in result.detections:
            validate_source_span(request.text, d.textSpan, d.evidence)
    return result
