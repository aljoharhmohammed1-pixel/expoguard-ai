"""Offline fire-only integration boundary. Caller supplies trusted metadata."""
from typing import Literal
from pydantic import model_validator
from ai.core.analysis import Contract, ImageRequest, Identifier
from ai.core.multimodal import UnifiedAnalysis, from_fire_analysis, validate_unified_for_request
from ai.inference.fire_adapter import FireAnalyzer


class IntegrationRequest(Contract):
    contractVersion: Literal['1.0.0'] = '1.0.0'
    inputType: Literal['IMAGE']
    analysisMode: Literal['FIRE_SMOKE']
    metadata: ImageRequest


class IntegrationError(Contract):
    code: Literal['INVALID_REQUEST', 'INVALID_MEDIA', 'MEDIA_READ_FAILED', 'INTERNAL_ERROR', 'INVALID_ANALYSIS']
    message: Identifier


class IntegrationResponse(Contract):
    contractVersion: Literal['1.0.0'] = '1.0.0'
    request: IntegrationRequest | None = None
    analysis: UnifiedAnalysis | None = None
    error: IntegrationError | None = None

    @model_validator(mode='after')
    def consistent(self):
        if (self.analysis is None) == (self.error is None):
            raise ValueError('Exactly one analysis or integration error is required')
        if self.analysis is not None:
            if self.request is None or self.analysis.inputType != 'IMAGE':
                raise ValueError('Image analysis requires its original request')
            validate_unified_for_request(self.analysis, self.request.metadata)
            if any(c.category != 'FIRE' for c in self.analysis.candidates):
                raise ValueError('Fire-only interface cannot advertise other categories')
        return self


def failure(code, message, request=None):
    return IntegrationResponse(request=request, error=IntegrationError(code=code, message=message))


class FireIntegrationService:
    """One instance per worker; synchronous, no external providers or side effects.

    Retries may repeat inference. Backend owns idempotent storage and review.
    """
    def __init__(self, analyzer=None):
        self.analyzer = analyzer if analyzer is not None else FireAnalyzer()

    def analyze(self, raw_request, image_bytes):
        try:
            request = IntegrationRequest.model_validate(raw_request)
        except (ValueError, TypeError):
            return failure('INVALID_REQUEST', 'Request does not match integration contract 1.0.0.')
        if not isinstance(image_bytes, bytes):
            return failure('INVALID_MEDIA', 'Image media must be bytes.', request)
        try:
            # Copy metadata: an injected analyzer must not mutate the returned request.
            result = self.analyzer.analyze(request.metadata.model_copy(deep=True), image_bytes)
        except Exception:
            return failure('INTERNAL_ERROR', 'Image analysis did not complete.', request)
        try:
            analysis = from_fire_analysis(result)
            return IntegrationResponse(request=request, analysis=analysis)
        except Exception:
            return failure('INVALID_ANALYSIS', 'Analysis failed the integration contract.', request)
