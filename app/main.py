import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Annotated
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from app.settings import Settings
from app.registry import ModelRegistry, CameraZoneConflict
from ai.core.analysis import Contract, Identifier, ImageRequest
from ai.core.multimodal import UnifiedAnalysis, TextRequest
from ai.text.classifier import ArabicTextClassifier
from ai.operations.correlation import CorrelationRequest
from app.operations import CorrelationResult, correlation_result
from ai.operations.summary import SummaryRequest, SummaryResult, summarize
from ai.operations.recommendations import IncidentFacts, RecommendationResult, recommend
from app.playbooks import load_catalog
from app.examples import crowd_example, failure_example

logger = logging.getLogger('expoguard.api')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False


class APIError(Contract):
    code: Identifier
    message: Identifier


class APIErrorResponse(Contract):
    error: APIError


def error(code, status):
    return JSONResponse(status_code=status, headers={'X-Error-Code':code},
                        content={'error':{'code':code, 'message':'Request could not be completed.'}})


class BodyLimit:
    """Bound the received body before multipart parsing, including chunked uploads."""
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] not in ('POST', 'PUT', 'PATCH'):
            return await self.app(scope, receive, send)
        limit = self.limit if scope['path'] == '/v1/analyze/image' else 256*1024
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            size += len(message.get('body', b''))
            if size > limit:
                return await error('REQUEST_TOO_LARGE', 413)(scope, receive, send)
            chunks.append(message.get('body', b''))
            if not message.get('more_body', False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {'type':'http.request', 'body':b''.join(chunks), 'more_body':False}
        await self.app(scope, replay, send)


def create_app(settings=None, registry=None):
    settings = settings or Settings.from_env()
    registry = registry or ModelRegistry(settings)

    @asynccontextmanager
    async def lifespan(app):
        app.state.playbooks = load_catalog(settings.playbook_dir)
        await run_in_threadpool(registry.start)
        app.state.registry = registry
        try:
            yield
        finally:
            registry.stop()

    app = FastAPI(title='ExpoGuard AI', version=settings.version, lifespan=lifespan,
                  responses={status:{'model':APIErrorResponse} for status in (413,415,422,500)})
    app.add_middleware(BodyLimit, limit=settings.max_image_mb*1024*1024+64*1024)

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    async def validation_error(request, exc):
        return error('INVALID_REQUEST', 422)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error('HTTP_ERROR', exc.status_code)

    @app.middleware('http')
    async def logging_boundary(request, call_next):
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            response = error('INTERNAL_ERROR', 500)
        elapsed = round((perf_counter()-started)*1000, 3)
        response.headers['X-Process-Time-Ms'] = str(elapsed)
        # Never log bodies, query strings, paths, raw exception messages or credentials.
        record = {'event':'ai_request', 'httpStatus':response.status_code, 'totalDurationMs':elapsed,
                  'errorCode':response.headers.get('X-Error-Code')}
        record.update(getattr(request.state, 'analysis_log', {}))
        logger.info(json.dumps(record, ensure_ascii=True))
        return response

    @app.get('/health')
    def health():
        return JSONResponse(status_code=200 if registry.ready else 503, content={
            'status':'ok' if registry.ready else 'unavailable', 'service':'expoguard-ai',
            'version':settings.version, 'device':registry.device, 'modelsReady':registry.ready})

    @app.get('/v1/models')
    def models():
        return {'models':[registry.metadata(), registry.person_metadata()]}

    @app.post('/v1/correlation/analyze', response_model=CorrelationResult)
    def analyze_correlation(payload: CorrelationRequest, request: Request):
        result = correlation_result(payload)
        request.state.analysis_log = {'analysisType':'CORRELATION', 'analysisStatus':'OK',
                                      'policyVersion':result.proposal.policyVersion}
        return result

    @app.post('/v1/summary', response_model=SummaryResult)
    def summary(payload: SummaryRequest, request: Request):
        result = summarize(payload)
        request.state.analysis_log = {'analysisType':'SUMMARY', 'analysisStatus':'OK', 'policyVersion':result.policyVersion}
        return result

    @app.post('/v1/recommend', response_model=RecommendationResult)
    def recommendations(payload: IncidentFacts, request: Request):
        result = recommend(payload, app.state.playbooks, now=datetime.now(timezone.utc))
        request.state.analysis_log = {'analysisType':'RECOMMENDATION', 'analysisStatus':result.status,
                                      'policyVersion':result.policyVersion}
        return result

    @app.post('/v1/analyze/text', response_model=UnifiedAnalysis, response_model_exclude={'crowdMetrics'})
    def analyze_text(payload: TextRequest, request: Request):
        result = ArabicTextClassifier().analyze(payload)
        request.state.analysis_log = {'requestId':payload.requestId[:200], 'analysisType':'TEXT',
            'modelVersion':result.modelVersion, 'analysisStatus':result.analysisStatus,
            'inferenceDurationMs':None, 'errorCode':result.error.code if result.error else None}
        return result

    @app.post('/v1/analyze/image', response_model=UnifiedAnalysis,
              responses={200:{'content':{'application/json':{'example':crowd_example()}}},
                         503:{'model':UnifiedAnalysis,'content':{'application/json':{'example':failure_example()}}}},
              description='Fire/smoke and optional Crowd V1 in one image. cameraId resolves host-configured normalized polygon ROI and capacity. Configured zone mapping is authoritative; conflicts return 422. Missing configuration safely skips crowd. HIGH/CRITICAL produce human-review candidates, never incidents. See README for DEMO-CROWD setup.')
    async def analyze_image(request: Request,
            file: Annotated[UploadFile, File()], requestId: Annotated[str, Form()],
            imageId: Annotated[str, Form()], source: Annotated[str, Form()],
            capturedAt: Annotated[str, Form()], cameraId: Annotated[str | None, Form()] = None,
            zoneId: Annotated[str | None, Form()] = None, assetId: Annotated[str | None, Form()] = None,
            eventId: Annotated[str | None, Form()] = None):
        metadata = ImageRequest(requestId=requestId, imageId=imageId, source=source,
            capturedAt=capturedAt, cameraId=cameraId, zoneId=zoneId, assetId=assetId, eventId=eventId)
        if file.content_type not in {'image/jpeg', 'image/png', 'image/webp', 'image/bmp'}:
            return error('UNSUPPORTED_MEDIA_TYPE', 415)
        media = await file.read(settings.max_image_mb*1024*1024+1)
        if len(media) > settings.max_image_mb*1024*1024:
            return error('IMAGE_TOO_LARGE', 413)
        try:
            result, timings = await run_in_threadpool(registry.analyze, metadata, media)
        except CameraZoneConflict:
            return error('CAMERA_ZONE_CONFLICT', 422)
        request.state.analysis_log = {'requestId':requestId[:200], 'analysisType':'IMAGE',
            'modelVersion':result.modelVersion, 'analysisStatus':result.analysisStatus,
            'inferenceDurationMs':sum(value or 0 for value in timings.values()), **timings,
            'errorCode':result.error.code if result.error else None}
        # Keep the pre-Crowd wire shape when Crowd was skipped or failed.
        payload = result.model_dump(mode='json', exclude={'crowdMetrics'} if result.crowdMetrics is None else set())
        return JSONResponse(status_code=503 if result.analysisStatus == 'ERROR' else 200, content=payload)

    # FastAPI's initial OpenAPI serialization strips nulls inside examples.
    # Restore the complete validated examples: nullable confidence is required.
    original_openapi = app.openapi
    def openapi_with_examples():
        schema = original_openapi()
        responses = schema['paths']['/v1/analyze/image']['post']['responses']
        responses['200']['content']['application/json']['example'] = crowd_example()
        responses['503']['content']['application/json']['example'] = failure_example()
        return schema
    app.openapi = openapi_with_examples
    return app


app = create_app()
