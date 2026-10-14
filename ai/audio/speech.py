from ai.core.multimodal import AudioRequest, TextRequest, Transcript, UnifiedAnalysis, UnifiedError, validate_unified_for_request
from ai.providers.interfaces import SpeechProvider, validated_provider_version
from ai.text.classifier import ArabicTextClassifier


class SpeechAnalyzer:
    def __init__(self, provider: SpeechProvider | None = None, classifier=None):
        self.model_version = validated_provider_version(provider, 'speech-provider-unconfigured')
        self.provider = provider
        self.classifier = classifier if classifier is not None else ArabicTextClassifier()

    def analyze(self, request, audio_bytes):
        request = AudioRequest.model_validate(request)
        if not isinstance(audio_bytes, bytes):
            raise TypeError('audio_bytes must be bytes')
        version = self.model_version
        base = dict(requestId=request.requestId, inputType='AUDIO', audioId=request.audioId,
                    modelVersion=version, policyVersion='speech-text-review-v0.1.0')
        def fail(code, message):
            return UnifiedAnalysis(**base, analysisStatus='ERROR', error=UnifiedError(code=code,message=message))
        if not audio_bytes or len(audio_bytes) > 25*1024*1024:
            return UnifiedAnalysis(**base, analysisStatus='UNUSABLE', limitations=['Audio is empty or exceeds 25 MiB.'])
        if self.provider is None:
            return fail('PROVIDER_NOT_CONFIGURED', 'Speech provider is not configured.')
        try:
            raw = self.provider.transcribe(audio_bytes=audio_bytes, mime_type=request.mimeType, language=request.language)
        except Exception:
            return fail('TRANSCRIPTION_FAILED', 'Speech transcription failed.')
        try:
            transcript = Transcript.model_validate(raw)
        except Exception:
            return fail('INVALID_TRANSCRIPT', 'Transcription output failed validation.')
        if not transcript.text.strip():
            return UnifiedAnalysis(**base, analysisStatus='UNUSABLE', transcript=transcript,
                                   limitations=['No usable speech transcript.'])
        try:
            classified = self.classifier.analyze(TextRequest(requestId=request.requestId, textId=request.audioId,
                text=transcript.text, source=request.source, zoneId=request.zoneId, assetId=request.assetId))
        except Exception:
            return fail('CLASSIFICATION_FAILED', 'Transcript classification failed.')
        try:
            if classified.requestId != request.requestId or classified.textId != request.audioId:
                raise ValueError('Classifier returned a different request')
            result = UnifiedAnalysis(**base, analysisStatus=classified.analysisStatus, transcript=transcript,
                detections=classified.detections, candidates=classified.candidates, error=classified.error,
                componentVersions={'speech':version, 'text':classified.modelVersion, 'textPolicy':classified.policyVersion},
                limitations=classified.limitations + ['Transcription may contain errors; review the audio before operational decisions.'])
            return validate_unified_for_request(result, request)
        except Exception:
            return fail('INVALID_CLASSIFICATION', 'Transcript classification failed contract validation.')
