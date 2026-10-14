from typing import Protocol


def validated_provider_version(provider, unconfigured):
    """Snapshot configuration before any provider call or error-envelope creation."""
    if provider is None:
        return unconfigured
    version = getattr(provider, 'model_version', None)
    if not isinstance(version, str) or not version.strip():
        raise ValueError('Provider model_version must be a non-empty string')
    return version


class VisionProvider(Protocol):
    model_version: str

    def analyze(self, *, image_bytes: bytes, instructions: str, output_schema: dict) -> dict:
        """Return the structured VisionResponse, never operational actions."""
        ...


class SpeechProvider(Protocol):
    model_version: str

    def transcribe(self, *, audio_bytes: bytes, mime_type: str, language: str) -> dict:
        """Return Transcript: text, language, nullable transcriptionConfidence."""
        ...


class TextProvider(Protocol):
    model_version: str

    def classify(self, *, text: str, instructions: str, output_schema: dict) -> dict:
        """Return structured TextResponse with verbatim source evidence and offsets."""
        ...
