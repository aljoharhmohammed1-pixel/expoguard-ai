"""Offline entrypoint for v2. Vision/speech providers must be wired via Python API."""
import argparse
import sys
from pathlib import Path
from ai.core.analysis import ImageRequest
from ai.core.multimodal import TextRequest, AudioRequest, validate_unified_for_request
from ai.text.classifier import ArabicTextClassifier
from ai.vision.general import GeneralVisionAnalyzer
from ai.audio.speech import SpeechAnalyzer


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['text','image','audio'], required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--media', type=Path)
    args = parser.parse_args()
    try:
        raw = args.request.read_text(encoding='utf-8-sig')
        if args.kind == 'text':
            request = TextRequest.model_validate_json(raw)
            result = ArabicTextClassifier().analyze(request)
        else:
            if args.media is None:
                parser.error('--media is required for image/audio')
            data = args.media.read_bytes()
            if args.kind == 'image':
                request = ImageRequest.model_validate_json(raw)
                result = GeneralVisionAnalyzer().analyze(request, data)
            else:
                request = AudioRequest.model_validate_json(raw)
                result = SpeechAnalyzer().analyze(request, data)
        result = validate_unified_for_request(result, request)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(result.model_dump_json(indent=2))
    return 0 if result.analysisStatus == 'OK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
