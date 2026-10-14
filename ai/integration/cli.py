"""JSON request file + local image -> versioned JSON response on stdout."""
import argparse
import sys
from pathlib import Path
from ai.integration.service import FireIntegrationService, IntegrationRequest, failure
from ai.quality.image_quality import MAX_BYTES

MAX_REQUEST_BYTES = 64 * 1024


def read_bounded(path, limit):
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Input exceeds size limit')
    return data


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--media', type=Path, required=True)
    args = parser.parse_args()
    try:
        request = IntegrationRequest.model_validate_json(read_bounded(args.request, MAX_REQUEST_BYTES).decode('utf-8-sig'))
    except (ValueError, OSError):
        response = failure('INVALID_REQUEST', 'Cannot read a valid request within 64 KiB.')
    else:
        try:
            media = read_bounded(args.media, MAX_BYTES)
        except ValueError:
            response = failure('INVALID_MEDIA', 'Image exceeds 20 MiB.', request)
        except OSError:
            response = failure('MEDIA_READ_FAILED', 'Cannot read image media.', request)
        else:
            response = FireIntegrationService().analyze(request, media)
    print(response.model_dump_json(indent=2))
    if response.error is not None:
        return 2
    return 0 if response.analysis.analysisStatus == 'OK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
