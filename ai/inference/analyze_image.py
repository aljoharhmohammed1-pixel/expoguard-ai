"""Run from project root: python -m ai.inference.analyze_image --help."""
import argparse
from pathlib import Path
from ai.core.analysis import ImageRequest
from ai.inference.fire_adapter import FireAnalyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    try:
        request = ImageRequest.model_validate_json(args.metadata.read_text(encoding='utf-8-sig'))
        data = args.image.read_bytes()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    result = FireAnalyzer().analyze(request, data)
    print(result.model_dump_json(indent=2))
    return 0 if result.analysisStatus == 'OK' else 1


if __name__ == '__main__':
    raise SystemExit(main())
