"""Offline advisory API. No incident writes, messages or task execution."""
import argparse
from pathlib import Path
import sys
from ai.operations.correlation import CorrelationRequest, correlate
from ai.operations.recommendations import RecommendationRequest, recommend


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['correlate','recommend'])
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.input.read_text(encoding='utf-8-sig')
        if args.operation == 'correlate':
            request = CorrelationRequest.model_validate_json(raw)
            result = correlate(request.left, request.right, request.policy)
        else:
            request = RecommendationRequest.model_validate_json(raw)
            result = recommend(request.facts, request.playbooks, now=request.now)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(result.model_dump_json(indent=2))


if __name__ == '__main__':
    main()
