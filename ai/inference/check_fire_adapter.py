"""Local six-image regression check, not an independent accuracy benchmark."""
import hashlib
import json
from pathlib import Path
from ai.core.analysis import ImageRequest
from ai.inference.fire_adapter import FireAnalyzer, MODEL_PATH, MODEL_VERSION, POLICY_VERSION


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'inference/runs/fire_adapter_check'
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((root / 'inference/runs/fire_smoke_samples/report.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    analyzer = FireAnalyzer()
    records = []
    labels = {'fire': 'visible_flame', 'smoke': 'suspected_smoke'}
    for row in baseline['results']:
        request = ImageRequest(requestId='regression-' + row['file'], imageId=row['file'],
                               source='VISITOR_APP', capturedAt='2026-10-13T00:00:00Z',
                               context={'metadataIsSynthetic': True})
        analysis = analyzer.analyze(request, (root / 'datasets/fire_smoke_samples' / row['file']).read_bytes())
        previous = sorted((labels[d['class']], d['confidence']) for d in row['detections'])
        current = sorted((d.label, d.confidence) for d in analysis.detections)
        matches = len(previous) == len(current) and all(
            a[0] == b[0] and abs(a[1] - b[1]) <= .0001 for a, b in zip(previous, current))
        records.append(dict(file=row['file'], status=analysis.analysisStatus,
                            detections=len(current), matchesBaseline=matches))
        (output / (Path(row['file']).stem + '.json')).write_text(analysis.model_dump_json(indent=2), encoding='utf-8')
    unchanged = digest == hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    passed = unchanged and all(r['status'] == 'OK' and r['matchesBaseline'] for r in records)
    report = dict(modelVersion=MODEL_VERSION, policyVersion=POLICY_VERSION, weightsSha256=digest,
                  weightsUnchanged=unchanged, passed=passed, results=records,
                  note='Local regression only; metadata is synthetic, not capture provenance. No retraining or downloads.')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
