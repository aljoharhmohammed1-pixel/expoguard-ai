"""python -m ai.evaluation.benchmark --manifest PATH --output NEW_DIRECTORY"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Literal
from pydantic import AwareDatetime, Field, StrictBool, model_validator
from ai.core.analysis import Contract, Identifier, ImageRequest
from ai.inference.fire_adapter import FireAnalyzer, MODEL_PATH, MODEL_VERSION, POLICY_VERSION
from ai.evaluation.metrics import summarize


class Sample(Contract):
    imageId: Identifier
    path: Identifier
    scenario: Identifier = 'unassigned'
    sourceHint: str | None = None
    reviewStatus: Literal['PENDING', 'REVIEWED'] = 'PENDING'
    expectedFire: StrictBool | None = None
    reviewer: Identifier | None = None
    reviewedAt: AwareDatetime | None = None
    reviewNotes: str = ''

    @model_validator(mode='after')
    def valid_review(self):
        if self.reviewStatus == 'REVIEWED':
            if self.expectedFire is None or self.reviewer is None or self.reviewedAt is None:
                raise ValueError('Reviewed labels require expectedFire, reviewer and reviewedAt')
        elif self.expectedFire is not None or self.reviewer is not None or self.reviewedAt is not None:
            raise ValueError('Pending samples cannot supply accepted labels or reviewer metadata')
        return self


class Dataset(Contract):
    datasetVersion: Identifier
    independence: Literal['UNKNOWN', 'INDEPENDENT', 'TRAINING_OVERLAP'] = 'UNKNOWN'
    provenance: Identifier
    samples: list[Sample] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_samples(self):
        if len({s.imageId for s in self.samples}) != len(self.samples):
            raise ValueError('Duplicate imageId')
        return self


def run(manifest, output):
    manifest = Path(manifest).resolve()
    raw = manifest.read_bytes()
    dataset = Dataset.model_validate_json(raw)
    paths = [(manifest.parent / s.path).resolve() for s in dataset.samples]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate image paths bias evaluation')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    analyzer = FireAnalyzer()
    rows, queue = [], []
    started = datetime.now(timezone.utc)
    for index, (sample, path) in enumerate(zip(dataset.samples, paths)):
        request = ImageRequest(requestId=f'benchmark-{index+1}', imageId=sample.imageId,
            source='VISITOR_APP', capturedAt=started, context={'metadataIsSynthetic': True})
        t0 = time.perf_counter()
        try:
            data = path.read_bytes()
        except OSError:
            analysis = None
            status, error = 'ERROR', 'INPUT_READ_FAILED'
            digest = None
        else:
            analysis = analyzer.analyze(request, data)
            status = analysis.analysisStatus
            error = analysis.error.code if analysis.error else None
            digest = hashlib.sha256(data).hexdigest()
        elapsed = (time.perf_counter()-t0)*1000
        predicted = bool(analysis.candidates) if analysis is not None and status == 'OK' else None
        row = dict(imageId=sample.imageId, path=str(path), scenario=sample.scenario,
            expectedFire=sample.expectedFire, predictedFire=predicted, status=status,
            error=error, latencyMs=elapsed, imageSha256=digest)
        rows.append(row)
        if analysis is not None:
            (output/f'analysis-{index+1}.json').write_text(analysis.model_dump_json(indent=2), encoding='utf-8')
        review = sample.model_dump(mode='json')
        review['path'] = str(path)
        queue.append(review)
    summary = summarize(rows)
    report = dict(createdAt=started.isoformat(), datasetVersion=dataset.datasetVersion,
        independence=dataset.independence, manifestSha256=hashlib.sha256(raw).hexdigest(),
        modelVersion=MODEL_VERSION, policyVersion=POLICY_VERSION,
        weightsSha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() if MODEL_PATH.exists() else None,
        environment=dict(python=platform.python_version(), platform=platform.platform(), device='cpu'),
        summary=summary, scenarios={s:summarize([r for r in rows if r['scenario']==s]) for s in sorted({r['scenario'] for r in rows})},
        coldFirstRequestMs=rows[0]['latencyMs'], subsequentRequestsMs=summarize(rows[1:])['latencyMs'],
        rows=rows, limitations=[
            'Image-level FIRE candidate evaluation at confidence threshold 0.35; not detector mAP or incident-level evaluation.',
            'Only manually reviewed labels count; UNKNOWN independence is not an independent accuracy benchmark.',
            'ERROR and UNUSABLE abstain; conditional metrics exclude them and end-to-end recall counts positive abstentions as misses.',
            'mAP requires independent object-level box annotations; camera-hour rates require continuous observation and temporal alert policy.',
            'Latency includes file reading, image validation, inference, result validation and image hashing, but not report writes; first request includes lazy model loading.',
            'Synthetic request capture metadata; no alarms sent. Human review remains required.'])
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    review_dataset = dataset.model_dump(mode='json')
    review_dataset['samples'] = queue
    (output/'human-review.json').write_text(json.dumps(review_dataset, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.output)


if __name__ == '__main__':
    main()
