"""Build a local readiness manifest; never certify external integration or production safety."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def source_digest():
    records = []
    for directory, folders, files in os.walk(ROOT):
        folders[:] = sorted(d for d in folders if d not in {'.venv','__pycache__','datasets','runs','models','reports','.git'})
        for name in sorted(files):
            path = Path(directory)/name
            if path.suffix in {'.py','.json'}:
                records.append((path.relative_to(ROOT).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    # HTTP service lives beside ai; its edits must invalidate earlier test reports.
    app_root = ROOT.parent/'app'
    if app_root.is_dir():
        for path in sorted(app_root.rglob('*.py')):
            records.append(('app/'+path.relative_to(app_root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    for name in ['requirements.txt', 'Dockerfile', 'config/cameras.json']:
        path = ROOT.parent/name
        if path.is_file():
            records.append((name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(sorted(records)).encode()).hexdigest()


def build_readiness(test_report):
    digest = source_digest()
    report = json.loads(Path(test_report).read_text(encoding='utf-8'))
    tests_current = report.get('sourceDigest') == digest
    tests_ok = (tests_current and report.get('passed') is True and report.get('integration') is True
                and report.get('skipped') == 0 and report.get('testsRun',0) > 0
                and not report.get('errors') and not report.get('failures'))
    dependencies = {}
    for name in ['pydantic','Pillow','ultralytics','torch','numpy','opencv-python']:
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    model = ROOT/'models/infernoguard_best.pt'
    return dict(createdAt=datetime.now(timezone.utc).isoformat(), sourceDigest=digest,
        status='READY_FOR_BACKEND_REVIEW' if tests_ok and all(dependencies.values()) and model.is_file() else 'LOCAL_CHECKS_INCOMPLETE',
        productionReady=False, localIntegrationTestsCurrent=tests_current,
        localIntegrationTestsPassed=tests_ok, testReport=str(Path(test_report).resolve()),
        testCount=report.get('testsRun'), dependencies=dependencies,
        fireWeightsSha256=hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else None,
        schemas=sorted(p.name for p in (ROOT/'contracts').glob('*.schema.json')),
        externalAcceptanceRequired=[
            'Actual backend API/enums/auth/idempotency and version negotiation are not tested here.',
            'Vision/text/speech provider ports have no configured remote clients; new-provider quality is unmeasured.',
            'No approved operational playbooks were supplied; recommendation input must come from an authenticated approved catalogue.',
            'Independent reviewed data, detection metrics and operational thresholds remain pending.',
            'Fresh environment installation and deployment/load tests have not been verified.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_readiness(args.test_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'])
    return 0 if result['status'] == 'READY_FOR_BACKEND_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
