"""Run fast tests by default; --integration also loads the local real weights."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import unittest
from ai.handoff import source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    os.environ['EXPOGUARD_REAL_TESTS'] = '1' if args.integration else '0'
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root/'tests'))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if args.report:
        report = dict(timestamp=datetime.now(timezone.utc).isoformat(), integration=args.integration,
                      sourceDigest=source_digest(),
                      testsRun=result.testsRun, skipped=len(result.skipped),
                      failures=[{'test': str(t), 'detail': detail} for t, detail in result.failures],
                      errors=[{'test': str(t), 'detail': detail} for t, detail in result.errors],
                      seconds=round(time.perf_counter()-started, 3), passed=result.wasSuccessful())
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
