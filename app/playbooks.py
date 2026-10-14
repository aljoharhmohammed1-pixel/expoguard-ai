"""Load only host-managed catalogs; API callers cannot submit approvals."""
import json
from pathlib import Path
from ai.operations.recommendations import Playbook


def load_catalog(directory):
    books = []
    try:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError('Catalog directory missing')
        paths = sorted(directory.glob('*.json'))
        if len(paths) > 100:
            raise ValueError('Catalog too large')
        for path in paths:
            with path.open('rb') as stream:
                raw = stream.read(256*1024+1)
            if len(raw) > 256*1024:
                raise ValueError('Playbook too large')
            books.append(Playbook.model_validate(json.loads(raw)))
        if len({(b.playbookId,b.version) for b in books}) != len(books):
            raise ValueError('Duplicate playbook identity')
        return books
    except Exception:
        raise RuntimeError('Playbook catalog failed validation') from None
