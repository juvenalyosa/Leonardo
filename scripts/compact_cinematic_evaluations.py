#!/usr/bin/env python3
"""Losslessly compact completed, disposable per-evaluation working files.

Every original file is retained inside its evaluation's ZIP; CRC validation
precedes removal of the uncompressed duplicate. result.json stays searchable.
"""

import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'artifacts/cinematic_v2/qmmm/evaluations'
folders = sorted(root.glob('e*'))[:-100]
count = 0
for folder in folders:
    if (folder / 'source.zip').exists() or not (folder / 'result.json').exists():
        continue
    try:
        json.loads((folder / 'result.json').read_text())
    except (ValueError, OSError):
        continue
    files = [p for p in folder.iterdir() if p.is_file() and p.name != 'result.json']
    temporary = folder / 'source.tmp'
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError(f'CRC failure: {folder}')
        for path in files:
            if archive.read(path.name) != path.read_bytes():
                raise RuntimeError(f'Archive mismatch: {path}')
    temporary.replace(folder / 'source.zip')
    for path in files:
        path.unlink()
    count += 1
print('Losslessly compacted', count, 'completed evaluations.')
