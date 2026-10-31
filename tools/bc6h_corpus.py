#!/usr/bin/env python3
"""Fetch the pinned CC0 HDR corpus and recreate its unscaled RGB32 crops.

Requires Python numpy and opencv-python. Usage: bc6h_corpus.py OUTPUT_DIRECTORY
Image bytes stay outside the repository. Source hashes are verified before use.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import cv2

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
manifest = json.loads((Path(__file__).parent.parent / 'tests/bc6h_manifest.json').read_text())
for entry in manifest:
    path = root / (entry['id'] + '.hdr')
    if not path.exists():
        subprocess.run(['curl', '-LfsS', entry['url'], '-o', str(path)], check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('source hash mismatch: ' + entry['id'])
    rgb = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    x, y, w, h = entry['crop']
    data = rgb[y:y+h, x:x+w].astype('<f4').tobytes()
    if hashlib.sha256(data).hexdigest() != entry['rgb32_sha256']:
        raise ValueError('crop hash mismatch: ' + entry['id'])
    (root / (entry['id'] + '.rgb32')).write_bytes(data)
