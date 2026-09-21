#!/usr/bin/env python3
from pathlib import Path
import hashlib, sys

EXTS={'.png','.jpg','.jpeg','.webp','.svg','.pdf'}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):
            h.update(chunk)
    return h.hexdigest()

root=Path('chapters')
groups={}
for p in root.rglob('*'):
    if p.is_file() and p.suffix.lower() in EXTS and '/figures/' in p.as_posix():
        parts=p.parts
        try:
            i=parts.index('chapters')
            chapter=parts[i+1]
        except Exception:
            chapter='unknown'
        groups.setdefault(chapter,{}).setdefault(digest(p),[]).append(p)

bad=0
for chapter, hashes in sorted(groups.items()):
    for files in hashes.values():
        if len(files)>1:
            bad+=1
            print(f'DUPLICATE [{chapter}]')
            for f in files: print('  ',f)
print(f'exact_duplicate_groups={bad}')
sys.exit(1 if bad else 0)
