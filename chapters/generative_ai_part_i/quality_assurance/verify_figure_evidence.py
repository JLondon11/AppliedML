"""Fail-closed evidence gate for chapter figure claims (Python standard library only).

Usage: python verify_figure_evidence.py evidence.json [--root DIR]
Exit 0 only if every listed item has verifiable, matching evidence.
This does NOT certify empirical validity, which requires scientific review.
"""
import argparse
import hashlib
import json
from pathlib import Path


def verify(manifest, root):
    issues = []
    if not isinstance(manifest, dict) or not isinstance(manifest.get('figures'), list):
        return ['Manifest must contain a figures array']
    if not manifest['figures']:
        return ['Manifest has no figures']
    seen = set()
    for row in manifest['figures']:
        if not isinstance(row, dict):
            issues.append('Figure record is not an object')
            continue
        fid = row.get('figure_id', '<missing>')
        if fid in seen:
            issues.append(f'{fid}: duplicate figure ID')
        seen.add(fid)
        for field in ('figure_id', 'manuscript_domain', 'evidence_domain',
                      'dataset', 'source_code', 'output', 'output_sha256'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                issues.append(f'{fid}: missing {field}')
        if row.get('manuscript_domain') != row.get('evidence_domain'):
            issues.append(f'{fid}: DOMAIN MISMATCH: manuscript={row.get("manuscript_domain")!r}; evidence={row.get("evidence_domain")!r}')
        for field in ('source_code', 'output'):
            rel = row.get(field)
            if not isinstance(rel, str) or not rel:
                continue
            path = (root / rel).resolve()
            if not path.is_relative_to(root.resolve()):
                issues.append(f'{fid}: {field} escapes repository root')
            elif not path.is_file():
                issues.append(f'{fid}: missing {field}: {rel}')
            elif field == 'output':
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row.get('output_sha256'):
                    issues.append(f'{fid}: output SHA-256 mismatch')
        if row.get('experiment_executed') is not True:
            issues.append(f'{fid}: executed experiment not attested')
        if row.get('dataset_provenance_verified') is not True:
            issues.append(f'{fid}: dataset provenance not verified')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    issues = verify(json.loads(args.manifest.read_text(encoding='utf-8')), args.root)
    for issue in issues:
        print('FAIL:', issue)
    if issues:
        print(f'BLOCKED: {len(issues)} issue(s)')
        return 1
    print('Evidence file and metadata checks passed; independent scientific review remains required.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
