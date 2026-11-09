"""Validate scientific figure provenance manifests without third-party dependencies.

Usage: python validate_provenance.py PATH [PATH ...]
Exit status is nonzero if any manifest violates the evidence contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CATEGORIES = frozenset({'empirical', 'simulation', 'illustration', 'architecture'})
STATUSES = frozenset({'unverified', 'verified', 'certified'})
REQUIRED = {'figure', 'category', 'status', 'generator', 'dataset', 'protocol', 'artifact'}


def validate_record(record: object, index: int) -> list[str]:
    prefix = f'figures[{index}]'
    if not isinstance(record, dict):
        return [f'{prefix}: expected an object']
    errors = []
    missing = REQUIRED - record.keys()
    if missing:
        errors.append(f'{prefix}: missing fields: {", ".join(sorted(missing))}')
    for field in REQUIRED & record.keys():
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            errors.append(f'{prefix}.{field}: expected a nonempty string')
    if record.get('category') not in CATEGORIES:
        errors.append(f'{prefix}.category: unsupported category')
    if record.get('status') not in STATUSES:
        errors.append(f'{prefix}.status: unsupported status')
    if record.get('category') in {'illustration', 'architecture'} and record.get('status') == 'certified':
        errors.append(f'{prefix}: conceptual figures cannot receive empirical certification')
    if record.get('status') == 'certified':
        for field in ('dataset', 'protocol', 'metrics_file', 'test_command', 'test_result'):
            if not isinstance(record.get(field), str) or not record[field].strip() or record[field] == 'not applicable':
                errors.append(f'{prefix}.{field}: certification requires recorded evidence')
    return errors


def validate_manifest(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f'{path}: cannot read JSON: {exc}']
    if not isinstance(data, dict) or not isinstance(data.get('figures'), list):
        return [f'{path}: top-level figures must be an array']
    if not data['figures']:
        return [f'{path}: figures array cannot be empty']
    errors = []
    seen = set()
    for index, record in enumerate(data['figures']):
        errors.extend(validate_record(record, index))
        if isinstance(record, dict) and isinstance(record.get('figure'), str):
            if record['figure'] in seen:
                errors.append(f'{path}: duplicate figure: {record["figure"]}')
            seen.add(record['figure'])
    return [f'{path}: {error}' for error in errors]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', type=Path, nargs='+')
    parser.add_argument('--require-real-data', action='store_true',
                        help='reject conceptual-only or simulation-only figure records')
    args = parser.parse_args(argv)
    errors = [error for path in args.manifests for error in validate_manifest(path)]
    if args.require_real_data:
        for path in args.manifests:
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                for i, record in enumerate(data.get('figures', [])):
                    if not isinstance(record, dict) or record.get('category') != 'empirical':
                        errors.append(f'{path}: figures[{i}] is not classified as real-data empirical evidence')
            except (OSError, UnicodeError, ValueError, AttributeError):
                pass  # validate_manifest already records malformed manifests
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f'Validated {len(args.manifests)} provenance manifest(s).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
