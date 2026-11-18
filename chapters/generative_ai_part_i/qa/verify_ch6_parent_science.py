"""Fail-closed Chapter 6 parent-application scientific release audit.

Narrow Level A/B demonstration registers are not parent approvals. A parent study
can be certified only against its exact manuscript title and original task data.
This checks source/evidence relationships and documented R1-R10 status;
it cannot substitute for an independent scientific review of experiments.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

CHAPTER = Path('chapters/generative_ai_part_i')
FUNDAMENTAL = {'R1', 'R3', 'R4', 'R8', 'R9', 'R10'}
VALID = {'PASS', 'N/A — JUSTIFIED'}


def _real_file(root: Path, relative) -> bool:
    if not isinstance(relative, str) or not relative.strip():
        return False
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        return False
    path = (root / rel).resolve()
    return path.is_relative_to(root.resolve()) and path.is_file()


def audit(root: Path, inventory: Path | None = None) -> dict:
    root = root.resolve()
    path = inventory or root / CHAPTER / 'quality_assurance/production_inventory.json'
    if not path.is_file():
        return {'status': 'BLOCKED', 'certified_parent_applications': 0,
                'issues': ['No current manuscript-derived parent application inventory']}
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        return {'status': 'BLOCKED', 'certified_parent_applications': 0,
                'issues': [f'Parent application inventory unreadable: {exc}']}
    if not isinstance(document, dict):
        return {'status': 'BLOCKED', 'certified_parent_applications': 0,
                'issues': ['Parent inventory must be JSON object']}
    parents = document.get('parent_applications')
    figs = document.get('figures')
    if not isinstance(parents, list) or not isinstance(figs, list):
        return {'status': 'BLOCKED', 'certified_parent_applications': 0,
                'issues': ['Missing genuine parent or figure inventory']}
    issues = []
    if document.get('compiled_source_reviewed') is not True:
        issues.append('Current manuscript title/claims have not been independently inspected')
    if len(parents) != 14:
        issues.append(f'Expected exactly 14 audited parent studies; found {len(parents)}')
    ids = [p.get('unit_id') for p in parents if isinstance(p, dict)]
    if len(ids) != len(parents) or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x.strip() for x in ids):
        issues.append('Parent application identities are missing or duplicated')
    experimental = {f.get('figure_id') for f in figs if isinstance(f, dict) and f.get('kind') == 'experimental'
                    and f.get('scientific_reviewed') is True and f.get('dataset_provenance_verified') is True}
    certified = 0
    for i, unit in enumerate(parents):
        if not isinstance(unit, dict):
            issues.append(f'Parent #{i+1}: malformed record')
            continue
        uid = unit.get('unit_id') or f'parent #{i+1}'
        local = []
        if not unit.get('exact_manuscript_title') or unit.get('scope_reviewed') is not True or unit.get('title_claim_match_verified') is not True:
            local.append('actual title and full task claims not independently reconciled')
        if unit.get('scientific_status') != 'SCIENTIFIC_CERTIFIED':
            local.append('parent application lacks a scientific certificate')
        if unit.get('level') not in ('B', 'C'):
            local.append('named empirical parent requires B/C task-matched evidence')
        requirements = unit.get('r1_r10')
        if not isinstance(requirements, dict) or set(requirements) != {f'R{n}' for n in range(1,11)}:
            local.append('incomplete parent R1-R10 assessment')
        else:
            explanations = unit.get('na_justifications') or {}
            for code, value in requirements.items():
                if value not in VALID or (code in FUNDAMENTAL and value != 'PASS'):
                    local.append(f'{code} does not support this parent scope')
                if value == 'N/A — JUSTIFIED' and (not isinstance(explanations, dict) or
                        not isinstance(explanations.get(code), str) or len(explanations[code].strip()) < 20):
                    local.append(f'{code} lacks a substantive N/A justification')
        for key in ('dataset_provenance', 'executed_model_source', 'original_model_outputs',
                    'quantitative_evaluation', 'scientific_review_record'):
            if not _real_file(root, unit.get(key)):
                local.append(f'missing verified source artifact: {key}')
        figure_ids = unit.get('figure_ids')
        if not isinstance(figure_ids, list) or not figure_ids or not set(map(str, figure_ids)).intersection(experimental):
            local.append('no reviewed original-data result figure linked to parent scope')
        if not isinstance(unit.get('code_commit'), str) or len(unit['code_commit']) != 40 or any(c not in '0123456789abcdef' for c in unit['code_commit']):
            local.append('missing exact 40-character source commit SHA')
        if local:
            issues.extend(f'{uid}: {description}' for description in local)
        else:
            certified += 1
    return {'status': 'BLOCKED' if issues else 'PARENT_SCOPE_RECORDS_COMPLETE_INDEPENDENT_REVIEW_REQUIRED',
            'certified_parent_applications': certified, 'expected_parent_applications': 14,
            'issues': issues,
            'caveat': 'Source-linked records are necessary, not sufficient proof of experimental scientific truth'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--inventory', type=Path)
    args = parser.parse_args()
    result = audit(args.repo_root, args.inventory)
    print(json.dumps(result, indent=2))
    return 1 if result['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
