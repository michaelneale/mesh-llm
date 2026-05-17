#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

ALLOWED_STATUSES = {
    "PASS",
    "EXPECTED_REJECTED_PASS",
    "BLOCKED_STAGED_LOCAL",
    "BLOCKED_NETWORK_LOCAL",
    "BLOCKED_ASSET_LOCAL",
    "BLOCKED_EVIDENCE_GAP",
}
PLACEHOLDER_RE = re.compile(r"<[^>]+>")
PORTABLE_ENV_RE = re.compile(r"\$MESH_LLM_SMOKE_(MODEL|DRAFT)_PATH")
FORBIDDEN_TRACKED_RE = re.compile(
    r"(^/Users/|^/home/|/\.cache/huggingface/|models--[^/]+/snapshots/|/var/folders/|/private/var/folders/)",
    re.IGNORECASE,
)


def extract_keys(config_path: Path):
    rows = []
    for line in config_path.read_text().splitlines():
        if not line.startswith('|') or 'Config key path' in line or line.startswith('|---'):
            continue
        cells = [c.strip() for c in line.strip().split('|')[1:-1]]
        if len(cells) < 3:
            continue
        key_cell = cells[2]
        if key_cell == '—' or key_cell.startswith('see ') or key_cell.startswith('shared '):
            continue
        for raw in re.findall(r'`([^`]+)`', key_cell):
            for part in raw.split('<br>'):
                part = part.strip()
                if part:
                    rows.append(part)
    return rows


def is_non_empty_file(path: Path) -> bool:
    return path.exists() and bool(path.read_text().strip())


def has_forbidden_tracked_path(value: str) -> bool:
    return bool(FORBIDDEN_TRACKED_RE.search(value))


def main() -> int:
    parser = argparse.ArgumentParser(description='Check Task 11 manual smoke manifest coverage.')
    parser.add_argument('--matrix', default='docs/skippy/CONFIGURATION.md')
    parser.add_argument('--manifest', default='docs/skippy/manual-smoke/manifest.tsv')
    args = parser.parse_args()

    matrix_keys = extract_keys(Path(args.matrix))
    with Path(args.manifest).open(newline='') as fh:
        rows = list(csv.DictReader(fh, delimiter='	'))

    manifest_keys = [row['key_path'] for row in rows]
    missing = [key for key in matrix_keys if key not in manifest_keys]
    duplicates = sorted({key for key in manifest_keys if manifest_keys.count(key) > 1})
    extras = [key for key in manifest_keys if key not in matrix_keys]
    failures = []

    for row in rows:
        key = row['key_path']
        status = row['pass_fail_status'].strip()
        if status not in ALLOWED_STATUSES:
            failures.append(f'invalid status for {key}: {status or "<blank>"}')
        fixture_path = row['fixture_path'].strip()
        if not fixture_path or not Path(fixture_path).exists():
            failures.append(f'missing fixture for {key}: {row["fixture_path"]}')
        for field in ('fixture_path', 'startup_apply_command', 'model_identifier', 'verification_command', 'expected_result', 'actual_evidence_path'):
            value = row[field].strip()
            if not value:
                failures.append(f'blank {field} for {key}')
                continue
            if PLACEHOLDER_RE.search(value):
                failures.append(f'placeholder value in {field} for {key}: {value}')
            if field != 'actual_evidence_path' and has_forbidden_tracked_path(value):
                failures.append(f'machine-local path in {field} for {key}: {value}')
        evidence_path = row['actual_evidence_path'].strip()
        if evidence_path:
            p = Path(evidence_path)
            if not is_non_empty_file(p):
                failures.append(f'empty or missing evidence file for {key}: {evidence_path}')
        command = row['startup_apply_command'].strip()
        model = row['model_identifier'].strip()
        expected = row['expected_result'].strip()
        if status in {'PASS', 'EXPECTED_REJECTED_PASS'}:
            if 'not-run locally' in command.lower():
                failures.append(f'non-executable command marked successful for {key}')
        if status == 'PASS':
            if '--model-path $MESH_LLM_SMOKE_MODEL_PATH' not in command:
                failures.append(f'portable model env var missing for PASS row {key}')
        if key.startswith('speculative.') and status == 'PASS':
            if '--draft-path $MESH_LLM_SMOKE_DRAFT_PATH' not in command:
                failures.append(f'portable draft env var missing for speculative PASS row {key}')
        if status.startswith('BLOCKED_') and not expected:
            failures.append(f'blocked row missing blocker explanation for {key}')
        if PORTABLE_ENV_RE.search(model) and status != 'PASS':
            failures.append(f'env var leaked into non-PASS model identifier for {key}')

    evidence_files = {
        Path('.sisyphus/evidence/task-11-manual-runtime-smoke.txt'),
        Path('.sisyphus/evidence/task-11-manual-runtime-smoke-error.txt'),
    }
    for evidence in evidence_files:
        if not is_non_empty_file(evidence):
            failures.append(f'evidence file empty: {evidence}')
        else:
            content = evidence.read_text()
            if 'COMMAND:' not in content and 'Coverage check' not in content:
                failures.append(f'evidence file missing command/result markers: {evidence}')

    if missing or duplicates or extras or failures:
        if missing:
            print('Missing key paths:')
            for key in missing:
                print(key)
        if duplicates:
            print('Duplicate manifest keys:')
            for key in duplicates:
                print(key)
        if extras:
            print('Extra manifest keys:')
            for key in extras:
                print(key)
        if failures:
            print('Manifest/evidence failures:')
            for failure in failures:
                print(failure)
        return 1

    print(f'Coverage OK: {len(matrix_keys)} matrix keys, {len(rows)} manifest rows')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
