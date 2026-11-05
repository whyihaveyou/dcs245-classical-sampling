"""Meaningful corruption tests against a clearly synthetic real-worker fixture.

No GPU use or quantum result. --fixture must contain SYNTHETIC_ONLY.json.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from verify import sha256, verify_run


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def mutate_jsonl(path, mutator):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    mutator(rows)
    Path(path).write_text(''.join(json.dumps(row)+'\n' for row in rows))


def repair_record_hashes(directory):
    """Update bookkeeping to ensure semantic checks, not only hashes, catch edits."""
    summary = read(directory/'summary.json')
    summary['samples_sha256'] = sha256(directory/'samples.jsonl')
    summary['audits_sha256'] = sha256(directory/'audits.jsonl')
    write(directory/'summary.json', summary)
    artifacts = read(directory/'artifacts.json')
    for name in artifacts['files']:
        path = directory/name
        artifacts['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    write(directory/'artifacts.json', artifacts)
    marker = read(directory/'completed.json')
    marker['summary_sha256'] = sha256(directory/'summary.json')
    marker['artifacts_sha256'] = sha256(directory/'artifacts.json')
    write(directory/'completed.json', marker)


def run(fixture):
    fixture = Path(fixture).resolve()
    declaration = read(fixture/'SYNTHETIC_ONLY.json')
    assert declaration['synthetic_only'] and declaration['GPU_experiments_run'] == 0
    bundle, original_artifacts = fixture/'bundle', fixture/'artifacts'
    baseline = verify_run(bundle, original_artifacts, synthetic=True)
    assert baseline['record_and_full_vector_validation_passed'], baseline['failures']
    assert not baseline['timing_validation_passed']
    assert {x['code'] for x in baseline['failures']} == {'central_timing_ledger_missing'}
    workers = read(bundle/'protocol.json')['worker_count']
    ledger = {'synthetic_only': True, 'scope': 'Invented test timing; no measured GPU experiment.',
              'terminal': 'Succeeded', 'log_return_code': 0, 'declared_gpu_limit': workers,
              'worker_count': workers, 'gpus_per_worker': 1, 'max_observed_running_workers': 1,
              'central_monotonic_submit_to_terminal_observed_seconds': 2.,
              'central_monotonic_submit_to_all_logs_returned_seconds': 3.,
              'central_monotonic_submit_to_all_artifacts_verified_seconds': 4.}
    tests = []
    with tempfile.TemporaryDirectory(prefix='dcs245_verify_test_') as temporary:
        root = Path(temporary)
        ledger_path = root/'synthetic_ledger.json'
        write(ledger_path, ledger)
        good = verify_run(bundle, original_artifacts, ledger_path, synthetic=True)
        assert good['validation_passed'], good['failures']
        tests.append({'case': 'valid_synthetic_complete_fixture', 'passed': True})

        def case(name, mutate, expected_code):
            artifacts = root/name
            shutil.copytree(original_artifacts, artifacts)
            dirs = sorted(p.parent for p in artifacts.rglob('source.json'))
            mutate(dirs[0])
            result = verify_run(bundle, artifacts, ledger_path, synthetic=True)
            codes = {x['code'] for x in result['failures']}
            assert not result['validation_passed'] and expected_code in codes, (name, result['failures'])
            tests.append({'case': name, 'passed': True, 'detected_failure_codes': sorted(codes)})

        def change_rng(directory):
            def change(rows):
                rows[0]['draw_uniform'] = (rows[0]['draw_uniform'] + .25) % 1
            mutate_jsonl(directory/'samples.jsonl', change)
            repair_record_hashes(directory)
        case('changed_rng_draw_with_repaired_hashes', change_rng, 'sample_rng_draw_uniform')

        def change_cdf(directory):
            def change(rows):
                rows[0]['draw']['cdf_lower_fsum'] = rows[0]['draw_uniform'] + .01
            mutate_jsonl(directory/'samples.jsonl', change)
            repair_record_hashes(directory)
        case('invalid_CDF_interval_with_repaired_hashes', change_cdf, 'cdf_interval_fsum')

        def change_witness(directory):
            def change(rows):
                rows[0]['witness']['uncapped_overlap_witness'] += .2
            mutate_jsonl(directory/'audits.jsonl', change)
            repair_record_hashes(directory)
        case('forged_quality_witness_with_repaired_hashes', change_witness, 'witness_uncapped_overlap_witness')

        def remove_audit(directory):
            mutate_jsonl(directory/'audits.jsonl', lambda rows: rows.pop())
            repair_record_hashes(directory)
        case('missing_preselected_audit_with_repaired_hashes', remove_audit, 'all_audit_ids_exactly_once')

        def change_anchor(directory):
            anchor = read(directory/'anchor.json')
            anchor['actual_amplitudes_sha256'] = '0'*64
            write(directory/'anchor.json', anchor)
            repair_record_hashes(directory)
        case('different_worker_anchor_with_repaired_hashes', change_anchor, 'worker_anchor_vector_hash')

        def change_vector(directory):
            path = sorted(directory.glob('approx_*.npy'))[0]
            blob = bytearray(path.read_bytes())
            blob[-1] ^= 1
            path.write_bytes(blob)
        case('changed_archived_vector_bytes', change_vector, 'vector_file_hash')
        invalid_ledger = dict(ledger, declared_gpu_limit=16)
        write(ledger_path, invalid_ledger)
        result = verify_run(bundle, original_artifacts, ledger_path, synthetic=True)
        assert not result['timing_validation_passed'] and any(x['code'] == 'central_gpu_layout' for x in result['failures'])
        tests.append({'case': 'incorrect_GPU_limit', 'passed': True})
    return {'scope': 'SYNTHETIC CORRUPTION DETECTION TESTS ONLY; ZERO GPU EXPERIMENTS',
            'passed': True, 'fixture': str(fixture), 'fixture_samples': baseline['sample_count'],
            'fixture_audits': baseline['audit_count'], 'tests': tests}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.fixture)
    write(args.out, result)
    print(json.dumps(result, indent=2))
