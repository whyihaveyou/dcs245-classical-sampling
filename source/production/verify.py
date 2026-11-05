"""Independent production-record and complete-vector audit verifier.

No GPU use. Replays declared RNG/CDF/mapping records and, where archived,
checks complete approximate/reference vectors. It does not regenerate circuit
amplitudes; saved-vector hashes alone do not prove a contraction was correct.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_rng_state(state):
    return {'bit_generator': state['bit_generator'],
            'state': {'state': int(state['state']['state']), 'inc': int(state['state']['inc'])},
            'has_uint32': int(state['has_uint32']), 'uinteger': int(state['uinteger'])}


def replay_generation(generation):
    """Independent implementation of the frozen NumPy draw schedule."""
    n = int(generation['n'])
    free = list(map(int, generation['free_vertices']))
    closed = [q for q in range(n) if q not in free]
    rng = np.random.Generator(np.random.PCG64(int(generation['seed'])))
    result = []
    for sid in range(int(generation['count'])):
        before = copy.deepcopy(rng.bit_generator.state)
        branch_uniform = float(rng.random())
        conditional = branch_uniform < float(generation['mixture_weight'])
        bits = [0] * n
        if conditional:
            prefix = rng.integers(0, 2, len(closed), dtype=np.uint8).tolist()
            for q, value in zip(closed, prefix):
                bits[q] = int(value)
            draw_uniform = float(rng.random())
        else:
            bits = list(map(int, rng.integers(0, 2, n, dtype=np.uint8)))
            draw_uniform = None
        result.append({'sample_id': sid, 'conditional_branch': conditional,
                       'branch_uniform': branch_uniform, 'draw_uniform': draw_uniform,
                       'input_data_bits': ''.join(map(str, bits)), 'rng_before': before,
                       'rng_after': copy.deepcopy(rng.bit_generator.state)})
    return result


def map_output(input_bits, free, index, ring):
    """Logical ring indices -> original logical Qiskit q0-right bitstring."""
    bits = [int(x) for x in input_bits]
    for j, q in enumerate(free):
        bits[q] = (index >> j) & 1
    original = [0] * len(bits)
    for logical, physical in enumerate(ring):
        original[physical] = bits[logical]
    return ''.join(map(str, bits)), ''.join(map(str, original[::-1]))


def vector_replay(amplitudes, uniform):
    """Reconstruct actual finite-53-bit CDF law and independent fsum interval."""
    v = np.asarray(amplitudes, dtype=np.complex128)
    if v.ndim != 1 or not len(v) or len(v) & (len(v)-1) or not np.isfinite(v).all():
        raise ValueError('Invalid complete amplitude vector')
    weights = np.asarray(v.real*v.real + v.imag*v.imag, dtype='<f8', order='C')
    cumulative = np.cumsum(weights, dtype=np.float64)
    cdf_mass = float(cumulative[-1])
    if not math.isfinite(cdf_mass) or cdf_mass <= 0:
        raise ValueError('Zero or nonfinite CDF mass')
    cumulative /= cdf_mass
    cumulative[-1] = 1.
    index = int(np.searchsorted(cumulative, uniform, side='right'))
    total = math.fsum(map(float, weights))
    prefix = math.fsum(map(float, weights[:index]))
    endpoint_counts = np.ceil(cumulative * float(2**53)).astype(np.uint64)
    counts = np.diff(endpoint_counts, prepend=np.uint64(0))
    if int(endpoint_counts[-1]) != 2**53 or int(counts.sum(dtype=np.uint64)) != 2**53:
        raise ValueError('Finite-grid CDF counts fail normalization')
    probabilities = counts.astype(np.float64) / float(2**53)
    phase = np.ones(len(v), dtype=np.complex128)
    nonzero = np.abs(v) > 0
    phase[nonzero] = v[nonzero] / np.abs(v[nonzero])
    actual_law_vector = phase * np.sqrt(probabilities)
    return {'A_index': index, 'weights': weights, 'actual_law_vector': actual_law_vector,
            'actual_grid_counts': counts,
            'actual_selected_conditional_probability': float(probabilities[index]),
            'scaled_mass_cdf': cdf_mass, 'scaled_mass_fsum': total,
            'selected_scaled_weight': float(weights[index]),
            'selected_conditional_probability': float(weights[index] / total),
            'cdf_lower_numpy': float(cumulative[index-1]) if index else 0.,
            'cdf_upper_numpy': float(cumulative[index]),
            'cdf_lower_fsum': prefix/total, 'cdf_upper_fsum': (prefix+float(weights[index]))/total,
            'weights_sha256': hashlib.sha256(weights.tobytes()).hexdigest(),
            'amplitudes_sha256': hashlib.sha256(v.astype('<c16', copy=False).tobytes()).hexdigest(),
            'finite_53bit_draw_TV_against_computed_normalized_weights':
                float(.5*np.sum(abs(probabilities-weights/total), dtype=np.float64))}


def witness(reference, approximate):
    ref = np.asarray(reference, dtype=np.complex128)
    app = np.asarray(approximate, dtype=np.complex128)
    if ref.shape != app.shape or ref.ndim != 1 or not np.isfinite(ref).all() or not np.isfinite(app).all():
        raise ValueError('Invalid full-vector witness pair')
    rm = float(np.vdot(ref, ref).real)
    am = float(np.vdot(app, app).real)
    if am <= 0 or not math.isfinite(am) or not math.isfinite(rm):
        raise ValueError('Invalid full-vector norm')
    score = float(abs(np.vdot(ref, app))/math.sqrt(am))
    overlap = min(1., score/math.sqrt(rm)) if rm > 0 else None
    return {'reference_scaled_mass': rm, 'approximate_input_mass': am,
            'uncapped_overlap_witness': score, 'clipped_weight': min(1., score),
            'loss': 1-min(1., score), 'normalized_conditional_overlap_abs': overlap,
            'normalized_conditional_fidelity': None if overlap is None else overlap*overlap}


def kl_upper(mean, count, alpha):
    if count == 0 or mean == 1:
        return 1.
    if mean == 0:
        return -math.expm1(math.log(alpha)/count)
    target = math.log(1/alpha)/count
    low, high = mean, 1.
    for _ in range(100):
        mid = (low+high)/2
        if mid in (low, high):
            break
        value = mean*math.log(mean/mid)+(1-mean)*(math.log1p(-mean)-math.log1p(-mid))
        if value >= target:
            high = mid
        else:
            low = mid
    return high


def mean_statistics(values, scope):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, 'scope': scope}
    mean = float(values.mean())
    se = float(values.std(ddof=1)/math.sqrt(len(values))) if len(values)>1 else None
    return {'n': len(values), 'mean': mean, 'standard_error': se,
            'normal_95_interval_approximate': None if se is None else [mean-1.96*se, mean+1.96*se],
            'min': float(values.min()), 'max': float(values.max()), 'scope': scope}


class Checks:
    def __init__(self):
        self.failures, self.warnings, self.hashes = [], [], {}

    def check(self, condition, code, **context):
        if not condition:
            self.failures.append({'code': code, **context})
        return bool(condition)

    def warning(self, message):
        self.warnings.append(message)

    def read(self, path, lines=False):
        try:
            self.hashes[str(path)] = sha256(path)
            text = Path(path).read_text()
            return [json.loads(line) for line in text.splitlines() if line.strip()] if lines else json.loads(text)
        except (OSError, ValueError) as error:
            self.check(False, 'unreadable_json', path=str(path), detail=str(error))
            return [] if lines else {}

    def close(self, actual, expected, code, rel=3e-13, abs_tol=3e-15, **context):
        try:
            ok = math.isfinite(float(actual)) and math.isfinite(float(expected)) and math.isclose(
                float(actual), float(expected), rel_tol=rel, abs_tol=abs_tol)
        except (TypeError, ValueError):
            ok = False
        return self.check(ok, code, actual=actual, expected=expected, **context)


def verify_bundle(bundle, checks, synthetic=False):
    bundle = Path(bundle).resolve()
    manifest = checks.read(bundle/'manifest.json')
    protocol = checks.read(bundle/'protocol.json')
    checks.check(manifest.get('schema_version') == 'dcs245-bundle-v1', 'bundle_schema')
    checks.check(protocol.get('schema_version') == 'dcs245-production-v1', 'protocol_schema')
    checks.check(synthetic or not (manifest.get('synthetic_fixture') or manifest.get('synthetic_only')), 'synthetic_fixture_in_real_run')
    if not synthetic:
        checks.check(manifest.get('target_qasm_sha256') ==
                     '7a0b8aa6bcef23b901720cb21c9b05cc5de42d4b0e872db4606eec03766f73e5', 'target_qasm_identity')
        try:
            checks.check(sha256(bundle/'inputs/circuit.qasm') == manifest.get('target_qasm_sha256'),
                         'archived_target_qasm_hash')
        except OSError as error:
            checks.check(False, 'target_qasm_not_archived', detail=str(error))
    files = manifest.get('files', {})
    checks.check(bool(files), 'empty_source_manifest')
    for relative, expected in files.items():
        path = (bundle/relative).resolve()
        if not checks.check(path.is_relative_to(bundle), 'source_path_escapes_bundle', path=relative):
            continue
        try:
            checks.check(sha256(path) == expected, 'frozen_source_hash', path=relative)
        except OSError as error:
            checks.check(False, 'missing_frozen_source', path=relative, detail=str(error))
    required = ['protocol.json', 'inputs/circuit.qasm', 'inputs/injected_graph.npz', 'inputs/plans.json',
                'source/production/worker.py', 'source/production/protocol_common.py',
                'source/production/verify.py', 'source/sampling/sampling_protocol.py',
                'source/review/full_vector_audit_certificate.py']
    if not synthetic:
        for relative in required:
            checks.check(relative in files, 'required_dependency_not_hash_bound', path=relative)
    generation = protocol.get('generation', {})
    audit = protocol.get('audit', {})
    n = generation.get('n')
    count = generation.get('count')
    workers = protocol.get('worker_count')
    checks.check(synthetic or (n == 64 and count == 1389 and workers == 8), 'production_layout')
    checks.check(isinstance(n, int) and n > 0 and isinstance(count, int) and count > 0
                 and isinstance(workers, int) and 1 <= workers <= 8, 'valid_dimensions')
    checks.check(generation.get('mixture_weight') == 1., 'frozen_lambda_one_required')
    resources = protocol.get('resources', {})
    checks.check(resources.get('gpus_per_worker') == 1 and resources.get('max_concurrent_gpus') == workers
                 and resources.get('n_qubits') == n and resources.get('global_sample_count') == count,
                 'frozen_resource_layout')
    checks.check(protocol.get('engine') in ('fp32', 'half'), 'engine')
    checks.check(protocol.get('generation_plan') ==
                 ('inplace_k2_fp32' if protocol.get('engine') == 'fp32' else 'half_k1'), 'engine_plan')
    checks.check(protocol.get('reference_plan') == 'inplace_k3_fp64', 'reference_plan')
    anchor = protocol.get('anchor') or {}
    checks.check(anchor.get('engine') == protocol.get('engine')
                 and anchor.get('source_plan') == protocol.get('generation_plan'), 'anchor_engine_plan')
    anchor_bits = anchor.get('input_data_bits')
    checks.check(isinstance(anchor_bits, str) and len(anchor_bits) == n and set(anchor_bits) <= {'0', '1'}, 'anchor_bits')
    checks.check(isinstance(anchor.get('expected_amplitudes_sha256'), str)
                 and re.fullmatch('[0-9a-f]{64}', anchor.get('expected_amplitudes_sha256', '')) is not None,
                 'anchor_hash_not_frozen')
    free = generation.get('free_vertices', [])
    checks.check(len(free) == len(set(free)) and all(isinstance(q, int) and 0 <= q < n for q in free), 'free_vertices')
    closed = [q for q in range(n) if q not in free]
    checks.check(generation.get('closed_vertices') == closed, 'closed_vertices')
    expected = replay_generation(generation)
    records = generation.get('records', [])
    checks.check(len(records) == len(expected), 'protocol_record_count')
    fields = ['sample_id', 'conditional_branch', 'branch_uniform', 'draw_uniform', 'input_data_bits']
    for sid, (actual, wanted) in enumerate(zip(records, expected)):
        for field in fields:
            checks.check(actual.get(field) == wanted[field], 'protocol_rng_'+field, sample_id=sid)
        for field in ('rng_before', 'rng_after'):
            try:
                checks.check(normalized_rng_state(actual[field]) == normalized_rng_state(wanted[field]),
                             'protocol_'+field, sample_id=sid)
            except (KeyError, TypeError, ValueError) as error:
                checks.check(False, 'invalid_protocol_rng_state', sample_id=sid, detail=str(error))
    checks.check(generation.get('seed') != audit.get('seed'), 'audit_seed_equals_generation_seed')
    generation_initial = np.random.PCG64(int(generation['seed'])).state
    audit_initial = np.random.PCG64(int(audit['seed'])).state
    checks.check(generation_initial['state']['inc'] != audit_initial['state']['inc'], 'audit_and_generation_PCG64_streams_not_distinct')
    checks.check(synthetic or audit.get('per_worker') == 8, 'audit_count_per_worker')
    checks.check(audit.get('alpha') == .05, 'audit_alpha_not_frozen_0_05')
    checks.check(audit.get('primary_witness') == 'actual_53bit_cdf_law', 'primary_witness_not_frozen')
    rng = np.random.Generator(np.random.PCG64(int(audit['seed'])))
    selected = []
    for worker in range(workers):
        before = copy.deepcopy(rng.bit_generator.state)
        candidates = list(range(worker, count, workers))
        ids = sorted(map(int, rng.choice(candidates, size=audit['per_worker'], replace=False)))
        selected.extend(ids)
        checks.check(audit.get('ids_by_worker', {}).get(str(worker)) == ids, 'audit_selection_ids', worker_id=worker)
        try:
            selection = audit['selection_rng'][worker]
            checks.check(selection['worker_id'] == worker, 'audit_selection_worker', worker_id=worker)
            checks.check(normalized_rng_state(selection['rng_before']) == normalized_rng_state(before),
                         'audit_rng_before', worker_id=worker)
            checks.check(normalized_rng_state(selection['rng_after']) == normalized_rng_state(rng.bit_generator.state),
                         'audit_rng_after', worker_id=worker)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            checks.check(False, 'invalid_audit_selection_record', worker_id=worker, detail=str(error))
    checks.check(audit.get('ids') == sorted(selected), 'pooled_audit_selection')
    if generation.get('numpy_version') != np.__version__:
        checks.warning('Verifier NumPy version differs from protocol; every actual PCG64 state/draw was compared.')
    with np.load(bundle/'inputs/injected_graph.npz', allow_pickle=False) as graph:
        ring = list(map(int, graph['ring']))
        checks.check(sorted(ring) == list(range(n)), 'ring_permutation')
        adjacency = graph['graph']
        checks.check(adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1]
                     and np.array_equal(adjacency, adjacency.T) and not np.diag(adjacency).any()
                     and np.isin(adjacency, [0, 1]).all(), 'graph_adjacency')
    plans = checks.read(bundle/'inputs/plans.json')
    checks.check(plans.get('open_vertices') == free, 'plans_free_vertices')
    return manifest, protocol, ring, expected


def verify_draw_record(draw, uniform, free_count, checks, sid):
    context = {'sample_id': sid}
    try:
        index = draw['A_index']
        checks.check(isinstance(index, int) and 0 <= index < 2**free_count, 'draw_index_range', **context)
        checks.check(draw['uniform'] == uniform, 'draw_uniform', **context)
        totals = [float(draw[k]) for k in ('scaled_mass_cdf', 'scaled_mass_fsum')]
        checks.check(all(math.isfinite(x) and x > 0 for x in totals), 'draw_positive_mass', **context)
        selected = float(draw['selected_scaled_weight'])
        checks.check(math.isfinite(selected) and 0 < selected <= totals[1], 'draw_selected_weight', **context)
        checks.close(draw['selected_conditional_probability'], selected/totals[1], 'draw_probability_identity', **context)
        for method in ('numpy', 'fsum'):
            lo, hi = float(draw['cdf_lower_'+method]), float(draw['cdf_upper_'+method])
            checks.check(0 <= lo <= uniform < hi <= 1, 'cdf_interval_'+method, **context)
        checks.close(draw['cdf_upper_fsum']-draw['cdf_lower_fsum'], selected/totals[1],
                     'cdf_fsum_interval_width', rel=1e-8, abs_tol=3e-15, **context)
        checks.close(draw['uniform_to_nearest_boundary'],
                     min(uniform-draw['cdf_lower_fsum'], draw['cdf_upper_fsum']-uniform),
                     'cdf_boundary_margin', **context)
        checks.check(draw['independent_cdf_check_passed'] is True, 'worker_full_cdf_check', **context)
        tv = float(draw['finite_53bit_draw_TV_against_computed_normalized_weights'])
        checks.check(math.isfinite(tv) and 0 <= tv <= 1, 'grid_tv_range', **context)
        for field in ('weights_sha256', 'amplitudes_sha256'):
            checks.check(isinstance(draw[field], str) and re.fullmatch('[0-9a-f]{64}', draw[field]) is not None,
                         'invalid_vector_commitment_'+field, **context)
        return index
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        checks.check(False, 'malformed_draw_record', detail=str(error), **context)
        return None


def load_bound_vector(directory, metadata, expected_length, checks, sid, kind):
    try:
        path = (directory/metadata['path']).resolve()
        if not checks.check(path.parent == directory.resolve(), 'vector_path_escape', sample_id=sid, kind=kind):
            return None
        checks.check(sha256(path) == metadata['file_sha256'], 'vector_file_hash', sample_id=sid, kind=kind)
        values = np.load(path, allow_pickle=False)
        checks.check(values.dtype == np.dtype('complex128') and values.shape == (expected_length,),
                     'vector_dtype_shape', sample_id=sid, kind=kind)
        checks.check(metadata.get('shape') == [expected_length] and metadata.get('dtype') == 'complex128',
                     'vector_metadata_shape', sample_id=sid, kind=kind)
        checks.check(hashlib.sha256(values.astype('<c16', copy=False).tobytes()).hexdigest() == metadata['payload_sha256'],
                     'vector_payload_hash', sample_id=sid, kind=kind)
        checks.check(np.isfinite(values).all(), 'vector_finite', sample_id=sid, kind=kind)
        return values
    except (OSError, KeyError, TypeError, ValueError) as error:
        checks.check(False, 'unreadable_bound_vector', sample_id=sid, kind=kind, detail=str(error))
        return None


def verify_run(bundle, artifacts, ledger_path=None, synthetic=False):
    checks = Checks()
    bundle, artifacts = Path(bundle).resolve(), Path(artifacts).resolve()
    try:
        manifest, protocol, ring, expected_records = verify_bundle(bundle, checks, synthetic)
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        checks.check(False, 'invalid_bundle', detail=str(error))
        return {'validation_passed': False, 'failures': checks.failures, 'warnings': checks.warnings}
    g, audit_protocol = protocol['generation'], protocol['audit']
    count, n, free, workers = g['count'], g['n'], g['free_vertices'], protocol['worker_count']
    D, vector_length = 2.**(n-len(free)), 2**len(free)
    manifest_hash, protocol_hash = sha256(bundle/'manifest.json'), sha256(bundle/'protocol.json')
    plans = checks.read(bundle/'inputs/plans.json')
    plan_map = {p['name']: p for p in plans.get('plans', [])}
    worker_paths = {}
    for path in sorted(artifacts.rglob('source.json')):
        source = checks.read(path)
        worker = source.get('worker_id')
        if checks.check(isinstance(worker, int) and 0 <= worker < workers, 'unexpected_worker_source', path=str(path)):
            checks.check(worker not in worker_paths, 'duplicate_worker_artifacts', worker_id=worker)
            worker_paths[worker] = (path.parent, source)
    checks.check(sorted(worker_paths) == list(range(workers)), 'missing_worker_artifacts')
    all_ids, all_audit_ids, approximate_scores, reference_scores, witnesses, worker_timings = [], [], [], [], [], []
    all_sample_rows, hardware_rows = {}, []
    for worker in range(workers):
        if worker not in worker_paths:
            continue
        directory, source = worker_paths[worker]
        checks.check(synthetic or not source.get('synthetic_only'), 'synthetic_worker_in_real_run', worker_id=worker)
        expected_ids = list(range(worker, count, workers))
        expected_audits = audit_protocol['ids_by_worker'][str(worker)]
        for key, expected in [('worker_count', workers), ('manifest_sha256', manifest_hash),
                              ('protocol_sha256', protocol_hash), ('sample_ids', expected_ids),
                              ('audit_ids', expected_audits), ('engine', protocol['engine'])]:
            checks.check(source.get(key) == expected, 'worker_source_'+key, worker_id=worker)
        checks.check(not (directory/'failure.json').exists(), 'worker_failure_marker', worker_id=worker)
        hardware = checks.read(directory/'hardware.json')
        hardware_rows.append({'worker_id': worker, **hardware})
        visible = hardware.get('visible_device_count')
        checks.check(visible == 1, 'worker_visible_gpu_count_not_one', worker_id=worker, actual=visible)
        anchor_record = checks.read(directory/'anchor.json')
        anchor = protocol['anchor']
        checks.check(anchor_record.get('worker_id') == worker and anchor_record.get('passed') is True,
                     'worker_anchor_failed', worker_id=worker)
        for key in ('input_data_bits', 'expected_amplitudes_sha256'):
            checks.check(anchor_record.get(key) == anchor.get(key), 'worker_anchor_'+key, worker_id=worker)
        checks.check(anchor_record.get('actual_amplitudes_sha256') == anchor['expected_amplitudes_sha256'],
                     'worker_anchor_vector_hash', worker_id=worker)
        for field, expected_key in [('device_name', 'expected_device_name'), ('numpy', 'expected_numpy_version'), ('cupy', 'expected_cupy_version')]:
            checks.check(hardware.get(field) == anchor.get(expected_key) == anchor_record.get(field),
                         'worker_anchor_software_'+field, worker_id=worker)
        for field in ('cuda_driver_version', 'cuda_runtime_version', 'nvrtc_version'):
            checks.check(hardware.get(field) is not None and hardware.get(field) == anchor_record.get(field),
                         'worker_anchor_runtime_'+field, worker_id=worker)
        rows = checks.read(directory/'samples.jsonl', lines=True)
        audits = checks.read(directory/'audits.jsonl', lines=True)
        summary = checks.read(directory/'summary.json')
        completed = checks.read(directory/'completed.json')
        artifact_manifest = checks.read(directory/'artifacts.json')
        checks.check(artifact_manifest.get('worker_id') == worker, 'artifact_manifest_worker', worker_id=worker)
        expected_artifacts = {'source.json', 'hardware.json', 'anchor.json', 'samples.jsonl', 'audits.jsonl', 'summary.json'}
        expected_artifacts.update(f'approx_{sid:06d}.npy' for sid in expected_audits)
        expected_artifacts.update(f'reference_{sid:06d}.npy' for sid in expected_audits)
        checks.check(expected_artifacts <= set(artifact_manifest.get('files', {})), 'missing_required_artifact_manifest_entry', worker_id=worker)
        for relative, entry in artifact_manifest.get('files', {}).items():
            artifact_path = (directory/relative).resolve()
            if not checks.check(artifact_path.parent == directory.resolve(), 'artifact_manifest_path_escape', worker_id=worker, path=relative):
                continue
            try:
                checks.check(artifact_path.stat().st_size == entry['bytes'] and sha256(artifact_path) == entry['sha256'],
                             'required_artifact_hash', worker_id=worker, path=relative)
            except (OSError, KeyError, TypeError) as error:
                checks.check(False, 'required_artifact_missing', worker_id=worker, path=relative, detail=str(error))
        checks.check([r.get('sample_id') for r in rows] == expected_ids, 'worker_sample_ids_count_order', worker_id=worker)
        checks.check([r.get('audit_id') for r in audits] == expected_audits, 'worker_audit_ids_count_order', worker_id=worker)
        for row in rows:
            sid = row.get('sample_id')
            all_ids.append(sid)
            if not isinstance(sid, int) or not 0 <= sid < count:
                checks.check(False, 'invalid_sample_id', worker_id=worker, sample_id=sid)
                continue
            try:
                expected = expected_records[sid]
                checks.check(row['worker_id'] == worker and sid % workers == worker, 'sample_worker_assignment', sample_id=sid)
                for field in ('manifest_sha256', 'protocol_sha256'):
                    checks.check(row[field] == source[field], 'sample_'+field, sample_id=sid)
                for field in ('conditional_branch', 'branch_uniform', 'draw_uniform', 'input_data_bits'):
                    checks.check(row[field] == expected[field], 'sample_rng_'+field, sample_id=sid)
                for field in ('rng_before', 'rng_after'):
                    checks.check(normalized_rng_state(row[field]) == normalized_rng_state(expected[field]),
                                 'sample_'+field, sample_id=sid)
                is_audit = sid in expected_audits
                checks.check(row['is_preselected_audit'] is is_audit, 'audit_flag', sample_id=sid)
                checks.check(('approximate_vector_file' in row) == is_audit, 'audit_vector_presence', sample_id=sid)
                index = verify_draw_record(row['draw'], expected['draw_uniform'], len(free), checks, sid)
                if index is None:
                    continue
                ring_bits, qiskit_bits = map_output(expected['input_data_bits'], free, index, ring)
                checks.check(row['data_bits_ring_order'] == ring_bits and row['bitstring_qiskit'] == qiskit_bits,
                             'output_bit_mapping', sample_id=sid)
                draw = row['draw']
                p_approx = float(row['approximate_circuit_probability_estimate'])
                if not math.isfinite(p_approx) or p_approx < 0:
                    raise ValueError('Invalid approximate point probability')
                checks.close(p_approx, draw['selected_scaled_weight']/D, 'approximate_probability_identity',
                             abs_tol=0., sample_id=sid)
                grid_count = math.ceil(draw['cdf_upper_numpy']*2**53)-math.ceil(draw['cdf_lower_numpy']*2**53)
                checks.check(grid_count > 0, 'selected_grid_probability_zero', sample_id=sid)
                checks.close(row['actual_53bit_conditional_probability'], grid_count/2**53,
                             'actual_grid_probability', abs_tol=0., sample_id=sid)
                checks.close(row['actual_joint_sampler_probability'], grid_count/2**53/D,
                             'actual_joint_probability', abs_tol=0., sample_id=sid)
                contraction = row['contraction']
                generation_plan = plan_map[protocol['generation_plan']]
                checks.check(contraction['parts'] == 1 and contraction['part'] == 0
                             and contraction['slices_evaluated'] == 2**len(generation_plan['operations'])
                             and contraction['open_vertices'] == free, 'generation_full_coherent_plan', sample_id=sid)
                checks.check(contraction['prefix_dtype'] == 'complex64', 'generation_prefix_dtype', sample_id=sid)
                if protocol['engine'] == 'half':
                    checks.check(contraction.get('intermediate_storage') == 'float16 real/imaginary components',
                                 'half_storage_annotation', sample_id=sid)
                for field in ('cdf_seconds', 'sample_seconds_before_json_write'):
                    checks.check(math.isfinite(float(row[field])) and row[field] >= 0, 'sample_time_'+field, sample_id=sid)
                approximate_scores.append(2.**n*p_approx-1)
                all_sample_rows[sid] = row
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
                checks.check(False, 'malformed_sample', worker_id=worker, sample_id=sid, detail=str(error))
        for row in audits:
            sid = row.get('audit_id')
            all_audit_ids.append(sid)
            if sid not in all_sample_rows:
                checks.check(False, 'audit_without_valid_sample', sample_id=sid)
                continue
            try:
                sample = all_sample_rows[sid]
                checks.check(sid in expected_audits and row['worker_id'] == worker, 'audit_assignment', sample_id=sid)
                for field in ('manifest_sha256', 'protocol_sha256'):
                    checks.check(row[field] == source[field], 'audit_'+field, sample_id=sid)
                checks.check(row['approximate_vector_file'] == sample['approximate_vector_file'], 'audit_vector_identity', sample_id=sid)
                approx = load_bound_vector(directory, row['approximate_vector_file'], vector_length, checks, sid, 'approximate')
                ref = load_bound_vector(directory, row['reference_vector_file'], vector_length, checks, sid, 'reference')
                if approx is None or ref is None:
                    continue
                replay = vector_replay(approx, sample['draw_uniform'])
                checks.check(row['A_index'] == sample['draw']['A_index'] == replay['A_index'], 'full_vector_cdf_index', sample_id=sid)
                for key in ('weights_sha256', 'amplitudes_sha256'):
                    checks.check(replay[key] == sample['draw'][key], 'full_vector_'+key, sample_id=sid)
                for key in ('scaled_mass_cdf', 'scaled_mass_fsum', 'selected_scaled_weight',
                            'selected_conditional_probability', 'cdf_lower_numpy', 'cdf_upper_numpy',
                            'cdf_lower_fsum', 'cdf_upper_fsum', 'finite_53bit_draw_TV_against_computed_normalized_weights'):
                    checks.close(replay[key], sample['draw'][key], 'full_vector_'+key, sample_id=sid)
                checks.close(replay['actual_selected_conditional_probability'], sample['actual_53bit_conditional_probability'],
                             'full_vector_actual_grid_probability', abs_tol=0., sample_id=sid)
                law = replay['actual_law_vector']
                lawmeta = row['actual_cdf_law']
                for key, expected in [('grid_bits', 53), ('grid_total', 2**53),
                    ('grid_counts_sha256', hashlib.sha256(replay['actual_grid_counts'].astype('<u8').tobytes()).hexdigest()),
                    ('actual_law_vector_sha256', hashlib.sha256(law.astype('<c16').tobytes()).hexdigest()),
                    ('source_amplitudes_sha256', replay['amplitudes_sha256']),
                    ('source_weights_sha256', replay['weights_sha256'])]:
                    checks.check(lawmeta.get(key) == expected, 'actual_law_'+key, sample_id=sid)
                checks.close(lawmeta['source_cdf_total'], replay['scaled_mass_cdf'], 'actual_law_mass', sample_id=sid)
                checks.close(lawmeta['law_norm_squared'], float(np.vdot(law, law).real), 'actual_law_norm', sample_id=sid)
                actual_witness = witness(ref, law)
                raw_witness = witness(ref, approx)
                for label, values in [('witness', actual_witness), ('raw_vector_witness', raw_witness)]:
                    checks.check(row[label].get('audit_id') == sid, 'witness_audit_id', sample_id=sid)
                    for key, expected in values.items():
                        if expected is None:
                            checks.check(row[label].get(key) is None, 'witness_zero_mass_'+key, sample_id=sid)
                        else:
                            checks.close(row[label][key], expected, label+'_'+key, rel=1e-11, sample_id=sid)
                ref_plan = plan_map[protocol['reference_plan']]
                rc = row['reference_contraction']
                checks.check(rc['prefix_dtype'] == 'complex128' and rc['parts'] == 1 and rc['part'] == 0
                             and rc['slices_evaluated'] == 2**len(ref_plan['operations'])
                             and rc['open_vertices'] == free, 'fp64_reference_full_coherent_plan', sample_id=sid)
                p_ref = float(abs(ref[row['A_index']])**2/D)
                checks.close(row['reference_selected_probability'], p_ref, 'audited_fp64_probability', abs_tol=0., sample_id=sid)
                checks.close(row['reference_selected_linear_xeb'], 2.**n*p_ref-1, 'audited_fp64_xeb', sample_id=sid)
                checks.close(row['approximate_selected_probability_estimate'], sample['approximate_circuit_probability_estimate'],
                             'audited_approximate_probability', abs_tol=0., sample_id=sid)
                witnesses.append({'audit_id': sid, **actual_witness})
                reference_scores.append(2.**n*p_ref-1)
                del approx, ref, law, replay
            except (OSError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
                checks.check(False, 'malformed_audit', worker_id=worker, sample_id=sid, detail=str(error))
        try:
            checks.check(summary['completed'] is True and completed['completed'] is True,
                         'worker_not_complete', worker_id=worker)
            checks.check(summary['worker_id'] == completed['worker_id'] == worker, 'summary_worker_id', worker_id=worker)
            checks.check(summary['sample_count'] == len(expected_ids) and summary['audit_count'] == len(expected_audits),
                         'summary_counts', worker_id=worker)
            for key in ('manifest_sha256', 'protocol_sha256', 'engine', 'sample_ids', 'audit_ids'):
                checks.check(summary[key] == source[key], 'summary_'+key, worker_id=worker)
            checks.check(summary['samples_sha256'] == sha256(directory/'samples.jsonl'), 'samples_file_hash', worker_id=worker)
            checks.check(summary['audits_sha256'] == sha256(directory/'audits.jsonl'), 'audits_file_hash', worker_id=worker)
            checks.check(completed['summary_sha256'] == sha256(directory/'summary.json'), 'summary_file_hash', worker_id=worker)
            checks.check(completed['artifacts_sha256'] == sha256(directory/'artifacts.json'), 'artifacts_manifest_hash', worker_id=worker)
            timekeys = ['generation_setup_seconds', 'generation_seconds', 'generation_contraction_seconds',
                        'generation_cdf_seconds', 'generation_write_seconds', 'reference_setup_seconds',
                        'audit_seconds', 'audit_contraction_seconds', 'audit_compare_seconds', 'audit_write_seconds',
                        'worker_seconds_to_summary_build', 'anchor_seconds']
            for key in timekeys:
                checks.check(math.isfinite(float(summary[key])) and summary[key] >= 0, 'worker_time_'+key, worker_id=worker)
            total_phases = sum(summary[k] for k in ['generation_setup_seconds', 'anchor_seconds', 'generation_seconds', 'reference_setup_seconds', 'audit_seconds'])
            checks.check(total_phases <= summary['worker_seconds_to_summary_build']+1e-6, 'worker_phase_times_exceed_process', worker_id=worker)
            checks.check(completed['worker_seconds_to_complete_marker'] >= summary['worker_seconds_to_summary_build'],
                         'completion_before_summary', worker_id=worker)
            checks.check(completed['end_unix'] >= summary['process_start_unix'], 'worker_wallclock_order', worker_id=worker)
            checks.check(summary['anchor_amplitudes_sha256'] == anchor['expected_amplitudes_sha256'],
                         'summary_anchor_hash', worker_id=worker)
            worker_timings.append({'worker_id': worker, **{k: summary[k] for k in timekeys},
                                   'process_start_unix': summary['process_start_unix'],
                                   'process_end_unix': completed['end_unix'],
                                   'worker_seconds_to_complete_marker': completed['worker_seconds_to_complete_marker']})
        except (OSError, KeyError, TypeError, ValueError) as error:
            checks.check(False, 'malformed_worker_summary', worker_id=worker, detail=str(error))
    checks.check(sorted(all_ids) == list(range(count)), 'all_sample_ids_exactly_once')
    checks.check(sorted(all_audit_ids) == audit_protocol['ids'], 'all_audit_ids_exactly_once')
    checks.check(sorted(w['audit_id'] for w in witnesses) == audit_protocol['ids'], 'all_audit_vectors_verified')
    for field in ('device_name', 'compute_capability', 'numpy', 'cupy', 'cuda_driver_version', 'cuda_runtime_version', 'nvrtc_version'):
        values = [json.dumps(row.get(field), sort_keys=True) for row in hardware_rows]
        checks.check(len(set(values)) <= 1 and all(row.get(field) is not None for row in hardware_rows),
                     'heterogeneous_worker_'+field)
    if not synthetic:
        checks.check(all('H200' in row.get('device_name', '') for row in hardware_rows), 'worker_gpu_model_not_H200')
    data_passed = not checks.failures
    losses = [w['loss'] for w in witnesses]
    alpha = audit_protocol['alpha']
    mean_loss = math.fsum(losses)/len(losses) if losses else None
    upper = kl_upper(mean_loss, len(losses), alpha) if losses else 1.
    overlap = max(0., 1-upper)
    fidelity = g['mixture_weight']*overlap**2+(1-g['mixture_weight'])/2**n
    quality = {'valid_only_if_all_records_and_vectors_pass': data_passed,
               'audit_count': len(losses), 'alpha': alpha, 'observed_mean_loss': mean_loss,
               'population_mean_loss_upper': upper, 'overlap_lower': overlap,
               'state_fidelity_lower': fidelity, 'above_0_349': fidelity > .349, 'above_0_38': fidelity > .38,
               'scope': 'Actual 53-bit CDF-law witness with numerical complex128 reference blocks. '
                        'Statistical confidence is conditional on correct reference arithmetic; '
                        'this verifier does not independently regenerate circuit amplitudes.'}
    timing_start_failures = len(checks.failures)
    timing = verify_timing(ledger_path, worker_timings, hardware_rows, workers, checks, synthetic)
    timing_passed = len(checks.failures) == timing_start_failures
    result = {'schema_version': 'dcs245-independent-verification-v1', 'synthetic_fixture': synthetic,
              'validation_passed': data_passed and timing_passed,
              'record_and_full_vector_validation_passed': data_passed,
              'timing_validation_passed': timing_passed, 'sample_count': len(all_ids),
              'audit_count': len(witnesses), 'manifest_sha256': manifest_hash, 'protocol_sha256': protocol_hash,
              'executed_verifier_sha256': sha256(__file__),
              'frozen_verifier_sha256': manifest.get('files', {}).get('source/production/verify.py'),
              'quality': quality, 'timing': timing,
              'approximate_probability_linear_xeb_all_outputs': mean_statistics(approximate_scores,
                 'Diagnostic only: approximate generation amplitudes; NOT independent FP64 or exact all-output XEB.'),
              'fp64_reference_linear_xeb_audited_outputs_only': mean_statistics(reference_scores,
                 'Preselected audited subset only, numerical FP64 reference; does not score all outputs.'),
              'worker_timings': worker_timings, 'hardware': hardware_rows,
              'checks_scope': {'rng': 'Complete generation and audit-selection PCG64 states/draws replayed.',
                 'all_outputs': 'IDs, source hashes, prefix/output mapping, stored CDF intervals and probability identities.',
                 'audited_outputs': 'Archived full vectors loaded and hashed; full CDF/grid law and FP64 overlap recomputed.',
                 'remaining': 'Unaudited vectors require later frozen GPU regeneration to check commitments. '
                              'No quantum amplitudes were independently regenerated by this CPU verifier.',
                 'anchor': 'Each worker matched the frozen pilot block hash and software/device versions. '
                           'This supports reproducibility but is not a proof of determinism on every input.'},
              'failures': checks.failures, 'warnings': checks.warnings, 'input_hashes': checks.hashes}
    result['internal_time_and_quality_targets_met'] = bool(result['validation_passed'] and fidelity > .349
                                                          and timing.get('primary_seconds', math.inf) <= 870)
    return result


def verify_timing(ledger_path, worker_timings, hardware_rows, workers, checks, synthetic):
    if ledger_path is None:
        checks.check(False, 'central_timing_ledger_missing')
        return {'primary_seconds': None, 'scope': 'No central timing inference was made.'}
    ledger = checks.read(Path(ledger_path))
    checks.check(ledger.get('terminal') == 'Succeeded', 'central_terminal_not_succeeded')
    checks.check(ledger.get('log_return_code') == 0, 'central_log_return_code')
    checks.check(ledger.get('declared_gpu_limit') == workers and workers <= 8
                 and ledger.get('worker_count') == workers and ledger.get('gpus_per_worker') == 1,
                 'central_gpu_layout')
    keys = ['central_monotonic_submit_to_terminal_observed_seconds',
            'central_monotonic_submit_to_all_logs_returned_seconds',
            'central_monotonic_submit_to_all_artifacts_verified_seconds']
    values = {}
    for key in keys:
        value = ledger.get(key)
        checks.check(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                     'missing_or_invalid_'+key)
        values[key] = value
    primary = values[keys[-1]]
    if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
        checks.check(primary >= values[keys[0]] and primary >= values[keys[1]], 'artifacts_verified_before_terminal_or_logs')
    observed = ledger.get('max_observed_running_workers')
    checks.check(isinstance(observed, int) and 0 <= observed <= workers, 'observed_worker_concurrency')
    events = []
    for row in worker_timings:
        events.extend([(row['process_start_unix'], 1), (row['process_end_unix'], -1)])
    active, peak = 0, 0
    for _, delta in sorted(events, key=lambda pair: (pair[0], pair[1])):
        active += delta
        peak = max(peak, active)
    checks.check(peak <= 8, 'worker_interval_concurrency_exceeds_8')
    if observed == workers and all(h.get('visible_device_count') == 1 for h in hardware_rows):
        concurrency_scope = 'Status snapshot observed all workers Running; each worker reports one visible GPU.'
    else:
        concurrency_scope = 'Status snapshots give an observed lower bound; declared one-GPU task layout gives the upper bound.'
    return {'primary_seconds': primary,
            'primary_scope': 'Continuous central monotonic submit-to-required-artifacts-verified duration, including queue/deployment/collection.',
            **values, 'within_870_seconds': isinstance(primary, (int, float)) and primary <= 870,
            'max_worker_generation_seconds': max((r['generation_seconds'] for r in worker_timings), default=None),
            'sum_worker_generation_seconds': sum(r['generation_seconds'] for r in worker_timings),
            'sum_worker_audit_seconds': sum(r['audit_seconds'] for r in worker_timings),
            'sum_worker_process_seconds_one_GPU_each': sum(r['worker_seconds_to_complete_marker'] for r in worker_timings),
            'GPU_seconds_scope': 'Worker process duration on one visible GPU each; excludes allocation before process start and after its completion marker.',
            'max_observed_running_workers': observed, 'declared_gpu_limit': ledger.get('declared_gpu_limit'),
            'peak_overlapping_worker_process_intervals_from_host_UTC': peak,
            'concurrency_scope': concurrency_scope,
            'worker_interval_scope': 'UTC-based diagnostic, subject to cross-host clock alignment; not a replacement for central monotonic timing.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--records-only', action='store_true',
                        help='Exit on record/vector result while the continuous launcher has not finalized its timing ledger.')
    args = parser.parse_args()
    result = verify_run(args.bundle, args.artifacts, args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result.get(k) for k in ['validation_passed', 'record_and_full_vector_validation_passed',
          'timing_validation_passed', 'sample_count', 'audit_count', 'quality', 'timing', 'failures']}, indent=2))
    passed = result.get('record_and_full_vector_validation_passed') if args.records_only else result.get('validation_passed')
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()

