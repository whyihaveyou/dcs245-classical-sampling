#!/usr/bin/env python3
"""Score a fixed list of Qiskit bitstrings with the frozen FP64 block plan."""
import argparse, base64, gzip, hashlib, json, os, sys, time, subprocess
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--samples', type=Path, required=True)
    ap.add_argument('--worker-id', type=int, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    bundle = a.bundle.resolve()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'scoring_manifest.json').read_text())
    for filename, digest in manifest['files'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == digest, filename
    anchors = json.loads((root/'audit_anchors.json').read_text())
    source = bundle / 'source'
    for p in (source, source/'sampling', source/'kernel_opt'):
        sys.path.insert(0, str(p))
    from conditional_block_inplace_gpu import InplaceConditionalBlockGPU
    import cupy as cp

    protocol = json.loads((bundle/'protocol.json').read_text())
    plans = json.loads((bundle/'inputs/plans.json').read_text())
    plan = next(x for x in plans['plans'] if x['name'] == protocol['reference_plan'])
    graph = np.load(bundle/'inputs/injected_graph.npz')
    ring = list(map(int, graph['ring']))
    free = list(map(int, protocol['generation']['free_vertices']))
    n = int(protocol['generation']['n'])
    D = 2.0 ** (n - len(free))
    rows = [x.strip() for x in a.samples.read_text().splitlines() if x.strip()]
    if len(rows) != 1389 or any(len(x) != n or set(x) - {'0', '1'} for x in rows):
        raise ValueError('invalid fixed sample list')
    assigned = [(i, rows[i]) for i in range(len(rows)) if i % a.workers == a.worker_id]
    if a.smoke:
        assigned = [(i, rows[i]) for i in (8,16)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists(): raise RuntimeError('Refusing to overwrite completed scores')
    checkpoint = a.out.with_suffix('.jsonl').open('x')
    hardware = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'], text=True).strip()
    print('SCORE_START:'+json.dumps({'worker_id':a.worker_id, 'smoke':a.smoke, 'hardware':hardware, 'numpy':np.__version__, 'cupy':cp.__version__, 'manifest_sha256':hashlib.sha256((root/'scoring_manifest.json').read_bytes()).hexdigest(), 'samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest()}),flush=True)
    assert np.__version__ == '2.2.6'
    assert cp.__version__ == '13.6.0'
    start = time.perf_counter()
    cp.cuda.Device(0).use()
    engine = InplaceConditionalBlockGPU(graph['graph'], graph['unitaries'], n,
                                        graph['order'], plan['operations'], free)
    engine.alloc(dtype=np.complex128)
    setup = time.perf_counter() - start
    out = []
    for sid, qiskit_bits in assigned:
        begin = time.perf_counter()
        physical = [int(x) for x in qiskit_bits[::-1]]
        logical = [physical[ring[q]] for q in range(n)]
        closed = logical.copy()
        for q in free:
            closed[q] = 0
        v, contraction = engine.block(np.asarray(closed, dtype=np.uint8))
        host = cp.asnumpy(v)
        index = sum(logical[q] << j for j, q in enumerate(free))
        p = float(abs(host[index]) ** 2 / D)
        out.append({'sample_id': sid, 'bitstring_qiskit': qiskit_bits,
                    'logical_bits': ''.join(map(str, logical)),
                    'A_index': index, 'probability_fp64': p,
                    'linear_xeb_point': float(2.0 ** n * p - 1.0),
                    'amplitude_real': float(host[index].real), 'amplitude_imag': float(host[index].imag),
                    'contraction_seconds': float(contraction['seconds']),
                    'wall_seconds': time.perf_counter() - begin})
        if str(sid) in anchors:
            expected=anchors[str(sid)]
            assert index == expected['A_index']
            rel=abs(p/expected['probability_fp64']-1)
            out[-1]['anchor_relative_error']=rel
            assert rel < 1e-8, (sid, rel)
        line=json.dumps(out[-1], separators=(',', ':'))
        checkpoint.write(line+'\n');checkpoint.flush();os.fsync(checkpoint.fileno())
        print('SCORE_ROW:'+line,flush=True)
        del host
    result = {'worker_id': a.worker_id, 'sample_count': len(out),
              'hardware':hardware, 'numpy':np.__version__, 'cupy':cp.__version__,
              'sample_list_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),
              'manifest_sha256':hashlib.sha256((root/'scoring_manifest.json').read_bytes()).hexdigest(),
              'smoke':a.smoke, 'setup_seconds': setup, 'score_seconds': time.perf_counter() - start,
              'rows': out, 'scope': 'FP64 numerical reference amplitudes using frozen inplace_k3_fp64 plan.'}
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    payload = base64.b64encode(gzip.compress(json.dumps(result, separators=(',', ':')).encode(), compresslevel=6)).decode()
    print(f'SCORE_ARCHIVE:{a.worker_id}:{payload}', flush=True)
    print(json.dumps({'worker_id': a.worker_id, 'completed': True,
                      'sample_count': len(out), 'score_seconds': result['score_seconds']}), flush=True)


if __name__ == '__main__':
    main()
