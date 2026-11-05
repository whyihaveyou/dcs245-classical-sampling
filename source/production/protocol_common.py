"""Pure CPU protocol construction and actual finite-grid CDF-law reconstruction."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'sampling'))
from sampling_protocol import make_protocol


def canonical_json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha256_bytes(value):return hashlib.sha256(value).hexdigest()
def file_sha256(path):return sha256_bytes(Path(path).read_bytes())


def make_run_protocol(seed,audit_seed,count,workers,audits_per_worker,n,free,engine,anchor=None):
    if seed==audit_seed:raise ValueError('Use distinct generation and audit seeds')
    if engine not in ('fp32','half'):raise ValueError('Unknown numerical engine')
    if workers<1 or audits_per_worker<1:raise ValueError('Invalid worker/audit count')
    rng=np.random.Generator(np.random.PCG64(audit_seed))
    by_worker={};selection=[]
    # Selection happens before generation; no prefix, output or score is inspected.
    for worker in range(workers):
        ids=list(range(worker,count,workers))
        if len(ids)<audits_per_worker:raise ValueError('Too few records in an audit stratum')
        before=rng.bit_generator.state
        chosen=sorted(map(int,rng.choice(ids,size=audits_per_worker,replace=False)))
        by_worker[str(worker)]=chosen
        selection.append({'worker_id':worker,'rng_before':before,'rng_after':rng.bit_generator.state})
    generation=make_protocol(seed,count,n,free,mixture_weight=1.)
    return {'schema_version':'dcs245-production-v1','execution_status':'not_run',
            'engine':engine,'worker_count':workers,'generation':generation,'anchor':anchor,
            'resources':{'gpus_per_worker':1,'max_concurrent_gpus':workers,'gpu_model':'NVIDIA H200',
                         'n_qubits':n,'global_sample_count':count},
            'generation_plan':'inplace_k2_fp32' if engine=='fp32' else 'half_k1',
            'reference_plan':'inplace_k3_fp64',
            'audit':{'seed':int(audit_seed),'per_worker':audits_per_worker,'alpha':.05,
                     'primary_witness':'actual_53bit_cdf_law',
                     'ids':sorted(i for ids in by_worker.values() for i in ids),
                     'ids_by_worker':by_worker,'selection_rng':selection,
                     'method':'Independent audit PCG64; uniform subset without replacement within each fixed worker stratum; selected before generation'},
            'law':'Fresh uniform closed bits per output, one conditional CDF draw; approximate joint sampler',
            'scope':'Protocol only. No experimental results, statistical certificate or performance claim.'}


def actual_cdf_law_vector(amplitudes):
    """Construct amplitudes whose probabilities equal the sampler's 53-bit law.

    The source vector phases are retained. Grid interval counts are exact
    uint64 integers; square roots/phases remain ordinary numerical arithmetic.
    """
    a=np.asarray(amplitudes,dtype=np.complex128)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Invalid vector')
    weights=np.asarray(a.real*a.real+a.imag*a.imag,dtype='<f8',order='C')
    cumulative=np.cumsum(weights,dtype=np.float64)
    if not cumulative[-1]>0:raise ArithmeticError('Zero conditional vector; no silent replacement')
    total=float(cumulative[-1]);cumulative/=total;cumulative[-1]=1.
    endpoints=np.ceil(cumulative*2.**53).astype(np.uint64)
    counts=np.diff(np.concatenate((np.zeros(1,np.uint64),endpoints)))
    if int(counts.sum(dtype=np.uint64))!=1<<53:raise AssertionError('Grid counts do not normalize')
    phases=np.ones(len(a),np.complex128);nonzero=np.abs(a)>0
    phases[nonzero]=a[nonzero]/np.abs(a[nonzero])
    law=phases*np.sqrt(counts.astype(np.float64)/2.**53)
    return law,{'grid_bits':53,'grid_total':1<<53,'grid_counts_sha256':sha256_bytes(counts.astype('<u8',copy=False).tobytes()),
                'actual_law_vector_sha256':sha256_bytes(law.astype('<c16',copy=False).tobytes()),
                'source_amplitudes_sha256':sha256_bytes(a.astype('<c16',copy=False).tobytes()),
                'source_weights_sha256':sha256_bytes(weights.tobytes()),'source_cdf_total':total,
                'law_norm_squared':float(np.vdot(law,law).real)}


def validate_bundle(bundle):
    root=Path(bundle).resolve();manifest=json.loads((root/'manifest.json').read_text())
    for relative,expected in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Manifest path escapes bundle')
        if file_sha256(path)!=expected:raise ValueError('Frozen file mismatch: '+relative)
    protocol=json.loads((root/'protocol.json').read_text());g=protocol['generation'];a=protocol['audit']
    expected=make_run_protocol(g['seed'],a['seed'],g['count'],protocol['worker_count'],a['per_worker'],
                               g['n'],g['free_vertices'],protocol['engine'],protocol['anchor'])
    if protocol!=expected:raise ValueError('Full protocol/RNG replay mismatch')
    return manifest,protocol
