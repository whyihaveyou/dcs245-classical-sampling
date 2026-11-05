"""Frozen production worker: generate all assigned samples, then all fixed audits."""
import time
PROCESS_START=time.perf_counter();PROCESS_UNIX_START=time.time()
import argparse,gc,hashlib,json,math,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent
for path in [SOURCE,SOURCE/'sampling',SOURCE/'kernel_opt',SOURCE/'review']:
    sys.path.insert(0,str(path))
from protocol_common import actual_cdf_law_vector,file_sha256,validate_bundle
from sampling_protocol import draw_checked,assemble_output
from conditional_block_inplace_gpu import InplaceConditionalBlockGPU
from conditional_block_half_gpu import HalfConditionalBlockGPU
from full_vector_audit_certificate import audit_vector_pair


def write_json(path,value):
    start=time.perf_counter()
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return time.perf_counter()-start


def save_vector(path,value):
    start=time.perf_counter()
    with Path(path).open('xb') as f:
        np.save(f,np.asarray(value,dtype=np.complex128),allow_pickle=False);f.flush();os.fsync(f.fileno())
    return {'path':Path(path).name,'file_sha256':file_sha256(path),
            'payload_sha256':hashlib.sha256(np.asarray(value,dtype='<c16').tobytes()).hexdigest(),
            'shape':list(value.shape),'dtype':'complex128'},time.perf_counter()-start


def append_row(stream,row):
    t=time.perf_counter();stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
    return time.perf_counter()-t


def execute(bundle,worker_id,outdir):
    bundle=Path(bundle).resolve();outdir=Path(outdir)
    if Path(__file__).resolve()!=(bundle/'source/production/worker.py').resolve():
        raise ValueError('Execute the worker from inside its frozen bundle')
    manifest,protocol=validate_bundle(bundle)
    if not 0<=worker_id<protocol['worker_count']:raise ValueError('Invalid worker ID')
    outdir.mkdir(parents=True,exist_ok=False)
    g=protocol['generation'];plans=json.loads((bundle/'inputs/plans.json').read_text())
    generation_plan=next(p for p in plans['plans'] if p['name']==protocol['generation_plan'])
    reference_plan=next(p for p in plans['plans'] if p['name']==protocol['reference_plan'])
    free=g['free_vertices'];n=g['n'];D=2.**(n-len(free))
    graph=np.load(bundle/'inputs/injected_graph.npz');ring=list(map(int,graph['ring']))
    assigned=[r for r in g['records'] if r['sample_id']%protocol['worker_count']==worker_id]
    audit_ids=protocol['audit']['ids_by_worker'][str(worker_id)]
    anchor=protocol['anchor']
    if not anchor or len(anchor.get('input_data_bits',''))!=n:raise ValueError('Missing or invalid frozen pilot anchor')
    identity={'worker_id':worker_id,'worker_count':protocol['worker_count'],'synthetic_only':manifest.get('synthetic_only',False),
              'manifest_sha256':file_sha256(bundle/'manifest.json'),'protocol_sha256':file_sha256(bundle/'protocol.json'),
              'sample_ids':[r['sample_id'] for r in assigned],'audit_ids':audit_ids,'engine':protocol['engine']}
    write_seconds=write_json(outdir/'source.json',identity)
    import cupy as cp
    cp.cuda.Device(0).use();hw=cp.cuda.runtime.getDeviceProperties(0)
    try:
        smi=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],
                           capture_output=True,text=True,timeout=10,check=False)
        smi_evidence={'returncode':smi.returncode,'stdout':smi.stdout,'stderr':smi.stderr}
    except (OSError,subprocess.TimeoutExpired) as exc:smi_evidence={'error':str(exc)}
    try:
        from cupy.cuda import nvrtc
        nvrtc_version=list(map(int,nvrtc.getVersion()))
    except (ImportError,AttributeError,RuntimeError):nvrtc_version=None
    hardware={'device_name':hw['name'].decode(),
        'total_bytes':int(hw['totalGlobalMem']),'compute_capability':[int(hw['major']),int(hw['minor'])],
        'numpy':np.__version__,'cupy':cp.__version__,'visible_device_count':int(cp.cuda.runtime.getDeviceCount()),
        'selected_device_index':0,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_driver_version':int(cp.cuda.runtime.driverGetVersion()),
        'cuda_runtime_version':int(cp.cuda.runtime.runtimeGetVersion()),'nvrtc_version':nvrtc_version,
        'nvidia_smi_allocation_evidence':smi_evidence}
    write_seconds+=write_json(outdir/'hardware.json',hardware)
    for got,want in [('device_name','expected_device_name'),('numpy','expected_numpy_version'),('cupy','expected_cupy_version')]:
        if hardware[got]!=anchor[want]:raise ValueError('Worker software/device differs from pilot anchor: '+got)
    if hardware['visible_device_count']!=protocol['resources']['gpus_per_worker']:
        raise ValueError('Visible GPU count differs from frozen per-worker configuration')
    selected={};samples={};sample_write_seconds=0.;generation_contraction_seconds=0.;cdf_seconds=0.
    phase='setup_generation';phase_start=time.perf_counter()
    try:
        cls=HalfConditionalBlockGPU if protocol['engine']=='half' else InplaceConditionalBlockGPU
        engine=cls(graph['graph'],graph['unitaries'],n,graph['order'],generation_plan['operations'],free)
        engine.alloc(dtype=np.complex64)
        generation_setup_seconds=time.perf_counter()-phase_start
        if engine.inplace_rmax!=generation_plan['rmax']:raise ValueError('Compiled generation rank mismatch')
        phase='fixed_anchor';phase_start=time.perf_counter()
        v,anchor_contraction=engine.block(np.asarray(list(map(int,anchor['input_data_bits'])),np.uint8))
        anchor_host=cp.asnumpy(v);del v
        anchor_hash=hashlib.sha256(anchor_host.astype('<c16',copy=False).tobytes()).hexdigest()
        anchor_result={'worker_id':worker_id,'input_data_bits':anchor['input_data_bits'],
            'expected_amplitudes_sha256':anchor['expected_amplitudes_sha256'],'actual_amplitudes_sha256':anchor_hash,
            'passed':anchor_hash==anchor['expected_amplitudes_sha256'],'contraction':anchor_contraction,
            'device_name':hardware['device_name'],'numpy':hardware['numpy'],'cupy':hardware['cupy'],
            'cuda_driver_version':hardware['cuda_driver_version'],'cuda_runtime_version':hardware['cuda_runtime_version'],
            'nvrtc_version':hardware['nvrtc_version'],
            'scope':'Fixed block outside 1389 production outputs and 64 audit IDs; no production RNG consumed'}
        write_seconds+=write_json(outdir/'anchor.json',anchor_result);del anchor_host
        anchor_seconds=time.perf_counter()-phase_start
        if not anchor_result['passed']:raise ArithmeticError('Frozen full-vector anchor hash mismatch; no retry')
        phase='generation';phase_start=time.perf_counter()
        with (outdir/'samples.jsonl').open('x') as stream:
            for record in assigned:
                if not record['conditional_branch']:raise ValueError('This worker requires lambda=1')
                begin=time.perf_counter();sid=record['sample_id'];x=np.asarray(list(map(int,record['input_data_bits'])),np.uint8)
                v,contraction=engine.block(x);host=cp.asnumpy(v);del v
                generation_contraction_seconds+=contraction['seconds']
                t=time.perf_counter();draw=draw_checked(host,record['draw_uniform']);draw_seconds=time.perf_counter()-t
                cdf_seconds+=draw_seconds
                bits,qiskit=assemble_output(record['input_data_bits'],free,draw['A_index'],ring)
                grid_count=math.ceil(draw['cdf_upper_numpy']*2.**53)-math.ceil(draw['cdf_lower_numpy']*2.**53)
                row={**record,'worker_id':worker_id,'manifest_sha256':identity['manifest_sha256'],
                     'protocol_sha256':identity['protocol_sha256'],'is_preselected_audit':sid in audit_ids,
                     'draw':draw,'contraction':contraction,'cdf_seconds':draw_seconds,
                     'data_bits_ring_order':bits,'bitstring_qiskit':qiskit,
                     'approximate_circuit_probability_estimate':draw['selected_scaled_weight']/D,
                     'actual_53bit_conditional_probability':grid_count/2.**53,
                     'actual_joint_sampler_probability':grid_count/2.**53/D,
                     'probability_scope':'Circuit estimate uses approximate amplitudes; sampler probability includes uniform prefix and the finite-grid CDF law.'}
                if sid in audit_ids:
                    selected[sid]=host
                    artifact,elapsed=save_vector(outdir/f'approx_{sid:06d}.npy',host)
                    row['approximate_vector_file']=artifact;row['approximate_vector_write_seconds']=elapsed
                    sample_write_seconds+=elapsed;write_seconds+=elapsed
                row['sample_seconds_before_json_write']=time.perf_counter()-begin
                elapsed=append_row(stream,row);sample_write_seconds+=elapsed;write_seconds+=elapsed
                samples[sid]=row
                print(json.dumps({'phase':phase,'worker_id':worker_id,'sample_id':sid,
                                  'sample_seconds':time.perf_counter()-begin}),flush=True)
                del host
        generation_seconds=time.perf_counter()-phase_start
        if len(samples)!=len(assigned) or set(selected)!=set(audit_ids):raise AssertionError('Generation count mismatch')
        del engine;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        phase='setup_reference';phase_start=time.perf_counter()
        reference=InplaceConditionalBlockGPU(graph['graph'],graph['unitaries'],n,graph['order'],reference_plan['operations'],free)
        reference.alloc(dtype=np.complex128)
        reference_setup_seconds=time.perf_counter()-phase_start
        if reference.inplace_rmax!=reference_plan['rmax']:raise ValueError('Compiled reference rank mismatch')
        phase='audit';phase_start=time.perf_counter();audit_contraction_seconds=0.;audit_compare_seconds=0.;audit_write_seconds=0.;audits=[]
        with (outdir/'audits.jsonl').open('x') as stream:
            for sid in audit_ids:
                begin=time.perf_counter();sample=samples[sid];approx=selected[sid]
                x=np.asarray(list(map(int,sample['input_data_bits'])),np.uint8)
                v,contraction=reference.block(x);ref=cp.asnumpy(v);del v
                audit_contraction_seconds+=contraction['seconds'];t=time.perf_counter()
                if hashlib.sha256(approx.astype('<c16',copy=False).tobytes()).hexdigest()!=sample['draw']['amplitudes_sha256']:
                    raise AssertionError('Actual sampled approximate vector changed')
                law,lawmeta=actual_cdf_law_vector(approx)
                if lawmeta['source_weights_sha256']!=sample['draw']['weights_sha256']:raise AssertionError('Audit weights changed')
                witness=audit_vector_pair(ref,law,sid);raw_witness=audit_vector_pair(ref,approx,sid)
                index=sample['draw']['A_index'];p_ref=float(abs(ref[index])**2/D)
                compare_elapsed=time.perf_counter()-t;audit_compare_seconds+=compare_elapsed
                artifact,elapsed=save_vector(outdir/f'reference_{sid:06d}.npy',ref)
                audit_write_seconds+=elapsed;write_seconds+=elapsed
                row={'audit_id':sid,'worker_id':worker_id,'manifest_sha256':identity['manifest_sha256'],
                     'protocol_sha256':identity['protocol_sha256'],'witness':witness,'raw_vector_witness':raw_witness,
                     'actual_cdf_law':lawmeta,'approximate_vector_file':sample['approximate_vector_file'],
                     'reference_vector_file':artifact,'reference_contraction':contraction,'A_index':index,
                     'reference_selected_probability':p_ref,'reference_selected_linear_xeb':float(2.**n*p_ref-1),
                     'approximate_selected_probability_estimate':sample['approximate_circuit_probability_estimate'],
                     'compare_seconds':compare_elapsed,'reference_vector_write_seconds':elapsed,
                     'audit_seconds_before_json_write':time.perf_counter()-begin,
                     'scope':'Preselected full-vector audit; XEB shown only for this audited subset; FP64 reference is numerical.'}
                elapsed=append_row(stream,row);audit_write_seconds+=elapsed;write_seconds+=elapsed;audits.append(row)
                print(json.dumps({'phase':phase,'worker_id':worker_id,'audit_id':sid,
                                  'audit_seconds':time.perf_counter()-begin,'loss':witness['loss']}),flush=True)
                del selected[sid],approx,ref,law
        audit_seconds=time.perf_counter()-phase_start
        del reference;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        result={'completed':True,**identity,'sample_count':len(samples),'audit_count':len(audits),
                'generation_setup_seconds':generation_setup_seconds,'reference_setup_seconds':reference_setup_seconds,
                'anchor_seconds':anchor_seconds,'anchor_amplitudes_sha256':anchor_hash,
                'generation_seconds':generation_seconds,'generation_contraction_seconds':generation_contraction_seconds,
                'generation_cdf_seconds':cdf_seconds,'generation_write_seconds':sample_write_seconds,
                'audit_seconds':audit_seconds,'audit_contraction_seconds':audit_contraction_seconds,
                'audit_compare_seconds':audit_compare_seconds,'audit_write_seconds':audit_write_seconds,
                'durable_write_seconds_before_summary':write_seconds,
                'worker_seconds_to_summary_build':time.perf_counter()-PROCESS_START,
                'process_start_unix':PROCESS_UNIX_START,'summary_build_unix':time.time(),
                'samples_sha256':file_sha256(outdir/'samples.jsonl'),'audits_sha256':file_sha256(outdir/'audits.jsonl'),
                'timing_scope':'Worker phases include durable writes and checks. Central deployment/queue/collection are recorded by the launcher separately.'}
        summary_write=write_json(outdir/'summary.json',result)
        artifact_manifest={'worker_id':worker_id,'files':{p.name:{'bytes':p.stat().st_size,'sha256':file_sha256(p)}
            for p in sorted(outdir.iterdir()) if p.is_file()},'scope':'Durable artifacts after generation and all fixed audits'}
        artifact_manifest_write=write_json(outdir/'artifacts.json',artifact_manifest)
        marker={'completed':True,'worker_id':worker_id,'worker_seconds_to_complete_marker':time.perf_counter()-PROCESS_START,
                'summary_write_seconds':summary_write,'artifact_manifest_write_seconds':artifact_manifest_write,
                'end_unix':time.time(),'summary_sha256':file_sha256(outdir/'summary.json'),
                'artifacts_sha256':file_sha256(outdir/'artifacts.json')}
        write_json(outdir/'completed.json',marker);print(json.dumps(marker),flush=True)
        return result
    except BaseException:
        write_json(outdir/'failure.json',{'completed':False,'phase':phase,'worker_id':worker_id,
                   'samples_preserved':len(samples),'worker_seconds':time.perf_counter()-PROCESS_START,
                   'traceback':traceback.format_exc(),'policy':'No resampling, output substitution or silent retry occurred.'})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--worker-id',type=int,required=True)
    p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();execute(a.bundle,a.worker_id,a.outdir)

if __name__=='__main__':main()
