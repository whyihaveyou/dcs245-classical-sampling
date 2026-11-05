"""Run REAL worker bookkeeping with a declared fake numerical/GPU backend.

All artifacts are explicitly synthetic. This exercises counts, retention,
durable files, audit/CDF integration and phase timing without using a GPU.
"""
import argparse,importlib.util,json,sys,time,types
from pathlib import Path
import numpy as np
from build_protocol import build_bundle
from protocol_common import make_run_protocol,file_sha256


class SyntheticEngine:
    def __init__(self,graph,unitaries,n,order,operations,free):
        self.n=n;self.free=list(free);self.closed=[q for q in range(n) if q not in free]
        self.operations=operations or []
        self.inplace_rmax=len(free);self.inplace_maxslots=len(free)
    def alloc(self,dtype=np.complex64,**kwargs):self.dtype=dtype
    def block(self,x):
        start=time.perf_counter();prefix=sum(int(x[q])<<j for j,q in enumerate(self.closed));j=np.arange(1<<len(self.free))
        weights=1+.2*np.cos(j+.3*prefix);weights/=weights.sum()
        mass=1+.01*np.cos(2*np.pi*prefix/(1<<len(self.closed)))
        v=np.sqrt(mass*weights)*np.exp(1j*(.13*j+.2*prefix))
        v=v.astype(self.dtype).astype(np.complex128)
        return v,{'seconds':time.perf_counter()-start,'prefix_dtype':np.dtype(self.dtype).name,
                  'rmax':self.inplace_rmax,'open_vertices':self.free,'part':0,'parts':1,
                  'slices_evaluated':1<<len(self.operations),'terminal_rank':len(self.free),'work_per_slice':0,
                  'vector_scale':'sqrt(2^(n-k)); sum all complex parts before probabilities',
                  'scope':'SYNTHETIC FAKE BACKEND; NOT CIRCUIT AMPLITUDES'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    out=a.outdir;out.mkdir(parents=True,exist_ok=False);bundle=out/'bundle';artifacts=out/'artifacts';artifacts.mkdir()
    n,free=6,[5,1,3];fake=SyntheticEngine(None,None,n,None,None,free);fake.alloc()
    anchor_vector,_=fake.block(np.zeros(n,np.uint8))
    import hashlib
    anchor={'engine':'fp32','source_plan':'inplace_k2_fp32','source_prefix_index':0,'input_data_bits':'0'*n,
            'expected_amplitudes_sha256':hashlib.sha256(anchor_vector.astype('<c16').tobytes()).hexdigest(),
            'expected_device_name':'SYNTHETIC DEVICE; NO GPU USED','expected_numpy_version':np.__version__,
            'expected_cupy_version':'SYNTHETIC NO CUPY','scope':'SYNTHETIC ANCHOR, NO EXPERIMENT'}
    build_bundle(bundle,7324501,7324502,'fp32',24,2,2,anchor)
    protocol=make_run_protocol(7324501,7324502,24,2,2,n,free,'fp32',anchor)
    (bundle/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    np.savez(bundle/'inputs/injected_graph.npz',graph=np.zeros((n,n),np.uint8),unitaries=np.asarray([np.eye(2)]*n),
             order=np.arange(n),ring=np.arange(n))
    plans={'open_vertices':free,'plans':[{'name':name,'operations':[],'rmax':3} for name in ['inplace_k2_fp32','inplace_k3_fp64']]}
    (bundle/'inputs/plans.json').write_text(json.dumps(plans,indent=2)+'\n')
    manifest=json.loads((bundle/'manifest.json').read_text());manifest.update({'synthetic_only':True,'n':n,'injected_vertices':n,
        'target_qasm_sha256':None,'scope':'SYNTHETIC TEST BUNDLE: no actual quantum circuit run'})
    manifest['files']={str(p.relative_to(bundle)):file_sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cp=types.ModuleType('cupy');cp.__version__='SYNTHETIC NO CUPY';cp.asnumpy=np.asarray
    cuda=types.ModuleType('cupy.cuda');cuda.Device=lambda index:types.SimpleNamespace(use=lambda:None)
    cuda.runtime=types.SimpleNamespace(getDeviceProperties=lambda index:{'name':b'SYNTHETIC DEVICE; NO GPU USED',
        'totalGlobalMem':0,'major':0,'minor':0},getDeviceCount=lambda:1,driverGetVersion=lambda:0,runtimeGetVersion=lambda:0)
    cuda.nvrtc=types.SimpleNamespace(getVersion=lambda:(0,0));cp.cuda=cuda
    cp.get_default_memory_pool=lambda:types.SimpleNamespace(free_all_blocks=lambda:None)
    sys.modules['cupy']=cp;sys.modules['cupy.cuda']=cuda
    for worker_id in range(2):
        name=f'synthetic_frozen_worker_{worker_id}'
        spec=importlib.util.spec_from_file_location(name,bundle/'source/production/worker.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.InplaceConditionalBlockGPU=SyntheticEngine;module.HalfConditionalBlockGPU=SyntheticEngine
        module.execute(bundle,worker_id,artifacts/f'worker{worker_id}')
    receipt={'synthetic_only':True,'GPU_experiments_run':0,'scope':'Fake numerical/GPU backend; real worker bookkeeping and audit integration',
             'bundle':str(bundle.resolve()),'artifacts':str(artifacts.resolve()),'sample_count':24,'audit_count':4}
    (out/'SYNTHETIC_ONLY.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
