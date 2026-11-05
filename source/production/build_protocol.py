"""Create a reviewable frozen source/input bundle from explicitly supplied seeds.

No seed is selected automatically and building a bundle does not execute it.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from protocol_common import make_run_protocol,file_sha256,validate_bundle

HERE=Path(__file__).resolve().parent


def build_bundle(out,seed,audit_seed,engine,count=1389,workers=8,audits_per_worker=8,anchor=None):
    if not anchor or anchor.get('engine')!=engine:raise ValueError('Supply the selected engine pilot anchor specification')
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    source=root/'source';inputs=root/'inputs';inputs.mkdir()
    dcs=HERE.parent;upstream=dcs.parent/'dcs-rankwidth'
    groups={'production':list(HERE.glob('*.py')),
            'sampling':[dcs/'sampling'/name for name in ['conditional_block_gpu.py','conditional_block_inplace_gpu.py',
                        'conditional_block_half_gpu.py','sampling_protocol.py']],
            'kernel_opt':[dcs/'kernel_opt'/name for name in ['inplace_plan.py','inplace_gpu.py','inplace_half_gpu.py']],
            'review':[dcs/'protocol/review'/name for name in ['full_vector_audit_certificate.py','conditional_mixture_certificate.py']],
            'provenance':[dcs/'protocol/verify_graph_provenance.py'],
            'dcs_rankwidth':list((upstream/'dcs_rankwidth').glob('*.py'))}
    for group,files in groups.items():
        target=source/group;target.mkdir(parents=True)
        for path in files:shutil.copy2(path,target/path.name)
    for name in ['LICENSE','NOTICE']:
        if (upstream/name).exists():shutil.copy2(upstream/name,source/name)
    shutil.copy2(dcs/'sampling/injected_graph.npz',inputs/'injected_graph.npz')
    shutil.copy2(dcs/'sampling/consolidated_pilot_plans.json',inputs/'plans.json')
    shutil.copy2(dcs.parent/'rgs_attack/dcs245_qasm/dcs_nq64_depth73_checks12_t314.qasm',inputs/'circuit.qasm')
    target_qasm_hash='7a0b8aa6bcef23b901720cb21c9b05cc5de42d4b0e872db4606eec03766f73e5'
    if file_sha256(inputs/'circuit.qasm')!=target_qasm_hash:raise ValueError('Target QASM hash mismatch')
    metadata=json.loads((dcs/'protocol/zenodo_22233383.json').read_text())
    shutil.copy2(dcs/'protocol/zenodo_22233383.json',inputs/'dataset_metadata.json')
    provenance={'circuit_file':'circuit.qasm','source_url':'https://zenodo.org/records/22233383',
        'original_archive_member':'v2/loop64_314t.qasm','QAT_issue':'https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/245',
        'sha256':target_qasm_hash,'dataset_title':metadata['metadata'].get('title'),
        'creators':metadata['metadata'].get('creators',[]),'data_license_as_declared_by_Zenodo':metadata['metadata'].get('license','unknown'),
        'license_evidence':'dataset_metadata.json, frozen snapshot of the source record',
        'license_scope':'Dataset license is separate from the Apache-2.0 license of upstream dcs-rankwidth code.'}
    (inputs/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (inputs/'NOTICE.txt').write_text('Input dataset: Dataset for Sampling hard circuits with verifiably high fidelity\n'
        'Source: https://zenodo.org/records/22233383\n'
        'License declared by the source metadata: Creative Commons Attribution 4.0 (CC-BY-4.0).\n'
        'License URL: https://creativecommons.org/licenses/by/4.0/\n'
        'Original member: v2/loop64_314t.qasm; copied without changes.\n'
        'Creators and frozen metadata evidence: PROVENANCE.json and dataset_metadata.json.\n'
        'The Apache-2.0 license copied for dcs-rankwidth applies to code, not this dataset.\n')
    evidence=json.loads((dcs/'protocol/graph_provenance_validation.json').read_text())
    if not evidence['passed'] or evidence['qasm_sha256']!=target_qasm_hash or evidence['graph_npz_sha256']!=file_sha256(inputs/'injected_graph.npz'):
        raise ValueError('Independent graph provenance evidence does not bind these inputs')
    shutil.copy2(dcs/'protocol/graph_provenance_validation.json',inputs/'graph_provenance_validation.json')
    plans=json.loads((inputs/'plans.json').read_text());graph=np.load(inputs/'injected_graph.npz')
    protocol=make_run_protocol(seed,audit_seed,count,workers,audits_per_worker,len(graph['ring']),plans['open_vertices'],engine,anchor)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    manifest={'schema_version':'dcs245-bundle-v1','target_qasm_sha256':target_qasm_hash,
              'n':len(graph['ring']),'injected_vertices':len(graph['graph']),
              'files':{str(p.relative_to(root)):file_sha256(p) for p in sorted(root.rglob('*')) if p.is_file()},
              'scope':'Frozen code/inputs/protocol only; no run or result implied'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_bundle(root)
    return {'bundle':str(root),'manifest_sha256':file_sha256(root/'manifest.json'),
            'protocol_sha256':file_sha256(root/'protocol.json'),'status':'not_run'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--audit-seed',type=int,required=True)
    p.add_argument('--engine',choices=['fp32','half'],required=True)
    p.add_argument('--anchor-json',type=Path,required=True)
    p.add_argument('--count',type=int,default=1389);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--audits-per-worker',type=int,default=8);a=p.parse_args()
    print(json.dumps(build_bundle(a.outdir,a.seed,a.audit_seed,a.engine,a.count,a.workers,a.audits_per_worker,
                                 json.loads(a.anchor_json.read_text())),indent=2))

if __name__=='__main__':main()
