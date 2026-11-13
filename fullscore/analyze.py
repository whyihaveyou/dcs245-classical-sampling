"""Independent CPU verification and fixed-bootstrap statistics for full r1 scores."""
import argparse,base64,csv,gzip,hashlib,json,math,pathlib,re
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--logs',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
base=pathlib.Path(__file__).resolve().parent;src=base/'source';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((src/'scoring_manifest.json').read_text())
for f,digest in manifest['files'].items():assert h(src/f)==digest,f
samples=list(map(json.loads,(src/'r1_sample_records.jsonl').read_text().splitlines()))
anchors=json.loads((src/'audit_anchors.json').read_text());protocol=json.loads((src/'bundle/protocol.json').read_text());free=protocol['generation']['free_vertices'];g=np.load(src/'bundle/inputs/injected_graph.npz');ring=list(map(int,g['ring']))
ledger=json.loads((a.logs/'central_timing.json').read_text());assert not ledger['submission_errors'];assert len(ledger['terminal'])==8
for v in ledger['terminal'].values():assert v['state']=='Succeeded' and v['logs_returncode']==0
rows=[];workers=[]
for w in range(8):
 text=(a.logs/f'logs{w}.txt').read_text()
 # rjob splits long stdout lines at 16 KiB, adding a log prefix to each piece.
 # Reassemble only contiguous pure-base64 continuation records; gzip CRC then verifies completeness.
 lines=[line.split(' >> ',1)[-1] for line in text.splitlines()]
 payloads=[]
 for i,line in enumerate(lines):
  marker='SCORE_ARCHIVE:'+str(w)+':'
  if line.startswith(marker):
   payload=line[len(marker):];j=i+1
   while j<len(lines) and re.fullmatch(r'[A-Za-z0-9+/=]+',lines[j]):
    payload+=lines[j];j+=1
   payloads.append(payload)
 assert len(payloads)==1,(w,len(payloads))
 result=json.loads(gzip.decompress(base64.b64decode(payloads[0],validate=True)))
 assert result['worker_id']==w and not result['smoke']
 assert result['manifest_sha256']==h(src/'scoring_manifest.json')
 assert result['sample_list_sha256']==h(src/'r1_samples.txt')
 assert result['sample_count']==len(result['rows'])==len(range(w,1389,8))
 assert [r['sample_id'] for r in result['rows']]==list(range(w,1389,8))
 checkpoint=[json.loads(x) for x in re.findall(r'SCORE_ROW:(\{[^\n]+\})',text)]
 assert checkpoint==result['rows']
 rows+=result.pop('rows');workers.append(result)
rows.sort(key=lambda r:r['sample_id']);assert [r['sample_id'] for r in rows]==list(range(1389));maxerr=0;anchor_count=0
for r,old in zip(rows,samples):
 sid=r['sample_id'];assert r['bitstring_qiskit']==old['bitstring_qiskit']
 physical=list(map(int,r['bitstring_qiskit'][::-1]));logical=[physical[ring[q]] for q in range(64)]
 assert ''.join(map(str,logical))==r['logical_bits']==old['data_bits_ring_order']
 index=sum(logical[q]<<j for j,q in enumerate(free));assert index==r['A_index']
 probability=(r['amplitude_real']**2+r['amplitude_imag']**2)/2**44
 assert math.isfinite(probability) and probability>=0
 assert math.isclose(probability,r['probability_fp64'],rel_tol=1e-14,abs_tol=0)
 assert math.isclose(2**64*probability-1,r['linear_xeb_point'],rel_tol=1e-12,abs_tol=1e-12)
 if str(sid) in anchors:
  e=abs(probability/anchors[str(sid)]['probability_fp64']-1);maxerr=max(maxerr,e);anchor_count+=1;assert e<1e-8
assert anchor_count==64
ibm_path=base.parent/'protocol/zenodo_22882048/zenodo_experiment1_data/experiment1_probabilities.csv'
ibm_rows=list(csv.DictReader(ibm_path.open()));assert len(ibm_rows)==1389
ibm_bits=(base.parent/'protocol/samples_314t_exp1.txt').read_text().splitlines();assert [r['x'] for r in ibm_rows]==ibm_bits
x=np.array([2**64*r['probability_fp64']-1 for r in rows]);y=np.array([2**64*float(r['p'])-1 for r in ibm_rows])
rng=np.random.default_rng(20261113);bootx=[];booty=[]
for i in range(200):
 bootx.extend(x[rng.integers(len(x),size=(100,len(x)))].mean(axis=1));booty.extend(y[rng.integers(len(y),size=(100,len(y)))].mean(axis=1))
def stats(z,b):
 return {'count':len(z),'linear_xeb':float(z.mean()),'standard_error_iid':float(z.std(ddof=1)/np.sqrt(len(z))),'bootstrap95_percentile_iid':list(map(float,np.quantile(b,[.025,.975]))),'minimum_point':float(z.min()),'maximum_point':float(z.max())}
approx=np.array([r['approximate_circuit_probability_estimate'] for r in samples]);reference=np.array([r['probability_fp64'] for r in rows]);rel=np.abs(approx/reference-1)
assert np.all(np.isfinite(rel))
summary={'analysis_numpy_version':np.__version__,'selected_point_FP32_vs_FP64':{'max_relative_error':float(rel.max()),'median_relative_error':float(np.median(rel)),'p99_relative_error':float(np.quantile(rel,.99)),'count_above_1e_4':int((rel>1e-4).sum()),'scope':'All selected points only; not a total-variation bound for the whole distribution.'},'validation_passed':True,'count':len(rows),'anchor_count':anchor_count,'max_anchor_relative_probability_error':maxerr,'classical_r1':stats(x,bootx),'IBM_hardware':stats(y,booty),'difference':{'linear_xeb':float(x.mean()-y.mean()),'standard_error_independent_iid':float(np.sqrt(x.var(ddof=1)/len(x)+y.var(ddof=1)/len(y))),'bootstrap95_independent_iid':list(map(float,np.quantile(np.array(bootx)-np.array(booty),[.025,.975])))},'bootstrap':{'repetitions':20000,'seed':20261113,'method':'percentile, sample-level iid resampling; hardware temporal correlations not modeled'},'workers':workers,'GPU_hours_scoring_process_sum':sum(w['score_seconds'] for w in workers)/3600,'max_worker_scoring_seconds':max(w['score_seconds'] for w in workers),'central_timing':ledger,'scope':'complex128 numerical reference with frozen r1 full contraction plan, separate recomputation from FP32 generation. Same graph and core contraction family; not a separate algorithm or interval proof. Score validation costs must not be called sampling runtime.','hashes':{'scoring_manifest':h(src/'scoring_manifest.json'),'r1_sample_list':h(src/'r1_samples.txt'),'ibm_csv':h(ibm_path)}}
a.out.mkdir(exist_ok=False,parents=True)
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.out/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
with (a.out/'scores.csv').open('w') as f:
 keys=['sample_id','bitstring_qiskit','A_index','probability_fp64','linear_xeb_point','wall_seconds'];cw=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');cw.writeheader();cw.writerows(rows)
print(json.dumps({k:summary[k] for k in ['validation_passed','count','max_anchor_relative_probability_error','classical_r1','IBM_hardware','difference','GPU_hours_scoring_process_sum','max_worker_scoring_seconds']},indent=2))
