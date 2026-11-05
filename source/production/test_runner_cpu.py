"""Synthetic early failure/empty success tests of the standard-library runner."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent

def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix='dcs245_runner_synthetic_') as temp:
        root=Path(temp);bundle=root/'bundle';entry=bundle/'source/production/worker.py';entry.parent.mkdir(parents=True)
        for i,source in enumerate(["raise RuntimeError('SYNTHETIC intentional pre-setup failure')\n","print('SYNTHETIC empty success')\n"]):
            entry.write_text(source);out=root/f'case{i}'
            cmd=[sys.executable,str(HERE/'run_worker.py'),'--bundle',str(bundle),'--worker-id','0','--outdir',str(out)]
            p=subprocess.run(cmd,capture_output=True,text=True)
            receipt=json.loads((out/'runner_exit.json').read_text())
            manifest=json.loads((out/'transport_artifacts.json').read_text())
            assert p.returncode!=0 and receipt['returncode']!=0 and receipt['worker_completed'] is False
            assert receipt['worker_log_sha256']==hashlib.sha256((out/'worker.log').read_bytes()).hexdigest()
            assert {'worker.log','runner_exit.json'}<=set(manifest['files'])
            before=(out/'runner_exit.json').read_bytes();retry=subprocess.run(cmd,capture_output=True,text=True)
            assert retry.returncode!=0 and (out/'runner_exit.json').read_bytes()==before
            results.append({'case':i,'expected_failure_preserved':True,'overwrite_rejected':True})
    result={'passed':True,'scope':'SYNTHETIC runner tests; no GPU or real production executed','cases':results}
    (HERE/'runner_cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
