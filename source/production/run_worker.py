"""Standard-library launcher preserving logs/exit evidence even before worker setup."""
import argparse,hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def durable_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--worker-id',type=int,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    if a.outdir.exists():raise FileExistsError('Worker output exists; no silent retry/overwrite')
    a.outdir.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();start_unix=time.time();child=None;error=None;code=1
    fd,log_name=tempfile.mkstemp(prefix=f'.worker{a.worker_id}_',suffix='.log',dir=a.outdir.parent)
    try:
        with os.fdopen(fd,'w') as log:
            try:
                child=subprocess.Popen([sys.executable,str(a.bundle/'source/production/worker.py'),
                    '--bundle',str(a.bundle),'--worker-id',str(a.worker_id),'--outdir',str(a.outdir)],
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in child.stdout:log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
                code=child.wait()
            except BaseException as exc:
                error=repr(exc);log.write(error+'\n');log.flush()
                if child is not None and child.poll() is None:child.terminate();code=child.wait()
            log.flush();os.fsync(log.fileno())
    finally:
        a.outdir.mkdir(parents=True,exist_ok=True)
        os.replace(log_name,a.outdir/'worker.log')
        child_code=code
        if code==0 and not (a.outdir/'completed.json').exists():
            code=70;error='Child exited zero without the required completed marker'
        receipt={'worker_id':a.worker_id,'returncode':code,'child_returncode':child_code,'runner_error':error,'start_unix':start_unix,
                 'end_unix':time.time(),'runner_monotonic_seconds_before_receipt':time.perf_counter()-start,
                 'worker_log_sha256':digest(a.outdir/'worker.log'),'worker_completed':(a.outdir/'completed.json').exists(),
                 'scope':'Includes child execution and log preservation; central scheduling/upload/collection are separate.'}
        durable_json(a.outdir/'runner_exit.json',receipt)
        files={f.name:{'bytes':f.stat().st_size,'sha256':digest(f)} for f in sorted(a.outdir.iterdir()) if f.is_file()}
        durable_json(a.outdir/'transport_artifacts.json',{'worker_id':a.worker_id,'files':files,'worker_returncode':code})
        print(json.dumps({'runner_receipt':receipt,'transport_manifest_sha256':digest(a.outdir/'transport_artifacts.json')}),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
