"""Run one frozen DCS245 worker and upload exactly its durable artifacts.

The production worker remains the scientific source of truth.  This wrapper
only adds a bounded, manifest-bound transport step after generation and audit
completion.  It never redraws, filters, overwrites, or regenerates outputs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import hashlib


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent.parent
TRANSPORT_PARENT = BUNDLE / "source"


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, check=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--allowlist", type=Path, required=True)
    p.add_argument("--receiver-url", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--token-file")
    g.add_argument("--token-env")
    args = p.parse_args()
    if args.outdir.exists() or args.receipt.exists():
        raise SystemExit("refusing to overwrite an existing worker output or receipt")
    if args.bundle.resolve() != BUNDLE.resolve():
        raise SystemExit("wrapper must be executed from its frozen bundle")
    # Bind transport to the allowlist frozen inside this exact bundle.  An
    # operator cannot substitute a broader/alternate allowlist at launch.
    frozen_allowlist = (BUNDLE / "inputs/transport_allowlist.json").resolve()
    if args.allowlist.resolve() != frozen_allowlist:
        raise SystemExit("allowlist must be the bundle's frozen transport_allowlist.json")
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    expected_allowlist_hash = manifest.get("files", {}).get("inputs/transport_allowlist.json")
    if not expected_allowlist_hash or sha256(frozen_allowlist) != expected_allowlist_hash:
        raise SystemExit("bundle transport allowlist hash does not match manifest")
    receipt_parent = args.receipt.parent.resolve()
    if receipt_parent.exists() and receipt_parent.is_symlink():
        raise SystemExit("receipt parent cannot be a symlink")
    receipt_parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    start = time.perf_counter()
    runner = [sys.executable, str(args.bundle / "source/production/run_worker.py"),
              "--bundle", str(args.bundle), "--worker-id", str(args.worker_id),
              "--outdir", str(args.outdir)]
    result = run(runner)
    worker_returncode = int(result.returncode)

    # Verify the durable chain before transmitting a successful worker.  A
    # failed worker is still uploaded with present_only evidence for diagnosis.
    chain_out = receipt_parent / f"worker{args.worker_id}_chain.json"
    chain_cmd = [sys.executable, "-m", "transport.verify_worker_chain",
                 "--worker-dir", str(args.outdir), "--out", str(chain_out)]
    if worker_returncode == 0:
        chain_cmd.append("--require-success")
    chain_rc = run(chain_cmd, env={**os.environ, "PYTHONPATH": str(TRANSPORT_PARENT)})
    if worker_returncode == 0 and chain_rc.returncode != 0:
        raise SystemExit("successful worker failed its durability-chain check")

    manifest_path = receipt_parent / f"worker{args.worker_id}_transfer_manifest.json"
    build_cmd = [sys.executable, "-m", "transport.build_manifest",
                 "--allowlist", str(args.allowlist), "--source-dir", str(args.outdir),
                 "--prefix", f"worker{args.worker_id}_", "--out", str(manifest_path)]
    if worker_returncode != 0:
        build_cmd.append("--present-only")
    build_rc = run(build_cmd, env={**os.environ, "PYTHONPATH": str(TRANSPORT_PARENT)})
    if build_rc.returncode != 0:
        raise SystemExit("could not freeze worker transport manifest")

    token_args = ["--token-file", args.token_file] if args.token_file else ["--token-env", args.token_env]
    upload_cmd = [sys.executable, "-m", "transport.upload", "--url", args.receiver_url,
                  "--manifest", str(manifest_path), "--source-dir", str(args.outdir),
                  "--receipt", str(args.receipt), "--strip-prefix", f"worker{args.worker_id}_",
                  *token_args]
    upload_rc = run(upload_cmd, env={**os.environ, "PYTHONPATH": str(TRANSPORT_PARENT)})
    record = {
        "worker_id": args.worker_id,
        "worker_returncode": worker_returncode,
        "chain_returncode": int(chain_rc.returncode),
        "manifest_returncode": int(build_rc.returncode),
        "upload_returncode": int(upload_rc.returncode),
        "elapsed_seconds": time.perf_counter() - start,
        "scope": "Frozen worker execution, durability check, manifest freeze and private receiver upload.",
    }
    record_path = receipt_parent / f"worker{args.worker_id}_upload_summary.json"
    if record_path.exists() or record_path.is_symlink():
        raise SystemExit("refusing to overwrite upload summary")
    with record_path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Preserve the scientific failure status.  A transport failure is also a
    # failure, since an unacknowledged artifact cannot count as collected.
    raise SystemExit(worker_returncode if worker_returncode else upload_rc.returncode)


if __name__ == "__main__":
    main()
