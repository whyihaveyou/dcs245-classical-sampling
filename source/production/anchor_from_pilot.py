"""Extract a chosen engine's fixed full-vector anchor from a successful pilot."""
import argparse,json
from pathlib import Path

def anchor_from_pilot(summary,prefixes,engine):
    if not summary['passed']:raise ValueError('Pilot did not pass')
    name={'fp32':'inplace_k2_fp32','half':'half_k1'}[engine]
    row=next(r for r in summary['rows'] if r['plan']==name and r['prefix_index']==0)
    return {'engine':engine,'source_plan':name,'source_prefix_index':0,
            'input_data_bits':prefixes['data_bits'][0],
            'expected_amplitudes_sha256':row['CDF_probe']['amplitudes_sha256'],
            'expected_device_name':summary['device'],'expected_numpy_version':summary['numpy'],
            'expected_cupy_version':summary['cupy'],
            'scope':'Fixed deterministic pilot block, outside production/audit sample counts; consumes no production RNG'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--prefix-manifest',type=Path,required=True);p.add_argument('--engine',choices=['fp32','half'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    anchor=anchor_from_pilot(json.loads(a.summary.read_text()),json.loads(a.prefix_manifest.read_text()),a.engine)
    with a.out.open('x') as f:json.dump(anchor,f,indent=2);f.write('\n')
