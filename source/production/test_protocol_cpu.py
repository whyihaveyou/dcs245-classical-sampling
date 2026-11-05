"""Protocol/CDF/bundle roundtrips. Every number here is synthetic test data."""
import json,math,sys,tempfile
from fractions import Fraction
from pathlib import Path
import numpy as np
from protocol_common import make_run_protocol,actual_cdf_law_vector,validate_bundle
from build_protocol import build_bundle
from sampling_protocol import draw_checked


def main():
    args=(91330245,12940245,19,3,2,6,[5,1,3],'half')
    p=make_run_protocol(*args)
    assert p==json.loads(json.dumps(p))==make_run_protocol(*args)
    assert [len([r for r in p['generation']['records'] if r['sample_id']%3==g]) for g in range(3)]==[7,6,6]
    assert len(p['audit']['ids'])==6==len(set(p['audit']['ids']))
    for g,ids in p['audit']['ids_by_worker'].items():assert len(ids)==2 and all(i%3==int(g) for i in ids)
    changed_generation=make_run_protocol(91330246,*args[1:])
    changed_audit=make_run_protocol(args[0],12940246,*args[2:])
    assert p['audit']==changed_generation['audit']
    assert p['generation']==changed_audit['generation']
    # Independent rational arithmetic gives exact finite-grid interval counts.
    checks=[]
    for a in [np.array([0,1,1j,1+1j,0,2,1-1j,0],complex),
              np.array([1e-15,1e-8j,1,1j,2+3j,.17,.21j,.01],complex)]:
        law,meta=actual_cdf_law_vector(a);weights=a.real*a.real+a.imag*a.imag
        cumulative=[];acc=0.
        for w in weights:acc=float(acc+float(w));cumulative.append(acc)
        cumulative=[float(c/acc) for c in cumulative];cumulative[-1]=1.
        endpoints=[math.ceil(Fraction.from_float(c)*(1<<53)) for c in cumulative]
        counts=[b-a for a,b in zip([0]+endpoints[:-1],endpoints)]
        assert sum(counts)==1<<53
        target=np.asarray(counts,dtype=float)/2**53
        assert np.max(np.abs(np.abs(law)**2-target))<4e-16
        draw=draw_checked(a,.381245)
        assert target[draw['A_index']]>0
        checks.append({'dimension':len(a),'norm':float(np.vdot(law,law).real),
                       'max_probability_roundoff':float(np.max(np.abs(np.abs(law)**2-target)))})
    for invalid in [np.zeros(8,complex),np.array([complex('nan'),1])]:
        try:actual_cdf_law_vector(invalid)
        except (ValueError,ArithmeticError):pass
        else:raise AssertionError('Invalid vector accepted')
    # Snapshot and tamper detection; temporary bundle is never executed.
    with tempfile.TemporaryDirectory(prefix='dcs245_synthetic_bundle_') as temp:
        out=Path(temp)/'bundle'
        receipt=build_bundle(out,91330245,12940245,'fp32',count=19,workers=3,audits_per_worker=2,
                             anchor={'engine':'fp32','scope':'SYNTHETIC TEST PLACEHOLDER; never executed'})
        assert receipt['status']=='not_run';validate_bundle(out)
        with (out/'inputs/plans.json').open('a') as f:f.write(' ')
        try:validate_bundle(out)
        except ValueError as exc:assert 'Frozen file mismatch' in str(exc)
        else:raise AssertionError('Modified frozen input accepted')
    result={'passed':True,'scope':'SYNTHETIC CPU TESTS ONLY; no production seed frozen and no GPU samples generated',
            'protocol_roundtrip':True,'independent_audit_stream':True,'balanced_audit_selection':True,
            'exact_rational_grid_count_checks':checks,'frozen_bundle_tamper_rejected':True}
    Path(__file__).with_name('cpu_roundtrip_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
