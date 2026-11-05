"""Deterministic protocol and CDF helpers for a future frozen production run.

No production seed/mixture/count is selected by this module. Conditional draws
are checked using independently summed mass intervals while their full weight
vectors are present; SHA-256 commitments enable later regeneration checks.
These checks do not certify the amplitudes against the target quantum circuit.
"""
import hashlib
import math

import numpy as np


def make_protocol(seed, count, n, free_vertices, mixture_weight=1.):
    free = list(map(int, free_vertices))
    if len(set(free)) != len(free) or not all(0 <= q < n for q in free):
        raise ValueError('Free vertices must be distinct data indices')
    if not 0 <= mixture_weight <= 1 or count < 1:
        raise ValueError('Invalid mixture weight or sample count')
    closed = [q for q in range(n) if q not in free]
    rng = np.random.Generator(np.random.PCG64(seed))
    records = []
    for sample_id in range(count):
        before = rng.bit_generator.state
        branch_u = float(rng.random())
        expensive = branch_u < mixture_weight
        x = np.zeros(n, np.uint8)
        if expensive:
            x[closed] = rng.integers(0, 2, len(closed), dtype=np.uint8)
            draw_u = float(rng.random())
        else:
            x[:] = rng.integers(0, 2, n, dtype=np.uint8)
            draw_u = None
        records.append({'sample_id':sample_id, 'conditional_branch':expensive,
                        'branch_uniform':branch_u, 'draw_uniform':draw_u,
                        'input_data_bits':''.join(map(str,x)),
                        'rng_before':before, 'rng_after':rng.bit_generator.state})
    return {'seed':int(seed), 'count':count, 'n':n, 'free_vertices':free,
            'closed_vertices':closed, 'mixture_weight':mixture_weight,
            'rng':'NumPy PCG64; branch float64, uint8 vector bits, conditional float64',
            'numpy_version':np.__version__, 'records':records}


def draw_checked(amplitudes, uniform):
    """Draw once and check its entire CDF interval with Python math.fsum.

    Primary algorithm: NumPy float64 squared magnitudes and sequential CDF.
    Independent draw check: accurately summed total/prefix masses. Disagreement
    is an error and must be retained/diagnosed; this helper never resamples.
    """
    v = np.asarray(amplitudes, dtype=np.complex128)
    if v.ndim != 1 or not len(v) or len(v) & (len(v)-1):
        raise ValueError('Expected a power-of-two one-dimensional amplitude vector')
    if not np.isfinite(v).all() or not 0 <= uniform < 1:
        raise ValueError('Nonfinite vector or invalid uniform')
    weights = np.asarray(v.real*v.real+v.imag*v.imag, dtype='<f8', order='C')
    cumulative = np.cumsum(weights, dtype=np.float64)
    total_cdf = float(cumulative[-1])
    if not total_cdf > 0 or not np.isfinite(total_cdf):
        raise ArithmeticError('Numerically zero/nonfinite block: do not replace the query')
    cumulative /= total_cdf
    cumulative[-1] = 1.
    index = int(np.searchsorted(cumulative, uniform, side='right'))
    if index >= len(weights) or weights[index] <= 0:
        raise ArithmeticError('CDF produced an invalid outcome')
    # Different summation implementation/order from the primary NumPy CDF.
    independent_mass = math.fsum(map(float, weights))
    prefix_mass = math.fsum(map(float, weights[:index]))
    lo, hi = prefix_mass/independent_mass, (prefix_mass+float(weights[index]))/independent_mass
    independent_index_agrees = lo <= uniform < hi
    if not independent_index_agrees:
        raise ArithmeticError('Independent accurately summed CDF interval disagrees; do not resample')
    # PCG64.random() uses a uniform 53-bit grid. Count those grid points in
    # every CDF bin, including zero-probability bins and exact boundaries.
    grid_ends = np.ceil(cumulative*2.**53)
    grid_masses = np.diff(grid_ends, prepend=0.)/2.**53
    draw_tv = float(.5*np.sum(np.abs(grid_masses-weights/independent_mass), dtype=np.float64))
    return {'A_index':index, 'uniform':float(uniform), 'scaled_mass_cdf':total_cdf,
            'scaled_mass_fsum':independent_mass, 'selected_scaled_weight':float(weights[index]),
            'selected_conditional_probability':float(weights[index]/independent_mass),
            'cdf_lower_numpy':float(cumulative[index-1]) if index else 0.,
            'cdf_upper_numpy':float(cumulative[index]), 'cdf_lower_fsum':lo, 'cdf_upper_fsum':hi,
            'uniform_to_nearest_boundary':min(uniform-lo,hi-uniform),
            'independent_cdf_check_passed':True,
            'finite_53bit_draw_TV_against_computed_normalized_weights':draw_tv,
            'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),
            'amplitudes_sha256':hashlib.sha256(v.astype('<c16',copy=False).tobytes()).hexdigest(),
            'replay_scope':'Full CDF checked in memory; hashes/interval retained. Independent regeneration requires rerunning the frozen block contraction.'}


def assemble_output(input_bits, free_vertices, A_index, ring=None):
    bits = list(map(int, input_bits))
    if not 0 <= A_index < 1 << len(free_vertices):
        raise ValueError('Conditional index outside free-output range')
    for j,q in enumerate(free_vertices): bits[q] = (A_index >> j)&1
    if ring is None: ring = list(range(len(bits)))
    if sorted(ring) != list(range(len(bits))): raise ValueError('Invalid ring permutation')
    original = [0]*len(bits)
    for q,actual in enumerate(ring): original[actual] = bits[q]
    return ''.join(map(str,bits)), ''.join(map(str,original[::-1]))


def selftest():
    for weights in [np.array([0.,1.,0.,2.,3.,0.,4.,0.]),
                    np.array([1e-100,1.,.5,0.,7.,2e-100,1.,.1])]:
        v = np.sqrt(weights)*np.exp(1j*np.arange(len(weights)))
        cdf = np.cumsum(weights)/np.sum(weights)
        for idx in np.flatnonzero(weights):
            lo = cdf[idx-1] if idx else 0.
            if cdf[idx] == lo: continue  # Below float64 CDF resolution.
            u = (lo+cdf[idx])/2
            result = draw_checked(v,u)
            assert result['A_index'] == idx
    m = make_protocol(733245,113,6,[5,1,3],.45)
    assert m == make_protocol(733245,113,6,[5,1,3],.45)
    assert any(r['conditional_branch'] for r in m['records'])
    assert any(not r['conditional_branch'] for r in m['records'])
    for i in range(8):
        bits,qiskit = assemble_output('000000',[5,1,3],i,[5,0,4,1,3,2])
        assert sum(int(bits[q]) << j for j,q in enumerate([5,1,3])) == i
        assert len(qiskit) == 6
    return {'passed':True, 'scope':'CPU protocol, CDF and bit-mapping tests; no quantum amplitudes generated'}


if __name__ == '__main__':
    import json
    print(json.dumps(selftest()))
