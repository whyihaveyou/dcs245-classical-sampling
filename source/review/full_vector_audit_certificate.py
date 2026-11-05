"""Sparse full-vector audits of an approximate conditional sampler.

An audit compares the approximate conditional vector ACTUALLY used by the
sampler with a complete reference block scaled by sqrt(D). The approximate
engine's error is included in the witness; reference arithmetic and CDF/RNG
correctness remain separate qualifications. No GPU dependency.
"""
import json
import math
from pathlib import Path

from conditional_mixture_certificate import certificate_from_losses


def audit_vector_pair(reference_scaled, approximate_vector, audit_id):
    """Return scalar audit metrics; NumPy is imported only for vector work.

    reference_scaled = sqrt(D) * psi[prefix,:], with original target norm.
    approximate_vector is the complete vector used to form conditional CDF.
    It may have any nonzero overall norm. An actual zero-vector fallback must
    be explicitly passed; this function never chooses a fallback silently.
    """
    import numpy as np
    ref = np.asarray(reference_scaled, dtype=np.complex128)
    approx = np.asarray(approximate_vector, dtype=np.complex128)
    if ref.ndim != 1 or ref.shape != approx.shape or not len(ref):
        raise ValueError('Need two nonempty full vectors with identical shape')
    if not np.isfinite(ref).all() or not np.isfinite(approx).all():
        raise ValueError('Nonfinite vector values; do not filter this audit')
    ref_mass = float(np.vdot(ref, ref).real)
    approx_mass = float(np.vdot(approx, approx).real)
    if not math.isfinite(ref_mass) or not math.isfinite(approx_mass) or approx_mass <= 0:
        raise ValueError('Invalid norm; supply the actual declared fallback if used')
    approx_norm = math.sqrt(approx_mass)
    score = float(abs(np.vdot(ref, approx)) / approx_norm)
    overlap_abs = min(1., score / math.sqrt(ref_mass)) if ref_mass > 0 else None
    weight = min(score, 1.)
    return {'audit_id': audit_id, 'vector_length': len(ref),
            'reference_scaled_mass': ref_mass, 'approximate_input_mass': approx_mass,
            'normalized_conditional_overlap_abs': overlap_abs,
            'normalized_conditional_fidelity': overlap_abs**2 if overlap_abs is not None else None,
            'uncapped_overlap_witness': score, 'clipped_weight': weight, 'loss': 1-weight,
            'reference_arithmetic': 'complex128 numerical reference; not an interval-arithmetic proof'}


def certificate_from_audits(audits, expected_audit_ids, mixture_weight=1.,
                            total_expensive_branches=None, n_qubits=64, alpha=.05,
                            reference_mean_witness_error_bound=0.):
    """KL certificate from ALL preselected vector audits, no output filtering.

    expected_audit_ids must come from a frozen selection made independently
    of prefixes, output bits, losses and scores. This helper verifies identity
    and count matching, not that the upstream random-selection claim is true.
    Optional reference error must be a justified global mean witness bound,
    not the maximum disagreement over these or other sampled vectors.
    """
    rows = list(audits)
    expected = list(expected_audit_ids)
    got = [row['audit_id'] for row in rows]
    if len(set(expected)) != len(expected) or len(set(got)) != len(got):
        raise ValueError('Repeated audit IDs are invalid; repeated PREFIXES are allowed')
    if set(got) != set(expected) or len(got) != len(expected):
        raise ValueError('Missing, extra or substituted preselected audit IDs')
    if total_expensive_branches is not None:
        if (not isinstance(total_expensive_branches, int)
                or total_expensive_branches < len(rows)):
            raise ValueError('Invalid total expensive branch count')
    losses = []
    for row in rows:
        score = float(row['uncapped_overlap_witness'])
        if not math.isfinite(score) or score < 0:
            raise ValueError('Invalid audit witness; do not filter this audit')
        losses.append(1-min(score, 1.))
    result = certificate_from_losses(
        losses, mixture_weight, n_qubits=n_qubits, alpha=alpha,
        mean_weight_error_bound=reference_mean_witness_error_bound)
    result['audit_count'] = result.pop('expensive_branch_count')
    result['total_expensive_branches'] = total_expensive_branches
    result['audit_ids'] = got
    result['scope'] = ('Exact-reference theorem; approximate conditional-vector error is included '
                       'in audited overlaps. FP64 reference correctness and CDF/RNG fidelity '
                       'to the audited vectors require separate validation.')
    result['requirements'] = [
        'Frozen sampler, fixed total outputs and independent mixture labels.',
        'Audit IDs selected independently before prefixes and outputs; all selected audits included.',
        'Fresh iid uniform prefix for every expensive output, including audited outputs.',
        'Audit the complete approximate vector actually used for its one conditional sample.',
        'Reference is the original full-circuit sqrt(D)-scaled block, not a normalized block.',
        'Per-prefix phase choice is an existence proof and does not alter classical output probabilities.',
        'Numerical reference errors and actual CDF/RNG implementation remain separately qualified.',
    ]
    return result


def synthetic_validation():
    """Independent synthetic identity check, including half-quantized vectors."""
    import numpy as np
    rng = np.random.default_rng(20261103)
    cases = []
    for D, K in [(2, 4), (4, 8), (8, 16)]:
        psi = rng.normal(size=(D, K)) + 1j * rng.normal(size=(D, K))
        psi[0, :] = 0
        psi /= np.linalg.norm(psi)
        approx = psi + .03 * (rng.normal(size=(D, K)) + 1j * rng.normal(size=(D, K)))
        approx = approx.real.astype(np.float16).astype(float) + 1j * approx.imag.astype(np.float16).astype(float)
        phi = np.empty_like(psi)
        rows = []
        for y in range(D):
            a = approx[y] / np.linalg.norm(approx[y])
            inner = np.vdot(psi[y], a)
            phase = np.conj(inner) / abs(inner) if inner else 1.
            phi[y] = phase * a / math.sqrt(D)
            rows.append(audit_vector_pair(math.sqrt(D) * psi[y], approx[y], y))
        expected = math.fsum(r['uncapped_overlap_witness'] for r in rows) / D
        observed = abs(np.vdot(psi, phi))
        assert abs(expected-observed) < 3e-15
        assert abs(np.linalg.norm(phi)-1) < 3e-15
        assert rows[0]['uncapped_overlap_witness'] == 0
        assert rows[0]['loss'] == 1
        assert np.allclose(abs(phi)**2, abs(approx)**2 / np.sum(abs(approx)**2, axis=1)[:, None] / D)
        certificate = certificate_from_audits(rows, list(range(D)), .45,
                                              total_expensive_branches=D, n_qubits=int(math.log2(D*K)))
        assert certificate['audit_count'] == D
        try:
            certificate_from_audits(rows[:-1], list(range(D)), .45)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing audit was accepted')
        cases.append({'D': D, 'K': K, 'sqrt_fidelity': observed,
                      'witness_mean': expected, 'zero_target_block_handled': True,
                      'phase_changes_preserve_output_probabilities': True})
    result = {'scope': 'SYNTHETIC ALGEBRA CHECK, NOT FULL-CIRCUIT RESULTS', 'passed': True, 'cases': cases}
    Path(__file__).with_name('full_vector_audit_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    synthetic_validation()
