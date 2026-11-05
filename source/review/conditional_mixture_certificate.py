"""KL confidence helper for full-circuit conditional sampling + uniform mixture.

Standard library only. This computes an exact-arithmetic statistical theorem;
it does not certify the floating-point block engine or the production RNG.
Inputs are z_y = D * p_y BEFORE conditional normalization, for every expensive
branch in a fixed-length iid Bernoulli-mixture experiment. No filtering.
"""
import argparse
import json
import math
from pathlib import Path


def kl_bernoulli(a, b):
    if not (math.isfinite(a) and math.isfinite(b) and 0 <= a <= 1 and 0 <= b <= 1):
        raise ValueError('KL arguments must be finite and in [0,1]')
    if a == b:
        return 0.
    if b in (0., 1.):
        return math.inf
    if a == 0:
        return -math.log1p(-b)
    if a == 1:
        return -math.log(b)
    return a * math.log(a / b) + (1 - a) * (math.log1p(-a) - math.log1p(-b))


def bounded_mean_upper(mean, n, alpha=.05):
    """One-sided KL-Chernoff upper bound for iid [0,1] observations.

    Returns 1 for n=0 (no information). The routine inverts the fixed-n
    inequality and must not be used with quality-dependent stopping.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError('n must be a nonnegative integer')
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    if n == 0:
        return 1.
    if mean is None or not math.isfinite(mean) or not 0 <= mean <= 1:
        raise ValueError('mean must be finite and in [0,1]')
    if mean == 1:
        return 1.
    if mean == 0:
        return -math.expm1(math.log(alpha) / n)
    lo, hi = mean, 1.
    threshold = math.log(1 / alpha) / n
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if kl_bernoulli(mean, mid) >= threshold:
            hi = mid
        else:
            lo = mid
    return hi


def certificate_from_losses(losses, mixture_weight, n_qubits=64, alpha=.05,
                            mean_weight_error_bound=0., conditional_state_l2_error_bound=0.):
    """Certify lambda |phi><phi| + (1-lambda) I/2**n_qubits.

    Optional error bounds must be independently justified deterministic global
    bounds. A max error over tested points is NOT such a bound. Defaults zero
    give the exact-arithmetic theorem only, not a numerical certification.
    """
    if not math.isfinite(mixture_weight) or not 0 <= mixture_weight <= 1:
        raise ValueError('mixture_weight must be in [0,1]')
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError('n_qubits must be a positive integer')
    for value in (mean_weight_error_bound, conditional_state_l2_error_bound):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Error bounds must be finite and nonnegative')
    losses = [float(v) for v in losses]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in losses):
        raise ValueError('Each loss must be finite and in [0,1]')
    n = len(losses)
    mean = math.fsum(losses) / n if n else None
    upper = bounded_mean_upper(mean, n, alpha)
    overlap = max(0., 1 - upper - mean_weight_error_bound - conditional_state_l2_error_bound)
    pure_lower = overlap**2
    uniform_floor = math.ldexp(1 - mixture_weight, -n_qubits)
    mixture_lower = mixture_weight * pure_lower + uniform_floor
    return {'statistic': 'one-sided bounded-loss KL-Chernoff', 'alpha': alpha,
            'expensive_branch_count': n, 'observed_mean_loss': mean,
            'population_mean_loss_upper': upper, 'overlap_lower': overlap,
            'conditional_state_fidelity_lower': pure_lower,
            'mixture_weight': mixture_weight, 'n_qubits': n_qubits,
            'uniform_state_fidelity_contribution': uniform_floor,
            'mixture_state_fidelity_lower': mixture_lower,
            'mean_weight_error_bound': mean_weight_error_bound,
            'conditional_state_l2_error_bound': conditional_state_l2_error_bound,
            'scope': 'Exact-arithmetic statistical theorem unless separately justified global numerical error bounds are supplied.',
            'requirements': ['Fixed total output count; independent Bernoulli labels with the stated mixture_weight.',
                             'Every expensive branch uses a fresh iid uniform prefix and one conditional output.',
                             'Original full-circuit block masses before conditional normalization; no filtering.',
                             'Partition, confidence function and algorithm fixed independently of production masses.',
                             'Numerical correctness and RNG/CDF replay require separate validation.']}


def certificate_from_scaled_masses(scaled_masses, mixture_weight, **kwargs):
    masses = [float(v) for v in scaled_masses]
    if any(not math.isfinite(v) or v < 0 for v in masses):
        raise ValueError('Scaled masses must be finite and nonnegative')
    losses = [1 - min(math.sqrt(v), 1.) for v in masses]
    result = certificate_from_losses(losses, mixture_weight, **kwargs)
    result['observed_min_scaled_mass'] = min(masses) if masses else None
    result['observed_max_scaled_mass'] = max(masses) if masses else None
    return result


def required_mean_loss(mixture_weight, n, target, alpha=.05, n_qubits=64):
    """Boundary ell: the certificate exceeds target when mean_loss < ell.

    Returns None if even zero observed loss cannot certify the target.
    No numerical-error penalty is included in this feasibility helper.
    """
    floor = math.ldexp(1 - mixture_weight, -n_qubits)

    def bound(mean):
        a = 1 - bounded_mean_upper(mean, n, alpha)
        return mixture_weight * a**2 + floor

    if bound(0.) <= target:
        return None
    lo, hi = 0., 1.
    for _ in range(100):
        mid = (lo + hi) / 2
        if bound(mid) > target:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--masses', type=Path, required=True,
                        help='JSON array, or JSON object with key scaled_masses')
    parser.add_argument('--mixture-weight', type=float, required=True)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--n-qubits', type=int, default=64)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    data = json.loads(args.masses.read_text())
    masses = data['scaled_masses'] if isinstance(data, dict) else data
    result = certificate_from_scaled_masses(masses, args.mixture_weight,
                                            n_qubits=args.n_qubits, alpha=args.alpha)
    body = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.write_text(body)
    print(body, end='')


if __name__ == '__main__':
    main()
