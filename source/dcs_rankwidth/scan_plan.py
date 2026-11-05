# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Scan plan: linear-order contraction of a graph state against a product state.

With the local Cliffords absorbed into the contraction vectors f_k = v_k U_k,

    <v|S> = 2^{-N/2} sum_y prod_k f_k(y_k) (-1)^{sum_{(a,b) in E} y_a y_b}.

Vertices are summed one at a time in a fixed order. After position j, the only information about
the processed vertices needed for the rest of the sum is the syndrome: for each future vertex, the
parity of its processed neighbours set to 1. The syndrome is written s = o + sum_i c_i b_i
(affine offset o, F2 basis b_i), so the environment E[c] has 2^{r_j} entries, where r_j is the cut
rank. Processing vertex v with value y multiplies by f_v(y) and by the sign
(-1)^{y (o_v + c.beta_v)}, then updates s -> s restricted to the future + y a_v.

The plan (bases, linear maps (c, y) -> c', a right inverse L and the kernel K) is compiled once in
exact integer arithmetic. Execution is then a gather with no atomics:

    E'[c'] = sum_{p in L c' + span K} E[p_c] f_v(p_y) sign.

This is the linear case of the rank-width contraction of Kuyanov and Kissinger
(arXiv:2603.06764, Prop. 4.1 and Cor. 4.2), with cost O(N 2^{r_max}).
"""
import math
import numpy as np

W_T = np.array([1.0, np.exp(1j * math.pi / 4)])


class Step:
    __slots__ = ("v", "free", "r_in", "r_out", "beta", "Lcols", "kernel")


def _reduce(vec, basis, pivots):
    """Reduce vec against the basis (insertion order). Returns (residual, coefficients)."""
    coef = 0
    for k, (b, p) in enumerate(zip(basis, pivots)):
        if (vec >> p) & 1:
            vec ^= b
            coef |= 1 << k
    return vec, coef


def future_masks(N, edges, order):
    """fut[j]: bitmask (over positions) of the neighbours of the vertex at position j that come later."""
    pos = {q: j for j, q in enumerate(order)}
    fut = [0] * N
    for a, b in edges:
        pa, pb = pos[a], pos[b]
        if pa < pb:
            fut[pa] |= 1 << pb
        else:
            fut[pb] |= 1 << pa
    return fut


def make_plan(N, edges, order, fixed=frozenset()):
    """Compile the scan. `fixed` holds vertices whose value y is deterministic (their contraction
    vector has a zero entry). Returns (steps, rank profile). Independent of x."""
    fut = future_masks(N, edges, order)
    basis = []
    steps, profile = [], []
    for j in range(N):
        v = order[j]
        st = Step(); st.v = v; st.free = v not in fixed; st.r_in = len(basis)
        st.beta = sum(((b >> j) & 1) << i for i, b in enumerate(basis))
        clear = ~(1 << j)
        ws = [b & clear for b in basis] + ([fut[j]] if st.free else [])
        nb, piv, mcols = [], [], []
        for w in ws:
            res, coef = _reduce(w, nb, piv)
            if res:
                piv.append((res & -res).bit_length() - 1)
                nb.append(res)
                coef |= 1 << (len(nb) - 1)
            mcols.append(coef)
        rp = len(nb); st.r_out = rp
        # pivot column J[k]: the first column whose leading bit is k (the one that created element k)
        J = {}
        for i, m in enumerate(mcols):
            if m and m.bit_length() - 1 not in J:
                J[m.bit_length() - 1] = i
        P = [0] * rp                               # right inverse: M P[k] = e_k
        for k in range(rp):
            i = J[k]; m = mcols[i]; p = 1 << i
            for l in range(k):
                if (m >> l) & 1:
                    p ^= P[l]
            P[k] = p
        st.Lcols = P
        pivcols = set(J.values())
        K = []                                     # kernel of M
        for i, m in enumerate(mcols):
            if i in pivcols:
                continue
            kv = 1 << i
            for k in range(rp):
                if (m >> k) & 1:
                    kv ^= P[k]
            K.append(kv)
        st.kernel = K
        basis = nb
        steps.append(st); profile.append(rp)
    return steps, profile


def local_vectors(U, N, n, x_bits, w=None):
    """Contraction vectors f_q = v_q U_q. Data qubits (q < n) use <x_q|; references use w, which is
    None (T gate everywhere), a single 2-vector, or a dict {reference: 2-vector}."""
    f = {}
    for q in range(N):
        if q < n:
            bra = np.eye(2)[x_bits[q]]
        elif isinstance(w, dict):
            bra = np.asarray(w[q], dtype=np.complex128)
        else:
            bra = W_T if w is None else np.asarray(w)
        f[q] = bra @ U[q]
    return f


def structurally_fixed(U, N, n):
    """Vertices whose value y is deterministic for every x (a zero entry in <e_b|U_q> or w U_q)."""
    fx = set()
    for q in range(N):
        vecs = [np.eye(2)[0] @ U[q], np.eye(2)[1] @ U[q]] if q < n else [W_T @ U[q]]
        if all(min(abs(v[0]), abs(v[1])) < 1e-12 for v in vecs):
            fx.add(q)
    return fx


def byte_tables(cols, nbits):
    """Lookup tables applying c -> XOR_{i: c_i = 1} cols[i], one byte of c at a time."""
    nb = max(1, (nbits + 7) // 8)
    T = np.zeros((nb, 256), dtype=np.uint64)
    for b in range(nb):
        for val in range(256):
            acc = 0
            for t in range(8):
                i = 8 * b + t
                if i < nbits and (val >> t) & 1:
                    acc ^= cols[i]
            T[b, val] = acc
    return T


def kernel_combos(kernel):
    combos = [0]
    for kv in kernel:
        combos = combos + [c ^ kv for c in combos]
    return combos


def _parity(x):
    for sh in (32, 16, 8, 4, 2, 1):
        x = x ^ (x >> np.uint64(sh))
    return x & np.uint64(1)


def sweep_reference(steps, f, yfix, fut, dtype=np.complex128):
    """Plain NumPy execution of a compiled plan (for tests). Returns (value, log_scale)."""
    E = np.ones(1, dtype=dtype); o = 0; logscale = 0.0
    for j, st in enumerate(steps):
        v = st.v
        f0, f1 = complex(f[v][0]), complex(f[v][1])
        ov = (o >> j) & 1
        o &= ~(1 << j)
        if not st.free:
            y = int(yfix[v]); fy = f1 if y else f0
            if y:
                o ^= fut[j]
        T = byte_tables(st.Lcols, st.r_out)
        cp = np.arange(1 << st.r_out, dtype=np.uint64)
        p0 = np.zeros_like(cp)
        for b in range(T.shape[0]):
            p0 ^= np.take(T[b], (cp >> np.uint64(8 * b)) & np.uint64(255))
        beta = np.uint64(st.beta); mask = np.uint64((1 << st.r_in) - 1)
        acc = np.zeros(1 << st.r_out, dtype=dtype)
        for kc in kernel_combos(st.kernel):
            p = p0 ^ np.uint64(kc); c = p & mask
            par = _parity(c & beta) ^ np.uint64(ov)
            val = np.take(E, c.astype(np.int64))
            if st.free:
                yb = (p >> np.uint64(st.r_in)) & np.uint64(1)
                sgn = 1.0 - 2.0 * (yb & par).astype(np.float64)
                acc += val * (f0 + (f1 - f0) * yb.astype(np.float64)) * sgn
            else:
                sgn = 1.0 - 2.0 * (np.uint64(y) & par).astype(np.float64)
                acc += val * fy * sgn
        E = acc
        m = float(np.max(np.abs(E))) if E.size else 0.0
        if m > 0:
            E /= m; logscale += math.log(m)
    return complex(E[0]), logscale


def amplitude_reference(tab, N, n, order, x_bits, w=None):
    """<x|U|0> up to a global phase, with the NumPy executor. For small instances and tests."""
    from .t_injection import graph_form
    edges, U = graph_form(tab, N)
    f = local_vectors(U, N, n, x_bits, w=w)
    fixed = structurally_fixed(U, N, n)
    steps, _ = make_plan(N, edges, order, fixed)
    yfix = {q: (0 if abs(f[q][0]) > 1e-12 else 1) for q in fixed}
    val, ls = sweep_reference(steps, f, yfix, future_masks(N, edges, order))
    return val * math.exp(ls) * 2 ** (-N / 2)
