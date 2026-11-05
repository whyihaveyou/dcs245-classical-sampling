# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Parallel CPU execution of a compiled scan plan (Numba).

Two buffers of 2^{r_max} entries are allocated once and swapped at every step. Each step is a
single pass over memory: gather, sign, and local weight. At r_max = 36 in complex64 this requires
about 1.1 TB, which fits on a 3 TB node.
"""
import math
import numpy as np
import numba as nb

from .t_injection import graph_form
from .scan_plan import (make_plan, future_masks, local_vectors, structurally_fixed,
                        byte_tables, kernel_combos)


@nb.njit(inline="always")
def _par64(x):
    x ^= x >> np.uint64(32); x ^= x >> np.uint64(16); x ^= x >> np.uint64(8)
    x ^= x >> np.uint64(4); x ^= x >> np.uint64(2); x ^= x >> np.uint64(1)
    return x & np.uint64(1)


@nb.njit(parallel=True, fastmath=False, cache=True)
def _step(E, out, n_out, T, nbytes, combos, r_in, beta, ov, free, f0, f1, yfix):
    mask_c = (np.uint64(1) << np.uint64(r_in)) - np.uint64(1)
    rin = np.uint64(r_in)
    one = np.uint64(1)
    for cp in nb.prange(n_out):
        c_ = np.uint64(cp)
        p0 = np.uint64(0)
        for b in range(nbytes):
            p0 ^= T[b, (c_ >> np.uint64(8 * b)) & np.uint64(255)]
        acc = 0j
        for k in range(combos.shape[0]):
            p = p0 ^ combos[k]
            c = p & mask_c
            par = _par64(c & beta) ^ ov
            if free:
                y = (p >> rin) & one
                w = f1 if y == one else f0
                if (y & par) == one:
                    w = -w
            else:
                w = f1 if yfix == 1 else f0
                if (yfix == 1) and (par == one):
                    w = -w
            acc += E[np.int64(c)] * w
        out[cp] = acc


@nb.njit(parallel=True, cache=True)
def _maxabs(E, n):
    m = 0.0
    for i in nb.prange(n):
        m = max(m, abs(E[i]))
    return m


@nb.njit(parallel=True, cache=True)
def _scale(E, n, s):
    for i in nb.prange(n):
        E[i] *= s


class CompiledScan:
    """Compile once per circuit and order, then evaluate any number of bitstrings."""

    def __init__(self, tab, N, n, order):
        self.N, self.n, self.order = N, n, order
        self.edges, self.U = graph_form(tab, N)
        self.fixed = structurally_fixed(self.U, N, n)
        self.steps, self.profile = make_plan(N, self.edges, order, self.fixed)
        self.fut = future_masks(N, self.edges, order)
        self.tables = [byte_tables(st.Lcols, st.r_out) for st in self.steps]
        self.combos = [np.array(kernel_combos(st.kernel), dtype=np.uint64) for st in self.steps]
        self.rmax = max(max(self.profile), 1)
        self.work = sum((1 << s.r_out) * (1 << len(s.kernel)) for s in self.steps)
        self.buf = None

    def alloc(self, dtype=np.complex64):
        self.buf = [np.empty(1 << self.rmax, dtype=dtype), np.empty(1 << self.rmax, dtype=dtype)]

    def _sweep(self, f, yfix, renorm_every=4):
        A, B = self.buf
        A[0] = 1.0
        o = 0; logscale = 0.0
        for j, st in enumerate(self.steps):
            v = st.v
            ov = (o >> j) & 1
            o &= ~(1 << j)
            y = 0
            if not st.free:
                y = int(yfix[v])
                if y:
                    o ^= self.fut[j]
            n_out = 1 << st.r_out
            T = self.tables[j]
            _step(A, B, n_out, T, T.shape[0], self.combos[j], st.r_in, np.uint64(st.beta),
                  np.uint64(ov), st.free, complex(f[v][0]), complex(f[v][1]), y)
            A, B = B, A
            if j % renorm_every == 0 or j == len(self.steps) - 1:
                m = _maxabs(A, n_out)
                if m > 0:
                    _scale(A, n_out, 1.0 / m); logscale += math.log(m)
        self.buf = [A, B]
        return complex(A[0]), logscale

    def amplitude(self, x_bits, w=None):
        """<x|U|0> up to a global phase. x_bits: data values (ring order) padded to length N."""
        if self.buf is None:
            self.alloc()
        f = local_vectors(self.U, self.N, self.n, x_bits, w=w)
        yfix = {q: (0 if abs(f[q][0]) > 1e-12 else 1) for q in self.fixed}
        val, ls = self._sweep(f, yfix)
        return val * math.exp(ls) * 2 ** (-self.N / 2)
