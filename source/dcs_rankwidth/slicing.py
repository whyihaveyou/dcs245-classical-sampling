# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Slicing: fix k vertices of the graph state and sum over their 2^k values.

Fixing vertex v to a in Eq. (sum over y of prod_k f_k(y_k) (-1)^{sum_E y_a y_b}) gives a factor f_v(a),
turns each edge (v, u) into a sign (-1)^{a y_u} absorbed into f_u, and removes v from the graph (as a
row and as a column of every cut). Each slice is a contraction on the reduced graph, whose cut ranks are
at most those of the full graph, so the memory drops from 2^rho_max to 2^rho'_max. The plan of the
reduced graph is compiled once; only the vectors f change from one slice to the next.
"""
import math, itertools
import numpy as np

from .t_injection import graph_form
from .scan_plan import (make_plan, future_masks, local_vectors, structurally_fixed,
                        byte_tables, kernel_combos)
from .scan_kernel import CompiledScan


def prefix_cut_ranks_graph(N, edges, order):
    """Cut ranks of every prefix of `order` for the graph (vertices not in `order` are ignored)."""
    pos = {v: i for i, v in enumerate(order)}; M = len(order)
    A = np.zeros((M, M), dtype=np.uint8)
    for a, b in edges:
        if a in pos and b in pos:
            A[pos[a], pos[b]] = A[pos[b], pos[a]] = 1
    G = np.zeros((M, 2 * M), dtype=np.uint8)
    G[:, 0::2] = np.eye(M, dtype=np.uint8); G[:, 1::2] = A
    r, piv = 0, np.zeros(2 * M, dtype=np.int64)
    for c in range(2 * M):
        nz = np.nonzero(G[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            G[[r, p]] = G[[p, r]]
        rows = np.nonzero(G[:, c])[0]; rows = rows[rows != r]
        G[rows] ^= G[r]
        piv[c] = 1; r += 1
        if r == M:
            break
    cum = np.cumsum(piv)
    return np.array([cum[2 * i - 1] - i for i in range(1, M)])


def choose_slices(N, edges, order, k, window=1, max_candidates=None, verbose=True):
    """Greedy choice of k vertices to fix: at each step, try the vertices that sit at or near the
    cuts of maximal rank and keep the one that lowers (r_max, plateau) the most."""
    S = []
    for step in range(k):
        o = [v for v in order if v not in S]
        r = prefix_cut_ranks_graph(N, edges, o); m = int(r.max())
        hot = [i for i, x in enumerate(r) if x >= m - window]
        lo, hi = max(0, min(hot) - 20), min(len(o), max(hot) + 22)
        cand = o[lo:hi]
        if max_candidates and len(cand) > max_candidates:
            cand = list(np.random.default_rng(step).choice(cand, max_candidates, replace=False))
        best = None
        for v in cand:
            o2 = [u for u in o if u != v]
            r2 = prefix_cut_ranks_graph(N, edges, o2); m2 = int(r2.max())
            c = (m2, float(np.sum(2.0 ** (r2 - m2))))
            if best is None or c < best[0]:
                best = (c, v)
        S.append(int(best[1]))
        if verbose:
            print(f"  slice {step + 1}: fix vertex {best[1]}  ->  r_max={best[0][0]}  plateau={best[0][1]:.1f}", flush=True)
    return S


class SlicedScan(CompiledScan):
    """Contraction summed over 2^k slices. Same interface as CompiledScan.amplitude."""

    def __init__(self, tab, N, n, order, slices, edges=None, U=None):
        self.N_full, self.n, self.slices = N, n, list(slices)
        if edges is None or U is None:
            edges, U = graph_form(tab, N)
        self.U_full, self.edges_full = U, edges
        S = set(self.slices)
        keep = [v for v in range(N) if v not in S]
        self.relabel = {v: i for i, v in enumerate(keep)}
        self.keep = keep
        self.nbrs_in_keep = {s: [] for s in self.slices}
        self.slice_edges = []
        red_edges = []
        for a, b in edges:
            if a in S and b in S:
                self.slice_edges.append((a, b))
            elif a in S:
                self.nbrs_in_keep[a].append(self.relabel[b])
            elif b in S:
                self.nbrs_in_keep[b].append(self.relabel[a])
            else:
                red_edges.append((self.relabel[a], self.relabel[b]))
        self.N = len(keep); self.edges = red_edges
        self.U = {self.relabel[v]: U[v] for v in keep}
        # structurally fixed vertices of the reduced graph (data vertices with a basis-state contraction)
        fx_full = structurally_fixed(U, N, n)
        self.fixed = {self.relabel[v] for v in fx_full if v in self.relabel}
        red_order = [self.relabel[v] for v in order if v in self.relabel]
        self.order = red_order
        self.steps, self.profile = make_plan(self.N, self.edges, red_order, self.fixed)
        self.fut = future_masks(self.N, self.edges, red_order)
        self.tables = [byte_tables(st.Lcols, st.r_out) for st in self.steps]
        self.combos = [np.array(kernel_combos(st.kernel), dtype=np.uint64) for st in self.steps]
        self.rmax = max(max(self.profile), 1)
        self.work = sum((1 << s.r_out) * (1 << len(s.kernel)) for s in self.steps)
        self.buf = None

    def amplitude(self, x_bits, w=None, part=0, parts=1):
        """Amplitude summed over the slices. With parts > 1, only the slices whose index is congruent to
        `part` modulo `parts` are summed, so that several devices can share the work; the partial
        amplitudes (complex) must then be added before taking the modulus."""
        if self.buf is None:
            self.alloc()
        f_full = local_vectors(self.U_full, self.N_full, self.n, x_bits, w=w)
        total = 0j
        for idx, a in enumerate(itertools.product((0, 1), repeat=len(self.slices))):
            if idx % parts != part:
                continue
            val = dict(zip(self.slices, a))
            coef = complex(np.prod([f_full[s][val[s]] for s in self.slices]))
            if coef == 0:
                continue
            coef *= (-1) ** sum(val[p] * val[q] for p, q in self.slice_edges)
            flip = np.zeros(self.N, dtype=int)
            for s in self.slices:
                if val[s]:
                    for u in self.nbrs_in_keep[s]:
                        flip[u] ^= 1
            f = {}
            for v in self.keep:
                i = self.relabel[v]; fv = np.array(f_full[v], dtype=np.complex128)
                if flip[i]:
                    fv = fv * np.array([1, -1])
                f[i] = fv
            yfix = {q: (0 if abs(f[q][0]) > 1e-12 else 1) for q in self.fixed}
            v_, ls = self._sweep(f, yfix)
            total += coef * v_ * math.exp(ls)
        return total * 2 ** (-self.N_full / 2)
