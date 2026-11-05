# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Parity slicing: fix a vertex after a local complementation or a pivot.

Measuring vertex w of the graph state in the Z basis removes it (vertex slicing). Measuring it in the Y
or X basis fixes a parity of w and its neighbours: it amounts to a local complementation at w (Y) or a
pivot on an edge (w, u) (X), followed by the removal of w. Local complementation preserves every cut
rank; only the effect of the removal changes. Every transformation is exact: with
|G*v> = exp(+i pi/4 X_v) prod_{u in N(v)} exp(-i pi/4 Z_u) |G> (up to a phase), the local Cliffords
are updated as U_v <- U_v exp(-i pi/4 X) and U_u <- U_u exp(+i pi/4 Z).
"""
import numpy as np
from .t_injection import graph_form
from .slicing import SlicedScan

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_RXm = np.eye(2) * np.cos(np.pi / 4) - 1j * np.sin(np.pi / 4) * _X     # exp(-i pi/4 X)
_RZp = np.eye(2) * np.cos(np.pi / 4) + 1j * np.sin(np.pi / 4) * _Z     # exp(+i pi/4 Z)


def lc(A, U, v):
    """Local complementation at v, in place, on the adjacency matrix A and the local Cliffords U."""
    nb = np.nonzero(A[v])[0]
    if len(nb) > 1:
        A[np.ix_(nb, nb)] ^= 1
        A[nb, nb] = 0
    if U is not None:
        U[v] = U[v] @ _RXm
        for u in nb:
            U[u] = U[u] @ _RZp


def prefix_ranks_matrix(A, order):
    """Cut ranks of all prefixes of `order` for the induced subgraph on `order`."""
    M = len(order); As = A[np.ix_(order, order)]
    G = np.zeros((M, 2 * M), dtype=np.uint8)
    G[:, 0::2] = np.eye(M, dtype=np.uint8); G[:, 1::2] = As
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


def _cost(A, order):
    r = prefix_ranks_matrix(A, order); m = int(r.max())
    return m, float(np.sum(2.0 ** (r - m)))


def choose_parity_slices(N, edges, order, k, window_pad=15, max_cand=60, n_pivots=4, seed=0, verbose=True):
    """Greedy: at each step, try for vertices near the cuts of maximal rank the Z, Y and X removals,
    and keep the one that lowers (r_max, plateau) the most. Returns the list of transformations
    [(kind, w, u)] and the removed vertices."""
    rng = np.random.default_rng(seed)
    A = np.zeros((N, N), dtype=np.uint8)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    ops, S = [], []
    alive = list(order)
    for step in range(k):
        r = prefix_ranks_matrix(A, alive); m = int(r.max())
        hot = [i for i, x in enumerate(r) if x == m]
        cand = alive[max(0, min(hot) - window_pad):min(len(alive), max(hot) + window_pad + 2)]
        if len(cand) > max_cand:
            cand = list(rng.choice(cand, max_cand, replace=False))
        best = None
        for w in cand:
            o = [x for x in alive if x != w]
            trials = [("Z", w, None)]
            trials.append(("Y", w, None))
            nb = [u for u in np.nonzero(A[w])[0] if u in set(alive)]
            for u in (rng.choice(nb, min(n_pivots, len(nb)), replace=False) if nb else []):
                trials.append(("X", w, int(u)))
            for kind, ww, u in trials:
                B = A.copy()
                if kind == "Y":
                    lc(B, None, ww)
                elif kind == "X":
                    lc(B, None, u); lc(B, None, ww); lc(B, None, u)
                c = _cost(B, o)
                if best is None or c < best[0]:
                    best = (c, kind, ww, u)
        c, kind, w, u = best
        if kind == "Y":
            lc(A, None, w)
        elif kind == "X":
            lc(A, None, u); lc(A, None, w); lc(A, None, u)
        ops.append((kind, int(w), None if u is None else int(u))); S.append(int(w))
        alive = [x for x in alive if x != w]
        if verbose:
            print(f"  slice {step + 1}: {kind} on vertex {w}" + (f" (pivot with {u})" if u is not None else "")
                  + f"  ->  r_max={c[0]}  plateau={c[1]:.1f}", flush=True)
    return ops, S


def build_parity_sliced(tab, N, n, order, ops):
    """Apply the transformations to the graph and the local Cliffords, then build the sliced scan."""
    edges, U = graph_form(tab, N)
    A = np.zeros((N, N), dtype=np.uint8)
    for a, b in edges:
        A[a, b] = A[b, a] = 1
    U = {k: np.array(v, dtype=complex) for k, v in U.items()}
    for kind, w, u in ops:
        if kind == "Y":
            lc(A, U, w)
        elif kind == "X":
            lc(A, U, u); lc(A, U, w); lc(A, U, u)
    iu = np.triu_indices(N, 1)
    new_edges = [(int(a), int(b)) for a, b in zip(*iu) if A[a, b]]
    S = [w for _, w, _ in ops]
    return SlicedScan(tab, N, n, order, S, edges=new_edges, U=U)
