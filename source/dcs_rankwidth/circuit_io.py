# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
Circuit input and cut-rank computation.

The circuit is read from OpenQASM, transpiled to the basis {cz, rz, sx, x}, and flattened into a
list of operations acting on qubits labelled by their position along the ring. Each operation is
    ("cz", [a, b], layer)
    ("c1", [q], layer, gate)          gate in {"SQRT_X", "X", "S", "Z", "S_DAG"}
    ("T",  [q], layer, theta)         diagonal non-Clifford rotation diag(1, e^{i theta})
where `layer` is the ASAP CZ layer index.
"""
import math
import numpy as np
from qiskit import QuantumCircuit, transpile

HALF_PI = math.pi / 2


def _load_qasm(path):
    try:
        qc = QuantumCircuit.from_qasm_file(path)
    except Exception:
        from qiskit import qasm3
        qc = qasm3.load(path)
    qc = qc.remove_final_measurements(inplace=False)
    return transpile(qc, basis_gates=["cz", "rz", "sx", "x"], optimization_level=0)


def _split_data_ancilla(qc):
    """Ancillas are degree-1 vertices of the interaction graph (each check acts on one data qubit)."""
    idx = {q: i for i, q in enumerate(qc.qubits)}
    nbr = {i: set() for i in range(qc.num_qubits)}
    for ins in qc.data:
        if len(ins.qubits) == 2:
            a, b = (idx[q] for q in ins.qubits)
            nbr[a].add(b); nbr[b].add(a)
    anc = {i for i, s in nbr.items() if len(s) == 1}
    data = [i for i in range(qc.num_qubits) if i not in anc and nbr[i]]
    return data, anc, nbr, idx


def _ring_order(data, nbr):
    """Order the data qubits along the ring."""
    dset = set(data)
    start = data[0]; order = [start]; prev = None; cur = start
    while True:
        nxt = [v for v in nbr[cur] if v in dset and v != prev and v not in order[1:]]
        nxt = [v for v in nxt if v != start] or ([start] if len(order) > 2 and start in nbr[cur] else [])
        if not nxt or nxt[0] == start:
            break
        prev, cur = cur, nxt[0]; order.append(cur)
    if len(order) != len(data):
        raise RuntimeError(f"could not reconstruct the ring ({len(order)}/{len(data)} qubits)")
    return order


def _to_ops(qc, anc, idx, order):
    """Drop ancillas (an ideal check acts as the identity on the data) and label each operation
    with its ASAP CZ layer. Returns (ops, depth, number of rotations with a generic angle)."""
    pos = {q: k for k, q in enumerate(order)}
    last = {k: 0 for k in range(len(order))}
    ops, generic = [], 0
    for ins in qc.data:
        qs = [idx[q] for q in ins.qubits]
        if any(q in anc for q in qs):
            continue
        name = ins.operation.name
        p = [pos[q] for q in qs]
        if name == "cz":
            L = max(last[p[0]], last[p[1]]) + 1
            last[p[0]] = last[p[1]] = L
            ops.append(("cz", p, L))
        elif name in ("sx", "x"):
            ops.append(("c1", p, last[p[0]], "SQRT_X" if name == "sx" else "X"))
        elif name == "rz":
            th = float(ins.operation.params[0]) % (2 * math.pi)
            k = th / HALF_PI
            if abs(k - round(k)) < 1e-9:
                g = ["I", "S", "Z", "S_DAG"][int(round(k)) % 4]
                if g != "I":
                    ops.append(("c1", p, last[p[0]], g))
            else:
                if abs(2 * k - round(2 * k)) >= 1e-9:
                    generic += 1
                ops.append(("T", p, last[p[0]], th))
        elif name in ("barrier", "delay", "measure", "reset", "id"):
            continue
        else:
            raise ValueError(f"unexpected gate {name}")
    return ops, max(last.values()), generic


def load_circuit(path):
    """Read a QASM file. Returns (ops, ring) where ring[k] is the QASM index of ring position k."""
    qc = _load_qasm(path)
    data, anc, nbr, idx = _split_data_ancilla(qc)
    ring = _ring_order(data, nbr)
    ops, depth, generic = _to_ops(qc, anc, idx, ring)
    return ops, ring


def circuit_depth(ops):
    return max(op[2] for op in ops)


def cut_ranks(sim, n, order=None):
    """Entanglement entropy (in bits) of the stabiliser state held by the stim TableauSimulator
    `sim`, across every prefix of `order`: r_i = S(first i qubits), i = 1..n-1.

    Uses S_A = rank_F2(G restricted to the columns of A) - |A| for the n x 2n generator matrix G,
    and obtains all prefix ranks from a single Gaussian elimination with pivots chosen in column
    order: the rank of the first k columns equals the number of pivot columns among them."""
    stabs = sim.canonical_stabilizers()
    X = np.array([s.to_numpy()[0] for s in stabs], dtype=np.uint8)
    Z = np.array([s.to_numpy()[1] for s in stabs], dtype=np.uint8)
    order = list(range(n)) if order is None else order
    G = np.empty((X.shape[0], 2 * n), dtype=np.uint8)
    G[:, 0::2] = X[:, order]; G[:, 1::2] = Z[:, order]
    piv_cols = []; r = 0
    for c in range(2 * n):
        nz = np.nonzero(G[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            G[[r, p]] = G[[p, r]]
        rows = np.nonzero(G[:, c])[0]; rows = rows[rows != r]
        G[rows] ^= G[r]
        piv_cols.append(c); r += 1
        if r == G.shape[0]:
            break
    piv = np.zeros(2 * n, dtype=int); piv[piv_cols] = 1
    cum = np.cumsum(piv)
    return np.array([cum[2 * i - 1] - i for i in range(1, n)])
