# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
T-gate injection and graph-state form.

Each diagonal rotation diag(1, e^{i theta}) on qubit q is replaced by a reference qubit prepared in
|0>, a CNOT from q to the reference, and a final contraction of the reference with the linear
functional w = (1, e^{i theta}). Since CNOT|b>|0> = |b>|b>, this reproduces the rotation exactly.
The rewritten circuit is Clifford, so

    <x|U|0> = (<x| (x) w (x) ... (x) w) |S>,

with |S> a stabiliser state on n + t qubits. |S> is then written as a graph state followed by one
local Clifford per qubit, |S> = (U_1 (x) ... (x) U_N) |G>.
"""
import math
import numpy as np
import stim


def build_injected(ops, n):
    """Returns (tableau, N, natural_order). References are numbered n, n+1, ... in the order in
    which the rotations appear; natural_order places each reference right after its data qubit."""
    t = sum(op[0] == "T" for op in ops)
    N = n + t
    sim = stim.TableauSimulator(); sim.set_num_qubits(N)
    gate = {"SQRT_X": sim.sqrt_x, "X": sim.x, "S": sim.s, "Z": sim.z, "S_DAG": sim.s_dag}
    refs_of = {i: [] for i in range(n)}; k = n
    for op in ops:
        if op[0] == "cz":
            sim.cz(*op[1])
        elif op[0] == "c1":
            gate[op[3]](op[1][0])
        else:
            q = op[1][0]; sim.cnot(q, k); refs_of[q].append(k); k += 1
    natural_order = []
    for i in range(n):
        natural_order.append(i); natural_order.extend(refs_of[i])
    return sim.current_inverse_tableau().inverse(), N, natural_order


def reference_angles(ops, n):
    """Angle of each reference qubit n, n+1, ..., in the allocation order of build_injected."""
    out = {}; k = n
    for op in ops:
        if op[0] == "T":
            out[k] = float(op[3]) if len(op) > 3 else math.pi / 4
            k += 1
    return out


_EXACT = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2 ** -0.5, -(2 ** -0.5)])


def _snap(M):
    """stim returns complex64 unitaries; project each entry onto its exact Clifford value."""
    M = np.asarray(M, dtype=np.complex128)
    snap = lambda x: _EXACT[np.argmin(np.abs(_EXACT[None, :] - x.reshape(-1, 1)), axis=1)].reshape(x.shape)
    return snap(M.real) + 1j * snap(M.imag)


def graph_form(tab, N):
    """Returns (edges, U): the graph-state edges and one exact 2x2 local Clifford per qubit,
    such that |S> = (U_1 (x) ... (x) U_N) CZ_edges |+>^N."""
    circ = tab.to_circuit(method="graph_state")
    edges = []; local = {k: stim.Circuit() for k in range(N)}
    for ins in circ:
        name = ins.name
        if name in ("TICK", "RX"):
            continue
        tg = [x.value for x in ins.targets_copy()]
        if name == "CZ":
            edges += list(zip(tg[0::2], tg[1::2]))
        else:
            for q in tg:
                local[q].append(name, [0])
    U = {k: (_snap(stim.Tableau.from_circuit(local[k]).to_unitary_matrix(endian="little"))
             if len(local[k]) else np.eye(2, dtype=np.complex128)) for k in range(N)}
    return edges, U
