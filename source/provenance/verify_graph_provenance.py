"""Independently parse the target QASM and check all injected stabilizers.

This does not call dcs_rankwidth's circuit parser or injection builder.
Global stabilizer-state phase is immaterial to the sampling task.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import qiskit
from qiskit import qasm2
import stim

TARGET = '7a0b8aa6bcef23b901720cb21c9b05cc5de42d4b0e872db4606eec03766f73e5'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(qasm, graph_file):
    started = time.perf_counter()
    if sha(qasm) != TARGET:
        raise ValueError('This verifier is bound to the published QAT 245 circuit')
    qc = qasm2.load(str(qasm), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    data = np.load(graph_file, allow_pickle=False)
    graph, local = data['graph'], data['unitaries']
    ring = list(map(int, data['ring']))
    counts = dict(qc.count_ops())
    n, N = qc.num_qubits, len(graph)
    assert n == 64 and N == 378 and counts.get('t') == 314
    assert sorted(ring) == list(range(n))
    assert local.shape == (N, 2, 2)
    assert graph.shape == (N, N) and np.array_equal(graph, graph.T)
    assert np.all(np.diag(graph) == 0) and np.all((graph == 0) | (graph == 1))
    physical_to_ring = {q: i for i, q in enumerate(ring)}
    original = stim.TableauSimulator()
    original.set_num_qubits(N)
    ref = n
    for ins in qc.data:
        qubits = [physical_to_ring[qc.find_bit(q).index] for q in ins.qubits]
        name = ins.operation.name
        if name == 't':
            assert len(qubits) == 1
            original.cnot(qubits[0], ref)
            ref += 1
        elif name == 'cz':
            original.cz(*qubits)
        elif name == 'h':
            original.h(*qubits)
        elif name == 's':
            original.s(*qubits)
        elif name == 'sx':
            original.sqrt_x(*qubits)
        else:
            raise ValueError('Unexpected target gate: '+name)
    assert ref == N
    reconstructed = stim.TableauSimulator()
    reconstructed.set_num_qubits(N)
    reconstructed.h(*range(N))
    edges = list(zip(*np.nonzero(np.triu(graph, 1))))
    for a, b in edges:
        reconstructed.cz(int(a), int(b))
    for i in range(N):
        unitary = local[i]
        assert np.max(np.abs(unitary.conj().T @ unitary - np.eye(2))) < 1e-12
        tableau = stim.Tableau.from_unitary_matrix(unitary, endian='little')
        reconstructed.do_tableau(tableau, [i])
    actual = list(map(str, original.canonical_stabilizers()))
    rebuilt = list(map(str, reconstructed.canonical_stabilizers()))
    if actual != rebuilt:
        raise AssertionError('Injected QASM state differs from graph/local-Clifford state')
    return {
        'passed': True, 'target': 'ibm_boston_dcs_nq64_depth73_checks12',
        'qasm_sha256': sha(qasm), 'graph_npz_sha256': sha(graph_file),
        'logical_qubits': n, 'injected_qubits': N, 'edges': len(edges),
        'qiskit_gate_counts': counts, 'reference_rotations': ref-n,
        'reference_angles': 'All 314 are +pi/4; default T contraction weights are applicable.',
        'canonical_stabilizers_checked': len(actual),
        'canonical_stabilizers_sha256': hashlib.sha256('\n'.join(actual).encode()).hexdigest(),
        'ring_order': ring, 'qiskit_version': qiskit.__version__, 'stim_version': stim.__version__,
        'verification_seconds_excluding_imports': time.perf_counter()-started,
        'scope': 'Independent Qiskit parsing and Stim simulation of the full injected Clifford state versus the stored graph and local Cliffords; no upstream parser or injection builder called. The T bra identity is exact; this does not certify later floating-point contraction.'
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qasm', type=Path, required=True)
    p.add_argument('--graph', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = verify(a.qasm, a.graph)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
