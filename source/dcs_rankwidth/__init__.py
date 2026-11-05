# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""Exact amplitudes of doped Clifford circuits by T injection and linear rank-width contraction."""
from .circuit_io import load_circuit, cut_ranks
from .t_injection import build_injected, graph_form, reference_angles
from .scan_plan import make_plan, local_vectors, structurally_fixed, amplitude_reference
from .scan_kernel import CompiledScan

__all__ = ["load_circuit", "cut_ranks", "build_injected", "graph_form", "reference_angles",
           "make_plan", "local_vectors", "structurally_fixed", "amplitude_reference", "CompiledScan"]
