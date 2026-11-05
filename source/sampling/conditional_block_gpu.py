"""Coherent, parity-sliced conditional output blocks on one GPU.

Uses the Apache-2.0 dcs-rankwidth GPU kernel by Kim Renaud; the partial sweep
adapts its scalar executor while preserving its power-of-two normalization.
No sampling or fidelity claim is made by this numerical engine alone.
"""
import importlib.util
import itertools
import math
import time
from pathlib import Path

import numpy as np

from dcs_rankwidth.gpu_kernel import GPUMixin
from dcs_rankwidth.parity_slicing import lc
from dcs_rankwidth.scan_plan import (
    _reduce, byte_tables, future_masks, kernel_combos, local_vectors, make_plan,
    structurally_fixed,
)
from dcs_rankwidth.slicing import SlicedScan


class ConditionalBlockGPU(GPUMixin, SlicedScan):
    def __init__(self, graph, unitaries, n, order, parity_operations, open_vertices):
        N = len(graph)
        graph = np.asarray(graph, np.uint8).copy()
        U = {i: np.asarray(u, complex).copy() for i, u in enumerate(unitaries)}
        self.open_full = list(map(int, open_vertices))
        if len(set(self.open_full)) != len(self.open_full) or not all(0 <= q < n for q in self.open_full):
            raise ValueError('Free vertices must be distinct data vertices')
        S = [int(v) for _, v, _ in parity_operations]
        if set(S) & set(range(n)):
            raise ValueError('This adapter only permits slicing reference vertices')
        for kind, v, u in parity_operations:
            if kind == 'Y':
                lc(graph, U, v)
            elif kind == 'X':
                lc(graph, U, u); lc(graph, U, v); lc(graph, U, u)
            elif kind != 'Z':
                raise ValueError(kind)
        edges = [(int(a), int(b)) for a, b in zip(*np.nonzero(np.triu(graph, 1)))]
        SlicedScan.__init__(self, None, N, n, order, S, edges=edges, U=U)
        self.open_red = [self.relabel[q] for q in self.open_full]
        O = set(self.open_red)
        self.order = [v for v in self.order if v not in O] + self.open_red
        self.fixed -= O
        all_steps, self.rank_profile = make_plan(self.N, self.edges, self.order, self.fixed)
        self.stop = self.N-len(O)
        self.steps = all_steps[:self.stop]
        self.fut = future_masks(self.N, self.edges, self.order)
        self.tables = [byte_tables(st.Lcols, st.r_out) for st in self.steps]
        self.combos = [np.asarray(kernel_combos(st.kernel), np.uint64) for st in self.steps]
        self.rmax = max([1]+[max(s.r_in, s.r_out) for s in self.steps])
        self.work = sum((1 << s.r_out)*(1 << len(s.kernel)) for s in self.steps)
        self.profile = False
        basis = []
        for j, st in enumerate(self.steps):
            ws = [b & ~(1 << j) for b in basis]+([self.fut[j]] if st.free else [])
            nb, piv = [], []
            for old in ws:
                residual, _ = _reduce(old, nb, piv)
                if residual:
                    piv.append((residual & -residual).bit_length()-1); nb.append(residual)
            basis = nb
        self.terminal_basis = [b >> self.stop for b in basis]
        self.terminal_rank = len(basis)
        self.open_edges = [(self.open_red.index(a), self.open_red.index(b))
                           for a, b in self.edges if a in O and b in O]

    def alloc(self, dtype=np.complex64, device=0, fp64_kernel_path=None):
        self.use_fp64 = np.dtype(dtype) == np.dtype(np.complex128)
        if self.use_fp64:
            if fp64_kernel_path is None:
                fp64_kernel_path = Path(__file__).resolve().parents[2]/'rgs_attack/upstream_scan/gpu_kernel_fp64.py'
            spec = importlib.util.spec_from_file_location('validated_fp64_scan', fp64_kernel_path)
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            mod.GPUMixin.alloc(self, dtype=dtype, device=device)
        else:
            GPUMixin.alloc(self, dtype=dtype, device=device)
        cp = self.cp
        ids = cp.arange(1 << self.terminal_rank, dtype=cp.uint64)
        self.d_syndrome_map = cp.zeros_like(ids)
        for j, b in enumerate(self.terminal_basis):
            self.d_syndrome_map ^= ((ids >> cp.uint64(j)) & cp.uint64(1))*cp.uint64(b)
        self.d_phase = cp.ones(1 << len(self.open_full), cp.float64)
        ids = cp.arange(len(self.d_phase), dtype=cp.uint64)
        for a, b in self.open_edges:
            self.d_phase *= 1-2*(((ids >> cp.uint64(a)) & (ids >> cp.uint64(b)) & cp.uint64(1)).astype(cp.float64))
        cp.cuda.get_current_stream().synchronize()

    def _prefix(self, f, yfix):
        cp = self.cp
        A, B = self.d_buf
        A[0] = 1
        offset, logscale, maximum = 0, 0., 1.
        scale_type = np.float64 if self.use_fp64 else np.float32
        max_uint, max_float = (np.uint64, np.float64) if self.use_fp64 else (np.uint32, np.float32)
        for j, st in enumerate(self.steps):
            ov = (offset >> j) & 1
            offset &= ~(1 << j)
            y = 0
            if not st.free:
                y = int(yfix[st.v])
                if y: offset ^= self.fut[j]
            f0, f1 = map(complex, f[st.v])
            exponent = math.frexp(maximum)[1]
            scale = math.ldexp(1., -exponent)
            logscale += exponent*math.log(2.)
            self.d_max.fill(0)
            self._kernel(self._grid, self._block,
                (A, B, np.uint64(1 << st.r_out), self.d_tables[j], np.int32(self.tables[j].shape[0]),
                 self.d_combos[j], np.int32(len(self.combos[j])), np.int32(st.r_in),
                 np.uint64(st.beta), np.uint64(ov), np.int32(st.free),
                 np.float64(f0.real), np.float64(f0.imag), np.float64(f1.real), np.float64(f1.imag),
                 np.int32(y), scale_type(scale), self.d_max))
            A, B = B, A
            maxbits = int(cp.asnumpy(self.d_max)[0])
            max2 = float(np.asarray([maxbits], dtype=max_uint).view(max_float)[0])
            if not np.isfinite(max2): raise ArithmeticError('Nonfinite prefix table')
            if max2 == 0:
                self.d_buf = [A, B]
                return None, logscale, 0
            maximum = math.sqrt(max2)
        self.d_buf = [A, B]
        return A[:1 << self.terminal_rank], logscale, offset >> self.stop

    def block(self, x_bits, w=None, part=0, parts=1):
        """Return scaled conditional amplitudes and timing, optionally one slice partition.

        The scale is sqrt(D), D=2^(n-k). Only the COHERENT sum over all parts
        has squared norm z_y=D*p(prefix). Do not sum probabilities of parts.
        """
        cp = self.cp
        start = time.perf_counter()
        f_full = local_vectors(self.U_full, self.N_full, self.n, x_bits, w=w)
        total = cp.zeros(1 << len(self.open_full), cp.complex128)
        count = 0
        for idx, bits in enumerate(itertools.product((0, 1), repeat=len(self.slices))):
            if idx % parts != part: continue
            count += 1
            values = dict(zip(self.slices, bits))
            coef = complex(np.prod([f_full[s][values[s]] for s in self.slices]))
            coef *= (-1)**sum(values[a]*values[b] for a, b in self.slice_edges)
            if not coef: continue
            flips = np.zeros(self.N, np.uint8)
            for s in self.slices:
                if values[s]:
                    for v in self.nbrs_in_keep[s]: flips[v] ^= 1
            f = {self.relabel[v]: np.asarray(f_full[v])*np.array([1, 1-2*int(flips[self.relabel[v]])])
                 for v in self.keep}
            yfix = {q: (0 if abs(f[q][0]) > 1e-12 else 1) for q in self.fixed}
            env, logscale, offset = self._prefix(f, yfix)
            if env is None: continue
            freeflip = sum(int(flips[q]) << j for j, q in enumerate(self.open_red))
            indices = self.d_syndrome_map ^ cp.uint64(offset ^ freeflip)
            coef *= math.exp(logscale-(self.N_full-self.n+len(self.open_full))*.5*math.log(2.))
            total[indices] += env.astype(cp.complex128)*np.complex128(coef)
        # The transform is linear, so it can be applied once after the slice sum.
        k = len(self.open_full)
        for q in range(k):
            v = total.reshape(-1, 2, 1 << q)
            lo, hi = v[:, 0].copy(), v[:, 1].copy()
            v[:, 0], v[:, 1] = lo+hi, lo-hi
        total *= self.d_phase
        for q, original in enumerate(self.open_full):
            u = self.U_full[original]
            v = total.reshape(-1, 2, 1 << q)
            lo, hi = v[:, 0].copy(), v[:, 1].copy()
            v[:, 0] = u[0, 0]*lo+u[0, 1]*hi
            v[:, 1] = u[1, 0]*lo+u[1, 1]*hi
        cp.cuda.get_current_stream().synchronize()
        return total, {'seconds': time.perf_counter()-start, 'part': part, 'parts': parts,
                       'slices_evaluated': count, 'prefix_dtype': 'complex128' if self.use_fp64 else 'complex64',
                       'rmax': self.rmax, 'terminal_rank': self.terminal_rank,
                       'work_per_slice': self.work, 'open_vertices': self.open_full,
                       'vector_scale': 'sqrt(2^(n-k)); sum all complex parts before probabilities'}
