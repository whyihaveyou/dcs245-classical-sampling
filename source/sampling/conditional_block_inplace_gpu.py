"""Conditional blocks using the one-buffer preserved-coordinate GPU executor.

The parity sum and Fourier tail reuse the independently checked canonical
adapter. Only prefix execution and its final syndrome-coordinate basis change.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'kernel_opt'))
from inplace_gpu import InplaceGPUMixin
from conditional_block_gpu import ConditionalBlockGPU


class InplaceConditionalBlockGPU(InplaceGPUMixin, ConditionalBlockGPU):
    def alloc(self, dtype=np.complex64, device=0, fp64_kernel_path=None):
        # Both precisions use the same newly reviewed orbit kernel; the legacy
        # fp64_kernel_path argument is accepted only for interface compatibility.
        InplaceGPUMixin.alloc(self, dtype=dtype, device=device)
        self.use_fp64 = self.inplace_fp64
        self.terminal_basis = [b >> self.stop for _,b in sorted(self.inplace_terminal_basis.items())]
        self.terminal_rank = len(self.terminal_basis)
        cp = self.cp
        ids = cp.arange(1 << self.terminal_rank, dtype=cp.uint64)
        self.d_syndrome_map = cp.zeros_like(ids)
        for j,b in enumerate(self.terminal_basis):
            self.d_syndrome_map ^= ((ids >> cp.uint64(j)) & cp.uint64(1))*cp.uint64(b)
        self.d_phase = cp.ones(1 << len(self.open_full), cp.float64)
        ids = cp.arange(len(self.d_phase), dtype=cp.uint64)
        for a,b in self.open_edges:
            self.d_phase *= 1-2*(((ids >> cp.uint64(a)) & (ids >> cp.uint64(b)) & cp.uint64(1)).astype(cp.float64))
        cp.cuda.get_current_stream().synchronize()

    def _prefix(self, f, yfix):
        env, logscale, offset = self.prefix_inplace(f, yfix)
        return env, logscale, offset >> self.stop

