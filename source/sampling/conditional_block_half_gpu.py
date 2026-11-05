"""Experimental conditional blocks with float16 component storage only.

Local arithmetic and coherent slice accumulation remain FP64. This numerical
approximation must be tested against whole FP64 conditional vectors.
"""
import numpy as np
from conditional_block_inplace_gpu import InplaceConditionalBlockGPU
from inplace_half_gpu import HalfStorageInplaceGPUMixin


class HalfConditionalBlockGPU(HalfStorageInplaceGPUMixin, InplaceConditionalBlockGPU):
    def alloc(self,dtype=np.complex64,device=0,fp64_kernel_path=None):
        HalfStorageInplaceGPUMixin.alloc(self,dtype=dtype,device=device)
        self.use_fp64=False
        self.terminal_basis=[b>>self.stop for _,b in sorted(self.inplace_terminal_basis.items())]
        self.terminal_rank=len(self.terminal_basis)
        cp=self.cp
        ids=cp.arange(1<<self.terminal_rank,dtype=cp.uint64)
        self.d_syndrome_map=cp.zeros_like(ids)
        for j,b in enumerate(self.terminal_basis):
            self.d_syndrome_map^=((ids>>cp.uint64(j))&cp.uint64(1))*cp.uint64(b)
        self.d_phase=cp.ones(1<<len(self.open_full),cp.float64)
        ids=cp.arange(len(self.d_phase),dtype=cp.uint64)
        for a,b in self.open_edges:
            self.d_phase*=1-2*(((ids>>cp.uint64(a))&(ids>>cp.uint64(b))&cp.uint64(1)).astype(cp.float64))
        cp.cuda.get_current_stream().synchronize()

    def block(self,*args,**kwargs):
        vector,meta=super().block(*args,**kwargs)
        meta.update({'intermediate_storage':'float16 real/imaginary components',
                     'local_arithmetic':'FP64 then float32-to-float16 rounded storage',
                     'coherent_accumulation':'complex128',
                     'approximation_scope':'Whole-vector quality requires independent FP64 validation'})
        return vector,meta
