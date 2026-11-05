"""Experimental IEEE binary16 STORAGE, FP64 arithmetic in the orbit executor.

This is an approximate numerical variant, not an exact sampler. Each complex
entry is two independently rounded IEEE float16 components (4 bytes total).
Weights and local sums are double; rescaling and normalization statistics use
the stored values. Baseline inplace_gpu.py is imported but never modified.
"""
import math
import numpy as np
from dcs_rankwidth.scan_plan import byte_tables
from inplace_plan import compile_inplace
from inplace_gpu import _CODE, InplaceGPUMixin, orbit_descriptor

_HALF_HELPERS=r'''
__device__ __forceinline__ float half_to_float_(unsigned short x){
 float y;asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(x));return y;
}
__device__ __forceinline__ unsigned short float_to_half_(float x){
 unsigned short y;asm("cvt.rn.f16.f32 %0, %1;" : "=h"(y) : "f"(x));return y;
}
'''
_HALF_READ=r'''__device__ __forceinline__ double2 read_(const ushort2* E,unsigned long long c){ushort2 x=E[c];return make_double2((double)half_to_float_(x.x),(double)half_to_float_(x.y));}'''
_HALF_WRITE=r'''__device__ __forceinline__ float write_(ushort2* E,unsigned long long c,double2 x,double scale){
 unsigned short hr=float_to_half_((float)(x.x*scale)),hi=float_to_half_((float)(x.y*scale));
 E[c]=make_ushort2(hr,hi);float r=half_to_float_(hr),i=half_to_float_(hi);return r*r+i*i;
}'''

def half_code():
    text=_CODE
    lo=text.index('__device__ __forceinline__ double2 read_(')
    hi=text.index('__device__ __forceinline__ int sign_(',lo)
    text=text[:lo]+_HALF_HELPERS+_HALF_READ+'\n'+_HALF_WRITE+'\n'+text[hi:]
    for key,val in {'VALTYPE':'ushort2','REALTYPE':'float','MAKEVAL':'make_ushort2','MAXTYPE':'unsigned int','MAXFUN':'fmaxf','MAXBITS':'__float_as_uint'}.items():
        text=text.replace(key,val)
    return text

class HalfStorageInplaceGPUMixin(InplaceGPUMixin):
    def alloc(self,dtype=np.complex64,device=0):
        # dtype describes the exported terminal vector; stored components are
        # always float16. Do not mistake this for the FP32 baseline allocator.
        if np.dtype(dtype)!=np.dtype(np.complex64):raise ValueError('Half storage exports complex64; use baseline executor for complex128')
        import cupy as cp
        self.cp=cp;cp.cuda.Device(device).use();self.inplace_fp64=False;self.inplace_storage='float16-components'
        self.inplace_steps,self.inplace_terminal_basis,self.inplace_maxslots=compile_inplace(self.order,self.fut,self.fixed,len(self.steps))
        self.inplace_rmax=max([0]+[len(s.active_out) for s in self.inplace_steps])
        if self.inplace_maxslots!=self.inplace_rmax:raise AssertionError('Unexpected physical slot expansion')
        self.d_inplace=cp.empty((1<<self.inplace_maxslots,2),dtype=cp.uint16)
        self.d_max=cp.zeros(1,dtype=cp.uint32);self._inplace_kernel=cp.RawKernel(half_code(),'inplace_step')
        self.inplace_launch=[]
        for st in self.inplace_steps:
            kind,omit=orbit_descriptor(st);slots=[p for p in st.active_in if p not in omit]
            T=byte_tables([1<<p for p in slots],len(slots));identity=slots==list(range(len(slots)))
            self.inplace_launch.append((kind,1<<len(slots),cp.asarray(T.reshape(-1)),T.shape[0],identity))
        self.inplace_grid=cp.cuda.Device(device).attributes['MultiProcessorCount']*32
        active=tuple(sorted(self.inplace_terminal_basis));ids=cp.arange(1<<len(active),dtype=cp.uint64);physical=cp.zeros_like(ids)
        for j,p in enumerate(active):physical|=((ids>>cp.uint64(j))&cp.uint64(1))<<cp.uint64(p)
        self.d_inplace_terminal_indices=physical;self.buf=True

    def prefix_inplace(self,f,yfix):
        cp=self.cp;self.d_inplace[0,0]=np.uint16(0x3c00);self.d_inplace[0,1]=np.uint16(0)
        offset=0;ls=0.;maximum=1.
        for st,(kind,n,T,nb,identity) in zip(self.inplace_steps,self.inplace_launch):
            ov=(offset>>st.j)&1;offset&=~(1<<st.j);y=0
            if not st.free:
                y=int(yfix[st.v])
                if y:offset^=self.fut[st.j]
            f0,f1=map(complex,f[st.v]);e=math.frexp(maximum)[1];scale=math.ldexp(1.,-e);ls+=e*math.log(2.)
            self.d_max.fill(0)
            self._inplace_kernel((min(self.inplace_grid,(n+255)//256),),(256,),
                (self.d_inplace,np.uint64(n),T,np.int32(nb),np.int32(identity),np.int32(kind),np.uint64(st.K),np.uint64(st.Ka),np.int32(st.pivot),
                 np.uint64(st.beta),np.int32(ov),np.float64(f0.real),np.float64(f0.imag),np.float64(f1.real),np.float64(f1.imag),np.int32(y),np.float64(scale),self.d_max))
            bits=int(cp.asnumpy(self.d_max)[0]);m2=float(np.asarray([bits],np.uint32).view(np.float32)[0])
            if not np.isfinite(m2):raise ArithmeticError('Nonfinite half-storage environment')
            if m2==0:return None,ls,offset
            maximum=math.sqrt(m2)
        values=self.d_inplace[self.d_inplace_terminal_indices].view(cp.float16).astype(cp.float32)
        env=values[:,0].astype(cp.complex64);env.imag=values[:,1]
        return env,ls,offset
