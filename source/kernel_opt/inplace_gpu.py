"""One-buffer GPU executor for the preserved-syndrome plan.

Based on the graph-scan formalism in dcs-rankwidth (Kim Renaud, Apache-2.0).
Novel executor groups each step into disjoint 1-, 2-, or 4-entry orbits. It
loads the complete orbit before writing outputs in that same orbit. No GPU
runtime or distribution-quality claim is made by this source alone.
"""
import math
import numpy as np
from dcs_rankwidth.scan_plan import byte_tables
from inplace_plan import compile_inplace

_CODE=r'''
__device__ __forceinline__ double2 mul(double2 a,double2 b){return make_double2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ __forceinline__ double2 add(double2 a,double2 b){return make_double2(a.x+b.x,a.y+b.y);}
__device__ __forceinline__ double2 signed_(double2 a,int s){return make_double2(a.x*s,a.y*s);}
__device__ __forceinline__ double2 read_(const VALTYPE* E,unsigned long long c){VALTYPE x=E[c];return make_double2((double)x.x,(double)x.y);}
__device__ __forceinline__ REALTYPE write_(VALTYPE* E,unsigned long long c,double2 x,double scale){REALTYPE r=(REALTYPE)(x.x*scale),i=(REALTYPE)(x.y*scale);E[c]=MAKEVAL(r,i);return r*r+i*i;}
__device__ __forceinline__ int sign_(unsigned long long c,unsigned long long beta,int ov){return 1-2*((__popcll(c&beta)&1)^ov);}
extern "C" __global__ void inplace_step(VALTYPE* __restrict__ E,unsigned long long norbits,
 const unsigned long long* __restrict__ T,int nb,int identity_map,int kind,unsigned long long K,unsigned long long Ka,int pivot,
 unsigned long long beta,int ov,double f0r,double f0i,double f1r,double f1i,int fixed_y,double scale,MAXTYPE* dmax){
 unsigned long long stride=(unsigned long long)blockDim.x*gridDim.x;REALTYPE lm=0;
 double2 f0=make_double2(f0r,f0i),f1=make_double2(f1r,f1i);
 for(unsigned long long x=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;x<norbits;x+=stride){
  unsigned long long c=identity_map?x:0;
  if(!identity_map)for(int b=0;b<nb;++b)c^=T[256*b+((x>>(8*b))&255ULL)];
  double2 a=read_(E,c),o0,o1;
  if(kind==0){ // grow: one input, two outputs in the same expansion pair
   o0=mul(f0,a);o1=mul(f1,signed_(a,sign_(c,beta,ov)));
   lm=MAXFUN(lm,write_(E,c,o0,scale));lm=MAXFUN(lm,write_(E,c|(1ULL<<pivot),o1,scale));
  }else if(kind==1){ // mix, nontrivial Pauli shift
   unsigned long long d=c^Ka;double2 b=read_(E,d);
   o0=add(mul(f0,a),mul(f1,signed_(b,sign_(d,beta,ov))));
   o1=add(mul(f0,b),mul(f1,signed_(a,sign_(c,beta,ov))));
   lm=MAXFUN(lm,write_(E,c,o0,scale));lm=MAXFUN(lm,write_(E,d,o1,scale));
  }else if(kind==2){
   o0=mul(add(f0,signed_(f1,sign_(c,beta,ov))),a);lm=MAXFUN(lm,write_(E,c,o0,scale));
  }else if(kind==3){ // replace: old kernel K and newly independent neighbor vector
   unsigned long long d=c^K;double2 b=read_(E,d);
   o0=mul(f0,add(a,b));o1=mul(f1,add(signed_(a,sign_(c,beta,ov)),signed_(b,sign_(d,beta,ov))));
   lm=MAXFUN(lm,write_(E,c,o0,scale));lm=MAXFUN(lm,write_(E,d,o1,scale));
  }else if(kind==4){
   o0=mul(fixed_y?signed_(f1,sign_(c,beta,ov)):f0,a);lm=MAXFUN(lm,write_(E,c,o0,scale));
  }else if(kind==5){
   unsigned long long d=c^K;double2 b=read_(E,d);
   o0=fixed_y?mul(f1,add(signed_(a,sign_(c,beta,ov)),signed_(b,sign_(d,beta,ov)))):mul(f0,add(a,b));
   lm=MAXFUN(lm,write_(E,c,o0,scale));
  }else if(kind==6){ // shrink: read entire four-entry orbit before any output write
   unsigned long long c1=c^K,d=c^Ka,d1=d^K;
   double2 b=read_(E,c1),e=read_(E,d),g=read_(E,d1);
   o0=add(mul(f0,add(a,b)),mul(f1,add(signed_(e,sign_(d,beta,ov)),signed_(g,sign_(d1,beta,ov)))));
   o1=add(mul(f0,add(e,g)),mul(f1,add(signed_(a,sign_(c,beta,ov)),signed_(b,sign_(c1,beta,ov)))));
   lm=MAXFUN(lm,write_(E,c,o0,scale));lm=MAXFUN(lm,write_(E,d,o1,scale));
  }else{ // shrink with Ka=0
   unsigned long long d=c^K;double2 b=read_(E,d);
   o0=add(mul(add(f0,signed_(f1,sign_(c,beta,ov))),a),mul(add(f0,signed_(f1,sign_(d,beta,ov))),b));
   lm=MAXFUN(lm,write_(E,c,o0,scale));
  }
 }
 for(int off=16;off>0;off>>=1)lm=MAXFUN(lm,__shfl_xor_sync(0xffffffffu,lm,off));
 if((threadIdx.x&31)==0)atomicMax(dmax,MAXBITS(lm));
}
'''

def code(fp64=False):
    values={'VALTYPE':'double2' if fp64 else 'float2','REALTYPE':'double' if fp64 else 'float','MAKEVAL':'make_double2' if fp64 else 'make_float2','MAXTYPE':'unsigned long long' if fp64 else 'unsigned int','MAXFUN':'fmax' if fp64 else 'fmaxf','MAXBITS':'__double_as_longlong' if fp64 else '__float_as_uint'}
    text=_CODE
    for key,val in values.items():text=text.replace(key,val)
    # Cast to unsigned for the 64-bit atomicMax overload.
    if fp64:text=text.replace('atomicMax(dmax,__double_as_longlong(lm))','atomicMax(dmax,(unsigned long long)__double_as_longlong(lm))')
    return text

def orbit_descriptor(st):
    kind=st.kind
    if kind=='grow':return 0,()
    if kind=='mix':return (1,(st.pivot_a,)) if st.Ka else (2,())
    if kind=='replace':return 3,(st.pivot,)
    if kind=='fixed':return 4,()
    if kind=='fixed_shrink':return 5,(st.pivot,)
    if kind=='shrink':return (6,(st.pivot,st.pivot_a)) if st.Ka else (7,(st.pivot,))
    raise ValueError(kind)

class InplaceGPUMixin:
    def alloc(self,dtype=np.complex64,device=0):
        import cupy as cp
        self.cp=cp;cp.cuda.Device(device).use();self.inplace_fp64=np.dtype(dtype)==np.dtype(np.complex128)
        if np.dtype(dtype) not in (np.dtype(np.complex64),np.dtype(np.complex128)):raise ValueError(dtype)
        stop=len(self.steps)
        self.inplace_steps,self.inplace_terminal_basis,self.inplace_maxslots=compile_inplace(self.order,self.fut,self.fixed,stop)
        self.inplace_rmax=max([0]+[len(s.active_out) for s in self.inplace_steps])
        if self.inplace_maxslots!=self.inplace_rmax:raise AssertionError('Unexpected physical slot expansion')
        self.d_inplace=cp.empty(1<<self.inplace_maxslots,dtype=dtype)
        self.d_max=cp.zeros(1,dtype=cp.uint64 if self.inplace_fp64 else cp.uint32)
        self._inplace_kernel=cp.RawKernel(code(self.inplace_fp64),'inplace_step')
        self.inplace_launch=[]
        for st in self.inplace_steps:
            kind,omit=orbit_descriptor(st);slots=[p for p in st.active_in if p not in omit]
            cols=[1<<p for p in slots];T=byte_tables(cols,len(slots));identity=slots==list(range(len(slots)))
            self.inplace_launch.append((kind,1<<len(slots),cp.asarray(T.reshape(-1)),T.shape[0],identity))
        self.inplace_grid=cp.cuda.Device(device).attributes['MultiProcessorCount']*32
        active=tuple(sorted(self.inplace_terminal_basis));ids=cp.arange(1<<len(active),dtype=cp.uint64);physical=cp.zeros_like(ids)
        for j,p in enumerate(active):physical|=((ids>>cp.uint64(j))&cp.uint64(1))<<cp.uint64(p)
        self.d_inplace_terminal_indices=physical
        self.buf=True

    def prefix_inplace(self,f,yfix):
        cp=self.cp;self.d_inplace[0]=1;offset=0;ls=0.;maximum=1.
        max_uint,max_float=(np.uint64,np.float64) if self.inplace_fp64 else (np.uint32,np.float32)
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
            bits=int(cp.asnumpy(self.d_max)[0]);m2=float(np.asarray([bits],max_uint).view(max_float)[0])
            if not np.isfinite(m2):raise ArithmeticError('Nonfinite in-place environment')
            if m2==0:return None,ls,offset
            maximum=math.sqrt(m2)
        return self.d_inplace[self.d_inplace_terminal_indices],ls,offset

    def _sweep(self,f,yfix,renorm_every=None):
        env,ls,_=self.prefix_inplace(f,yfix)
        return (0j if env is None else complex(self.cp.asnumpy(env[:1])[0])),ls
