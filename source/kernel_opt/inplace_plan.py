"""Preserved-coordinate syndrome plan with orbit-local in-place updates.

Exact arithmetic compiler/reference. No GPU implementation or runtime claim.
Physical coordinate slots are reused, never compacted. Every output of an
orbit update lies in that same orbit; all orbit inputs must be loaded before
any writes. This is what makes one-buffer GPU execution race-free.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class InplaceStep:
    v:int
    j:int
    kind:str
    active_in:tuple
    active_out:tuple
    beta:int
    K:int=0
    Ka:int=0
    pivot:int=-1
    pivot_a:int=-1
    free:bool=True

def eliminate(columns):
    piv={};deps=[]
    for slot,vec in sorted(columns.items()):
        x=int(vec);coef=1<<slot
        while x:
            p=x.bit_length()-1
            if p not in piv:
                piv[p]=(x,coef);break
            a,b=piv[p];x^=a;coef^=b
        if not x:deps.append(coef)
    return piv,deps

def solve(vec,piv):
    x=int(vec);coef=0
    while x:
        p=x.bit_length()-1
        if p not in piv:break
        a,b=piv[p];x^=a;coef^=b
    return x,coef

def compile_inplace(order,future,fixed,stop=None):
    basis={};steps=[];maxslots=0
    for j,v in enumerate(order[:stop]):
        beta=sum(((b>>j)&1)<<p for p,b in basis.items())
        active=tuple(sorted(basis));restricted={p:b&~(1<<j) for p,b in basis.items()}
        piv,deps=eliminate(restricted)
        if len(deps)>1:raise AssertionError('Deleting one vertex cannot drop more than one rank')
        K=deps[0] if deps else 0;pk=K.bit_length()-1
        free=v not in fixed
        st=InplaceStep(v,j,'',active,(),beta,K=K,pivot=pk,free=free)
        if not free:
            st.kind='fixed_shrink' if K else 'fixed'
            if K:del restricted[pk]
        else:
            residual,Ka=solve(future[j],piv)
            if residual:
                if K:
                    st.kind='replace';restricted[pk]^=future[j]
                else:
                    st.kind='grow';p=0
                    while p in restricted:p+=1
                    st.pivot=p;restricted[p]=future[j]
            else:
                if K:
                    st.kind='shrink'
                    if Ka>>pk&1:Ka^=K
                    st.Ka=Ka;st.pivot_a=Ka.bit_length()-1
                    del restricted[pk]
                else:
                    st.kind='mix';st.Ka=Ka;st.pivot_a=Ka.bit_length()-1
        basis=restricted;st.active_out=tuple(sorted(basis));steps.append(st)
        if basis:maxslots=max(maxslots,max(basis)+1)
    return steps,basis,maxslots

def addresses(active,omit=()):
    slots=[p for p in active if p not in omit];x=np.arange(1<<len(slots),dtype=np.uint64);out=np.zeros_like(x)
    for j,p in enumerate(slots):out|=((x>>np.uint64(j))&np.uint64(1))<<np.uint64(p)
    return out

def signs(x,beta,offset):
    y=x&np.uint64(beta)
    for sh in (32,16,8,4,2,1):y^=y>>np.uint64(sh)
    return 1-2*((y&np.uint64(1))^np.uint64(offset)).astype(np.int64)

def execute_inplace(steps,f,yfix,future,maxslots,normalize=True,return_trace=False):
    E=np.zeros(1<<maxslots,np.complex128);E[0]=1;offset=0;ls=0.;trace=[]
    for st in steps:
        j=st.j;ov=(offset>>j)&1;offset&=~(1<<j)
        f0,f1=map(complex,f[st.v]);kind=st.kind
        y=int(yfix[st.v]) if not st.free else 0
        if y:offset^=future[j]
        if kind=='grow':
            c=addresses(st.active_in);a=E[c].copy();sg=signs(c,st.beta,ov)
            E[c]=f0*a;E[c|np.uint64(1<<st.pivot)]=f1*a*sg
        elif kind=='mix':
            if st.Ka:
                c=addresses(st.active_in,(st.pivot_a,));d=c^np.uint64(st.Ka)
                a,b=E[c].copy(),E[d].copy();s0,s1=signs(c,st.beta,ov),signs(d,st.beta,ov)
                E[c]=f0*a+f1*s1*b;E[d]=f0*b+f1*s0*a
            else:
                c=addresses(st.active_in);E[c]*=f0+f1*signs(c,st.beta,ov)
        elif kind=='replace':
            c=addresses(st.active_in,(st.pivot,));d=c^np.uint64(st.K)
            a,b=E[c].copy(),E[d].copy();s0,s1=signs(c,st.beta,ov),signs(d,st.beta,ov)
            E[c]=f0*(a+b);E[d]=f1*(s0*a+s1*b)
        elif kind=='fixed':
            c=addresses(st.active_in);E[c]*=(f1*signs(c,st.beta,ov) if y else f0)
        elif kind=='fixed_shrink':
            c=addresses(st.active_in,(st.pivot,));d=c^np.uint64(st.K)
            a,b=E[c].copy(),E[d].copy()
            E[c]=f1*(signs(c,st.beta,ov)*a+signs(d,st.beta,ov)*b) if y else f0*(a+b)
        elif kind=='shrink':
            if st.Ka:
                c=addresses(st.active_in,(st.pivot,st.pivot_a));d=c^np.uint64(st.Ka);c1=c^np.uint64(st.K);d1=d^np.uint64(st.K)
                a,b,e,g=E[c].copy(),E[c1].copy(),E[d].copy(),E[d1].copy()
                E[c]=f0*(a+b)+f1*(signs(d,st.beta,ov)*e+signs(d1,st.beta,ov)*g)
                E[d]=f0*(e+g)+f1*(signs(c,st.beta,ov)*a+signs(c1,st.beta,ov)*b)
            else:
                c=addresses(st.active_in,(st.pivot,));d=c^np.uint64(st.K)
                a,b=E[c].copy(),E[d].copy()
                E[c]=(f0+f1*signs(c,st.beta,ov))*a+(f0+f1*signs(d,st.beta,ov))*b
        else:raise ValueError(kind)
        outids=addresses(st.active_out)
        if normalize:
            m=np.max(np.abs(E[outids]))
            if m:
                exponent=math.frexp(float(m))[1];E[outids]*=math.ldexp(1.,-exponent);ls+=exponent*math.log(2.)
        if return_trace:trace.append((E[outids].copy(),ls,offset))
    active=steps[-1].active_out if steps else ()
    return E[addresses(active)],ls,offset,trace
