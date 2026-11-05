# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""
GPU execution of a compiled scan (CuPy). Same algorithm as scan_kernel._step, written in CUDA C.

Each output entry cp of the new table reads at most four entries of the previous one:
    p0 = XOR_b T[b][byte_b(cp)],  p = p0 ^ combo,  c = p & mask,  parity = popcount(c & beta) ^ ov,
    w  = f_v(y) with y = bit r_in of p (free vertex) or the fixed value, negated if y & parity,
    out[cp] = sum_combos E[c] * w.
Two buffers of 2^r_max complex64 entries live on the GPU.

Renormalisation is fused into the scan kernel (scan_step_scaled): each output is multiplied by a power
of two chosen from the maximum modulus of the previous step (exact in floating point), and the maximum
modulus of what the kernel writes is obtained in registers (warp reduction, one atomicMax per warp).
There is no separate reduction or rescaling pass over the table, and no dependency on CuPy's CUB
reductions, which need CUDA headers more recent than some cluster modules provide.
"""
import math
import numpy as np

_SRC = r"""
extern "C" __global__ void scan_step(const float2* __restrict__ E, float2* __restrict__ out,
    const unsigned long long n_out, const unsigned long long* __restrict__ T, const int nbytes,
    const unsigned long long* __restrict__ combos, const int ncombos, const int r_in,
    const unsigned long long beta, const unsigned long long ov, const int is_free,
    const float f0r, const float f0i, const float f1r, const float f1i, const int yfix)
{
    const unsigned long long mask = (r_in >= 64) ? ~0ULL : ((1ULL << r_in) - 1ULL);
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long cp = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; cp < n_out; cp += stride) {
        unsigned long long p0 = 0ULL;
        for (int b = 0; b < nbytes; ++b)
            p0 ^= T[b * 256 + ((cp >> (8 * b)) & 255ULL)];
        float ar = 0.f, ai = 0.f;
        for (int k = 0; k < ncombos; ++k) {
            const unsigned long long p = p0 ^ combos[k];
            const unsigned long long c = p & mask;
            const unsigned long long par = ((unsigned long long)(__popcll(c & beta) & 1)) ^ ov;
            float wr, wi;
            if (is_free) {
                const unsigned long long y = (p >> r_in) & 1ULL;
                if (y) { wr = f1r; wi = f1i; } else { wr = f0r; wi = f0i; }
                if (y & par) { wr = -wr; wi = -wi; }
            } else {
                if (yfix) { wr = f1r; wi = f1i; } else { wr = f0r; wi = f0i; }
                if (yfix && par) { wr = -wr; wi = -wi; }
            }
            const float2 e = E[c];
            ar += e.x * wr - e.y * wi;
            ai += e.x * wi + e.y * wr;
        }
        out[cp] = make_float2(ar, ai);
    }
}
"""


_SRC_SCALED = r"""
extern "C" __global__ void scan_step_scaled(const float2* __restrict__ E, float2* __restrict__ out,
    const unsigned long long n_out, const unsigned long long* __restrict__ T, const int nbytes,
    const unsigned long long* __restrict__ combos, const int ncombos, const int r_in,
    const unsigned long long beta, const unsigned long long ov, const int is_free,
    const double f0r, const double f0i, const double f1r, const double f1i, const int yfix,
    const float scale, unsigned int* __restrict__ d_max)
{
    // Weights and sums are in double precision: the kernel is limited by memory, so this is free, and it
    // removes the systematic bias that float32 weights (1/sqrt(2), ...) would add, about 5e-8 per vertex.
    const unsigned long long mask = (r_in >= 64) ? ~0ULL : ((1ULL << r_in) - 1ULL);
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    const double dscale = (double)scale;
    float lm = 0.f;
    for (unsigned long long cp = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; cp < n_out; cp += stride) {
        unsigned long long p0 = 0ULL;
        for (int b = 0; b < nbytes; ++b)
            p0 ^= T[b * 256 + ((cp >> (8 * b)) & 255ULL)];
        double ar = 0.0, ai = 0.0;
        for (int k = 0; k < ncombos; ++k) {
            const unsigned long long p = p0 ^ combos[k];
            const unsigned long long c = p & mask;
            const unsigned long long par = ((unsigned long long)(__popcll(c & beta) & 1)) ^ ov;
            double wr, wi;
            if (is_free) {
                const unsigned long long y = (p >> r_in) & 1ULL;
                if (y) { wr = f1r; wi = f1i; } else { wr = f0r; wi = f0i; }
                if (y & par) { wr = -wr; wi = -wi; }
            } else {
                if (yfix) { wr = f1r; wi = f1i; } else { wr = f0r; wi = f0i; }
                if (yfix && par) { wr = -wr; wi = -wi; }
            }
            const float2 e = E[c];
            ar += (double)e.x * wr - (double)e.y * wi;
            ai += (double)e.x * wi + (double)e.y * wr;
        }
        const float far = (float)(ar * dscale), fai = (float)(ai * dscale);
        out[cp] = make_float2(far, fai);
        lm = fmaxf(lm, far * far + fai * fai);
    }
    for (int off = 16; off > 0; off >>= 1)
        lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, off));
    if ((threadIdx.x & 31) == 0)
        atomicMax(d_max, __float_as_uint(lm));
}
"""


class GPUMixin:
    """Mix into CompiledScan or SlicedScan to run the sweep on a GPU."""

    profile = False

    def alloc(self, dtype=np.complex64, device=0):
        import cupy as cp
        self.cp = cp
        cp.cuda.Device(device).use()
        self._kernel = cp.RawKernel(_SRC_SCALED, "scan_step_scaled")
        size = 1 << self.rmax
        self.d_buf = [cp.empty(size, dtype=np.complex64), cp.empty(size, dtype=np.complex64)]
        self.d_max = cp.zeros(1, dtype=np.uint32)
        self.d_tables = [cp.asarray(T.reshape(-1)) for T in self.tables]
        self.d_combos = [cp.asarray(c) for c in self.combos]
        sm = cp.cuda.Device(device).attributes["MultiProcessorCount"]
        self._grid, self._block = (sm * 32,), (256,)
        self.buf = True

    def _sweep(self, f, yfix, renorm_every=None):
        """Returns (value, log-scale): the amplitude is value * exp(log-scale) (up to 2^(-N/2)).
        Before step j the table holds the previous result divided by 2^(e_j), where m_j = f * 2^(e_j),
        f in [0.5, 1), is the maximum modulus of the table: the scaling is exact."""
        cp = self.cp
        A, B = self.d_buf
        A[0] = 1.0
        o = 0; logscale = 0.0; m = 1.0
        ev = []
        for j, st in enumerate(self.steps):
            v = st.v
            ov = (o >> j) & 1
            o &= ~(1 << j)
            y = 0
            if not st.free:
                y = int(yfix[v])
                if y:
                    o ^= self.fut[j]
            n_out = 1 << st.r_out
            T = self.tables[j]
            f0, f1 = complex(f[v][0]), complex(f[v][1])
            e = math.frexp(m)[1]
            scale = math.ldexp(1.0, -e)
            logscale += e * math.log(2.0)
            self.d_max.fill(0)
            if self.profile:
                s_ev, e_ev = cp.cuda.Event(), cp.cuda.Event(); s_ev.record()
            self._kernel(self._grid, self._block,
                         (A, B, np.uint64(n_out), self.d_tables[j], np.int32(T.shape[0]),
                          self.d_combos[j], np.int32(len(self.combos[j])), np.int32(st.r_in),
                          np.uint64(st.beta), np.uint64(ov), np.int32(1 if st.free else 0),
                          np.float64(f0.real), np.float64(f0.imag), np.float64(f1.real), np.float64(f1.imag),
                          np.int32(y), np.float32(scale), self.d_max))
            if self.profile:
                e_ev.record(); ev.append((s_ev, e_ev, j, st.r_out))
            A, B = B, A
            bits = int(cp.asnumpy(self.d_max)[0])
            m2 = float(np.array([bits], dtype=np.uint32).view(np.float32)[0])
            if not m2 > 0.0:
                self.d_buf = [A, B]
                return 0j, logscale
            m = math.sqrt(m2)
        self.d_buf = [A, B]
        if self.profile:
            self._events = [(cp.cuda.get_elapsed_time(s_, e_) / 1000.0, j_, ro_) for s_, e_, j_, ro_ in ev]
        return complex(cp.asnumpy(A[0:1])[0]), logscale
