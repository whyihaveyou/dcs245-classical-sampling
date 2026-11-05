# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Kim Renaud (Calcul Québec)
"""Kernel variants for the GPU micro-benchmark (timing experiments only; the production kernel is
gpu_kernel._SRC). All variants compute the same quantity as scan_step."""

_ARGS = """const float2* __restrict__ E, float2* __restrict__ out, const unsigned long long n_out,
    const unsigned long long* __restrict__ T, const unsigned long long* __restrict__ combos, const int ncombos,
    const int r_in, const unsigned long long beta, const unsigned long long ov, const int is_free,
    const float f0r, const float f0i, const float f1r, const float f1i, const int yfix"""

_COMMON = """
__device__ __forceinline__ void pick_weight(const unsigned long long p, const unsigned long long c,
    const int r_in, const unsigned long long beta, const unsigned long long ov, const int is_free,
    const float f0r, const float f0i, const float f1r, const float f1i, const int yfix, float& wr, float& wi)
{
    const unsigned long long par = ((unsigned long long)(__popcll(c & beta) & 1)) ^ ov;
    if (is_free) {
        const unsigned long long y = (p >> r_in) & 1ULL;
        if (y) { wr = f1r; wi = f1i; } else { wr = f0r; wi = f0i; }
        if (y & par) { wr = -wr; wi = -wi; }
    } else {
        if (yfix) { wr = f1r; wi = f1i; } else { wr = f0r; wi = f0i; }
        if (yfix && par) { wr = -wr; wi = -wi; }
    }
}
"""

# pure streaming copy: the bandwidth ceiling of a read-then-write pass over the table
STREAM = """
extern "C" __global__ void stream_copy(const float2* __restrict__ E, float2* __restrict__ out,
    const unsigned long long n_out, const unsigned long long mask)
{
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long cp = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; cp < n_out; cp += stride)
        out[cp] = E[cp & mask];
}
"""

# tables in shared memory, byte loop fully unrolled (NB is a compile-time constant)
SMEM = _COMMON + """
extern "C" __global__ void scan_smem(""" + _ARGS + """)
{
    __shared__ unsigned long long sT[NB * 256];
    for (int i = threadIdx.x; i < NB * 256; i += blockDim.x) sT[i] = T[i];
    __syncthreads();
    const unsigned long long mask = (r_in >= 64) ? ~0ULL : ((1ULL << r_in) - 1ULL);
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    for (unsigned long long cp = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; cp < n_out; cp += stride) {
        unsigned long long p0 = 0ULL;
        #pragma unroll
        for (int b = 0; b < NB; ++b) p0 ^= sT[b * 256 + ((cp >> (8 * b)) & 255ULL)];
        float ar = 0.f, ai = 0.f;
        for (int k = 0; k < ncombos; ++k) {
            const unsigned long long p = p0 ^ combos[k];
            const unsigned long long c = p & mask;
            float wr, wi;
            pick_weight(p, c, r_in, beta, ov, is_free, f0r, f0i, f1r, f1i, yfix, wr, wi);
            const float2 e = E[c];
            ar += e.x * wr - e.y * wi;
            ai += e.x * wi + e.y * wr;
        }
        out[cp] = make_float2(ar, ai);
    }
}
"""

# same, with 4 independent outputs per thread and per iteration (more loads in flight)
ILP4 = _COMMON + """
extern "C" __global__ void scan_ilp4(""" + _ARGS + """)
{
    __shared__ unsigned long long sT[NB * 256];
    for (int i = threadIdx.x; i < NB * 256; i += blockDim.x) sT[i] = T[i];
    __syncthreads();
    const unsigned long long mask = (r_in >= 64) ? ~0ULL : ((1ULL << r_in) - 1ULL);
    const unsigned long long stride = (unsigned long long)blockDim.x * gridDim.x;
    const unsigned long long first = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned long long base = first; base < n_out; base += 4ULL * stride) {
        unsigned long long p0[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const unsigned long long cp = base + (unsigned long long)u * stride;
            unsigned long long x = 0ULL;
            #pragma unroll
            for (int b = 0; b < NB; ++b) x ^= sT[b * 256 + ((cp >> (8 * b)) & 255ULL)];
            p0[u] = x;
        }
        float ar[4] = {0.f, 0.f, 0.f, 0.f}, ai[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k = 0; k < ncombos; ++k) {
            const unsigned long long cb = combos[k];
            float2 e[4]; float wr[4], wi[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned long long p = p0[u] ^ cb;
                const unsigned long long c = p & mask;
                pick_weight(p, c, r_in, beta, ov, is_free, f0r, f0i, f1r, f1i, yfix, wr[u], wi[u]);
                e[u] = E[c];
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                ar[u] += e[u].x * wr[u] - e[u].y * wi[u];
                ai[u] += e[u].x * wi[u] + e[u].y * wr[u];
            }
        }
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const unsigned long long cp = base + (unsigned long long)u * stride;
            if (cp < n_out) out[cp] = make_float2(ar[u], ai[u]);
        }
    }
}
"""
