// The reference RBM in C++.  Build: c++ -O2 -shared -fPIC -I.. rbm.cpp -o librbm.so
// --state c:librbm.so --params rbm.bin  (rbm.bin: int32 N, int32 H, then complex128 W (N x H,
// row major), b (H), a (N))
#include "qmbl_state.h"
#include <complex>
#include <cstdio>
#include <vector>
using C = std::complex<double>;
static int N = 0, H = 0;
static std::vector<C> W, b, a;

extern "C" int qmbl_init(const char *path) {
    FILE *f = std::fopen(path, "rb");
    int32_t nh[2];
    if (!f || std::fread(nh, 4, 2, f) != 2) return 1;
    N = nh[0]; H = nh[1]; W.resize(size_t(N) * H); b.resize(H); a.resize(N);
    size_t got = std::fread(W.data(), 16, W.size(), f) + std::fread(b.data(), 16, H, f) + std::fread(a.data(), 16, N, f);
    std::fclose(f);
    return got == W.size() + H + N ? 0 : 2;
}

static C log_cosh(C z) { if (z.real() < 0) z = -z; return z + std::log(1.0 + std::exp(-2.0 * z)) - std::log(2.0); }

extern "C" void logpsi(const int8_t *x, int n_conf, int n_sites, double *re, double *im) {
    for (int k = 0; k < n_conf; ++k) {
        if (n_sites != N) { re[k] = im[k] = 0.0 / 0.0; continue; }    // wrong lattice: NaN
        const int8_t *s = x + size_t(k) * n_sites;
        C out = 0;
        for (int i = 0; i < N; ++i) out += a[i] * double(s[i]);
        for (int j = 0; j < H; ++j) {
            C t = b[j];
            for (int i = 0; i < N; ++i) t += W[size_t(i) * H + j] * double(s[i]);
            out += log_cosh(t);
        }
        re[k] = out.real(); im[k] = out.imag();
    }
}
