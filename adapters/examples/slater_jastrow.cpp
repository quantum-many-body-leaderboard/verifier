// The reference Slater-Jastrow state in C++.  Build: c++ -O2 -shared -fPIC -I.. slater_jastrow.cpp -o libsj.so
// --state c:libsj.so --params sj.bin  (sj.bin: int32 n_orb, int32 n_f, then float64 M (n_orb x n_f,
// row major) and V (n_orb x n_orb))
#include "qmbl_state.h"
#include <cmath>
#include <cstdio>
#include <vector>
static int NO = 0, NF = 0;
static std::vector<double> M, V;

extern "C" int qmbl_init(const char *path) {
    FILE *f = std::fopen(path, "rb");
    int32_t h[2];
    if (!f || std::fread(h, 4, 2, f) != 2) return 1;
    NO = h[0]; NF = h[1]; M.resize(size_t(NO) * NF); V.resize(size_t(NO) * NO);
    size_t got = std::fread(M.data(), 8, M.size(), f) + std::fread(V.data(), 8, V.size(), f);
    std::fclose(f);
    return got == M.size() + V.size() ? 0 : 2;
}

// log|det A| and the sign of det A by LU with partial pivoting (A is n x n, overwritten).
static double logabsdet(std::vector<double> &A, int n, int &sign) {
    double la = 0.0; sign = 1;
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r) if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c])) p = r;
        if (A[p * n + c] == 0.0) { sign = 0; return -INFINITY; }
        if (p != c) { for (int k = 0; k < n; ++k) std::swap(A[p * n + k], A[c * n + k]); sign = -sign; }
        double d = A[c * n + c];
        if (d < 0) sign = -sign;
        la += std::log(std::fabs(d));
        for (int r = c + 1; r < n; ++r) {
            double m = A[r * n + c] / d;
            for (int k = c + 1; k < n; ++k) A[r * n + k] -= m * A[c * n + k];
        }
    }
    return la;
}

extern "C" void logpsi(const int8_t *x, int n_conf, int n_sites, double *re, double *im) {
    std::vector<double> A(size_t(NF) * NF);
    for (int k = 0; k < n_conf; ++k) {
        const int8_t *s = x + size_t(k) * n_sites;
        int row = 0;
        double jas = 0.0;
        for (int i = 0; i < n_sites && n_sites == NO; ++i) {
            if (!s[i]) continue;
            if (row < NF) for (int j = 0; j < NF; ++j) A[row * NF + j] = M[size_t(i) * NF + j];
            ++row;
            for (int j = 0; j < n_sites; ++j) if (s[j]) jas += 0.5 * V[size_t(i) * NO + j];
        }
        if (row != NF) { re[k] = im[k] = NAN; continue; }    // wrong lattice or filling
        int sign;
        re[k] = logabsdet(A, NF, sign) + jas;
        im[k] = sign < 0 ? M_PI : 0.0;
    }
}
