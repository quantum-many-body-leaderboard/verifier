/* qmbl_state.h - the C ABI of a state for the QMBL verifier (op_verify.py --state c:libstate.so).
 *
 * The verifier loads the shared library with ctypes and calls, from one thread at a time:
 *
 *   qmbl_init(path)   once, only when --params PATH is given (optional export);
 *                     return 0 on success, anything else aborts the measurement.
 *   logpsi(x, n_conf, n_sites, re, im)
 *                     x: n_conf configurations of n_sites int8 entries each, row by row
 *                        (x[k * n_sites + i] is entry i of configuration k). Spin models: +1/-1 per
 *                        site, n_sites = number of sites. Hubbard models: 0/1 per orbital in NetKet's
 *                        SpinOrbitalFermions order, so n_sites = 2 x number of sites.
 *                        Site index i = x * ly + y.
 *                     re[k], im[k]: real and imaginary part of log psi of configuration k
 *                        (log|psi| and the phase; the phase is read modulo 2 pi).
 *
 * logpsi must be deterministic and keep no state between calls. It may be called with any
 * n_conf up to the verifier's chunk size (8192 by default, --host-chunk).
 */
#ifndef QMBL_STATE_H
#define QMBL_STATE_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int qmbl_init(const char *params_path);
void logpsi(const int8_t *x, int n_conf, int n_sites, double *re, double *im);

#ifdef __cplusplus
}
#endif
#endif
