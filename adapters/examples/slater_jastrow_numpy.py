"""The reference Slater-Jastrow state as a numpy host function.
--state python:slater_jastrow_numpy.py:build --params sj.npz"""

import numpy as np


def build(params_path):
    z = np.load(params_path)
    M, V = z["M"], z["V"]
    n_f = M.shape[1]

    def logpsi(x):                                    # x: int8, (n_conf, 2 N)
        occ = np.nonzero(x > 0)[1].reshape(x.shape[0], n_f)
        sign, logabs = np.linalg.slogdet(M[occ])
        xf = x.astype(np.float64)
        return logabs + 0.5 * ((xf @ V) * xf).sum(-1) + 1j * np.where(sign < 0, np.pi, 0.0)
    return logpsi
