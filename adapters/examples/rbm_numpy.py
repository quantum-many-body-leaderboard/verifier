"""The reference RBM as a numpy host function.  --state python:rbm_numpy.py:build --params rbm.npz"""

import numpy as np


def build(params_path):
    z = np.load(params_path)
    W, b, a = z["W"], z["b"], z["a"]

    def logpsi(x):                                    # x: int8, (n_conf, n_sites)
        x = x.astype(np.complex128)
        t = x @ W + b
        t = np.where(t.real < 0, -t, t)
        return (t + np.log(1.0 + np.exp(-2.0 * t)) - np.log(2.0)).sum(-1) + x @ a
    return logpsi
