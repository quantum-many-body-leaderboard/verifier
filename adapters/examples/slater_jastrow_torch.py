"""The reference Slater-Jastrow state in PyTorch.  --state torch:slater_jastrow_torch.py:build --params sj.pt"""

import math

import torch


class SlaterJastrow(torch.nn.Module):
    def __init__(self, n_orb, n_f):
        super().__init__()
        self.n_f = n_f
        self.M = torch.nn.Parameter(torch.zeros(n_orb, n_f, dtype=torch.float64))
        self.V = torch.nn.Parameter(torch.zeros(n_orb, n_orb, dtype=torch.float64))

    def forward(self, x):                             # x: int8, (n_conf, 2 N), entries 0/1
        occ = torch.nonzero(x > 0)[:, 1].reshape(x.shape[0], self.n_f)     # increasing per row
        sign, logabs = torch.linalg.slogdet(self.M[occ])
        xf = x.to(torch.float64)
        jas = 0.5 * ((xf @ self.V) * xf).sum(-1)
        phase = torch.where(sign < 0, torch.full_like(logabs, math.pi), torch.zeros_like(logabs))
        return torch.complex(logabs + jas, phase)


def build(instance):
    n = int(instance["n_sites"])
    return SlaterJastrow(2 * n, 2 * int(instance["params"]["Nf"]))
