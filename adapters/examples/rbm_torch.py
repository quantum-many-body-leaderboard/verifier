"""The reference RBM in PyTorch.  --state torch:rbm_torch.py:RBM --params rbm.pt
(rbm.pt = torch.save(model.state_dict(), "rbm.pt"))."""

import math

import torch


class RBM(torch.nn.Module):
    def __init__(self, instance, alpha=1):           # the verifier passes the QMBL instance (a dict)
        super().__init__()
        n = int(instance["n_sites"])
        c = torch.complex128
        self.W = torch.nn.Parameter(torch.zeros(n, alpha * n, dtype=c))
        self.b = torch.nn.Parameter(torch.zeros(alpha * n, dtype=c))
        self.a = torch.nn.Parameter(torch.zeros(n, dtype=c))

    def forward(self, x):                             # x: int8, (n_conf, n_sites), entries +-1
        x = x.to(torch.complex128)
        t = x @ self.W + self.b
        t = torch.where(t.real < 0, -t, t)
        return (t + torch.log(1.0 + torch.exp(-2.0 * t)) - math.log(2.0)).sum(-1) + x @ self.a
