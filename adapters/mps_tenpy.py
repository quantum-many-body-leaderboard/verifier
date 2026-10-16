"""Write a TeNPy MPS to the verifier's file format (MPS_FORMAT.md). Needs physics-tenpy and h5py.

    from adapters.mps_tenpy import write_tenpy
    E = model.H_MPO.expectation_value(psi)          # in your model's units
    write_tenpy("state.h5", psi, site_order, energy=4 * E,   # spins: QMBL counts Pauli units
                instance_id="J1J2/square_16_P_0.5")

`site_order[k]` is the QMBL site (x * ly + y) of MPS site k. Supported sites: SpinHalfSite (any
`conserve`) and SpinHalfFermionSite (any `cons_N`, `cons_Sz`); the local states are taken by label
('up', 'down'; 'empty', 'up', 'down', 'full'), so TeNPy's charge-sorted order does not matter. TeNPy
orders fermions by MPS site with a site's up before its down mode (|full> = c+_up c+_dn |0>), which
is the format's order: no signs change. The MPS must be finite (bc 'finite').
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mps_numpy import write_mps  # noqa: E402

LABELS = {"spin-half": ("up", "down"), "electron": ("empty", "up", "down", "full")}


def _basis(site) -> str:
    from tenpy.networks.site import SpinHalfFermionSite, SpinHalfSite
    if isinstance(site, SpinHalfSite):
        return "spin-half"
    if isinstance(site, SpinHalfFermionSite):
        return "electron"
    raise TypeError(f"site type {type(site).__name__}: use SpinHalfSite or SpinHalfFermionSite")


def tenpy_tensors(psi) -> tuple[list, str]:
    """Dense tensors (left, physical, right) in the format's local basis, and the basis name."""
    if psi.bc != "finite":
        raise ValueError("only finite MPS")
    basis = _basis(psi.sites[0])
    out = []
    for k in range(psi.L):
        site = psi.sites[k]
        if _basis(site) != basis:
            raise TypeError("all sites must be of one type")
        perm = [site.state_labels[lab] for lab in LABELS[basis]]
        B = psi.get_B(k, form="Th" if k == 0 else "B").transpose(["vL", "p", "vR"])
        a = B.to_ndarray()[:, perm, :]
        if k == 0:
            a = a * psi.norm
        out.append(a)
    return out, basis


def write_tenpy(path: str, psi, site_order, *, energy: float | None = None, instance_id: str | None = None,
                producer: str | None = None) -> str:
    import tenpy
    tensors, basis = tenpy_tensors(psi)
    return write_mps(path, tensors, site_order, basis, energy=energy, instance_id=instance_id,
                     producer=producer or f"TeNPy {tenpy.__version__}")
