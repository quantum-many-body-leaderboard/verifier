"""Cross-check of the impurity Hamiltonians (qrec_verify/impurity.py) by DMRG in TeNPy, against VarBench's forkTPS
ground-state energies on the instances without an exact row.

TeNPy builds the same Hamiltonian from the parameters with its own fermion operators (SpinHalfFermionSite, Jordan-
Wigner strings as in tests/test_mps.py), one TeNPy site per spatial orbital in the verifier's order
(impurity orbitals, then the bath orbitals of orbital a = 0 .. M-1); the first block ('down'/'ud_0') is TeNPy's
down spin. The particle number is conserved (N from the rows' dof) and S_z is free (cons_Sz=None) unless --sz is given.
`--hyb conj` uses V* d+ c + h.c. instead of V d+ c + h.c. (the convention question of the spin-orbit case).

    python tests/xcheck_impurity_tenpy.py INSTANCE_FILE [--chi 400] [--sweeps 20] [--hyb plain|conj]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(inst: dict, chi: int = 400, sweeps: int = 20, hyb: str = "plain", log=print, par: dict | None = None,
        n_el: int | None = None) -> dict:
    """DMRG of the instance's impurity Hamiltonian built in TeNPy (or of `par`, e.g. a truncated bath, with the
    instance's electron number, or n_el)."""
    from tenpy.algorithms import dmrg
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingModel, MPOModel
    from tenpy.networks.mps import MPS as TMPS
    from tenpy.networks.site import SpinHalfFermionSite
    from qrec_verify import impurity as IM
    par = par or IM.read(IM.file_for(inst))
    if hyb == "conj":
        par = dict(par, bath={b: [[(e, np.conj(V)) for e, V in sites] for sites in per] for b, per in par["bath"].items()})
    T = IM.one_body(par)
    M, nb = par["M"], par["Nb"]
    n = M * (1 + nb)
    N = n_el or IM.n_electrons(inst)
    site = SpinHalfFermionSite(cons_N="N", cons_Sz=None)
    lat = Chain(n, site, bc="open", bc_MPS="finite")
    sp = {par["blocks"][0]: ("d", "Cdd", "Cd"), par["blocks"][1]: ("u", "Cdu", "Cu")}
    U, J, Up = par["U"], par["J"], par["Up"]

    class Model(CouplingModel, MPOModel):
        def __init__(self):
            CouplingModel.__init__(self, lat, explicit_plus_hc=False)
            for b, (s, cd, c) in sp.items():
                for i in range(n):
                    if abs(T[b][i, i]) > 0:
                        self.add_onsite_term(float(np.real(T[b][i, i])), i, "N" + s)
                    for j in range(n):
                        if i != j and abs(T[b][i, j]) > 1e-14:
                            # T_ij c+_i c_j, Jordan-Wigner string between i and j
                            # (as tests/test_mps.py: TeNPy 1.1's add_coupling_term inserts no string itself)
                            if i < j:
                                self.add_coupling_term(T[b][i, j], i, j, cd + " JW", c, op_string="JW")
                            else:                   # c+_i c_j = - c_j c+_i with j < i
                                self.add_coupling_term(-T[b][i, j], j, i, c + " JW", cd, op_string="JW")
            for m in range(M):
                self.add_onsite_term(U, m, "NuNd")
            for m in range(M):
                for mp in range(M):
                    if m == mp:
                        continue
                    lo, hi = min(m, mp), max(m, mp)
                    self.add_coupling_term(Up, m if m < mp else mp, hi, "Nu" if m < mp else "Nd", "Nd" if m < mp else "Nu")
                    if mp > m:
                        for s in ("u", "d"):
                            self.add_coupling_term(Up - J, m, mp, "N" + s, "N" + s)
                    if J:
                        # -J S+_m S-_mp (bosonic composites, no string); +J P+_m P_mp with P+ = Cdu Cdd
                        self.add_coupling_term(-J, lo, hi, "Sp" if lo == m else "Sm", "Sm" if lo == m else "Sp")
                        self.add_coupling_term(J, lo, hi, "Cdu Cdd" if lo == m else "Cd Cu", "Cd Cu" if lo == m else "Cdu Cdd")
            MPOModel.__init__(self, lat, self.calc_H_MPO())

    t0 = time.perf_counter()
    model = Model()
    # initial product state with N electrons: fill the lowest bath levels first
    T0 = T[par["blocks"][0]]
    order = np.argsort(np.real(np.diag(T0)))
    occ = ["empty"] * n
    k = 0
    for i in order:
        if k + 2 <= N:
            occ[i] = "full"
            k += 2
    for i in order:
        if k < N and occ[i] == "empty":
            occ[i] = "up"
            k += 1
    psi = TMPS.from_product_state(lat.mps_sites(), occ, bc="finite", dtype=complex if par["soc"] else float)
    params = {"mixer": True, "max_E_err": 1e-12, "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
              "max_sweeps": sweeps, "min_sweeps": min(6, sweeps),
              "max_trunc_err": 1.0}   # TeNPy 1.1 aborts above 1e-4 by default; the variance reports the truncation
    info = dmrg.run(psi, model, params)
    E = float(np.real(info["E"]))
    log(json.dumps({"instance": inst["instance_id"], "energy": E, "hyb": hyb, "stage": "dmrg done"}))
    H2 = model.H_MPO.variance(psi) if hasattr(model.H_MPO, "variance") else None
    out = {"instance": inst["instance_id"], "energy": E, "chi": chi, "variance": None if H2 is None else float(np.real(H2)),
           "N": float(np.sum(psi.expectation_value("Ntot"))), "Sz": float(np.real(np.sum(psi.expectation_value("Sz")))),
           "hyb": hyb, "wall_s": time.perf_counter() - t0,
           "varbench": sorted(r["energy"] for r in inst["rows"] if r.get("bound_type") == "variational")[:1]}
    log(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_file")
    ap.add_argument("--chi", type=int, default=400)
    ap.add_argument("--sweeps", type=int, default=20)
    ap.add_argument("--hyb", default="plain", choices=("plain", "conj"))
    a = ap.parse_args()
    run(json.load(open(a.instance_file)), a.chi, a.sweeps, a.hyb)
