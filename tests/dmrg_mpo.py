"""DMRG (TeNPy, requirements-mps.txt) on the verifier's own MPO (qrec_verify.mps.build_mpo) of an instance, without
charge conservation (v1.9).

    python tests/dmrg_mpo.py INSTANCE_ID [--chi 128] [--sweeps 30]

Prints a JSON line with the DMRG energy and QMBL's exact rows. Used for exact rows beyond full ED in 1D
(TFIsing chains): the MPO is the verifier's Hamiltonian (tests/test_mps.py checks it against NetKet's operator).
"""
import argparse, json, pathlib, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("iid")
ap.add_argument("--chi", type=int, default=128); ap.add_argument("--sweeps", type=int, default=30)
a = ap.parse_args()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qrec_verify import instance as I, mps as M
from tenpy.linalg import np_conserved as npc
from tenpy.networks.mpo import MPO as TMPO
from tenpy.networks.mps import MPS as TMPS
from tenpy.networks.site import Site
from tenpy.linalg.charges import LegCharge
from tenpy.algorithms import dmrg

inst = I.load(a.iid)
n = inst["n_sites"]
mpo = M.build_mpo(inst, np.arange(n))
W = list(mpo.W) if hasattr(mpo, "W") else list(mpo[0])
d = W[0].shape[-1]
leg = LegCharge.from_trivial(d)
site = Site(leg, [str(k) for k in range(d)])
sites = [site] * n
Ws = []
for k, w in enumerate(W):
    w = np.asarray(w, dtype=complex if np.iscomplexobj(w) else float)
    if k == n - 1:
        w = w[:, 1:2]                     # keep the "done" column only
    Ws.append(npc.Array.from_ndarray_trivial(w, labels=["wL", "wR", "p", "p*"]))
IdL = [0] + [0] * (n - 1) + [None]
IdR = [None] + [1] * (n - 1) + [0]
H = TMPO(sites, Ws, bc="finite", IdL=IdL, IdR=IdR)
rng = np.random.default_rng(1)
psi = TMPS.from_product_state(sites, [int(rng.integers(d)) for _ in range(n)], bc="finite")
t0 = time.time()
eng = dmrg.TwoSiteDMRGEngine(psi, type("Mdl", (), {"H_MPO": H})(), {
    "mixer": True, "max_sweeps": a.sweeps, "min_sweeps": 8, "max_E_err": 1e-13,
    "trunc_params": {"chi_max": a.chi, "svd_min": 1e-12}})
E, psi = eng.run()
ex = [r["energy"] for r in inst["rows"] if str(r.get("bound_type", "")).startswith("exact")]
print(json.dumps({"instance": a.iid, "energy": float(np.real(E)), "chi": a.chi, "max_chi": max(psi.chi),
                  "qmbl_exact": ex, "rel": abs(float(np.real(E)) - ex[0]) / abs(ex[0]) if ex else None,
                  "wall_s": time.time() - t0}))
