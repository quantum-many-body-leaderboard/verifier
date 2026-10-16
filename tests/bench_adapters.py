"""Speed of the adapters on the scorer's frozen spin sampler, CPU only.

A complex RBM (alpha = 1) on the 10x10 J1-J2 model (J2 = 0.2), written in every framework of
adapters/examples/. For each adapter: time per sweep of the frozen parallel-tempering sampler
(256 chains x 16 replicas, 100 Metropolis steps per sweep, each a log psi call on 4096
configurations) and time of the local energies of one block (4096 samples), and from them the
samples per second and the estimated wall time of one frozen score (4096 + 256 sweeps, 16 blocks).

    python tests/bench_adapters.py [--only torch,c] [--sweeps 4]
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_adapters as T  # noqa: E402  (sets jax x64)
import netket as nk  # noqa: E402

from qrec_verify import external as E, hamiltonians as Hm, instance as I, score as SC  # noqa: E402


def main(argv):
    kinds = [k for k in T.ADAPTERS if T.have(k)]
    if "--only" in argv:
        want = argv[argv.index("--only") + 1].split(",")
        kinds = [k for k in kinds if k in want]
    sweeps = int(argv[argv.index("--sweeps") + 1]) if "--sweeps" in argv else 4
    ws = pathlib.Path(tempfile.mkdtemp(prefix="op-bench-", dir=os.environ.get("TMPDIR") or None))
    try:
        inst = I.load("J1J2/square_100_P_0.2")
        system = Hm.build(inst)
        n = 100
        rng = np.random.default_rng(7)
        P = {"W": T.cplx(rng, (n, n), 0.02), "b": T.cplx(rng, (n,), 0.02), "a": T.cplx(rng, (n,), 0.02)}
        T.export(ws, "rbm", P, (n, n), "rbm.cpp")
        specs = {"flax": (f"{T.EX}/rbm_flax.py:build_model", ws / "rbm.msgpack"),
                 "jax": (f"jax:{T.EX}/rbm_jax.py:build", ws / "rbm.npz"),
                 "python": (f"python:{T.EX}/rbm_numpy.py:build", ws / "rbm.npz"),
                 "torch": (f"torch:{T.EX}/rbm_torch.py:RBM", ws / "rbm.pt"),
                 "c": (f"c:{ws}/librbm.so", ws / "rbm.bin"),
                 "julia": (f"julia:{T.EX}/rbm.jl", ws / "rbm.jld2")}
        cfg = SC.SPIN
        n_blocks = cfg["min_samples"] // (cfg["n_chains"] * cfg["block"])
        n_sweeps_score = cfg["n_burn"] + cfg["min_samples"] // cfg["n_chains"]
        out = {}
        for k in kinds:
            spec, params = specs[k]
            st = E.load_state(spec, str(params), inst)
            try:
                smp = SC.spin_sampler(system)
                vs = nk.vqs.MCState(smp, model=st.model, variables=st.variables, sampler_seed=1,
                                    n_samples=cfg["n_chains"] * cfg["block"], n_discard_per_chain=0)
                t0 = time.perf_counter()
                np.asarray(vs.sample(chain_length=1, n_discard_per_chain=0))          # compile
                t_compile = time.perf_counter() - t0
                t0 = time.perf_counter()
                np.asarray(vs.sample(chain_length=1, n_discard_per_chain=sweeps - 1))
                t_sweep = (time.perf_counter() - t0) / sweeps
                vs.sample(chain_length=cfg["block"], n_discard_per_chain=0)
                le = vs.local_estimators(system.hamiltonian)                          # compile
                np.asarray(getattr(le, "data", le))
                t0 = time.perf_counter()
                le = vs.local_estimators(system.hamiltonian)
                e = np.asarray(getattr(le, "data", le))
                t_eloc = time.perf_counter() - t0
            finally:
                st.close()
            block_s = cfg["block"] * t_sweep + t_eloc
            r = {"s_per_sweep": t_sweep, "s_local_energy_per_block": t_eloc, "compile_s": t_compile,
                 "samples_per_s": cfg["n_chains"] * cfg["block"] / block_s,
                 "configs_per_s_in_sampler": cfg["n_chains"] * cfg["n_replicas"] * n / t_sweep,
                 "score_hours_estimate": (n_sweeps_score * t_sweep + n_blocks * t_eloc) / 3600,
                 "mean_e_loc_per_site": float(np.real(e).mean()) / 400}
            out[k] = r
            print(f"  {k}: {json.dumps(r)}", flush=True)
        print("BENCH " + json.dumps(out))
    finally:
        shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:])
