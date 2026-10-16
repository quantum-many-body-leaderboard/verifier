"""Synthetic checks of qrec_verify.stats: every gate must pass on clean data and fire on the
failure it exists for. numpy/scipy only; runs with pytest or as `python tests/test_stats.py`."""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from qrec_verify import stats as S  # noqa: E402

M, N = 64, 2048


def ar1(rng, m, n, phi, burn=2000):
    """Stationary AR(1) chains with unit marginal variance."""
    e = rng.standard_normal((m, n + burn)) * math.sqrt(1 - phi ** 2)
    x = np.empty_like(e)
    x[:, 0] = rng.standard_normal(m)
    for t in range(1, n + burn):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x[:, burn:]


def gate(d, name):
    return next(g for g in d.gates if g.name == name).status


def test_iid_passes():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((M, N))
    d = S.diagnose(x)
    assert d.status == "pass", [(g.name, g.status, g.detail) for g in d.gates]
    assert 0.85 < d.values["iat"] < 1.2
    assert d.values["sigma_ratio_max_min"] < 1.3
    assert d.values["alpha"] > 3.0


def test_ar1_iat_and_sigma():
    rng = np.random.default_rng(2)
    phi = 0.9
    x = ar1(rng, M, 4096, phi)
    true_iat = (1 + phi) / (1 - phi)                       # 19
    e = S.ess(x)
    assert abs(e["iat"] / true_iat - 1) < 0.15, e["iat"]
    s = S.sigma_estimates(x)
    true_sigma = math.sqrt(true_iat / x.size)
    for k in ("sigma_iat", "sigma_blocking", "sigma_chain_means"):
        assert abs(s[k] / true_sigma - 1) < 0.25, (k, s[k], true_sigma)
    tau, w = S.iat_sokal(x[0])
    assert abs(tau / true_iat - 1) < 0.4, tau


def test_stuck_chains_fail_rhat():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((M, N))
    x[:8] += 3.0                                          # 8 chains in another mode
    d = S.diagnose(x)
    assert gate(d, "rhat") == "fail", d.values["e_rhat"]
    assert d.status == "fail"


def test_transient_fails_thermalisation():
    rng = np.random.default_rng(4)
    x = ar1(rng, M, N, 0.5)
    x += 5.0 * np.exp(-np.arange(N) / 300.0)              # burn-in not discarded
    d = S.diagnose(x)
    assert gate(d, "thermalisation") == "fail", (d.values["geweke_z"], d.values["mser_cut_fraction"])


def test_tails():
    rng = np.random.default_rng(5)
    heavy = S.tails(rng.standard_t(1.5, size=(M, N)))
    mid = S.tails(rng.standard_t(2.5, size=(M, N)))
    light = S.tails(rng.standard_normal((M, N)))
    assert heavy["alpha"] < 2.0, heavy
    assert 2.0 <= mid["alpha"] < 3.0, mid
    assert light["alpha"] >= 3.0, light
    d = S.diagnose(rng.standard_t(1.5, size=(M, N)))
    assert gate(d, "tails") == "warn" and "below 2" in [g.detail for g in d.gates if g.name == "tails"][0]


def test_acceptance_frozen_and_decay():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((M, N))
    acc = np.full((M, 64), 0.3)
    assert gate(S.diagnose(x, acc=acc), "acceptance") == "pass"
    frozen = acc.copy()
    frozen[3, 40:] = 0.0                                  # one chain stops moving
    assert gate(S.diagnose(x, acc=frozen), "acceptance") == "fail"
    decay = acc * np.linspace(1.0, 0.2, 64)[None, :]      # acceptance collapses over the run
    assert gate(S.diagnose(x, acc=decay), "acceptance") == "warn"


def test_modes():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, N))
    la = rng.standard_normal((M, N)) * 0.3
    assert gate(S.diagnose(x, logabs=la), "modes") == "pass"
    hi = la.copy()
    hi[:6] += 6.0                                         # 6 of 64 chains in a high-|psi| mode
    d = S.diagnose(x, logabs=hi)
    assert gate(d, "modes") == "fail", d.values["mode_weight"]
    small = la.copy()
    small[:2] += 1.5
    assert gate(S.diagnose(x, logabs=small), "modes") == "warn"


def test_precision_gate():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((M, N))
    sig = S.sigma_estimates(x)["sigma"]
    assert gate(S.diagnose(x, sigma_target_per_site=2 * sig), "precision") == "pass"
    assert gate(S.diagnose(x, sigma_target_per_site=0.5 * sig), "precision") == "fail"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
