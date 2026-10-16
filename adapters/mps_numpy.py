"""Write an MPS from plain arrays to the verifier's file format (MPS_FORMAT.md). numpy and h5py only.

For codes that dump their tensors (C++, Fortran, Julia without ITensors):

    from adapters.mps_numpy import write_mps, snake_order
    write_mps("state.h5", tensors, site_order=snake_order(lx, ly), local_basis="electron",
              energy=E_total, instance_id="Hubbard/square_16_P_4_8", producer="my-dmrg 2.1")

tensors[k] is the tensor of chain position k with legs (left bond, physical, right bond); the first
and last may also be given without their size-1 boundary leg. `leg_order` names another layout, e.g.
"plr" for (physical, left, right). The local basis must already be the format's (spin-half: [up, down];
electron: [empty, up, down, up+down] with |up+down> = c+_up c+_dn |0>, creation operators ordered by
chain position, up before down); `basis_map` reorders a different local order, and `double_sign=-1`
converts a code that defines |up+down> = c+_dn c+_up |0>.
"""

from __future__ import annotations

import numpy as np

FORMAT, FORMAT_VERSION = "qmbl-mps", 1
DIMS = {"spin-half": 2, "electron": 4}


def site_index(x: int, y: int, ly: int) -> int:
    """QMBL's site index on an lx x ly lattice."""
    return x * ly + y


def column_order(lx: int, ly: int) -> list:
    """Chain position k -> site: x-major, y fastest (QMBL's own numbering, order[k] = k)."""
    return [site_index(x, y, ly) for x in range(lx) for y in range(ly)]


def snake_order(lx: int, ly: int) -> list:
    """Boustrophedon: y runs up in even columns x and down in odd ones."""
    return [site_index(x, y if x % 2 == 0 else ly - 1 - y, ly) for x in range(lx) for y in range(ly)]


def _canon(t: np.ndarray, k: int, n: int, leg_order: str) -> np.ndarray:
    t = np.asarray(t)
    if t.ndim == 2 and n > 1 and k in (0, n - 1):
        legs = leg_order.replace("l", "") if k == 0 else leg_order.replace("r", "")
        t = np.expand_dims(t, axis=0)
        legs = ("l" if k == 0 else "r") + legs
    elif t.ndim == 3:
        legs = leg_order
    else:
        raise ValueError(f"tensor {k}: {t.ndim} legs")
    return np.transpose(t, [legs.index(c) for c in "lpr"])


def write_mps(path: str, tensors, site_order, local_basis: str, *, energy: float | None = None,
              instance_id: str | None = None, producer: str | None = None, leg_order: str = "lpr",
              basis_map=None, double_sign: int = 1, quantum_numbers: dict | None = None,
              dtype=None) -> str:
    """Write the file. `energy` is the state's energy in QMBL's units (total; spins in Pauli units,
    i.e. 4 x the S.S total), as the producing code computes it for these tensors; the verifier
    reports whether it reproduces it. `basis_map[v]` = the code's local index of the format's index
    v. `quantum_numbers` (optional dict name -> array) is stored under /quantum_numbers and ignored."""
    import h5py
    if local_basis not in DIMS:
        raise ValueError(f"local_basis must be one of {sorted(DIMS)}")
    n = len(tensors)
    order = np.asarray(site_order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(n)):
        raise ValueError("site_order must be a permutation of 0..n-1 with one entry per tensor")
    out = []
    for k, t in enumerate(tensors):
        a = _canon(t, k, n, leg_order)
        if basis_map is not None:
            a = a[:, list(basis_map), :]
        if double_sign == -1:
            if local_basis != "electron":
                raise ValueError("double_sign applies to the electron basis")
            a = a.copy()
            a[:, 3, :] *= -1
        if a.shape[1] != DIMS[local_basis]:
            raise ValueError(f"tensor {k}: physical dimension {a.shape[1]}, {local_basis} needs {DIMS[local_basis]}")
        if out and out[-1].shape[2] != a.shape[0]:
            raise ValueError(f"bond between tensors {k - 1} and {k} does not match")
        out.append(a)
    if out[0].shape[0] != 1 or out[-1].shape[2] != 1:
        raise ValueError("the outer bonds must have dimension 1")
    if dtype is None:
        dtype = np.complex128 if any(np.iscomplexobj(a) for a in out) else np.float64
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["local_basis"] = local_basis
        f.attrs["n_sites"] = n
        if instance_id is not None:
            f.attrs["instance_id"] = instance_id
        if energy is not None:
            f.attrs["energy"] = float(energy)
        if producer is not None:
            f.attrs["producer"] = producer
        f.create_dataset("site_order", data=order)
        g = f.create_group("tensors")
        for k, a in enumerate(out):
            g.create_dataset(str(k), data=np.ascontiguousarray(a, dtype=dtype))
        if quantum_numbers:
            q = f.create_group("quantum_numbers")
            for name, v in quantum_numbers.items():
                q.create_dataset(name, data=np.asarray(v))
    return path
