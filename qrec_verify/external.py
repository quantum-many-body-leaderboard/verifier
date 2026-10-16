"""States from any framework.

The scorer (`score.py`) samples and measures a flax module. A state written in another framework
reaches it through one of two wrappers, so the samplers, the estimator and the gates stay the same
for every entry:

- a host function `logpsi(x) -> log psi(x)` (numpy in, numpy out), called from inside the compiled
  sampler through `jax.pure_callback` (`HostState`). PyTorch, Julia, C/C++ and plain Python states
  use this path.
- a pure JAX function `apply_fn(params, x)` with its parameter tree, traced and compiled like a flax
  module (`JaxFunctionState`). JAX states without flax (Equinox, Haiku, plain functions) use it.

The host contract, the same for every adapter:
- `x` is an `int8` array of shape `(n_conf, n_dof)`, C order, one configuration per row, in the
  verifier's convention (README.md, "Conventions"): spins +1/-1 per site, fermions 0/1 per
  orbital in NetKet's `SpinOrbitalFermions` order. `n_dof` is the number of sites for spin models
  and twice that for Hubbard models.
- the return value is `log psi(x)` as `complex128` of shape `(n_conf,)` (a real result is taken as
  a positive amplitude). The phase is read modulo 2 pi.
- the function may be called many times with up to `chunk` configurations; it must be
  deterministic and must not keep state between calls.

State specs (`op_verify.py --state`):

    file.py:build_model                    flax (the default; also `flax:file.py:build_model`)
    torch:file.py:NAME     --params x.pt   NAME(instance) or NAME() returns a torch.nn.Module
    jax:file.py:NAME       --params x.npz  NAME(instance) or NAME() returns apply_fn(params, x)
    julia:file.jl         [--params P]     the file defines logpsi(x::Matrix{Int8})::Vector{ComplexF64}
    c:libstate.so         [--params P]     the library exports logpsi (adapters/qmbl_state.h)
    python:file.py:NAME   [--params P]     NAME(params_path) or NAME(params_path, instance) returns logpsi
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pathlib
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

KINDS = ("flax", "torch", "jax", "julia", "c", "python")
HOST_KINDS = ("torch", "julia", "c", "python")
DEFAULT_CHUNK = 8192
STATE_COLLECTION = "qmbl_state"        # holds the sha256 of the state's files, so state_hash differs per state


# ------------------------------------------------------------------ spec

def parse_spec(spec: str) -> tuple[str, str]:
    """'torch:model.py:Net' -> ('torch', 'model.py:Net'); 'model.py:build_model' -> ('flax', ...)."""
    head, sep, rest = spec.partition(":")
    if sep and head in KINDS:
        return head, rest
    return "flax", spec


def n_dof(inst: dict) -> int:
    """Length of one configuration: sites for spin models, 2 x sites (orbitals) for Hubbard."""
    from . import instance as I
    n = int(inst["n_sites"])
    return n if inst["model"] in I.SPIN_MODELS else 2 * n


def files_sha256(paths) -> np.ndarray:
    """sha256 over the bytes of the state's files (code and parameters), as 32 uint8."""
    h = hashlib.sha256()
    for p in paths:
        if p is None:
            continue
        p = pathlib.Path(p)
        h.update(p.name.encode() + b"\0")
        h.update(p.read_bytes())
    return np.frombuffer(h.digest(), dtype=np.uint8).copy()


def wants_argument(f: Callable) -> bool:
    """True when f has a positional parameter without a default (then it gets the instance)."""
    try:
        ps = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.default is inspect.Parameter.empty and
               p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in ps)


def n_positional(f: Callable) -> int:
    try:
        ps = inspect.signature(f).parameters.values()
    except (TypeError, ValueError):
        return 0
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in ps)


# ------------------------------------------------------------------ host functions

class HostLogPsi:
    """A host function `fn(x int8 (n, d)) -> complex128 (n,)`, made safe to call from the sampler:
    any leading batch shape, chunks of at most `chunk` configurations, one call at a time, output
    shape and dtype checked. Hashed by identity (it is a static field of the flax module)."""

    def __init__(self, fn: Callable, n_dof: int, chunk: int = DEFAULT_CHUNK, name: str = "",
                 close: Callable | None = None):
        self.fn, self.n_dof, self.chunk, self.name = fn, int(n_dof), int(chunk), name
        self._lock = threading.Lock()
        self._close = close
        self.n_calls = 0
        self.n_configs = 0

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x)
        lead = x.shape[:-1]
        if x.shape[-1] != self.n_dof:
            raise ValueError(f"{self.name}: configurations of length {x.shape[-1]}, expected {self.n_dof}")
        flat = np.ascontiguousarray(np.rint(x.reshape(-1, self.n_dof)).astype(np.int8))
        out = np.empty(flat.shape[0], dtype=np.complex128)
        with self._lock:
            for k in range(0, flat.shape[0], self.chunk):
                part = flat[k:k + self.chunk]
                r = np.asarray(self.fn(part))
                if r.shape != (part.shape[0],):
                    raise ValueError(f"{self.name}: logpsi returned shape {r.shape} for {part.shape[0]} "
                                     "configurations, expected (n_conf,)")
                out[k:k + part.shape[0]] = r
                self.n_calls += 1
            self.n_configs += flat.shape[0]
        return out.reshape(lead)

    def close(self):
        if self._close is not None:
            self._close()
            self._close = None


def _host_module():
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class HostState(nn.Module):
        """log psi from a host function, through jax.pure_callback (vectorised: one host call per
        batch, any vmap dimensions folded into the batch)."""
        host: Any

        @nn.compact
        def __call__(self, x):
            out = jax.ShapeDtypeStruct(x.shape[:-1], jnp.complex128)
            return jax.pure_callback(self.host, out, x, vmap_method="broadcast_all")

    return HostState


def _jax_module():
    import flax.linen as nn
    import jax.numpy as jnp

    class JaxFunctionState(nn.Module):
        """log psi = apply_fn(params, x) with a parameter tree stored as the variable params/tree."""
        apply_fn: Callable

        @nn.compact
        def __call__(self, x):
            p = self.get_variable("params", "tree")
            lead = x.shape[:-1]
            y = self.apply_fn(p, jnp.asarray(x).reshape(-1, x.shape[-1]).astype(jnp.int8))
            return jnp.asarray(y).reshape(lead).astype(jnp.complex128)

    return JaxFunctionState


# ------------------------------------------------------------------ loaded state

@dataclass
class LoadedState:
    kind: str
    model: Any                       # flax module
    variables: dict
    host: HostLogPsi | None = None
    files: list = field(default_factory=list)

    def close(self):
        if self.host is not None:
            self.host.close()


def load_state(spec: str, params: str | None, inst: dict, *, chunk: int | None = None) -> LoadedState:
    """Any state spec -> a flax module and its variables, ready for score.evaluate."""
    from . import score as SC
    kind, rest = parse_spec(spec)
    chunk = chunk or int(os.environ.get("QMBL_HOST_CHUNK", DEFAULT_CHUNK))
    d = n_dof(inst)
    if kind == "flax":
        if not params:
            raise ValueError("a flax state needs --params (msgpack)")
        return LoadedState("flax", SC.build_model(rest, inst), SC.load_variables(params), files=[params])
    if kind == "jax":
        return _load_jax(rest, params, inst)
    if kind == "torch":
        fn, files, close = _torch_host(rest, params, inst)
    elif kind == "c":
        fn, files, close = _c_host(rest, params)
    elif kind == "python":
        fn, files, close = _python_host(rest, params, inst)
    elif kind == "julia":
        fn, files, close = _julia_host(rest, params)
    else:                                                    # pragma: no cover
        raise ValueError(kind)
    host = HostLogPsi(fn, d, chunk, name=f"{kind}:{rest}", close=close)
    model = _host_module()(host=host)
    variables = {"params": {}, STATE_COLLECTION: {"sha256": files_sha256(files)}}
    return LoadedState(kind, model, variables, host=host, files=files)


def _file_and_name(rest: str) -> tuple[str, str]:
    path, _, name = rest.rpartition(":")
    if not path or not name:
        raise ValueError(f"expected <file.py>:<name>, got {rest!r}")
    return path, name


# ------------------------------------------------------------------ JAX without flax

def load_npz_tree(path: str) -> dict:
    """An .npz whose keys are '/'-joined paths ('dense/w') -> nested dicts of arrays
    (adapters/jax_npz.py writes this layout)."""
    tree: dict = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            node = tree
            parts = key.split("/")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = np.asarray(z[key])
    return tree


def _load_jax(rest: str, params: str | None, inst: dict) -> LoadedState:
    from . import score as SC
    import jax.numpy as jnp
    import jax
    path, name = _file_and_name(rest)
    factory = SC.load_callable(rest)
    mod = sys.modules[factory.__module__]
    apply_fn = factory(inst) if wants_argument(factory) else factory()
    if hasattr(mod, "load_params"):
        tree = mod.load_params(params)
    elif params is None:
        raise ValueError("a jax state needs --params (.npz or .msgpack), or load_params(path) in its file")
    elif params.endswith(".npz"):
        tree = load_npz_tree(params)
    else:
        import flax
        tree = flax.serialization.msgpack_restore(pathlib.Path(params).read_bytes())
    tree = jax.tree_util.tree_map(jnp.asarray, tree)
    model = _jax_module()(apply_fn=apply_fn)
    return LoadedState("jax", model, {"params": {"tree": tree}}, files=[path] + ([params] if params else []))


# ------------------------------------------------------------------ PyTorch

def _torch_host(rest: str, params: str | None, inst: dict):
    from . import score as SC
    import torch
    path, _ = _file_and_name(rest)
    if not params:
        raise ValueError("a torch state needs --params (a state_dict saved with torch.save)")
    make = SC.load_callable(rest)
    module = make(inst) if wants_argument(make) else make()
    sd = torch.load(params, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    module.load_state_dict(sd)
    dev = os.environ.get("QMBL_TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(dev).eval()

    def fn(x: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            y = module(torch.from_numpy(x).to(dev))
        y = y.detach().to("cpu")
        y = y.to(torch.complex128) if y.is_complex() else y.to(torch.float64)
        return y.numpy().astype(np.complex128, copy=False)

    return fn, [path, params], None


# ------------------------------------------------------------------ C / C++ (ctypes)

def _c_host(lib: str, params: str | None):
    import ctypes
    p = str(pathlib.Path(lib).resolve())
    so = ctypes.CDLL(p)
    f = so.logpsi
    f.restype = None
    f.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    if params is not None:
        try:
            init = so.qmbl_init
        except AttributeError:
            raise ValueError(f"{lib}: --params given but the library exports no qmbl_init") from None
        init.restype = ctypes.c_int
        init.argtypes = [ctypes.c_char_p]
        rc = init(str(pathlib.Path(params).resolve()).encode())
        if rc != 0:
            raise RuntimeError(f"{lib}: qmbl_init({params}) returned {rc}")

    def fn(x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        re = np.empty(n, dtype=np.float64)
        im = np.empty(n, dtype=np.float64)
        f(x.ctypes.data, n, d, re.ctypes.data, im.ctypes.data)
        return re + 1j * im

    return fn, [p] + ([params] if params else []), None


# ------------------------------------------------------------------ plain Python

def _python_host(rest: str, params: str | None, inst: dict):
    from . import score as SC
    path, _ = _file_and_name(rest)
    make = SC.load_callable(rest)
    fn = make(params, inst) if n_positional(make) >= 2 else make(params)
    return fn, [path] + ([params] if params else []), None


# ------------------------------------------------------------------ Julia (juliacall, own process)

class _Worker:
    """A host function served by a child Python process (`qrec_verify.hostproc`). Julia must run on
    the thread that started it, and the sampler calls back from another thread, so the Julia state
    lives in a process of its own; configurations go over a pipe."""

    def __init__(self, kind: str, target: str, params: str | None):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
        # Own pipes, not stdin/stdout: Julia's runtime switches fds 0-2 to non-blocking mode.
        r_req, w_req = os.pipe()
        r_res, w_res = os.pipe()
        self.proc = subprocess.Popen([sys.executable, "-m", "qrec_verify.hostproc", kind, target, params or "",
                                      str(r_req), str(w_res)], stdin=subprocess.DEVNULL, stdout=sys.stderr,
                                     env=env, pass_fds=(r_req, w_res))
        os.close(r_req)
        os.close(w_res)
        self.w = os.fdopen(w_req, "wb")
        self.r = os.fdopen(r_res, "rb")
        msg = self._read_frame()
        if not msg.startswith(b"ready"):
            raise RuntimeError(f"{kind} state {target}: worker failed to start: {msg.decode(errors='replace')}")

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            b = self.r.read(n - len(buf))
            if not b:
                rc = self.proc.poll()
                raise RuntimeError(f"state worker ended (exit {rc})")
            buf += b
        return bytes(buf)

    def _read_frame(self) -> bytes:
        (n,) = struct.unpack("<q", self._read_exact(8))
        if n < 0:                                       # error message
            raise RuntimeError("state worker: " + self._read_exact(-n).decode(errors="replace"))
        return self._read_exact(n)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n, d = x.shape
        self.w.write(struct.pack("<qq", n, d) + x.tobytes())
        self.w.flush()
        return np.frombuffer(self._read_frame(), dtype=np.complex128).copy()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.w.write(struct.pack("<qq", -1, 0))
                self.w.close()
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self.proc.kill()


def _julia_host(jl: str, params: str | None):
    p = str(pathlib.Path(jl).resolve())
    w = _Worker("julia", p, None if params is None else str(pathlib.Path(params).resolve()))
    return w, [p] + ([params] if params else []), w.close
