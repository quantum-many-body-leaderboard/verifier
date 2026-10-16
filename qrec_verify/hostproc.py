"""Child process that serves one state's log psi over a pipe (see external._Worker).

    python -m qrec_verify.hostproc julia /abs/state.jl /abs/params|"" READ_FD WRITE_FD

Protocol on two pipes the parent passes (READ_FD, WRITE_FD; not stdin/stdout, which Julia's runtime
switches to non-blocking mode), little endian: the parent sends (int64 n, int64 d) and n*d int8
configurations; the child answers with an int64 byte count and n complex128 values, or with a
negative count and an error message. n = -1 ends the process. Anything the state prints goes to
stderr, so it cannot corrupt the pipe.
"""

from __future__ import annotations

import os
import struct
import sys
import traceback

import numpy as np


def _julia(path: str, params: str | None):
    from juliacall import Main as jl
    m = jl.seval("Module(:QMBLState)")
    jl.Base.include(m, path)
    if not jl.isdefined(m, jl.Symbol("logpsi")):
        raise RuntimeError(f"{path} defines no logpsi")
    if params:
        if not jl.isdefined(m, jl.Symbol("init")):
            raise RuntimeError(f"{path}: --params given but the file defines no init(path)")
        m.init(params)
    call = jl.seval("(f, x) -> Vector{ComplexF64}(Base.invokelatest(f, Matrix{Int8}(x)))")
    f = m.logpsi

    def fn(x: np.ndarray) -> np.ndarray:
        return np.asarray(call(f, x), dtype=np.complex128)
    return fn


def main() -> int:
    kind, target, params, rfd, wfd = sys.argv[1:6]
    params = params or None
    rfd, wfd = int(rfd), int(wfd)
    inp, out = os.fdopen(rfd, "rb"), os.fdopen(wfd, "wb")
    sys.stdout = sys.stderr                   # prints of the state go to stderr

    def send(b: bytes, err: bool = False):
        out.write(struct.pack("<q", -len(b) if err else len(b)) + b)
        out.flush()

    try:
        if kind != "julia":
            raise ValueError(f"unknown worker kind {kind!r}")
        fn = _julia(target, params)
    except Exception:  # noqa: BLE001
        send(traceback.format_exc().encode(), err=True)
        return 1
    os.set_blocking(rfd, True)
    os.set_blocking(wfd, True)
    send(b"ready")
    while True:
        h = inp.read(16)
        if len(h) < 16:
            return 0
        n, d = struct.unpack("<qq", h)
        if n < 0:
            return 0
        raw = inp.read(n * d)
        x = np.frombuffer(raw, dtype=np.int8).reshape(n, d)
        try:
            r = np.ascontiguousarray(fn(x), dtype=np.complex128)
            if r.shape != (n,):
                raise ValueError(f"logpsi returned shape {r.shape} for {n} configurations")
            send(r.tobytes())
        except Exception:  # noqa: BLE001
            send(traceback.format_exc().encode(), err=True)


if __name__ == "__main__":
    sys.exit(main())
