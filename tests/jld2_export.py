"""Write the arrays of an .npz into a JLD2 file, in a process of its own (juliacall must not share a
process with torch).  python tests/jld2_export.py in.npz out.jld2"""

import sys

import numpy as np


def main(src, dst):
    from juliacall import Main as jl
    jl.seval("using JLD2")
    z = np.load(src)
    f = jl.JLD2.jldopen(dst, "w")
    put = jl.seval("(f, n, x, c) -> (f[n] = c ? Array{ComplexF64}(x) : Array{Float64}(x); nothing)")
    for n in z.files:
        put(f, n, np.ascontiguousarray(z[n]), bool(np.iscomplexobj(z[n])))
    jl.close(f)
    print("jld2", dst, sorted(z.files))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
