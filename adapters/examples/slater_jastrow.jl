# The reference Slater-Jastrow state in Julia.  --state julia:slater_jastrow.jl --params sj.jld2
# (sj.jld2 = JLD2.jldsave("sj.jld2"; M, V), M::Matrix{Float64} (2N, n_f), V::Matrix{Float64} (2N, 2N)).
using JLD2, LinearAlgebra

const M = Ref{Matrix{Float64}}()
const V = Ref{Matrix{Float64}}()

function init(path::String)
    d = JLD2.load(path)
    M[], V[] = d["M"], d["V"]
end

function logpsi(x::Matrix{Int8})::Vector{ComplexF64}   # x: (n_conf, 2N), entries 0/1
    out = Vector{ComplexF64}(undef, size(x, 1))
    for k in axes(x, 1)
        n = Float64.(x[k, :])
        la, s = logabsdet(M[][findall(>(0), x[k, :]), :])
        out[k] = la + 0.5 * dot(n, V[] * n) + (s < 0 ? im * pi : 0.0)
    end
    return out
end
