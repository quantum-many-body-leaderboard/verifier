# The reference RBM in Julia.  --state julia:rbm.jl --params rbm.jld2
# (rbm.jld2 = JLD2.jldsave("rbm.jld2"; W, b, a) with W::Matrix{ComplexF64} of size (n_sites, n_hidden)).
using JLD2

const W = Ref{Matrix{ComplexF64}}()
const b = Ref{Vector{ComplexF64}}()
const a = Ref{Vector{ComplexF64}}()

function init(path::String)                       # called once with the --params path
    d = JLD2.load(path)
    W[], b[], a[] = d["W"], d["b"], d["a"]
end

logcosh(z) = (z = real(z) < 0 ? -z : z; z + log(1 + exp(-2z)) - log(2.0))

function logpsi(x::Matrix{Int8})::Vector{ComplexF64}   # x: (n_conf, n_sites), entries +-1
    X = ComplexF64.(x)
    T = X * W[] .+ transpose(b[])
    return vec(sum(logcosh.(T); dims=2)) .+ X * a[]
end
