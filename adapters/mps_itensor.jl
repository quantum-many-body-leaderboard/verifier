# Write an ITensors.jl MPS to the verifier's file format (MPS_FORMAT.md). Needs ITensors, ITensorMPS, HDF5.
#
#     include("adapters/mps_itensor.jl")
#     E = inner(psi', H, psi) / inner(psi, psi)        # in your OpSum's units
#     write_qmbl_mps("state.h5", psi, site_order; energy = 4E,   # spins: QMBL counts Pauli units
#                    instance_id = "J1J2/square_16_P_0.5")
#
# site_order[k] is the QMBL site (x * ly + y, counted from 0) of MPS site k (k = 1..N). Site types
# "S=1/2" (states Up, Dn) and "Electron" (Emp, Up, Dn, UpDn) map one to one onto the format's local
# bases: ITensor orders fermions by site with up before down (|UpDn> = c+_up c+_dn |0>), which is
# the format's order, and its OpSum puts the Jordan-Wigner strings accordingly. QN blocks are
# written densely.
using ITensors, ITensorMPS, HDF5

function qmbl_basis(s::Index)
    hastags(s, "Electron") && return "electron"
    (hastags(s, "S=1/2") || hastags(s, "S=½")) && return "spin-half"
    error("site index $(tags(s)): use site type \"S=1/2\" or \"Electron\"")
end

function write_qmbl_mps(path::AbstractString, psi::MPS, site_order::AbstractVector{<:Integer};
                        energy = nothing, instance_id = nothing, producer = "ITensorMPS")
    N = length(psi)
    length(site_order) == N || error("site_order needs one entry per site")
    sort(collect(site_order)) == collect(0:N-1) || error("site_order must be a permutation of 0..N-1")
    sites = siteinds(psi)
    basis = qmbl_basis(sites[1])
    h5open(path, "w") do f
        attributes(f)["format"] = "qmbl-mps"
        attributes(f)["format_version"] = 1
        attributes(f)["local_basis"] = basis
        attributes(f)["n_sites"] = N
        instance_id === nothing || (attributes(f)["instance_id"] = instance_id)
        energy === nothing || (attributes(f)["energy"] = Float64(energy))
        attributes(f)["producer"] = producer
        f["site_order"] = Vector{Int64}(site_order)
        g = create_group(f, "tensors")
        for k in 1:N
            qmbl_basis(sites[k]) == basis || error("all sites must be of one type")
            T = dense(psi[k])
            l = k > 1 ? linkind(psi, k - 1) : nothing
            r = k < N ? linkind(psi, k) : nothing
            inds_ = filter(!isnothing, [l, sites[k], r])
            A = Array(T, inds_...)
            k == 1 && (A = reshape(A, 1, size(A)...))
            k == N && (A = reshape(A, size(A)..., 1))
            # Julia is column-major: store (right, physical, left) so that row-major readers
            # (h5py, C) see (left, physical, right).
            g[string(k - 1)] = permutedims(A, (3, 2, 1))
        end
    end
    return path
end
