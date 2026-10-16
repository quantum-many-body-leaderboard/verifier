# DMRG with ITensors.jl for tests/test_mps.py: reads an HDF5 job (Hamiltonian terms already in chain
# order, written by the test from the verifier's bond lists), runs DMRG with ITensor's own OpSum and
# writes the state with adapters/mps_itensor.jl.   julia tests/itensor_dmrg.jl job.h5
using ITensors, ITensorMPS, HDF5
include(joinpath(@__DIR__, "..", "adapters", "mps_itensor.jl"))

function main(jobfile)
    job = h5open(jobfile, "r") do f
        a = attributes(f)
        Dict(:basis => read(a["basis"]), :n_sites => Int(read(a["n_sites"])), :chi => Int(read(a["chi"])),
             :sweeps => Int(read(a["sweeps"])), :out => read(a["out"]), :instance_id => read(a["instance_id"]),
             :order => read(f["order"]), :init => read(f["init"]),
             :on_k => read(f["on_k"]), :on_c => read(f["on_c"]),
             :p => read(f["p"]), :q => read(f["q"]), :kind => read(f["kind"]), :c => read(f["c"]))
    end
    N = job[:n_sites]
    sites = job[:basis] == "electron" ? siteinds("Electron", N; conserve_qns = true) :
            siteinds("S=1/2", N; conserve_qns = true)
    os = OpSum()
    for (k, c) in zip(job[:on_k], job[:on_c])          # U n_up n_dn, k counted from 0
        os += c, "Nupdn", k + 1
    end
    for (p0, q0, kind, c) in zip(job[:p], job[:q], job[:kind], job[:c])
        p, q = p0 + 1, q0 + 1
        if kind == 0                        # Pauli s.s = 4 Sz Sz + 2 (S+ S- + S- S+)
            os += 4c, "Sz", p, "Sz", q
            os += 2c, "S+", p, "S-", q
            os += 2c, "S-", p, "S+", q
        elseif kind == 1                    # c (c+_p c_q + c+_q c_p) per spin
            os += c, "Cdagup", p, "Cup", q
            os += c, "Cdagup", q, "Cup", p
            os += c, "Cdagdn", p, "Cdn", q
            os += c, "Cdagdn", q, "Cdn", p
        elseif kind == 2
            os += c, "Ntot", p, "Ntot", q
        else
            error("term kind $kind")
        end
    end
    H = MPO(os, sites)
    psi0 = MPS(sites, Vector{Int}(job[:init]))
    nsw = job[:sweeps]
    maxdim = [min(job[:chi], 32 * 2^(k - 1)) for k in 1:nsw]
    noise = [k <= nsw ÷ 3 ? 1e-4 : k <= nsw - 4 ? 1e-6 : 0.0 for k in 1:nsw]
    t0 = time()
    energy, psi = dmrg(H, psi0; nsweeps = nsw, maxdim = maxdim, cutoff = 1e-12, noise = noise, outputlevel = 1)
    E = real(inner(psi', H, psi) / inner(psi, psi))
    println("ITENSOR_ENERGY ", E, " dmrg ", energy, " maxlinkdim ", maxlinkdim(psi), " wall ", time() - t0)
    write_qmbl_mps(job[:out], psi, Vector{Int}(job[:order]); energy = E, instance_id = job[:instance_id],
                   producer = "ITensorMPS (tests/itensor_dmrg.jl)")
    println("ITENSOR_WROTE ", job[:out])
end

main(ARGS[1])
