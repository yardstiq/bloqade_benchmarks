using JSON
using CUDA
using Adapt
using Bloqade
using BloqadeCUDA
using BenchmarkTools
CUDA.allowscalar(false)

# Simulate nonequilibrium dynamics of a 12-site ring of Rydberg atoms.
#  This script generates Figure 2 in the example RydbergBlockade.

function ring_benchmark_problem(nsites::Int, distance::Float64)
    R = distance/(2*sin(2*pi/(nsites)/2))                                       # Radius of the circle, using a little trigonometry
    pos = [(R*sin(i*2*pi/(nsites)), R*cos(i*2*pi/(nsites)) ) for i in 1:nsites] # Positions of each atom
    atoms = BloqadeLattices.AtomList(pos)                                         # Define the atom positions as an AtomList.

    # Define the pulse by specifying a maximum omega and delta.
    # Note that these parameters are computed using pulser_compare.py
    Omega_max= 9.179086064116243
    delta_0 =-20.397969031369428
    delta_f =  9.179086064116243

    # Define the timescales of the pulse
    Trise = 2.
    Tramp = 6.
    Tfall = 2.
    total_time = Trise + Tramp + Tfall

    # Define the detuning and Rabi fields as a function of time
    Δ = piecewise_linear(clocks=[0.0, Trise, Trise + Tramp, total_time], values=[delta_0,delta_0,delta_f,delta_f]);
    Ω = piecewise_linear(clocks=[0.0, Trise, Trise + Tramp, total_time], values=[0,Omega_max,Omega_max,0]);

    # Define the Hamiltonian and problem instance
    h = rydberg_h(atoms; C = 5420158.53 , Δ, Ω)
    reg = zero_state(nsites);
    return SchrodingerProblem(reg, total_time, h)
end

@info "setting up problems"
problems = map(4:25) do n
    ring_benchmark_problem(n, 9.0)
end

results = Dict{String, Any}()

@info "benchmarking CPU"
results["CPU"] = map(problems) do prob
    @info "benchmarking..." prob
    t = @benchmark emulate!($prob) setup=(set_zero_state!($(prob.reg)))
    minimum(t).time
end

@info "benchmarking CUDA"
results["CUDA"] = map(problems) do prob
    dprob = adapt(CuArray, prob)
    @info "benchmarking..." prob=dprob
    t = @benchmark CUDA.@sync(emulate!($dprob)) setup=(set_zero_state!($(dprob.reg)))
    minimum(t).time
end

write("data.json", JSON.json(results))
