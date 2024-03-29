using JSON
using CUDA
using Adapt
using Bloqade
using BloqadeCUDA
using Dates
using Logging
using LoggingExtras
using TerminalLoggers
using BenchmarkTools
CUDA.allowscalar(false)

timestamp_logger(logger) = TransformerLogger(logger) do log
    date_info = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    merge(log, (; message = "$date_info $(log.message)"))
end

log_stream = open("benchmark.log", "w+")
TerminalLogger(log_stream; always_flush=true) |> timestamp_logger |> Base.global_logger

@info "package loaded"

# Define the pulse by specifying a maximum omega and delta.
# Note that these parameters are computed using pulser_compare.py
Omega_max= 9.179086064116243
delta_0 = -20.397969031369428
delta_f =  9.179086064116243

# Define the timescales of the pulse
Trise = 0.2
Tramp = 3.6
Tfall = 0.2
total_time = Trise + Tramp + Tfall

# Define the detuning and Rabi fields as a function of time
Δ = piecewise_linear(clocks=[0.0, Trise, Trise + Tramp, total_time], values=[delta_0,delta_0,delta_f,delta_f]);
Ω = piecewise_linear(clocks=[0.0, Trise, Trise + Tramp, total_time], values=[0,Omega_max,Omega_max,0]);

function chain_benchmark_problem(nsites::Int, distance::Float64)
    atoms = generate_sites(ChainLattice(), nsites, scale=distance)
    h = rydberg_h(atoms; C = 5420158.53 , Δ, Ω)
    reg = zero_state(nsites);
    return SchrodingerProblem(reg, total_time, h)
end

function chain_benchmark_subspace_problem(nsites::Int, distance::Float64)
    atoms = generate_sites(ChainLattice(), nsites, scale=distance)
    space = blockade_subspace(atoms, distance)
    h = rydberg_h(atoms; C = 5420158.53 , Δ, Ω)
    reg = zero_state(space);
    return SchrodingerProblem(reg, total_time, h)
end

function ring_benchmark_problem(nsites::Int, distance::Float64)
    R = distance/(2*sin(2*pi/(nsites)/2))                                       # Radius of the circle, using a little trigonometry
    pos = [(R*sin(i*2*pi/(nsites)), R*cos(i*2*pi/(nsites)) ) for i in 1:nsites] # Positions of each atom
    atoms = BloqadeLattices.AtomList(pos)                                         # Define the atom positions as an AtomList.

    # Define the Hamiltonian and problem instance
    h = rydberg_h(atoms; C = 5420158.53 , Δ, Ω)
    reg = zero_state(nsites);
    return SchrodingerProblem(reg, total_time, h)
end

function set_zero_state_cuda!(reg)
    CUDA.allowscalar() do
        set_zero_state!(reg)
    end
end

function flush_data(results)
    @info "flushing data"
    if isfile("data.json")
        d = open("data.json") do f
            JSON.parse(f)
        end
        merge!(results, d)
    end

    write("data.json", JSON.json(results))
end

@info "benchmark start"

if isfile("data.json")
    @info "removing old data file"
    rm("data.json")
end

nqubits = 4:20
results = Dict{String, Any}()
results["ring (CPU)"] = map(nqubits) do n
    prob = ring_benchmark_problem(n, 9.0)
    @info "benchmarking..." prob
    t = @benchmark emulate!($prob) setup=(set_zero_state_cuda!($(prob.reg)))
    minimum(t).time
end

flush_data(results)

results["ring (CUDA)"] = map(nqubits) do n
    prob = ring_benchmark_problem(n, 9.0)
    dprob = adapt(CuArray, prob)
    @info "benchmarking..." prob=dprob
    t = @benchmark CUDA.@sync(emulate!($dprob)) setup=(set_zero_state_cuda!($(dprob.reg)))
    minimum(t).time
end

flush_data(results)

results["chain (CPU)"] = map(nqubits) do n
    prob = chain_benchmark_problem(n, 5.7)
    @info "benchmarking..." prob
    t = @benchmark emulate!($prob) setup=(set_zero_state_cuda!($(prob.reg)))
    minimum(t).time
end

flush_data(results)

results["chain (CUDA)"] = map(nqubits) do n
    prob = chain_benchmark_problem(n, 5.7)
    dprob = adapt(CuArray, prob)
    @info "benchmarking..." prob=dprob
    t = @benchmark CUDA.@sync(emulate!($dprob)) setup=(set_zero_state_cuda!($(dprob.reg)))
    minimum(t).time
end

flush_data(results)

results["chain (subspace,CPU)"] = map(nqubits) do n
    prob = chain_benchmark_subspace_problem(n, 5.7)
    @info "benchmarking..." prob
    t = @benchmark emulate!($prob) setup=(set_zero_state_cuda!($(prob.reg)))
    minimum(t).time
end

flush_data(results)

results["chain (subspace,CUDA)"] = map(nqubits) do n
    prob = chain_benchmark_subspace_problem(n, 5.7)
    dprob = adapt(CuArray, prob)
    @info "benchmarking..." prob=dprob
    t = @benchmark CUDA.@sync(emulate!($dprob)) setup=(set_zero_state_cuda!($(dprob.reg)))
    minimum(t).time
end

flush_data(results)
