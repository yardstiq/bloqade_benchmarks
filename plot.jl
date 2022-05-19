using JSON

bloqade_d = open("bloqade/data.json") do f
    JSON.parse(f)
end

pulser_d = open("pulser/data/Linux-CPython-3.9-64bit/0002_data.json") do f
    JSON.parse(f)
end

pulser_cpu_d = [pulser_d["benchmarks"][i]["stats"]["min"] * 1e9 for i in 1:16]
pulser_cpu_d[end]/1e9
bloqade_d["CUDA"][end-1]/1e9

pulser_cpu_d ./ bloqade_d["CPU"][1:end-1]
pulser_cpu_d ./ bloqade_d["CUDA"][1:end-1]
bloqade_d["CPU"]
pulser_cpu_d
findfirst(isequal(14), 4:20)
5.357647256896598e10 / bloqade_d["CPU"][11]

5.357647256896598e10 / bloqade_d["CUDA"][11]

bloqade_d["CPU"] ./ bloqade_d["CUDA"]
