from benchmarks import generate_chain_seq
from pulser_simulation import Simulation
import cProfile


sim = Simulation(generate_chain_seq(14), sampling_rate=0.1)

pr = cProfile.Profile()
pr.enable()
sim.run()
pr.disable()
pr.print_stats(sort="time")