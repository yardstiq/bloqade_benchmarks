from benchmarks import generate_chain_evolution
import cProfile



sim = generate_chain_evolution(19)

pr = cProfile.Profile()
pr.enable()
sim.run()
pr.disable()
pr.print_stats(sort="time")