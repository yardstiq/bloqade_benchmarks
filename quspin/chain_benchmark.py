
from quspin.operators import hamiltonian
import numpy as np
import pytest
# import cProfile
from quspin_rydberg import rydberg_hamiltonian_args


class Simulation:
    def __init__(self,psi_t_iter):
        self._psi_t_iter = psi_t_iter

    def run(self):
        for psi in self._psi_t_iter:
            pass


# nqubits = 10
# scale = 10
# C = 862690
# T = 10.0
def gen_chain_hamiltonian(L, scale):
    # T - total duration
    # L - length of the chain in terms of num qubits, some integer
    # scale - distance between Rydberg atoms

    t_rise = 0.2 
    t_fall = 0.2
    t_sweep = 3.6

    T = t_rise + t_fall + t_sweep

    # uniform problem
    C = 862690

    positions = scale * np.array([[i,0] for i in range(L)])

    # need to allow user variability here
    # args are: start, stop, number of vals (evenly spaced from [start, stop])
    t_vals = np.linspace(0,T,11)

    # Need to split the waveforms here, assume units of rad/microseconds stay the same

    # Specific to Omega
    Omega_max= 9.179086064116243
    # Specific to Delta
    delta_0 = -20.397969031369428
    delta_f =  9.179086064116243

    # Delta
    xp = [0.0, t_rise, t_rise + t_sweep, t_rise + t_sweep + t_fall]
    fp = [delta_0, delta_0, delta_f, delta_f]
    Delta = lambda t:np.interp(t, xp, fp)

    # Omega
    xp = [0.0, t_rise, t_rise + t_sweep, t_rise + t_fall + t_sweep]
    fp = [0.0, Omega_max, Omega_max, 0.0]
    Omega = lambda t:np.interp(t, xp, fp)

    # Delta = lambda t:np.sin(2*np.pi*t/T)**2
    # Omega = lambda t:np.cos(2*np.pi*t/T)**2

    kwargs = rydberg_hamiltonian_args(positions,Delta=Delta,Omega=Omega,C=C)
    h_Rydberg = hamiltonian(**kwargs)


    psi0 = np.ones(h_Rydberg.Ns)/np.sqrt(h_Rydberg.Ns)

    psi_t_iter = h_Rydberg.evolve(psi0,0,t_vals,iterate=True)

    return psi_t_iter

    """
    pr = cProfile.Profile()
    pr.enable()
    for i,psi_t in enumerate(psi_t_iter):
        print("t/T= {:.4f} norm= {:.6e}".format(t_vals[i]/T,np.linalg.norm(psi_t)))
    pr.disable()
    pr.print_stats(sort="time")
    """

nqubits_list = range(4,21)
@pytest.mark.parametrize('nqubits', nqubits_list)
def test_chain(benchmark, nqubits):
    benchmark.group = "chain"
    sim = Simulation(gen_chain_hamiltonian(nqubits, 10))
    benchmark(sim.run)
