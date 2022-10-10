
from quspin.operators import hamiltonian
import numpy as np
import pytest
import time
# import cProfile
from quspin_rydberg import rydberg_hamiltonian_args


class Simulation:
    def __init__(self,H,t_vals):
        self._H = H
        self._t_vals = t_vals

    def run(self):
        psi0 = np.zeros(self._H.Ns)
        psi0[-1] = 1
        psi_t_iter = self._H.evolve(psi0,0,self._t_vals,
        iterate=True,atol=1e-6,rtol=1e-3)
        for psi in psi_t_iter:
            pass

def generate_chain_evolution(L):
    # T - total duration
    # L - length of the chain in terms of num qubits, some integer
    # scale - distance between Rydberg atoms

    t_rise = 0.2 
    t_fall = 0.2
    t_sweep = 3.6

    T = t_rise + t_fall + t_sweep
    t_vals = np.arange(0,T+0.05,0.1)
    
    # uniform problem
    C = 862690
    scale = 5.7

    positions = scale * np.array([[i,0] for i in range(L)])

    # need to allow user variability here
    # args are: start, stop, number of vals (evenly spaced from [start, stop])

    # Need to split the waveforms into separate Omega and Delta from 
    # Pulser's "Pulse" format where they're combined

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

    return Simulation(h_Rydberg,t_vals)

def generate_ring_evolution(L):

    t_rise = 0.2 
    t_fall = 0.2
    t_sweep = 3.6

    T = t_rise + t_fall + t_sweep

    # taken from the Chain hamiltonian
    t_vals = np.arange(0,T+0.05,0.1)

    ground_state_spacing = 1
    R_interatomic = 9
    radius = R_interatomic/(2*np.sin(np.pi/L * (ground_state_spacing)))
    th = np.linspace(0,2*np.pi,L+1)[0:-1]
    positions = np.array([radius * np.sin(th) , radius * np.cos(th)]).T

    print('\nParameters of the atoms:')
    print('Number of atoms:  {:0.0f}'.format(L))
    print('Distance of atoms from origin: {:0.6f}um'.format(radius))
    print('Unit disk radius: {:0.6f}um'.format(R_interatomic))

    # MockDevice.interaction_coeff = C_6 / \hbar (from the pulser example)
    # Blockade Radius = C_6 / R_b^6 = \Omega (seems like this is what they want to find)
    # Just to be safe:
    mock_device_interaction_coeff = 5420158.53
    rydberg_blockade_energy = mock_device_interaction_coeff / R_interatomic**6

    Omega_max = 0.9*rydberg_blockade_energy

    delta_0 = -2*rydberg_blockade_energy
    delta_f = 0.9*rydberg_blockade_energy

    # Delta
    xp = [0.0, t_rise, t_rise + t_sweep, t_rise + t_sweep + t_fall]
    fp = [delta_0, delta_0, delta_f, delta_f]
    Delta = lambda t:np.interp(t, xp, fp)

    # Omega
    xp = [0.0, t_rise, t_rise + t_sweep, t_rise + t_fall + t_sweep]
    fp = [0.0, Omega_max, Omega_max, 0.0]
    Omega = lambda t:np.interp(t, xp, fp)

    kwargs = rydberg_hamiltonian_args(positions,Delta=Delta,Omega=Omega,C=mock_device_interaction_coeff)
    h_Rydberg = hamiltonian(**kwargs)

    return Simulation(h_Rydberg,t_vals)



nqubits_list = range(4,21)
@pytest.mark.parametrize('nqubits', nqubits_list)
def test_chain(benchmark, nqubits):
    benchmark.group = "chain"
    sim = generate_chain_evolution(nqubits)
    benchmark(sim.run)

@pytest.mark.parametrize('nqubits', nqubits_list)
def test_ring(benchmark, nqubits):
    benchmark.group = "ring"
    sim = generate_chain_evolution(nqubits)
    benchmark(sim.run)


