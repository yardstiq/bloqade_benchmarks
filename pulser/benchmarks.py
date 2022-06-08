from pulser import Pulse, Sequence, Register
from pulser_simulation import Simulation
from pulser.waveforms import RampWaveform
from pulser.devices import MockDevice

import pytest
import numpy as np

# from matplotlib.pyplot import *
# close('all')

def generate_chain_seq(L, distance=5.7):
    coords = np.array([i * distance for i in range(L)])
    coords = np.array([coords, np.array([distance for i in range(L)])]).T
    reg = Register.from_coordinates(coords, prefix='atom')

    Omega_max= 9.179086064116243
    delta_0 = -20.397969031369428
    delta_f =  9.179086064116243

    t_rise = 200   # Rise time, in nanoseconds
    t_fall = 200   # Fall time, in nanoseconds
    t_sweep =3600   # Sweep time, in nanoseconds

    rise = Pulse.ConstantDetuning(RampWaveform(t_rise, 0., Omega_max), delta_0, 0.)
    sweep = Pulse.ConstantAmplitude(Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.)
    fall = Pulse.ConstantDetuning(RampWaveform(t_fall, Omega_max, 0.), delta_f, 0.)

    # Define a sequence using the pulser interface
    seq = Sequence(reg, MockDevice)
    seq.declare_channel('ising', 'rydberg_global')
    seq.add(rise, 'ising')
    seq.add(sweep, 'ising')
    seq.add(fall, 'ising')
    return seq


seq = generate_chain_seq(5)

def generate_ring_seq(L):
    ground_state_spacing = 1
    R_interatomic = 9
    radius = R_interatomic/(2*np.sin(np.pi/L * (ground_state_spacing)))
    th = np.linspace(0,2*np.pi,L+1)[0:-1]
    coords = np.array([radius * np.sin(th) , radius * np.cos(th)]).T

    print('\nParameters of the atoms:')
    print('Number of atoms:  {:0.0f}'.format(L))
    print('Distance of atoms from origin: {:0.6f}um'.format(radius))
    print('Unit disk radius: {:0.6f}um'.format(R_interatomic))

    reg = Register.from_coordinates(coords, prefix='atom')
    # reg.draw(blockade_radius=R_interatomic, draw_half_radius=True, draw_graph = True)

    rydberg_blockade_energy = MockDevice.interaction_coeff / R_interatomic**6

    Omega_max = 0.9*rydberg_blockade_energy

    delta_0 = -2*rydberg_blockade_energy
    delta_f = 0.9*rydberg_blockade_energy

    t_rise = 200   # Rise time, in nanoseconds
    t_fall = 200   # Fall time, in nanoseconds
    t_sweep =3600   # Sweep time, in nanoseconds

    rise = Pulse.ConstantDetuning(RampWaveform(t_rise, 0., Omega_max), delta_0, 0.)
    sweep = Pulse.ConstantAmplitude(Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.)
    fall = Pulse.ConstantDetuning(RampWaveform(t_fall, Omega_max, 0.), delta_f, 0.)

    # Define a sequence using the pulser interface
    seq = Sequence(reg, MockDevice)
    seq.declare_channel('ising', 'rydberg_global')
    seq.add(rise, 'ising')
    seq.add(sweep, 'ising')
    seq.add(fall, 'ising')
    return seq


# Simulate evolution
nqubits_list = range(4,21) # 4:20
@pytest.mark.parametrize('nqubits', nqubits_list)
def test_ring(benchmark, nqubits):
    benchmark.group = "ring"
    sim = Simulation(generate_ring_seq(nqubits), sampling_rate=0.1)
    benchmark(sim.run, progress_bar=False)


@pytest.mark.parametrize('nqubits', nqubits_list)
def test_chain(benchmark, nqubits):
    benchmark.group = "chain"
    sim = Simulation(generate_chain_seq(nqubits), sampling_rate=0.1)
    benchmark(sim.run, progress_bar=False)


# sim = Simulation(seq, sampling_rate=0.1)
# results = sim.run()
# print(np.sort(np.abs(np.array(results.get_final_state()))**2,axis=0)[-10::])