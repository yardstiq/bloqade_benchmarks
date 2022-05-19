# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:06:07 2022

Copied and adapted from Pasqual pulser example
https://pulser.readthedocs.io/en/stable/tutorials/simulating.html

@author: jwurtz
"""

from pulser import Pulse, Sequence, Register
from pulser_simulation import Simulation
from pulser.waveforms import RampWaveform
from pulser.devices import MockDevice

import pytest
import time
import numpy as np
import qutip
import matplotlib.pyplot as plt

# from matplotlib.pyplot import *
# close('all')

def generate_seq(L):
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
    reg.draw(blockade_radius=R_interatomic, draw_half_radius=True, draw_graph = True)

    rydberg_blockade_energy = MockDevice.interaction_coeff / R_interatomic**6

    Omega_max = 0.9*rydberg_blockade_energy

    delta_0 = -2*rydberg_blockade_energy
    delta_f = 0.9*rydberg_blockade_energy

    t_rise = 2000   # Rise time, in nanoseconds
    t_fall = 2000   # Fall time, in nanoseconds
    t_sweep =6000   # Sweep time, in nanoseconds

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
nqubits_list = range(4,26) # 4:20
@pytest.mark.parametrize('nqubits', nqubits_list)
def test_sim(benchmark, nqubits):
    benchmark.group = "ring"
    sim = Simulation(generate_seq(nqubits), sampling_rate=0.1)
    benchmark(sim.run, progress_bar=False)
