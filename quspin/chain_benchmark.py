
from quspin.operators import hamiltonian
import numpy as np
import cProfile
from quspin_rydberg import rydberg_hamiltonian_args



N = 10
scale = 10
C = 862690

T = 10.0

# uniform problem

positions = scale*np.array([[i,0] for i in range(N)])


t_vals = np.linspace(0,T,11)


Delta = lambda t:np.sin(2*np.pi*t/T)**2
Omega = lambda t:np.cos(2*np.pi*t/T)**2

kwargs = rydberg_hamiltonian_args(positions,Delta=Delta,Omega=Omega,C=C)
h_Rydberg = hamiltonian(**kwargs)


psi0 = np.ones(h_Rydberg.Ns)/np.sqrt(h_Rydberg.Ns)

psi_t_iter = h_Rydberg.evolve(psi0,0,t_vals,iterate=True)


pr = cProfile.Profile()
pr.enable()
for i,psi_t in enumerate(psi_t_iter):
	print("t/T= {:.4f} norm= {:.6e}".format(t_vals[i]/T,np.linalg.norm(psi_t)))
pr.disable()
pr.print_stats(sort="time")
