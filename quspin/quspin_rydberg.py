import numpy as np
from quspin.basis import boson_basis_general

__export__ = ["rydberg_hamiltonian_args"]


def _process_static_and_dynamic_parameters(Natoms,Pars_list):
	# this is a helper function to process the input lists for the parameters
	# It parses the list (or scalar value) passed in. 

	try:
		N_pars = len(Pars_list)
		Pars_list = list(Pars_list)
	except TypeError:
		Pars_list = Natoms*[Pars_list]


	if len(Pars_list) != Natoms:
		raise ValueError("Omega must be a scalar or list with same length as positions.")


	List_static = {i:ele for i,ele in enumerate(Pars_list) if not callable(ele)}
	List_dynamic = {i:ele for i,ele in  enumerate(Pars_list) if callable(ele)}

	return List_static,List_dynamic




def rydberg_hamiltonian_args(positions,Omega=1.0,Delta=1.0,C=862690,checks=False):

	positions = np.array(positions).astype(np.float64)
	print(positions.shape)

	if positions.ndim != 1 and positions.ndim != 2:
		raise TypeError("expecting positions to be 1D or 2D array")

	if positions.ndim == 2 and positions.shape[1] != 2:
		raise TypeError("2D array for positions must have shape (Natoms,2)")

	Natoms = positions.shape[0]
	basis = boson_basis_general(Natoms,sps=2,Nb=None)


	Omega_static,Omega_dynamic = _process_static_and_dynamic_parameters(Natoms,Omega)
	Delta_static,Delta_dynamic = _process_static_and_dynamic_parameters(Natoms,Delta)

	# pick out functions with unique object id, this makes sure functions are not repeated. 
	# QuSpin will combine any objects that have functions with the same id value
	Omega_cc = {}
	for i,Omega in Omega_dynamic.items():
		func_id = id(Omega)
		if func_id not in Omega_cc:
			Omega_cc[func_id] = lambda t:np.conj(Omega(t))

	Omega_cc_dynamic = {i:Omega_cc[id(Omega)] for i,Omega in Omega_dynamic.items()}


	# generator that loops over interactions
	x = positions[:,0]
	y = positions[:,1]
	r_ij = [(i,j,np.hypot(x[i]-x[j],y[i]-y[j])) for i in range(Natoms) for j in range(Natoms) if j>i]

	RydbergInteract = [[2*np.pi*C/r**6,i,j] for i,j,r in r_ij]

	RydbergInteract = [[J,i,j] for J,i,j in RydbergInteract if J>np.finfo(np.float64).eps]

	Detuning_static_list = [[2*np.pi*Delta,i] for i,Delta in Delta_static.items()]

	Rabi_static_list = [[np.pi*Omega,i] for i,Omega in Omega_static.items()]
	Rabi_cc_static_list = [[np.pi*np.conj(Omega),i] for i,Omega_cc in Omega_static.items()]

	# this is not optimal for real-value drives, in general one should simplify these
	Rabi_dynamic = [["+",[[np.pi,i]],Omega,()] for i,Omega in Omega_dynamic.items()]
	Rabi_cc_dynamic = [["-",[[np.pi,i]],Omega_cc,()] for i,Omega_cc in Omega_cc_dynamic.items()]
	Detuning_dynamic = [["n",[[2*np.pi,i]],Delta,()] for i,Delta in Delta_dynamic.items()]

	args = dict(check_symm=checks,check_herm=checks,check_pcon=checks,basis=basis)


	args["static_list"] = [["nn",RydbergInteract],
					  ["n",Detuning_static_list],
					  ["+",Rabi_static_list],
					  ["-",Rabi_cc_static_list],
					  ]
	args["dynamic_list"] = Rabi_dynamic+Rabi_cc_dynamic+Detuning_dynamic

	if len(Rabi_static_list) > 0:
		# if static terms contain off diagonal components, use csr representation
		args["static_fmt"] = "csr"
	else:
		# otherwise, the static term is diagonal, hence use diagonal format
		args["static_fmt"] = "dia"

	dyanmic_fmt = {}

	for j,s,f,f_args in Detuning_dynamic:
		dyanmic_fmt[(f,())] = "dia"

	for j,s,f,f_args in Rabi_dynamic:
		dyanmic_fmt[(f,())] = "csr"

	for j,s,f,f_args in Rabi_cc_dynamic:
		dyanmic_fmt[(f,())] = "csr"

	args["dynamic_fmt"] = dyanmic_fmt

	return args














