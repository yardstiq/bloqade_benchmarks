from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

with open(os.path.join('bloqade', 'data.json')) as f:
    bloqade_json = json.load(f)


with open(os.path.join('pulser', 'data', 'Linux-CPython-3.9-64bit', '0004_data.json')) as f:
    pulser_json = json.load(f)


bloqade_json['chain (CPU)']
bloqade_json['chain (CUDA)']
bloqade_json['chain (subspace,CPU)']
bloqade_json['chain (subspace,CUDA)']

qutip_json = {
    'chain (CPU)': [pulser_json["benchmarks"][i]["stats"]["min"] * 1e9 for i in range(34) if pulser_json["benchmarks"][i]["group"] == "chain"],
    'ring (CPU)': [pulser_json["benchmarks"][i]["stats"]["min"] * 1e9 for i in range(34) if pulser_json["benchmarks"][i]["group"] == "ring"],
}

speedup = {
    'chain (CPU vs CPU)': [qutip/bloqade for qutip, bloqade in zip(qutip_json['chain (CPU)'], bloqade_json['chain (CPU)'])],
    'ring (CPU vs CPU)': [qutip/bloqade for qutip, bloqade in zip(qutip_json['ring (CPU)'], bloqade_json['ring (CPU)'])],
    'chain (CPU vs CUDA)': [qutip/bloqade for qutip, bloqade in zip(qutip_json['chain (CPU)'], bloqade_json['chain (CUDA)'])],
    'ring (CPU vs CUDA)': [qutip/bloqade for qutip, bloqade in zip(qutip_json['ring (CPU)'], bloqade_json['ring (CUDA)'])],
    'chain (CPU vs subspace,CPU)': [qutip/bloqade for qutip, bloqade in zip(qutip_json['chain (CPU)'], bloqade_json['chain (subspace,CPU)'])],
    'chain (CPU vs subspace,CUDA)': [qutip/bloqade for qutip, bloqade in zip(qutip_json['chain (CPU)'], bloqade_json['chain (subspace,CUDA)'])],
}

data = np.array([
        qutip_json['chain (CPU)'],
        bloqade_json['chain (CPU)'],
        bloqade_json['chain (CUDA)'],
        bloqade_json['chain (subspace,CPU)'],
        bloqade_json['chain (subspace,CUDA)']
    ])

df_chain_absolute = pd.DataFrame({
    'nqubits': np.arange(4, 21, dtype=int),
    'qutip (CPU)': qutip_json['chain (CPU)'],
    'bloqade (CPU)': bloqade_json['chain (CPU)'],
    'bloqade (CUDA)': bloqade_json['chain (CUDA)'],
    'bloqade (subspace,CPU)': bloqade_json['chain (subspace,CPU)'],
    'bloqade (subspace,CUDA)': bloqade_json['chain (subspace,CUDA)'],
})

df_chain_speedup = pd.DataFrame({
    'nqubits': np.arange(4, 21, dtype=int),
    'qutip (CPU)': np.ones(17),
    'bloqade (CPU)': speedup['chain (CPU vs CPU)'],
    'bloqade (CUDA)': speedup['chain (CPU vs CUDA)'],
})

df_chain_subspace_speedup = pd.DataFrame({
    'nqubits': np.arange(4, 21, dtype=int),
    'qutip (CPU)': np.ones(17),
    'bloqade (subspace,CPU)': speedup['chain (CPU vs subspace,CPU)'],
    'bloqade (subspace,CUDA)': speedup['chain (CPU vs subspace,CUDA)'],
})

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(18, 15))
df_chain_absolute.plot(
    x='nqubits',
    kind='bar',
    ax=ax1,
    stacked=False,
)

df_chain_speedup.plot(
    x='nqubits',
    kind='bar',
    ax=ax2,
    stacked=False,
)

df_chain_subspace_speedup.plot(
    x='nqubits',
    kind='bar',
    ax=ax3,
    stacked=False,
)

fig.suptitle('Benchmark of QuTiP (via Pulser) vs Bloqade on 1D Chain Lattice', fontsize=16)
ax1.set_title('absolute time')
ax2.set_title('Bloqade speedup of exact simulation (qutip exact = 1)')
ax3.set_title('Bloqade speedup of subspace approximation (qutip exact = 1)')

# yscales
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')

# ylabel
ax1.set_ylabel('ns (lower is better)')
ax2.set_ylabel('speedup (higher is better, qutip=1)')
ax3.set_ylabel('speedup (higher is better, qutip=1)')

xlabel = 'number of atoms'
ax1.set_xlabel(xlabel)
ax2.set_xlabel(xlabel)
ax3.set_xlabel(xlabel)


fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.5)
fig.savefig('chain.png')
