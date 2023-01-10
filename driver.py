import numpy as np
import time
import pickle
from tqdm import trange

from ising import run_ising, exact_solutions
from ising_numba import run_ising_numba
from ising_cython import run_ising_cython


# # Set the parameters
# J = 1.0
# L = [10, 20, 40, 80]
# T_low = 0.5
# T_high = 4.5
# nT = 50
# equil_steps = 5000
# mc_steps = 10000
# skip_steps = 100

# # Compute and store exact solutions (L-independent)
# T_arr, E_exact, M_exact, T_c = exact_solutions(J, L[0], T_low, T_high, nT)
# with open("data/ising_exact_data.pkl", "wb") as f:
#     pickle.dump((T_arr, E_exact, M_exact, T_c, L), f)

# #------------------------------------------------------------
# ### Monte Carlo simulation with Python + Numba
# #------------------------------------------------------------

# E_numba, M_numba, time_numba, spin_config_numba = [], [], [], []
# # Run the simulation
# for i in trange(len(L)):
#     start = time.time()
#     E, M, spin_config = run_ising_numba(J, L[i], T_low, T_high, nT, equil_steps, mc_steps, skip_steps)
#     end = time.time()
#     time_elapsed = end - start
#     E_numba.append(E)
#     M_numba.append(M)
#     time_numba.append(time_elapsed)
#     spin_config_numba.append(spin_config)
# # Pickle the data
# with open("data/ising_numba_data.pkl", "wb") as f:
#     pickle.dump((E_numba, M_numba, time_numba, spin_config_numba), f)

# #------------------------------------------------------------
# ### Monte Carlo simulation with Cython
# #------------------------------------------------------------

# E_cython, M_cython, time_cython, spin_config_cython = [], [], [], []
# # Run the simulation
# for i in trange(len(L)):
#     start = time.time()
#     E, M, spin_config = run_ising_cython(J, L[i], T_low, T_high, nT, equil_steps, mc_steps, skip_steps)
#     end = time.time()
#     time_elapsed = end - start
#     E_cython.append(E)
#     M_cython.append(M)
#     time_cython.append(time_elapsed)
#     spin_config_cython.append(spin_config)
# # Pickle the data
# with open("data/ising_cython_data.pkl", "wb") as f:
#     pickle.dump((E_cython, M_cython, time_cython, spin_config_cython), f)

#------------------------------------------------------------
### Benchmarking (scaling with system size): Pure Python, Python with Numba, and Cython
#------------------------------------------------------------

J = 1.0
L = [i for i in range(2, 13, 2)]
T_low = 0.5
T_high = 4.5
nT = 50
equil_steps = 5000
mc_steps = 10000
skip_steps = 100
time_py, time_numba, time_cython = [], [], []
for i in trange(len(L)):
    # Python
    start = time.time()
    _, _, _ = run_ising(J, L[i], T_low, T_high, nT, equil_steps, mc_steps, skip_steps)
    end = time.time()
    time_py.append(end - start)

    # Python + Numba
    start = time.time()
    _, _, _ = run_ising_numba(J, L[i], T_low, T_high, nT, equil_steps, mc_steps, skip_steps)
    end = time.time()
    time_numba.append(end - start)

    # Cython
    start = time.time()
    _, _, _ = run_ising_cython(J, L[i], T_low, T_high, nT, equil_steps, mc_steps, skip_steps)
    end = time.time()
    time_cython.append(end - start)

# Pickle the data
with open("data/ising_benchmark_data.pkl", "wb") as f:
    pickle.dump((time_py, time_numba, time_cython, L), f)


#------------------------------------------------------------

# E_py, M_py, time_py, spin_config_py = [], [], [], []
# # Run the simulation
# for i in trange(len(L)):
#     start = time.time()
#     E, M, spin_config = run_ising(J, L[i], T_low, T_high, nT, equil_steps, mc_steps, skip_steps)
#     end = time.time()
#     time_elapsed = end - start
#     E_py.append(E)
#     M_py.append(M)
#     time_py.append(time_elapsed)
#     spin_config_py.append(spin_config)
# # Pickle the data
# with open("data/ising_py_data.pkl", "wb") as f:
#     pickle.dump((E_py, M_py, time_py, spin_config_py), f)