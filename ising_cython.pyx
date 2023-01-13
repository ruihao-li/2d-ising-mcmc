cimport cython
import numpy as np

cimport numpy as np
from libc.stdlib cimport rand
from libc.math cimport exp, sqrt, pow
from cython.parallel import prange

cdef extern from "limits.h":
    int RAND_MAX

@cython.cdivision(True)
cdef int mod(int a, int b) nogil:
    if a < 0:
        return a % b + b
    else:
        return a % b

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int[:, :] random_spin_config(int L) nogil:
    """
    Generate a random spin configuration of size L x L.
    """
    cdef int i, j
    cdef int[:, :] spin_config
    with gil:
        spin_config = np.zeros((L, L), dtype=np.int32)
    for i in range(L):
        for j in range(L):
            spin_config[i, j] = 1 if rand() / RAND_MAX > 0.5 else -1
    return spin_config

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double compute_energy(double J, int[:, :] spin_config) nogil:
    """
    Compute the energy of the spin configuration.

    Args:
        J: Coupling constant
        L: Linear size of the lattice
        spin_config: Spin configuration

    Returns:
        Total energy
    """
    cdef int i, j
    cdef int L = spin_config.shape[0]
    cdef double energy = 0.0

    for i in prange(L, nogil=True):
        for j in range(L):
            energy += -J * spin_config[i, j] * (
                spin_config[i, mod(j + 1, L)] + spin_config[mod(i + 1, L), j] + spin_config[i, mod(j - 1, L)] + spin_config[mod(i - 1, L), j]
            )
    return energy * 0.5

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef double compute_magnetization(int[:, :] spin_config) nogil:
    """
    Compute the net absolute magnetization of the spin configuration.

    Args:
        L: Linear size of the lattice
        spin_config: Spin configuration

    Returns:
        Net absolute magnetization
    """
    cdef int i, j
    cdef int L = spin_config.shape[0]
    cdef int magnetization = 0

    for i in prange(L, nogil=True):
        for j in range(L):
            magnetization += spin_config[i, j]
    return sqrt(pow(magnetization, 2))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef int mc_update(float J, int[:, :] spin_config, double T) nogil:
    """
    Perform L^2 Metropolis steps to update the spin configuration.

    Args:
        J: Coupling constant
        L: Linear size of the lattice
        spin_config: Spin configuration
        T: Temperature

    Returns:
        Updated spin configuration
    """
    cdef int l, i, j
    cdef double delta_E
    cdef int L = spin_config.shape[0]

    for l in range(L * L):
        # Pick a random site
        i = rand() % L
        j = rand() % L
        # Compute the change in energy
        delta_E = 2 * J * spin_config[i, j] * (
                spin_config[i, mod(j + 1, L)] + spin_config[mod(i + 1, L), j] + spin_config[i, mod(j - 1, L)] + spin_config[mod(i - 1, L), j]
        )
        # Flip the spin if the energy decreases or if the Metropolis criterion is satisfied
        if delta_E <= 0:
            spin_config[i, j] *= -1
        elif rand() < exp(-delta_E / T) * RAND_MAX:
            spin_config[i, j] *= -1
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef run_ising_cython(double J, int L, double T_low, double T_high, int nT, int equil_steps, int mc_steps, int skip_steps):
    """
    Run the Monte Carlo simulation for a range of temperatures.

    Args:
        J: Coupling constant
        L: Linear size of the lattice
        T_low: Lower bound of the temperature range
        T_high: Upper bound of the temperature range
        nT: Number of temperatures to simulate
        equil_steps: Number of Monte Carlo steps to perform during equilibration
        mc_steps: Number of Monte Carlo steps to perform during measurement
        skip_steps: Number of Monte Carlo steps to skip between measurements

    Returns:
        Arrays of energies, magnetizations, and spin configurations
    """
    cdef double[:] T_array = np.linspace(T_low, T_high, nT, dtype=np.float64)
    cdef double[:] E_array = np.zeros(nT, dtype=np.float64)
    cdef double[:] M_array = np.zeros(nT, dtype=np.float64)
    cdef int[:, :, :] spin_config_array = np.zeros((nT, L, L), dtype=np.int32)
    cdef int i, j, k
    cdef double Et, Mt

    # Loop over temperatures
    for i in range(nT):
        Et = Mt = 0
        # Initialize a random spin configuration
        spin_config = random_spin_config(L)
        # Equilibrate the system
        for j in prange(equil_steps, nogil=True):
            mc_update(J, spin_config, T_array[i])
        # Perform Monte Carlo steps after equilibration
        for k in prange(mc_steps, nogil=True):
            mc_update(J, spin_config, T_array[i])
            # Skip the first few steps
            if k % skip_steps == 0:
                Et += compute_energy(J, spin_config)
                Mt += compute_magnetization(spin_config)
        # Average the energy and magnetization
        E_array[i] = Et / (mc_steps // skip_steps) / (L * L)
        M_array[i] = Mt / (mc_steps // skip_steps) / (L * L)
        # Store the final spin configuration for the given temperature
        spin_config_array[i, :, :] = spin_config
    return np.asarray(E_array), np.asarray(M_array), np.asarray(spin_config_array)