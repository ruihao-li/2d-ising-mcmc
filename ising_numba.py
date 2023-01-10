import numpy as np
from scipy.integrate import quad
import numba as nb
from tqdm import trange
from ising import Ising


@nb.jit(nopython=True)
def compute_energy(spin_config, J, L):
    """
    Compute the energy of the spin configuration (per site).
    """
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += (
                -J
                * spin_config[i, j]
                * (
                    spin_config[i, (j + 1) % L]
                    + spin_config[(i + 1) % L, j]
                    + spin_config[i, (j - 1) % L]
                    + spin_config[(i - 1) % L, j]
                )
            )
    return energy / 2 / L**2


@nb.jit(nopython=True)
def compute_magnetization(spin_config, L):
    """
    Compute the net absolute magnetization of the spin configuration (per site).
    """
    return np.abs(np.sum(spin_config)) / L**2


@nb.jit(nopython=True)
def mc_update(spin_config, J, L, T):
    """
    Perform L^2 Metropolis steps to update the spin configuration.
    """
    for _ in range(L**2):
        # Pick a random site
        i = np.random.randint(L)
        j = np.random.randint(L)
        # Compute the change in energy
        delta_E = (
            2
            * J
            * spin_config[i, j]
            * (
                spin_config[i, (j + 1) % L]
                + spin_config[(i + 1) % L, j]
                + spin_config[i, (j - 1) % L]
                + spin_config[(i - 1) % L, j]
            )
        )
        # Flip the spin if the energy decreases or if the Metropolis criterion is satisfied
        if delta_E <= 0:
            spin_config[i, j] *= -1
        elif np.random.random() < np.exp(-delta_E / T):
            spin_config[i, j] *= -1


class Ising_numba(Ising):
    """
    Numba-accelerated Python implementation of Metropolis Monte Carlo Simulation of the 2D Ising
    model (square lattice).
    """

    def __init__(self, J, L, T):
        super().__init__(J, L, T)

    def compute_energy(self):
        """
        Compute the energy of the spin configuration (per site).
        """
        return compute_energy(self.spin_config, self.J, self.L)

    def compute_magnetization(self):
        """
        Compute the net absolute magnetization of the spin configuration (per site).
        """
        return compute_magnetization(self.spin_config, self.L)

    def mc_update(self):
        """
        Perform L^2 Metropolis steps to update the spin configuration.
        """
        mc_update(self.spin_config, self.J, self.L, self.T)


def exact_solutions(J, L, T_low, T_high, nT):
    """
    Compute the exact solutions for the energy, magnetization, and critical temperature.

    Args:
        J (float): Coupling constant
        L (int): Linear size of the lattice
        T_low (float): Lower bound of the temperature range
        T_high (float): Upper bound of the temperature range
        nT (int): Number of temperatures to simulate

    Returns:
        (np.array, np.array, np.array, float): Arrays of temperatures, exact energies,
        magnetizations, and the critical temperature
    """
    T_array = np.linspace(T_low, T_high, nT)
    E_exact_array = np.zeros(nT)
    M_exact_array = np.zeros(nT)
    for i in range(nT):
        ising = Ising_numba(J, L, T_array[i])
        E_exact_array[i] = ising.exact_energy()
        M_exact_array[i] = ising.exact_magnetization()
    T_c = ising.exact_critical_temp()
    return T_array, E_exact_array, M_exact_array, T_c


def run_ising_numba(J, L, T_low, T_high, nT, equil_steps, mc_steps, skip_steps):
    """
    Run the Monte Carlo simulation for a range of temperatures.

    Args:
        J (float): Coupling constant
        L (int): Linear size of the lattice
        T_low (float): Lower bound of the temperature range
        T_high (float): Upper bound of the temperature range
        nT (int): Number of temperatures to simulate
        equil_steps (int): Number of Monte Carlo steps to perform during equilibration
        mc_steps (int): Number of Monte Carlo steps to perform during measurement
        skip_steps (int): Number of Monte Carlo steps to skip between measurements

    Returns:
        (np.array, np.array, np.array): Arrays of energies, magnetizations, and spin configurations
    """
    T_array = np.linspace(T_low, T_high, nT)
    # Initialize arrays to store the energies and magnetizations
    E_array = np.zeros(nT)
    M_array = np.zeros(nT)
    # Initialize arrays to store the final spin configurations for each temperature
    spin_config_array = np.zeros((nT, L, L))
    # Loop over temperatures
    for i in trange(nT):
        Et = Mt = 0
        # Initialize the Ising model
        ising = Ising_numba(J, L, T_array[i])
        # Equilibrate the system
        for _ in range(equil_steps):
            ising.mc_update()
        # Perform Monte Carlo steps after equilibration
        for j in range(mc_steps):
            ising.mc_update()
            # Skip the first few steps
            if j % skip_steps == 0:
                Et += ising.compute_energy()
                Mt += ising.compute_magnetization()
        # Average the energy and magnetization
        E_array[i] = Et / (mc_steps // skip_steps)
        M_array[i] = Mt / (mc_steps // skip_steps)
        # Store the final spin configuration
        spin_config_array[i] = ising.spin_config
    return E_array, M_array, spin_config_array
