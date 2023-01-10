import numpy as np
from scipy.integrate import quad
import time
import pickle
from tqdm import trange


class Ising:
    """
    Pure Python implementation of Metropolis Monte Carlo Simulation of the 2D Ising model (square
    lattice).
    """

    def __init__(self, J, L, T):
        """
        Initialize the Ising model.

        Args:
            J (float): Coupling constant
            L (int): Linear size of the lattice
            T (float): Temperature
        """
        self.J = J
        self.L = L
        self.T = T
        self.spin_config = self.generate_spin_config()

    def generate_spin_config(self):
        """
        Generate a random spin configuration.
        """
        return np.random.choice([-1, 1], size=(self.L, self.L))

    def compute_energy(self):
        """
        Compute the energy of the spin configuration (per site).
        """
        energy = 0
        for i in range(self.L):
            for j in range(self.L):
                energy += (
                    -self.J
                    * self.spin_config[i, j]
                    * (
                        self.spin_config[i, (j + 1) % self.L]
                        + self.spin_config[(i + 1) % self.L, j]
                        + self.spin_config[i, (j - 1) % self.L]
                        + self.spin_config[(i - 1) % self.L, j]
                    )
                )
        return energy / 2 / self.L**2

    def compute_magnetization(self):
        """
        Compute the net absolute magnetization of the spin configuration (per site).
        """
        return np.abs(np.sum(self.spin_config)) / self.L**2

    def mc_update(self):
        """
        Perform L^2 Metropolis steps to update the spin configuration.
        """
        for _ in range(self.L**2):
            # Pick a random site
            i = np.random.randint(self.L)
            j = np.random.randint(self.L)
            # Compute the change in energy
            delta_E = (
                2
                * self.J
                * self.spin_config[i, j]
                * (
                    self.spin_config[i, (j + 1) % self.L]
                    + self.spin_config[(i + 1) % self.L, j]
                    + self.spin_config[i, (j - 1) % self.L]
                    + self.spin_config[(i - 1) % self.L, j]
                )
            )
            # Flip the spin if the energy decreases or if the Metropolis criterion is satisfied
            if delta_E <= 0:
                self.spin_config[i, j] *= -1
            elif np.random.random() < np.exp(-delta_E / self.T):
                self.spin_config[i, j] *= -1

    def exact_critical_temp(self):
        """
        Compute the exact critical temperature.
        """
        return 2 * self.J / np.log(1 + np.sqrt(2))

    def exact_energy(self):
        """
        Compute the exact energy of the spin configuration (per site).
        """
        beta = 1 / self.T
        k = 1 / np.sinh(2 * beta * self.J) ** 2
        U = (
            -self.J
            * self.coth(2 * beta * self.J)
            * (
                1
                + 2
                / np.pi
                * (2 * self.tanh(2 * beta * self.J) ** 2 - 1)
                * quad(
                    lambda x: 1
                    / np.sqrt(1 - 4 * (k / (1 + k) ** 2) * (np.sin(x) ** 2)),
                    0,
                    np.pi / 2,
                )[0]
            )
        )
        return U

    def exact_magnetization(self):
        """
        Compute the exact spontaneous magnetization of the spin configuration (per site) for T < Tc;
        outputs 0 when T > Tc.
        """
        if self.T > self.exact_critical_temp():
            return 0
        else:
            beta = 1 / self.T
            M = (1 - np.sinh(2 * beta * self.J) ** (-4)) ** (1 / 8)
        return M

    @staticmethod
    def coth(x):
        return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


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
        ising = Ising(J, L, T_array[i])
        E_exact_array[i] = ising.exact_energy()
        M_exact_array[i] = ising.exact_magnetization()
    T_c = ising.exact_critical_temp()
    return T_array, E_exact_array, M_exact_array, T_c


def run_ising(J, L, T_low, T_high, nT, equil_steps, mc_steps, skip_steps):
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
        ising = Ising(J, L, T_array[i])
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
