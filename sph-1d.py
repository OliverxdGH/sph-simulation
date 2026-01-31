from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SPHSimulation:
    """
    One-dimensional Smoothed Particle Hydrodynamics (SPH) simulation.

    This script implements a simple 1D SPH solver (often used as a toy model
    for shock-tube-like problems). The state is evolved using:
      - Cubic spline SPH kernel (1D normalization used here)
      - Monaghan-style artificial viscosity
      - Fourth-order Runge–Kutta (RK4) time integration with fixed timestep

    State representation
    --------------------
    The per-particle state is stored in a matrix with columns:
        [x, v_x, rho, e, p, m]

    where:
      - x   : position
      - v_x : velocity
      - rho : density
      - e   : specific internal energy
      - p   : pressure (computed from EOS)
      - m   : mass

    Notes
    -----
    - This implementation uses an O(N^2) pairwise interaction approach.
    - The kernel normalization `a_d = 1/h` corresponds to a 1D setup.
    - Pressure is computed from the ideal-gas-like EOS:
          p = (gamma - 1) * rho * e
    """

    gamma: float
    dt: float
    n_steps: int
    h: float

    x_init: np.ndarray
    rho_init: np.ndarray
    v_init: np.ndarray
    e_init: np.ndarray
    p_init: np.ndarray
    m_init: np.ndarray

    N: int
    state_matrix: np.ndarray
    state_vec: np.ndarray
    final_state: Optional[np.ndarray]

    def __init__(self, gamma: float = 1.4, dt: float = 0.005, n_steps: int = 40, h: float = 0.0055) -> None:
        """
        Create an SPH simulation instance and initialize particle states.

        Parameters
        ----------
        gamma
            Adiabatic index in the equation of state.
        dt
            Fixed timestep for RK4 integration.
        n_steps
            Number of integration steps.
        h
            SPH smoothing length.
        """
        self.gamma = gamma
        self.dt = dt
        self.n_steps = n_steps
        self.h = h

        self.final_state = None

        # Initial setup (left + right regions)
        self.init_conditions()

    def init_conditions(self) -> None:
        """
        Set up a 1D two-region initial condition (shock-tube style).

        The domain is split into a left and right region with different
        density/pressure/energy values. Particles are placed uniformly
        within each region with different spacings.
        """
        # Left region
        N_left, dx_left = 320, 0.001875
        rho_left, v_left, e_left, p_left, m_left = 1.0, 0.0, 2.5, 1.0, 0.001875
        x_left = np.linspace(-0.6, 0.0 - dx_left, N_left)

        # Right region
        N_right, dx_right = 80, 0.0075
        rho_right, v_right, e_right, p_right, m_right = 0.25, 0.0, 1.795, 0.1795, 0.001875
        x_right = np.linspace(0.0, 0.6 - dx_right, N_right) + dx_right / 2.0

        # Concatenate into one particle set
        self.x_init = np.concatenate([x_left, x_right])
        self.rho_init = np.concatenate([np.full(N_left, rho_left), np.full(N_right, rho_right)])
        self.v_init = np.concatenate([np.full(N_left, v_left), np.full(N_right, v_right)])
        self.e_init = np.concatenate([np.full(N_left, e_left), np.full(N_right, e_right)])
        self.p_init = np.concatenate([np.full(N_left, p_left), np.full(N_right, p_right)])
        self.m_init = np.concatenate([np.full(N_left, m_left), np.full(N_right, m_right)])

        self.N = int(len(self.x_init))

        # State matrix: columns = [x, v_x, rho, e, p, m]
        self.state_matrix = np.column_stack((self.x_init, self.v_init, self.rho_init, self.e_init, self.p_init, self.m_init))

        # Recompute density and pressure from SPH and EOS
        rho0, _ = self.compute_density(self.state_matrix)
        self.state_matrix[:, 2] = rho0
        self.state_matrix[:, 4] = (self.gamma - 1.0) * self.state_matrix[:, 2] * self.state_matrix[:, 3]

        self.state_vec = self.vector_from_state(self.state_matrix)

    # -----------------------
    # Utilities
    # -----------------------
    def state_from_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        Reshape a flattened state vector into an (N, 6) state matrix.
        """
        return vec.reshape((self.N, 6))

    def vector_from_state(self, state: np.ndarray) -> np.ndarray:
        """
        Flatten an (N, 6) state matrix into a 1D state vector.
        """
        return state.flatten()

    # -----------------------
    # Kernel and helpers
    # -----------------------
    def compute_R_matrix(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pairwise separations and distance measures in 1D.

        Returns
        -------
        R
            Dimensionless distance matrix r/h (N, N).
        dx
            Pairwise signed separation x_i - x_j (N, N).
        r
            Pairwise absolute distances |x_i - x_j| (N, N).
        """
        x = state_matrix[:, 0]
        dx = x[:, None] - x[None, :]
        r = np.abs(dx)
        R = r / self.h
        return R, dx, r

    def cubic_spline_kernel(self, state_matrix: np.ndarray) -> np.ndarray:
        """
        Cubic spline SPH kernel W(r, h) in 1D form.

        Returns
        -------
        W
            Kernel values for all particle pairs (N, N).
        """
        R, _, _ = self.compute_R_matrix(state_matrix)
        W = np.zeros_like(R)
        a_d = 1.0 / self.h

        mask1 = (R >= 0.0) & (R < 1.0)
        W[mask1] = a_d * ((2.0 / 3.0) - R[mask1] ** 2 + 0.5 * R[mask1] ** 3)

        mask2 = (R >= 1.0) & (R < 2.0)
        W[mask2] = a_d * (1.0 / 6.0) * (2.0 - R[mask2]) ** 3

        return W

    def cubic_spline_kernel_derivative(self, state_matrix: np.ndarray) -> np.ndarray:
        """
        Compute ∂W/∂x (via dW/dr and chain rule) for all particle pairs in 1D.

        Returns
        -------
        gradWx
            Pairwise kernel gradient contribution (N, N).
        """
        R, dx, r = self.compute_R_matrix(state_matrix)
        dWdr = np.zeros_like(R)
        a_d = 1.0 / self.h
        r_safe = np.where(r == 0.0, 1e-12, r)

        mask1 = (R >= 0.0) & (R < 1.0)
        dWdr[mask1] = a_d * (-2.0 * R[mask1] + 1.5 * R[mask1] ** 2) / self.h

        mask2 = (R >= 1.0) & (R < 2.0)
        dWdr[mask2] = -a_d * 0.5 * (2.0 - R[mask2]) ** 2 / self.h

        gradWx = dWdr * dx / r_safe
        return gradWx

    def compute_density(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SPH density for each particle and the pairwise average density.

        Returns
        -------
        rho_calc
            Density for each particle (N,).
        rho_bar
            Symmetric pair-average density (N, N), useful for viscosity.
        """
        m = state_matrix[:, 5]
        W = self.cubic_spline_kernel(state_matrix)
        rho_calc = W.dot(m)
        rho_bar = 0.5 * (rho_calc[:, None] + rho_calc[None, :])
        return rho_calc, rho_bar

    def compute_viscosity(self, state_matrix: np.ndarray, alpha: float = 1.0, beta: float = 1.0, eps: float = 0.1) -> np.ndarray:
        """
        Compute Monaghan-style artificial viscosity Π_ij in 1D.

        Viscosity is applied only to particle pairs approaching each other.

        Returns
        -------
        Pi
            Pairwise viscosity matrix Π_ij (N, N).
        """
        _, dx, _ = self.compute_R_matrix(state_matrix)
        rho_calc, rho_bar = self.compute_density(state_matrix)
        e = state_matrix[:, 3]
        v_x = state_matrix[:, 1]

        c = np.sqrt(np.maximum((self.gamma - 1.0) * e, 1e-12))
        dv_x = v_x[:, None] - v_x[None, :]
        c_bar = 0.5 * (c[:, None] + c[None, :])

        denom = dx ** 2 + (eps * self.h) ** 2
        phi_ij = (self.h * dv_x * dx) / denom

        Pi = np.zeros_like(phi_ij)
        approaching = (dv_x * dx) < 0.0
        Pi[approaching] = (-alpha * c_bar[approaching] * phi_ij[approaching] + beta * (phi_ij[approaching] ** 2)) / rho_bar[approaching]
        return Pi

    def compute_accel_energy(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dv/dt and de/dt from pressure forces and artificial viscosity.

        Returns
        -------
        dvdt_x
            Acceleration for each particle (N,).
        dEdt
            Specific internal energy time derivative for each particle (N,).
        """
        v_x = state_matrix[:, 1]
        p = state_matrix[:, 4]
        m = state_matrix[:, 5]

        Pi = self.compute_viscosity(state_matrix)
        rho_calc, _ = self.compute_density(state_matrix)
        gradWx = self.cubic_spline_kernel_derivative(state_matrix)

        pij_term = (
            p[:, None] / (rho_calc[:, None] ** 2 + 1e-30)
            + p[None, :] / (rho_calc[None, :] ** 2 + 1e-30)
            + Pi
        )
        dvdt_x = -np.sum(m * pij_term * gradWx, axis=1)

        vel_diff_x = (v_x[:, None] - v_x[None, :])
        dEdt = 0.5 * np.sum(m * pij_term * (vel_diff_x * gradWx), axis=1)

        return dvdt_x, dEdt

    def compute_velocity(self, state_matrix: np.ndarray) -> np.ndarray:
        """
        Convenience function returning the velocity array v_x.
        """
        return state_matrix[:, 1]

    # -----------------------
    # RHS + Integrator
    # -----------------------
    def rhs(self, t: float, state_vec: np.ndarray) -> np.ndarray:
        """
        Right-hand side for the ODE system du/dt = F(t, u).

        The density and pressure are recomputed from the current state,
        then the acceleration and energy rate are computed.

        Returns
        -------
        dudt
            Flattened derivative state vector (same shape as state_vec).
        """
        state = self.state_from_vector(state_vec).copy()

        rho_calc, _ = self.compute_density(state)
        state[:, 2] = rho_calc
        state[:, 4] = (self.gamma - 1.0) * state[:, 2] * state[:, 3]

        dxdt = self.compute_velocity(state)
        dvdt_x, dEdt = self.compute_accel_energy(state)

        drhodt = np.zeros(self.N)
        dpdt = (self.gamma - 1.0) * state[:, 2] * dEdt
        dmdt = np.zeros(self.N)

        deriv_state = np.column_stack((dxdt, dvdt_x, drhodt, dEdt, dpdt, dmdt))
        return self.vector_from_state(deriv_state)

    def RK4step(self, f, t: float, u: np.ndarray, h: float) -> np.ndarray:
        """
        Perform one classic fourth-order Runge–Kutta step.

        Parameters
        ----------
        f
            RHS function f(t, u).
        t
            Current time.
        u
            Current state vector.
        h
            Timestep.

        Returns
        -------
        u_next
            Updated state vector after one RK4 step.
        """
        k1 = f(t, u)
        k2 = f(t + 0.5 * h, u + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, u + 0.5 * h * k2)
        k4 = f(t + h, u + h * k3)
        return u + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # -----------------------
    # Run Simulation
    # -----------------------
    def run(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run the simulation for `n_steps` RK4 steps with fixed timestep `dt`.

        Returns
        -------
        history
            List of state matrices (N, 6) stored after each step.
        times
            List of time values corresponding to each stored step.
        """
        t: float = 0.0
        state_vec: np.ndarray = self.state_vec
        history: List[np.ndarray] = []
        times: List[float] = []

        for step in range(self.n_steps):
            state_vec = self.RK4step(self.rhs, t, state_vec, self.dt)
            t += self.dt

            state = self.state_from_vector(state_vec)

            # Recompute density and pressure for stored state
            rho_new, _ = self.compute_density(state)
            state[:, 2] = rho_new
            state[:, 4] = (self.gamma - 1.0) * state[:, 2] * state[:, 3]
            state_vec = self.vector_from_state(state)

            if step == 0 or (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{self.n_steps}, t={t:.6f}")

            history.append(state.copy())
            times.append(t)

        self.final_state = state
        return history, times

    # -----------------------
    # Plot
    # -----------------------
    def plot_results(self) -> None:
        """
        Plot final density, energy, pressure, and velocity versus position.
        """
        if self.final_state is None:
            raise RuntimeError("No final state available. Run the simulation first.")

        state = self.final_state
        x, v, rho, e, p = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]

        plt.figure(figsize=(10, 4))

        plt.subplot(2, 2, 1)
        plt.plot(x, rho, ".", markersize=3)
        plt.xlabel("x")
        plt.ylabel("rho")
        plt.title("Density")
        plt.xlim(-0.4, 0.4)

        plt.subplot(2, 2, 2)
        plt.plot(x, e, ".", markersize=3)
        plt.xlabel("x")
        plt.ylabel("e")
        plt.title("Energy")
        plt.xlim(-0.4, 0.4)
        plt.ylim(1.8, 2.7)

        plt.subplot(2, 2, 3)
        plt.plot(x, p, ".", markersize=3)
        plt.xlabel("x")
        plt.ylabel("p")
        plt.title("Pressure")
        plt.xlim(-0.4, 0.4)
        plt.ylim(0, 1.2)

        plt.subplot(2, 2, 4)
        plt.plot(x, v, ".", markersize=3)
        plt.xlabel("x")
        plt.ylabel("v_x")
        plt.title("Velocity")
        plt.xlim(-0.4, 0.4)
        plt.ylim(0, 1.2)

        plt.tight_layout()
        plt.show()


def animate_results(history: List[np.ndarray], filename: str = "sph_simulation.gif") -> None:
    """
    Create and save a 2x2-panel animation of SPH results.

    Panels show:
      - Density vs x
      - Energy vs x
      - Pressure vs x
      - Velocity vs x

    Parameters
    ----------
    history
        List of state matrices (N, 6) produced by SPHSimulation.run().
    filename
        Output GIF filename.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 4))

    ax_rho, ax_e = axs[0, 0], axs[0, 1]
    ax_p, ax_v = axs[1, 0], axs[1, 1]

    scat_rho = ax_rho.plot([], [], ".", markersize=3)[0]
    scat_e = ax_e.plot([], [], ".", markersize=3)[0]
    scat_p = ax_p.plot([], [], ".", markersize=3)[0]
    scat_v = ax_v.plot([], [], ".", markersize=3)[0]

    ax_rho.set_xlim(-0.4, 0.4)
    ax_rho.set_ylim(0, 1.2)
    ax_rho.set_title("Density")

    ax_e.set_xlim(-0.4, 0.4)
    ax_e.set_ylim(1.8, 2.7)
    ax_e.set_title("Energy")

    ax_p.set_xlim(-0.4, 0.4)
    ax_p.set_ylim(0, 1.2)
    ax_p.set_title("Pressure")

    ax_v.set_xlim(-0.4, 0.4)
    ax_v.set_ylim(0, 1.2)
    ax_v.set_title("Velocity")

    plt.tight_layout()

    def init():
        scat_rho.set_data([], [])
        scat_e.set_data([], [])
        scat_p.set_data([], [])
        scat_v.set_data([], [])
        return scat_rho, scat_e, scat_p, scat_v

    def update(frame: int):
        state = history[frame]
        x, v, rho, e, p = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]

        scat_rho.set_data(x, rho)
        scat_e.set_data(x, e)
        scat_p.set_data(x, p)
        scat_v.set_data(x, v)
        return scat_rho, scat_e, scat_p, scat_v

    ani = animation.FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True, interval=200)
    ani.save(filename, writer="pillow", fps=10)
    plt.close(fig)
    print(f"Animation saved as {filename}")


if __name__ == "__main__":
    sim = SPHSimulation()
    history, times = sim.run()
    animate_results(history, filename="sph_simulation.gif")
