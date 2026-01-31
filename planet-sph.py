from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PlanetSimulation:
    """
    Smoothed Particle Hydrodynamics (SPH) toy model for a self-gravitating “planet”.

    The simulation evolves a set of particles with positions, velocities, masses, densities,
    and specific internal energies using:
      - A cubic spline SPH kernel for density and pressure forces
      - Monaghan-style artificial viscosity
      - Self-gravity with spline softening
      - Fourth-order Runge–Kutta (RK4) time integration
      - Adaptive timestep selection using a CFL condition and an acceleration limiter

    Notes
    -----
    - Units are whatever your input data file uses. The gravitational constant is SI.
      Make sure this is consistent with your initial conditions.
    - This code uses an O(N^2) pairwise interaction approach and is intended for relatively
      small particle counts.
    """

    gamma: float
    dt_fixed: Optional[float]
    n_steps: int
    h: Optional[float]
    eta: float
    cfl: float

    x_init: np.ndarray
    y_init: np.ndarray
    z_init: np.ndarray
    vx_init: np.ndarray
    vy_init: np.ndarray
    vz_init: np.ndarray
    m_init: np.ndarray
    rho_init: np.ndarray
    p_init: np.ndarray
    e_init: np.ndarray

    N: int
    state_matrix: np.ndarray
    state_vec: np.ndarray

    final_state: Optional[np.ndarray]
    last_dt: Optional[float]

    def __init__(self, filename: str = "Planet300.dat", gamma: float = 1.4, dt: Optional[float] = None,
                 n_steps: int = 40, h: Optional[float] = None, eta: float = 1.3, cfl: float = 0.3,
                 spin_period: Optional[float] = None, collision: bool = False, v_collide: float = 1e3) -> None:
        """
        Initialize simulation from a particle data file.

        Parameters
        ----------
        filename
            Path to the input data file. Expected columns:
            x, y, z, vx, vy, vz, m, rho, p (9 columns total).
        gamma
            Adiabatic index for the equation of state.
        dt
            Fixed timestep. If None, an adaptive timestep is used.
        n_steps
            Number of integration steps.
        h
            Global smoothing length. If None, it is estimated from initial conditions.
        eta
            Factor used when estimating smoothing length from initial conditions.
        cfl
            CFL factor used in adaptive timestep selection.
        spin_period
            If provided, add a solid-body rotation about the z-axis with this period.
        collision
            If True, duplicate the particle set to create two identical bodies that collide.
        v_collide
            Relative speed parameter used when setting up a collision.
        """
        self.gamma = gamma
        self.dt_fixed = dt
        self.n_steps = n_steps
        self.h = h
        self.eta = eta
        self.cfl = cfl

        self.final_state = None
        self.last_dt = None

        # Load planet data: x,y,z,vx,vy,vz,m,rho,p
        data: np.ndarray = np.loadtxt(filename)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        vx, vy, vz = data[:, 3], data[:, 4], data[:, 5]
        m, rho, p = data[:, 6], data[:, 7], data[:, 8]

        if collision:
            # Build two identical bodies separated in x and moving towards each other.
            offset: float = 0.25e8

            x1, y1, z1 = x - offset * 4.0, y, z
            vx1, vy1, vz1 = vx + v_collide, vy, vz

            x2, y2, z2 = x + offset * 4.0, y, z
            vx2, vy2, vz2 = vx - v_collide, vy, vz

            self.x_init = np.concatenate([x1, x2])
            self.y_init = np.concatenate([y1, y2])
            self.z_init = np.concatenate([z1, z2])

            self.vx_init = np.concatenate([vx1, vx2])
            self.vy_init = np.concatenate([vy1, vy2])
            self.vz_init = np.concatenate([vz1, vz2])

            self.m_init = np.concatenate([m.copy(), m.copy()])
            self.rho_init = np.concatenate([rho.copy(), rho.copy()])
            self.p_init = np.concatenate([p.copy(), p.copy()])
        else:
            self.x_init, self.y_init, self.z_init = x, y, z
            self.vx_init, self.vy_init, self.vz_init = vx, vy, vz
            self.m_init, self.rho_init, self.p_init = m, rho, p

        # Add spin if requested: solid-body rotation about z.
        if spin_period is not None:
            omega: float = 2.0 * np.pi / spin_period
            self.vx_init = self.vx_init - omega * self.y_init
            self.vy_init = self.vy_init + omega * self.x_init
            self.vz_init = self.vz_init + np.zeros_like(self.z_init)

        # Internal energy per mass from p = (gamma-1) rho e
        self.e_init = self.p_init / ((self.gamma - 1.0) * self.rho_init)

        self.N = int(len(self.x_init))

        # State matrix columns:
        # [x, y, z, vx, vy, vz, m, rho, e]
        self.state_matrix = np.column_stack(
            (self.x_init, self.y_init, self.z_init,
             self.vx_init, self.vy_init, self.vz_init,
             self.m_init, self.rho_init, self.e_init)
        )
        self.state_vec = self.vector_from_state(self.state_matrix)

        if self.h is None:
            self.choose_h_from_initial(self.eta)

    # -----------------------
    # Utilities
    # -----------------------
    def state_from_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        Reshape a flattened state vector into an (N, 9) state matrix.
        """
        return vec.reshape((self.N, 9))

    def vector_from_state(self, state: np.ndarray) -> np.ndarray:
        """
        Flatten an (N, 9) state matrix into a 1D state vector.
        """
        return state.flatten()

    def choose_h_from_initial(self, eta: float = 1.3) -> None:
        """
        Estimate a global smoothing length from the initial particle distribution.

        Uses the common SPH estimate:
            h_i = eta * (m_i / rho_i)^(1/3)
        and takes the median over particles to obtain a single global h.
        """
        h_i: np.ndarray = eta * (self.m_init / self.rho_init) ** (1.0 / 3.0)
        self.h = float(np.median(h_i))

    # -----------------------
    # Kernels and Physics
    # -----------------------
    def compute_R_matrix(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pairwise separation components and distances.

        Returns
        -------
        dx, dy, dz
            Pairwise coordinate differences (N, N).
        r
            Pairwise distance matrix (N, N).
        """
        x, y, z = state_matrix[:, 0], state_matrix[:, 1], state_matrix[:, 2]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dz = z[:, None] - z[None, :]
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return dx, dy, dz, r

    def cubic_spline_kernel(self, state_matrix: np.ndarray, h: float) -> np.ndarray:
        """
        Cubic spline SPH kernel W(r, h) in 3D.

        This is the standard M4 cubic spline kernel with compact support 2h.

        Returns
        -------
        W
            Kernel values for all particle pairs (N, N).
        """
        x, y, z = state_matrix[:, 0], state_matrix[:, 1], state_matrix[:, 2]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dz = z[:, None] - z[None, :]
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        q = r / h

        W = np.zeros_like(q)
        a_d = 1.0 / (np.pi * h ** 3)

        m1 = (q >= 0.0) & (q < 1.0)
        W[m1] = a_d * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)

        m2 = (q >= 1.0) & (q < 2.0)
        W[m2] = a_d * 0.25 * (2.0 - q[m2]) ** 3

        return W

    def cubic_spline_kernel_derivative(self, state_matrix: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gradient of the cubic spline kernel ∇W in 3D for all particle pairs.

        Implements:
            dW/dr = (sigma / h) f'(q), where q = r/h
            ∇W = (dW/dr) * r_hat

        Returns
        -------
        gradWx, gradWy, gradWz
            Components of ∇W for all particle pairs (N, N).
        """
        dx, dy, dz, r = self.compute_R_matrix(state_matrix)
        q = r / h

        dWdr = np.zeros_like(q)
        a_d = 1.0 / (np.pi * h ** 3)
        r_safe = np.where(r == 0.0, 1e-12, r)

        m1 = (q >= 0.0) & (q < 1.0)
        dWdr[m1] = (a_d / h) * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)

        m2 = (q >= 1.0) & (q < 2.0)
        dWdr[m2] = (a_d / h) * (-0.75 * (2.0 - q[m2]) ** 2)

        gradWx = dWdr * dx / r_safe
        gradWy = dWdr * dy / r_safe
        gradWz = dWdr * dz / r_safe
        return gradWx, gradWy, gradWz

    def compute_density(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SPH densities and pair-averaged density.

        Returns
        -------
        rho_calc
            Density for each particle (N,).
        rho_bar
            Symmetric pair-average of density (N, N), used in viscosity.
        """
        if self.h is None:
            raise ValueError("Smoothing length h is not set.")

        m = state_matrix[:, 6]
        W = self.cubic_spline_kernel(state_matrix, self.h)
        rho_calc = W.dot(m)
        rho_bar = 0.5 * (rho_calc[:, None] + rho_calc[None, :])
        return rho_calc, rho_bar

    def compute_viscosity(self, state_matrix: np.ndarray, alpha: float = 1.0, beta: float = 1.0, eps: float = 0.1) -> np.ndarray:
        """
        Monaghan (1992)-style artificial viscosity Π_ij.

        The viscosity is only applied to pairs that are approaching each other.

        Returns
        -------
        Pi
            Pairwise viscosity matrix Π_ij (N, N).
        """
        if self.h is None:
            raise ValueError("Smoothing length h is not set.")

        dx, dy, dz, r = self.compute_R_matrix(state_matrix)
        rho_calc, rho_bar = self.compute_density(state_matrix)

        e = state_matrix[:, 8]
        vx, vy, vz = state_matrix[:, 3], state_matrix[:, 4], state_matrix[:, 5]

        # Sound speed estimate (limited to avoid NaNs)
        c = np.sqrt(np.maximum((self.gamma - 1.0) * e, 1e-12))
        dvx = vx[:, None] - vx[None, :]
        dvy = vy[:, None] - vy[None, :]
        dvz = vz[:, None] - vz[None, :]
        c_bar = 0.5 * (c[:, None] + c[None, :])

        h_bar = float(self.h)
        denom = dx ** 2 + dy ** 2 + dz ** 2 + (eps * h_bar) ** 2
        mu_ij = (h_bar * (dvx * dx + dvy * dy + dvz * dz)) / denom

        Pi = np.zeros_like(mu_ij)
        approaching = (dvx * dx + dvy * dy + dvz * dz) < 0.0
        Pi[approaching] = (-alpha * c_bar[approaching] * mu_ij[approaching] + beta * (mu_ij[approaching] ** 2)) / rho_bar[approaching]
        return Pi

    def phi_term(self, r: np.ndarray, h: float) -> np.ndarray:
        """
        Derivative of the spline-softened gravitational potential.

        Returns dφ/dr for each pair distance r (vectorized).
        For r >= 2h, the Newtonian limit is used.

        Parameters
        ----------
        r
            Pairwise distances (N, N) or any array of distances.
        h
            Smoothing length.

        Returns
        -------
        dphidr
            Softened potential derivative dφ/dr (same shape as r).
        """
        R = r / h
        dphidr = np.zeros_like(R)

        mask1 = (R >= 0.0) & (R < 1.0)
        dphidr[mask1] = (1.0 / h ** 2) * ((4.0 / 3.0) * R[mask1] - (6.0 / 5.0) * R[mask1] ** 3 + 0.5 * R[mask1] ** 4)

        mask2 = (R >= 1.0) & (R < 2.0)
        dphidr[mask2] = (1.0 / h ** 2) * (
            (8.0 / 3.0) * R[mask2]
            - 3.0 * R[mask2] ** 2
            + (6.0 / 5.0) * R[mask2] ** 3
            - (1.0 / 6.0) * R[mask2] ** 4
            - (1.0 / (15.0 * R[mask2] ** 2))
        )

        mask3 = R >= 2.0
        dphidr[mask3] = 1.0 / (r[mask3] ** 2)
        return dphidr

    def gravity_accel(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gravitational acceleration for each particle using spline softening.

        Returns
        -------
        ax, ay, az
            Acceleration components for each particle (N,).
        """
        if self.h is None:
            raise ValueError("Smoothing length h is not set.")

        G: float = 6.6743e-11
        m = state_matrix[:, 6]
        dx, dy, dz, r = self.compute_R_matrix(state_matrix)
        r_safe = np.where(r == 0.0, 1e-12, r)

        dphi = self.phi_term(r_safe, float(self.h))
        dphidr_sym = 0.5 * (dphi + dphi)

        grad_factor = dphidr_sym / r_safe
        ax = -G * np.sum(m * grad_factor * dx, axis=1)
        ay = -G * np.sum(m * grad_factor * dy, axis=1)
        az = -G * np.sum(m * grad_factor * dz, axis=1)
        return ax, ay, az

    def compute_accel_energy(self, state_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute dv/dt and de/dt from pressure forces, viscosity, and gravity.

        Returns
        -------
        dvdt_x, dvdt_y, dvdt_z
            Acceleration components for each particle (N,).
        dedt
            Rate of change of specific internal energy (N,).
        """
        vx, vy, vz = state_matrix[:, 3], state_matrix[:, 4], state_matrix[:, 5]
        m = state_matrix[:, 6]
        e = state_matrix[:, 8]

        rho_calc, _ = self.compute_density(state_matrix)
        p = (self.gamma - 1.0) * rho_calc * e

        Pi = self.compute_viscosity(state_matrix)
        gradWx, gradWy, gradWz = self.cubic_spline_kernel_derivative(state_matrix, float(self.h))
        gx, gy, gz = self.gravity_accel(state_matrix)

        pij_term = (p[:, None] / (rho_calc[:, None] ** 2) + p[None, :] / (rho_calc[None, :] ** 2) + Pi)

        dvdt_x = -np.sum(m * pij_term * gradWx, axis=1) + gx
        dvdt_y = -np.sum(m * pij_term * gradWy, axis=1) + gy
        dvdt_z = -np.sum(m * pij_term * gradWz, axis=1) + gz

        dvx = vx[:, None] - vx[None, :]
        dvy = vy[:, None] - vy[None, :]
        dvz = vz[:, None] - vz[None, :]

        dedt = 0.5 * np.sum(m * pij_term * (dvx * gradWx + dvy * gradWy + dvz * gradWz), axis=1)
        return dvdt_x, dvdt_y, dvdt_z, dedt

    # -----------------------
    # RHS and Integrator
    # -----------------------
    def rhs(self, t: float, state_vec: np.ndarray) -> np.ndarray:
        """
        Right-hand side for ODE integration: du/dt = F(t, u).

        The state vector encodes an (N,9) matrix:
            [x, y, z, vx, vy, vz, m, rho, e]
        """
        state = self.state_from_vector(state_vec).copy()
        rho_calc, _ = self.compute_density(state)
        state[:, 7] = rho_calc

        dvdt_x, dvdt_y, dvdt_z, dedt = self.compute_accel_energy(state)

        dxdt, dydt, dzdt = state[:, 3], state[:, 4], state[:, 5]
        dmdt = np.zeros(self.N)

        deriv_state = np.column_stack(
            (dxdt, dydt, dzdt,
             dvdt_x, dvdt_y, dvdt_z,
             dmdt, np.zeros(self.N), dedt)
        )
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

    def compute_timestep(self, state_matrix: np.ndarray, cfl: Optional[float] = None) -> float:
        """
        Adaptive timestep from a CFL condition and an acceleration limiter.

        The timestep is chosen as:
          - dt_cfl ~ (cfl * h / c_s)
          - dt_acc ~ (cfl * sqrt(h / |a|))
        and then limited to a reasonable range.

        Returns
        -------
        dt
            Adaptive timestep.
        """
        if self.h is None:
            raise ValueError("Smoothing length h is not set.")

        cfl_val = self.cfl if cfl is None else cfl
        h_val = float(self.h)

        e = state_matrix[:, 8]
        c = np.sqrt(np.maximum((self.gamma - 1.0) * e, 1e-12))
        dt_cfl = cfl_val * h_val / np.maximum(c, 1e-10)

        dvdt_x, dvdt_y, dvdt_z, _ = self.compute_accel_energy(state_matrix)
        a_mag = np.sqrt(dvdt_x ** 2 + dvdt_y ** 2 + dvdt_z ** 2)
        dt_acc = cfl_val * np.sqrt(h_val / np.maximum(a_mag, 1e-20))

        dt = float(np.clip(np.min([np.min(dt_cfl), np.min(dt_acc)]), 1e-9, 1e6))
        return dt

    # -----------------------
    # Run Simulation
    # -----------------------
    def run(self) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run the simulation for `n_steps` integration steps.

        Returns
        -------
        history
            List of state matrices (N, 9) at each stored step.
        times
            List of times corresponding to each stored state.
        """
        t: float = 0.0
        state_vec: np.ndarray = self.state_vec
        history: List[np.ndarray] = []
        times: List[float] = []
        dts: List[float] = []

        for step in range(self.n_steps):
            state = self.state_from_vector(state_vec)
            state_vec = self.vector_from_state(state)

            dt = self.dt_fixed if self.dt_fixed is not None else self.compute_timestep(state)
            state_vec = self.RK4step(self.rhs, t, state_vec, dt)
            t += dt

            state = self.state_from_vector(state_vec)
            history.append(state.copy())
            times.append(t)
            dts.append(dt)

            if step == 0 or (step + 1) % 10 == 0:
                print(f"Step {step + 1}/{self.n_steps}  t={t:.6e}  dt={dt:.3e}")

        self.final_state = state
        self.last_dt = dts[-1] if dts else None
        return history, times

    # -----------------------
    # Plot and Animation
    # -----------------------
    def plot_positions_by_density(self, state_matrix: np.ndarray, history: Optional[List[np.ndarray]] = None,
                                 filename: str = "planet_density.png") -> None:
        """
        Plot 3D particle positions colored by density and save a PNG.

        Parameters
        ----------
        state_matrix
            State matrix (N, 9) to plot.
        history
            Unused, kept for compatibility with older call sites.
        filename
            Output image filename.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        x, y, z = state_matrix[:, 0], state_matrix[:, 1], state_matrix[:, 2]
        densities = state_matrix[:, 7]

        fig = plt.figure(figsize=(8, 7), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")

        sc = ax.scatter(x, y, z, c=densities, cmap="plasma", s=8)

        cb = plt.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label("Density", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

        ax.set_xlim(-5e8, 5e8)
        ax.set_ylim(-5e8, 5e8)
        ax.set_zlim(-5e8, 5e8)

        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.set_title("Planet Simulation", color="white")

        ax.grid(False)
        ax.tick_params(colors="white")
        ax.set_box_aspect([1, 1, 1])

        ax.xaxis.pane.set_facecolor((0, 0, 0, 1))
        ax.yaxis.pane.set_facecolor((0, 0, 0, 1))
        ax.zaxis.pane.set_facecolor((0, 0, 0, 1))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, facecolor="black")
        plt.show()

    def save_gif_3d(self, history: List[np.ndarray], times: List[float], gif_name: str = "planet_sim.gif",
                    fps: int = 20, point_size: int = 8) -> None:
        """
        Save a 3D animated GIF of the simulation.

        Parameters
        ----------
        history
            List of state matrices (N, 9).
        times
            List of times corresponding to each state.
        gif_name
            Output GIF filename.
        fps
            Frames per second for the output.
        point_size
            Marker size for particles.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        densities = history[0][:, 7]
        xyz0 = history[0][:, :3]

        fig = plt.figure(figsize=(8, 7), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")
        sc = ax.scatter(xyz0[:, 0], xyz0[:, 1], xyz0[:, 2], c=densities, cmap="plasma", s=point_size)

        ax.set_xlim(-5e8, 5e8)
        ax.set_ylim(-5e8, 5e8)
        ax.set_zlim(-5e8, 5e8)

        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_zlabel("Z", color="white")
        ax.set_title("Planet Simulation", color="white")
        ax.grid(False)
        ax.tick_params(colors="white")
        ax.set_box_aspect([1, 1, 1])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        cb = plt.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label("Density", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

        def update(frame: int):
            xyz = history[frame][:, :3]
            sc._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            time_hour = times[frame] / 3600.0
            ax.set_title(f"Planet Simulation  t = {time_hour:.2f} hours", color="white")
            return (sc,)

        ani = animation.FuncAnimation(fig, update, frames=len(history), blit=False, interval=1000 / fps)
        writer = animation.PillowWriter(fps=fps)
        ani.save(gif_name, writer=writer, dpi=120, savefig_kwargs={"facecolor": "black"})
        plt.close(fig)
        print(f"Saved GIF to {gif_name}")

    def save_gif_2d(self, history: List[np.ndarray], times: List[float], gif_name: str = "planet_sim_xy.gif",
                    fps: int = 20, point_size: int = 6) -> None:
        """
        Save a 2D (XY) animated GIF of the simulation.

        Parameters
        ----------
        history
            List of state matrices (N, 9).
        times
            List of times corresponding to each state.
        gif_name
            Output GIF filename.
        fps
            Frames per second for the output.
        point_size
            Marker size for particles.
        """
        densities = history[0][:, 7]
        xy0 = history[0][:, :2]

        fig, ax = plt.subplots(figsize=(7, 7), facecolor="black")
        sc = ax.scatter(xy0[:, 0], xy0[:, 1], c=densities, cmap="plasma", s=point_size)

        ax.set_xlim(-5e8, 5e8)
        ax.set_ylim(-5e8, 5e8)

        ax.set_xlabel("X", color="white")
        ax.set_ylabel("Y", color="white")
        ax.set_title("Planet Simulation (XY)", color="white")
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

        cb = plt.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("Density", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

        def update(frame: int):
            xy = history[frame][:, :2]
            sc.set_offsets(xy)
            time_hour = times[frame] / 3600.0
            ax.set_title(f"Planet Simulation (XY)  t = {time_hour:.2f} hours", color="white")
            return (sc,)

        ani = animation.FuncAnimation(fig, update, frames=len(history), blit=False, interval=1000 / fps)
        writer = animation.PillowWriter(fps=fps)
        ani.save(gif_name, writer=writer, dpi=120, savefig_kwargs={"facecolor": "black"})
        plt.close(fig)
        print(f"Saved GIF to {gif_name}")


if __name__ == "__main__":
    sim = PlanetSimulation("Planet600.dat", n_steps=600, spin_period=50000, eta=1.3, cfl=0.3,
                           collision=True, v_collide=1.5e4)

    history, times = sim.run()

    sim.save_gif_3d(history, times, gif_name="planet_sim.gif", fps=15, point_size=6)
    sim.save_gif_2d(history, times, gif_name="planet_sim_xy.gif", fps=20, point_size=4)
    sim.plot_positions_by_density(history[-1], history=history, filename="planet_density.png")
