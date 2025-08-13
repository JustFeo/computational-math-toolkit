from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

Array = np.ndarray


def solve_heat_explicit(
    u0: Array,
    alpha: float,
    dx: float,
    dt: float,
    steps: int,
) -> Array:
    """Solve the 1D heat equation u_t = alpha u_xx using an explicit scheme.

    This function uses forward Euler in time and central differences in space.
    Zero Neumann (no-flux) boundary conditions are enforced via symmetric ghost
    points (reflecting boundaries).

    Stability (CFL) condition
    -------------------------
    Let r = alpha * dt / dx^2. For the explicit scheme to be stable, require
    r <= 1/2.

    Parameters
    ----------
    u0 : ndarray, shape (n,)
        Initial condition at t=0.
    alpha : float
        Thermal diffusivity (positive).
    dx : float
        Spatial grid spacing (positive and uniform).
    dt : float
        Time step (positive).
    steps : int
        Number of time steps to advance.

    Returns
    -------
    U : ndarray, shape (steps+1, n)
        Time evolution, including the initial state as the first row.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if dx <= 0 or dt <= 0:
        raise ValueError("dx and dt must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    u = np.asarray(u0, dtype=float).ravel()
    n = u.size
    U = np.empty((steps + 1, n), dtype=float)
    U[0] = u

    r = alpha * dt / (dx * dx)
    if r > 0.5:
        warnings.warn(
            f"Explicit heat scheme likely unstable: r={r:.3f} > 0.5. Reduce dt or increase dx.",
            RuntimeWarning,
        )

    for k in range(steps):
        u_left_ghost = U[k, 1]  # Neumann: du/dx=0 -> mirror
        u_right_ghost = U[k, -2]
        lap = np.empty_like(U[k])
        # interior second differences
        lap[1:-1] = (U[k, 2:] - 2.0 * U[k, 1:-1] + U[k, :-2]) / (dx * dx)
        # boundaries with symmetric ghost points
        lap[0] = (U[k, 1] - 2.0 * U[k, 0] + u_left_ghost) / (dx * dx)
        lap[-1] = (u_right_ghost - 2.0 * U[k, -1] + U[k, -2]) / (dx * dx)
        U[k + 1] = U[k] + dt * alpha * lap

    return U


def solve_wave_1d(
    u0: Array,
    v0: Array,
    c: float,
    dx: float,
    dt: float,
    steps: int,
) -> Array:
    """Solve the 1D wave equation u_tt = c^2 u_xx with a second-order scheme.

    Scheme: central differences in time and space.
    Boundary conditions: fixed ends (Dirichlet) u=0 at both boundaries.

    Stability (CFL) condition
    -------------------------
    Let r = c * dt / dx. For stability, require r <= 1.

    Parameters
    ----------
    u0 : ndarray, shape (n,)
        Initial displacement at t=0.
    v0 : ndarray, shape (n,)
        Initial velocity at t=0.
    c : float
        Wave speed (positive).
    dx : float
        Spatial grid spacing (positive and uniform).
    dt : float
        Time step (positive).
    steps : int
        Number of time steps.

    Returns
    -------
    U : ndarray, shape (steps+1, n)
        Displacement over time, including the initial state.
    """
    if c <= 0:
        raise ValueError("c must be positive")
    if dx <= 0 or dt <= 0:
        raise ValueError("dx and dt must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    u0 = np.asarray(u0, dtype=float).ravel()
    v0 = np.asarray(v0, dtype=float).ravel()
    if u0.shape != v0.shape:
        raise ValueError("u0 and v0 must have the same shape")

    n = u0.size
    U = np.zeros((steps + 1, n), dtype=float)
    U[0] = u0

    r = c * dt / dx
    if r > 1.0:
        warnings.warn(
            f"Wave scheme likely unstable: r={r:.3f} > 1. Reduce dt or increase dx.",
            RuntimeWarning,
        )

    # First step using Taylor expansion
    lap0 = np.zeros_like(u0)
    lap0[1:-1] = (u0[2:] - 2.0 * u0[1:-1] + u0[:-2]) / (dx * dx)
    # Dirichlet: u=0 at boundaries -> enforce explicitly on all time levels
    U[0, 0] = 0.0
    U[0, -1] = 0.0
    U[1] = U[0] + dt * v0 + 0.5 * (c * c) * (dt * dt) * lap0
    U[1, 0] = 0.0
    U[1, -1] = 0.0

    for k in range(1, steps):
        lap = np.zeros(n, dtype=float)
        lap[1:-1] = (U[k, 2:] - 2.0 * U[k, 1:-1] + U[k, :-2]) / (dx * dx)
        U[k + 1] = 2.0 * U[k] - U[k - 1] + (c * c) * (dt * dt) * lap
        U[k + 1, 0] = 0.0
        U[k + 1, -1] = 0.0

    return U
