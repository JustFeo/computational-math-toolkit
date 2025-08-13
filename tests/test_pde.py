import numpy as np

from cmt.pde import solve_heat_explicit


def test_heat_neumann_mass_conservation():
    # domain
    n = 101
    dx = 1.0 / (n - 1)
    x = np.linspace(0.0, 1.0, n)
    alpha = 0.1
    dt = 0.2 * dx * dx / alpha  # stable r=0.2
    steps = 200
    # initial gaussian centered
    u0 = np.exp(-200.0 * (x - 0.5) ** 2)
    mass0 = u0.sum() * dx
    U = solve_heat_explicit(u0, alpha=alpha, dx=dx, dt=dt, steps=steps)
    mass_final = U[-1].sum() * dx
    assert abs(mass_final - mass0) <= 5e-3
