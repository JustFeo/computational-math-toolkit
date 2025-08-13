import numpy as np

from cmt.ode import rk4


def test_rk4_exponential_growth():
    k = 0.5
    f = lambda t, y: k * y
    t0, t1, dt = 0.0, 2.0, 1e-3
    t, y = rk4(f, y0=1.0, t0=t0, t1=t1, dt=dt)
    analytic = np.exp(k * t)
    error = np.max(np.abs(y - analytic))
    assert error <= 1e-4
