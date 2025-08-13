from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np

from .utils import ensure_1d_array, ensure_float_positive

Array = np.ndarray


def _build_time_grid(t0: float, t1: float, dt: float) -> Tuple[Array, float]:
    """Create a uniform time grid from t0 to t1 using approximately dt.

    If ``dt`` does not divide the interval exactly, the function adjusts the step size
    slightly so that the final time is exactly ``t1``.

    Returns the time grid and the effective step size used.
    """
    ensure_float_positive(dt, "dt")
    if not np.isscalar(t0) or not np.isscalar(t1):
        raise TypeError("t0 and t1 must be scalars")
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    num_steps = int(np.ceil((t1 - t0) / dt))
    dt_eff = (t1 - t0) / num_steps
    t = np.linspace(t0, t1, num_steps + 1)
    return t, dt_eff


def euler_forward(
    fun: Callable[[float, Array, Iterable], Array],
    y0: float | Array,
    t0: float,
    t1: float,
    dt: float,
    args: Iterable = (),
) -> Tuple[Array, Array]:
    """Solve an ODE initial value problem using forward Euler.

    Parameters
    ----------
    fun : callable
        Right-hand side function f(t, y, *args) returning the time derivative.
    y0 : float or ndarray
        Initial condition. Scalars or 1D arrays are supported.
    t0, t1 : float
        Start and end times with t1 > t0.
    dt : float
        Suggested time step size. The solver will adjust slightly so that the
        final time equals t1 exactly.
    args : Iterable, optional
        Additional parameters passed to ``fun``.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    y : ndarray
        Solution values at each time. Shape is (n_steps + 1,) for scalar y0
        or (n_steps + 1, n_vars) for vector y0.
    """
    t, dt_eff = _build_time_grid(t0, t1, dt)
    y0_arr = np.atleast_1d(y0).astype(float)
    is_scalar = y0_arr.size == 1

    y = np.empty((t.size, y0_arr.size), dtype=float)
    y[0] = y0_arr

    for k in range(t.size - 1):
        tk = t[k]
        yk = y[k]
        fval = np.asarray(fun(tk, yk if not is_scalar else yk[0], *args), dtype=float)
        fval = ensure_1d_array(fval, name="fun return value")
        y[k + 1] = yk + dt_eff * fval

    if is_scalar:
        return t, y.ravel()
    return t, y


def rk4(
    fun: Callable[[float, Array, Iterable], Array],
    y0: float | Array,
    t0: float,
    t1: float,
    dt: float,
    args: Iterable = (),
) -> Tuple[Array, Array]:
    """Solve an ODE initial value problem using classical Runge-Kutta (RK4).

    Parameters
    ----------
    fun : callable
        Right-hand side function f(t, y, *args) returning the time derivative.
    y0 : float or ndarray
        Initial condition. Scalars or 1D arrays are supported.
    t0, t1 : float
        Start and end times with t1 > t0.
    dt : float
        Suggested time step size. The solver will adjust slightly so that the
        final time equals t1 exactly.
    args : Iterable, optional
        Additional parameters passed to ``fun``.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    y : ndarray
        Solution values at each time. Shape is (n_steps + 1,) for scalar y0
        or (n_steps + 1, n_vars) for vector y0.

    Examples
    --------
    Solve dy/dt = k*y with analytic solution y(t)=y0*exp(k*t)::

        >>> import numpy as np
        >>> from cmt.ode import rk4
        >>> k = 0.5
        >>> f = lambda t, y: k * y
        >>> t, y = rk4(f, 1.0, 0.0, 2.0, 1e-3)
        >>> np.allclose(y[-1], np.exp(k * 2.0), rtol=1e-4)
        True
    """
    t, dt_eff = _build_time_grid(t0, t1, dt)
    y0_arr = np.atleast_1d(y0).astype(float)
    is_scalar = y0_arr.size == 1

    y = np.empty((t.size, y0_arr.size), dtype=float)
    y[0] = y0_arr

    for k in range(t.size - 1):
        tk = t[k]
        yk = y[k]

        def _f(tt: float, yy: Array) -> Array:
            out = np.asarray(
                fun(tt, yy if not is_scalar else yy[0], *args), dtype=float
            )
            return ensure_1d_array(out, name="fun return value")

        k1 = _f(tk, yk)
        k2 = _f(tk + 0.5 * dt_eff, yk + 0.5 * dt_eff * k1)
        k3 = _f(tk + 0.5 * dt_eff, yk + 0.5 * dt_eff * k2)
        k4 = _f(tk + dt_eff, yk + dt_eff * k3)

        y[k + 1] = yk + (dt_eff / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    if is_scalar:
        return t, y.ravel()
    return t, y
