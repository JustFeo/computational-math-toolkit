from __future__ import annotations

from typing import Callable

import numpy as np

Array = np.ndarray


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    maxiter: int = 100,
) -> float:
    """Find a root of f in [a, b] using the bisection method.

    Requires f(a) and f(b) to have opposite signs.
    """
    fa = f(a)
    fb = f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    if tol <= 0:
        raise ValueError("tol must be positive")

    left, right = a, b
    for _ in range(maxiter):
        mid = 0.5 * (left + right)
        fm = f(mid)
        if abs(fm) < tol or 0.5 * (right - left) < tol:
            return mid
        if fa * fm < 0:
            right = mid
            fb = fm
        else:
            left = mid
            fa = fm
    return 0.5 * (left + right)


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-8,
    maxiter: int = 50,
) -> float:
    """Find a root of f using Newton's method.

    Parameters
    ----------
    f : callable
        Function f(x).
    df : callable
        Derivative f'(x).
    x0 : float
        Initial guess.
    tol : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    """
    x = float(x0)
    if tol <= 0:
        raise ValueError("tol must be positive")

    for _ in range(maxiter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            raise ZeroDivisionError("Derivative is zero during Newton iteration")
        dx = fx / dfx
        x_new = x - dx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def simpson(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite Simpson's rule for integrating f on [a, b].

    ``n`` must be even. Uses uniform subintervals.
    """
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's rule")
    if a == b:
        return 0.0
    n = int(n)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = np.array([f(xi) for xi in x], dtype=float)
    s = fx[0] + fx[-1] + 4.0 * fx[1:-1:2].sum() + 2.0 * fx[2:-2:2].sum()
    return s * h / 3.0


def linear_interpolate(x: Array, y: Array, x_new: Array | float) -> Array | float:
    """Piecewise linear interpolation (vectorized).

    Parameters
    ----------
    x : ndarray, shape (n,)
        Strictly increasing x-coordinates.
    y : ndarray, shape (n,)
        Values at x.
    x_new : float or ndarray
        Points at which to interpolate.

    Returns
    -------
    y_new : float or ndarray
        Interpolated values.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")
    return np.interp(x_new, x, y)  # type: ignore[return-value]
