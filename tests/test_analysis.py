from math import pi, sqrt

import numpy as np

from cmt.analysis import bisection, newton, simpson


def test_bisection_sqrt2():
    f = lambda x: x * x - 2.0
    root = bisection(f, 0.0, 2.0, tol=1e-10, maxiter=200)
    assert abs(root - sqrt(2.0)) <= 1e-8


def test_newton_sqrt2():
    f = lambda x: x * x - 2.0
    df = lambda x: 2.0 * x
    root = newton(f, df, x0=1.0, tol=1e-12, maxiter=50)
    assert abs(root - sqrt(2.0)) <= 1e-8


def test_simpson_sin():
    res = simpson(np.sin, 0.0, pi, n=1000)
    assert abs(res - 2.0) <= 1e-8
