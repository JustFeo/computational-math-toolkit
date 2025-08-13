"""Computational Math Toolkit (CMT).

Core numerical algorithms for ODEs, PDEs, linear algebra, and numerical analysis.

This package is intended for learning and reference implementations, with clear
APIs, type hints, and tests.
"""

from .analysis import bisection, linear_interpolate, newton, simpson
from .la import eig_decompose, pca, svd_decompose
from .ode import euler_forward, rk4
from .pde import solve_heat_explicit, solve_wave_1d

__all__ = [
    "euler_forward",
    "rk4",
    "solve_heat_explicit",
    "solve_wave_1d",
    "svd_decompose",
    "pca",
    "eig_decompose",
    "bisection",
    "newton",
    "simpson",
    "linear_interpolate",
]
