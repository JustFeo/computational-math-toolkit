from __future__ import annotations

from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def ensure_float_positive(x: float, name: str) -> None:
    try:
        xf = float(x)
    except Exception as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if xf <= 0.0:
        raise ValueError(f"{name} must be positive")


def ensure_1d_array(a: Array, name: str = "array") -> Array:
    arr = np.asarray(a, dtype=float).ravel()
    return arr


def create_time_grid(t0: float, t1: float, dt: float) -> Tuple[Array, float]:
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")
    if dt <= 0:
        raise ValueError("dt must be positive")
    n = int(np.ceil((t1 - t0) / dt))
    dt_eff = (t1 - t0) / n
    t = np.linspace(t0, t1, n + 1)
    return t, dt_eff


def plot_time_series(
    t: Array, Y: Array, labels: Iterable[str] | None = None, title: str | None = None
) -> None:
    """Quick plotting helper for notebooks.

    Parameters
    ----------
    t : ndarray, shape (m,)
        Time grid.
    Y : ndarray, shape (m,) or (m, n)
        Series to plot. If 2D, each column is a series.
    labels : iterable of str, optional
        Legend labels for each series.
    title : str, optional
        Plot title.
    """
    t = np.asarray(t)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        plt.plot(t, Y, label=None if labels is None else next(iter(labels)))
    else:
        n_series = Y.shape[1]
        for j in range(n_series):
            lbl = None if labels is None else list(labels)[j]
            plt.plot(t, Y[:, j], label=lbl)
    if labels is not None:
        plt.legend()
    if title:
        plt.title(title)
    plt.xlabel("t")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
