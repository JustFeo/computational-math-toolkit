from __future__ import annotations

from typing import Tuple

import numpy as np

Array = np.ndarray


def svd_decompose(A: Array) -> Tuple[Array, Array, Array]:
    """Compute singular value decomposition of a 2D array.

    A = U @ diag(S) @ Vt, with U and Vt orthonormal (full_matrices=False).

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Input matrix.

    Returns
    -------
    U : ndarray, shape (m, r)
    S : ndarray, shape (r,)
    Vt : ndarray, shape (r, n)
        Where r = min(m, n).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U, S, Vt


def pca(X: Array, n_components: int) -> Tuple[Array, Array]:
    """Principal Component Analysis via SVD.

    Rows of X are samples, columns are features. Data are centered before SVD.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data matrix.
    n_components : int
        Number of principal components to return (1 <= n_components <= rank).

    Returns
    -------
    components : ndarray, shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance : ndarray, shape (n_components,)
        Variance explained by each component.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not (1 <= n_components <= min(X.shape)):
        raise ValueError(
            "n_components must be between 1 and min(n_samples, n_features)"
        )

    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # explained variance of each PC
    n_samples = X.shape[0]
    explained_variance = (S[:n_components] ** 2) / max(n_samples - 1, 1)
    components = Vt[:n_components]
    return components, explained_variance


def eig_decompose(A: Array) -> Tuple[Array, Array]:
    """Eigen decomposition of a square matrix.

    Wraps numpy.linalg.eig. For symmetric/hermitian matrices, prefer
    ``numpy.linalg.eigh`` for numerical stability and real eigenvalues.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Square matrix.

    Returns
    -------
    w : ndarray, shape (n,)
        Eigenvalues.
    V : ndarray, shape (n, n)
        Right eigenvectors stored column-wise.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D array")
    w, V = np.linalg.eig(A)
    return w, V
